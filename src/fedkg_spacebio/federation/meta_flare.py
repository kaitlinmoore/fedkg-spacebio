"""FLARE glue for phase 3a: a one-round controller that pools site DE summaries.

Checked against FLARE 2.9.1's installed source.

Why a custom controller (plan, approved in WS-07)
-------------------------------------------------
- ``FedTaskRecipe`` (model-free, one round) is the closest built-in. But its
  ``CmdTaskController`` **discards** client results, so pooling would have to happen outside
  FLARE from files the clients wrote, and Check 1 (federated = central) would test nothing.
- ``FedStatsRecipe`` computes named-feature statistics (count, mean, histogram) with noise and
  min-count filters. That's the wrong shape for ~15k-gene arrays of estimates, and its filters
  would alter them.
- The NumPy FedAvg recipe brings a persistor and a model lifecycle we don't have.

So ``MetaAnalysisController`` subclasses ``ModelController`` (as ``CmdTaskController`` does) and
in ``run()``:
1. sends one task to **every** client and waits for all of them (no partial pooling);
2. for each result, reads our metadata from ``meta[SUMMARY_META_KEY]``. FLARE adds its own keys
   (``client_name``, ``props``) to ``meta``, so ours are nested under one key. It checks that
   the client the summary claims to be is the client FLARE received it from, and runs
   ``validate_summary`` again on the server side;
3. calls ``stratum.pool_variants`` (our code), and writes the received summaries and the pooled
   tables under ``results_dir``.

The recipe mirrors ``FedTaskRecipe`` (a FedJob with the controller on the server and a
``ScriptRunner`` on each client), with our controller in place of ``CmdTaskController``.

Weighting: this is inverse-variance pooling, not FedAvg, so D-017's ``NUM_STEPS_CURRENT_ROUND``
isn't used. Each site still reports ``n_flight`` and ``n_ground``.

Job-spec runs (WS-13; CONTRACTS §6)
-----------------------------------
A run started from a job spec uses one of two further controllers, picked by the spec's
aggregation:
- ``SecureSumController`` (``secure_sum``): each client sends masked fixed-point vectors
  (``secure_sum.py``) under ``meta[MASKED_META_KEY]``. The server checks that **every** expected
  client answered (a sum over fewer leaves masks uncancelled; there is no dropout recovery), that
  each used the same plan, exclusions, f, m and row count, writes what arrived to
  ``server/received/``, and pools from totals only (``spec_server.pool_secure_sum``). It never
  receives a per-site estimate.
- ``SpecPlainController`` (``plain``): per-site summaries restricted to each client's plan genes,
  pooled over the plan's rows (``spec_server.pool_plain_spec``).
Both write ``server/per_gene.csv.gz`` with only the requested methods' columns, its schema file
(``methods_run``, ``aggregation``) and ``server/run.json``. The flag-driven controller above is
unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from nvflare import FedJob
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.workflows.model_controller import ModelController
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.fuel.utils.constants import FrameworkType
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.recipe.spec import Recipe

from fedkg_spacebio.federation import secure_sum as ss
from fedkg_spacebio.federation import spec_server, stratum
from fedkg_spacebio.federation.gene_plan import exclude_genes, read_gene_list, read_gene_plan
from fedkg_spacebio.federation.site_de import GENE_FIELDS, SiteSummary, validate_summary

TASK_NAME = "site_de_summary"
SUMMARY_META_KEY = "fedkg_site_summary"
MASKED_META_KEY = "fedkg_masked_contribution"


def summary_from_result(result: FLModel) -> SiteSummary:
    """Server side: turn one FLARE result back into a validated ``SiteSummary``."""
    site_meta = result.meta.get(SUMMARY_META_KEY)
    if site_meta is None:
        raise ValueError(f"result from {result.meta.get('client_name')} has no {SUMMARY_META_KEY}")
    sender = result.meta.get("client_name")
    if sender is not None and site_meta.get("client") != sender:
        raise ValueError(f"summary claims client {site_meta.get('client')!r} but came from "
                         f"{sender!r}")
    genes = {k: np.asarray(v) for k, v in (result.params or {}).items()}
    validate_summary(genes, site_meta)
    return SiteSummary(genes, site_meta)


def write_summaries(summaries: dict[str, SiteSummary], results_dir: Path) -> None:
    """The summaries as received: exact arrays (``.npz``), a readable CSV, and the metadata."""
    sdir = Path(results_dir) / "site_summaries"
    sdir.mkdir(parents=True, exist_ok=True)
    for client, s in summaries.items():
        np.savez(sdir / f"{client}.npz", **s.genes)
        s.to_frame().to_csv(sdir / f"{client}.csv.gz", float_format="%.17g")
        (sdir / f"{client}.meta.json").write_text(json.dumps(s.meta, indent=1))


def load_summaries(results_dir: Path) -> dict[str, SiteSummary]:
    """Read summaries written by ``write_summaries`` (or a client's ``sent/`` copy)."""
    out = {}
    for npz in sorted(Path(results_dir).glob("*.npz")):
        client = npz.stem
        with np.load(npz) as z:
            genes = {k: z[k] for k in z.files}
        meta = json.loads((npz.parent / f"{client}.meta.json").read_text())
        validate_summary(genes, meta)
        out[client] = SiteSummary(genes, meta)
    return out


def write_pooled(summaries: dict[str, SiteSummary], pooled: dict[str, pd.DataFrame],
                 results_dir: Path, source: str) -> None:
    results_dir = Path(results_dir)
    for name, table in pooled.items():
        vdir = results_dir / name
        vdir.mkdir(parents=True, exist_ok=True)
        table.to_csv(vdir / "meta_genes.csv.gz", index=False, float_format="%.17g")
        (vdir / "variant.json").write_text(json.dumps(table.attrs, indent=1))
    (results_dir / "pooling_run.json").write_text(json.dumps({
        "source": source, "clients": sorted(summaries),
        "n": {c: [s.meta["n_flight"], s.meta["n_ground"]] for c, s in summaries.items()},
        "variants": {k: {"n_genes": int(len(v)), **v.attrs} for k, v in pooled.items()},
    }, indent=1))


def variants_for(drop_variants: list[str]) -> tuple[stratum.Variant, ...]:
    extra = tuple(stratum.Variant(f"primary_without_{c}", "tested_mean10", None, (c,))
                  for c in drop_variants)
    return stratum.VARIANTS + extra


class _OneRoundController(ModelController):
    """One task to every expected client; wait for all of them."""

    def __init__(self, results_dir: str, expected_clients: list[str], task_name: str = TASK_NAME,
                 timeout: int = 0):
        super().__init__(persistor_id="")
        self.results_dir = results_dir
        self.expected_clients = sorted(expected_clients)
        self.task_name = task_name
        self.timeout = timeout

    def _collect(self) -> list[FLModel] | None:
        clients = self.sample_clients(None)
        if sorted(clients) != self.expected_clients:
            self.panic(f"connected clients {sorted(clients)} != expected {self.expected_clients}")
            return None
        task = FLModel(params={"task": self.task_name}, current_round=0, total_rounds=1)
        return self.send_model_and_wait(task_name=self.task_name, targets=clients,
                                        min_responses=len(clients), timeout=self.timeout,
                                        data=task)


class MetaAnalysisController(_OneRoundController):
    def __init__(self, results_dir: str, expected_clients: list[str],
                 drop_variants: list[str] | None = None, task_name: str = TASK_NAME,
                 timeout: int = 0):
        super().__init__(results_dir, expected_clients, task_name, timeout)
        self.drop_variants = list(drop_variants or [])

    def run(self):
        results = self._collect()
        if results is None:
            return
        summaries = {}
        for r in results:
            s = summary_from_result(r)
            summaries[s.meta["client"]] = s
        if sorted(summaries) != self.expected_clients:
            self.panic(f"got summaries from {sorted(summaries)}, expected "
                       f"{self.expected_clients}; refusing to pool a partial stratum")
            return
        write_summaries(summaries, Path(self.results_dir))  # before pooling, so transport
        frames = {c: s.to_frame() for c, s in summaries.items()}  # is checkable on its own
        pooled = stratum.pool_variants(frames, variants_for(self.drop_variants))
        write_pooled(summaries, pooled, Path(self.results_dir), source="flare")
        self.info(f"Finished {self.task_name}: pooled {len(summaries)} sites; "
                  f"primary genes = {len(pooled['primary'])}")


class _SpecController(_OneRoundController):
    """Shared set-up of the two job-spec controllers. Every argument is plain JSON (FLARE writes
    the controller's arguments into the server's job config)."""

    def __init__(self, results_dir: str, expected_clients: list[str], *, job_id: str,
                 job_spec_sha256: str, methods: list[str], flag: str, plan_path: str,
                 plan_sha256: str, exclude_path: str | None = None,
                 exclude_sha256: str | None = None, inclusion_flags_sha256: dict | None = None,
                 provenance: dict | None = None, annotation_path: str | None = None,
                 verify_annotation: bool = True, task_name: str = TASK_NAME, timeout: int = 0):
        super().__init__(results_dir, expected_clients, task_name, timeout)
        self.job_id, self.job_spec_sha256 = job_id, job_spec_sha256
        self.methods, self.flag = list(methods), flag
        self.plan_path, self.plan_sha256 = plan_path, plan_sha256
        self.exclude_path, self.exclude_sha256 = exclude_path, exclude_sha256
        self.inclusion_flags_sha256 = dict(inclusion_flags_sha256 or {})
        self.provenance = dict(provenance or {})
        self.annotation_path, self.verify_annotation = annotation_path, verify_annotation

    def _rows(self):
        """The plan minus exclusions: public coordinator files, hash-checked."""
        rows = read_gene_plan(self.plan_path, self.plan_sha256)
        if sorted(rows.columns) != self.expected_clients:
            raise ValueError(f"plan columns {sorted(rows.columns)} != {self.expected_clients}")
        if self.exclude_path:
            rows = exclude_genes(rows, read_gene_list(self.exclude_path, self.exclude_sha256))
        return rows

    def _symbols(self):
        if not self.annotation_path:
            return None
        from fedkg_spacebio.harmonize.gene_map import load_map
        gmap = load_map(Path(self.annotation_path), verify=self.verify_annotation)
        return gmap.table.dropna(subset=["ncbi_gene"]).set_index("ncbi_gene")["symbol"]

    def _check_flags_hash(self, client: str, meta_: dict) -> None:
        want = self.inclusion_flags_sha256.get(client)
        if want is not None and meta_.get("inclusion_flags_sha256") != want:
            raise ValueError(f"{client}: inclusion_flags_sha256 doesn't match the job spec")

    def _run_info(self, site_n: dict, extra: dict) -> dict:
        return {"source": "flare", "job_id": self.job_id,
                "job_spec_sha256": self.job_spec_sha256, "clients": self.expected_clients,
                "n": {c: list(v) for c, v in site_n.items()}, "gene_plan_sha256": self.plan_sha256,
                "genes_exclude_sha256": self.exclude_sha256, **extra}

    def _provenance(self, site_n: dict) -> dict:
        return {**self.provenance, "job_id": self.job_id,
                "job_spec_sha256": self.job_spec_sha256, "clients": self.expected_clients,
                "n_per_arm": {c: list(v) for c, v in site_n.items()},
                "inclusion_flag": self.flag, "gene_plan_sha256": self.plan_sha256,
                "genes_exclude_sha256": self.exclude_sha256}


class SecureSumController(_SpecController):
    """Pools masked contributions: totals only, every client required (see the module notes)."""

    # Every argument is spelled out: FLARE writes a component's arguments into the job config
    # from its class's own __init__ signature, so arguments passed through **kwargs would be lost.
    def __init__(self, results_dir: str, expected_clients: list[str], *, job_id: str,
                 job_spec_sha256: str, methods: list[str], flag: str, plan_path: str,
                 plan_sha256: str, fixed_point_bits: int, modulus_bits: int,
                 quantities: list[str], exclude_path: str | None = None,
                 exclude_sha256: str | None = None, inclusion_flags_sha256: dict | None = None,
                 provenance: dict | None = None, annotation_path: str | None = None,
                 verify_annotation: bool = True, task_name: str = TASK_NAME, timeout: int = 0):
        super().__init__(results_dir, expected_clients, job_id=job_id,
                         job_spec_sha256=job_spec_sha256, methods=methods, flag=flag,
                         plan_path=plan_path, plan_sha256=plan_sha256, exclude_path=exclude_path,
                         exclude_sha256=exclude_sha256,
                         inclusion_flags_sha256=inclusion_flags_sha256, provenance=provenance,
                         annotation_path=annotation_path, verify_annotation=verify_annotation,
                         task_name=task_name, timeout=timeout)
        self.fixed_point_bits, self.modulus_bits = fixed_point_bits, modulus_bits
        self.quantities = list(quantities)

    def run(self):
        results = self._collect()
        if results is None:
            return
        try:
            params = ss.Params(self.fixed_point_bits, self.modulus_bits)
            rows = self._rows()
            expected = {"job_id": self.job_id, "gene_plan_sha256": self.plan_sha256,
                        "genes_exclude_sha256": self.exclude_sha256, "n_rows": len(rows),
                        "fixed_point_bits": self.fixed_point_bits,
                        "modulus_bits": self.modulus_bits, "quantities": self.quantities}
            contributions = {}
            server = Path(self.results_dir) / "server"
            for r in results:
                m = (r.meta or {}).get(MASKED_META_KEY)
                sender = (r.meta or {}).get("client_name")
                if m is None:
                    raise ValueError(f"result from {sender} has no {MASKED_META_KEY}")
                if sender is not None and m.get("client") != sender:
                    raise ValueError(f"contribution claims {m.get('client')!r}, came from "
                                     f"{sender!r}")
                arrays = {k: np.asarray(v) for k, v in (r.params or {}).items()}
                # Record what arrived before any check, so a refused payload can be inspected.
                spec_server.write_received(server / "received", m["client"], arrays, m)
                self._check_flags_hash(m["client"], m)
                contributions[m["client"]] = (arrays, m)
            if sorted(contributions) != self.expected_clients:
                raise ValueError(f"contributions from {sorted(contributions)}, expected "
                                 f"{self.expected_clients}; refusing a partial sum")
            pooled, info = spec_server.pool_secure_sum(contributions, rows, params, expected)
            site_n = {c: (m["n_flight"], m["n_ground"]) for c, (_, m) in contributions.items()}
            aggregation = {"strategy": "secure_sum", **params.to_dict(),
                           "quantities": self.quantities, "mask_scheme": ss.MASK_SCHEME,
                           "simulation": "pairwise masks with seeds from the job config; all "
                                         "clients in one simulator; no dropout recovery; not a "
                                         "security claim (CONTRACTS §6.3)"}
            spec_server.write_outputs(server, pooled, methods=self.methods, rows=rows,
                                      site_n=site_n, aggregation=aggregation,
                                      provenance=self._provenance(site_n),
                                      run_info=self._run_info(site_n, info),
                                      symbols=self._symbols())
        except ValueError as exc:  # includes SecureSumError and SpecPoolError
            self.panic(f"secure sum refused: {exc}")
            return
        self.info(f"Finished secure sum: {len(contributions)} clients, {len(rows)} genes")


class SpecPlainController(_SpecController):
    """Pools per-site summaries restricted to each client's plan genes."""

    def run(self):
        results = self._collect()
        if results is None:
            return
        try:
            rows = self._rows()
            summaries = {}
            server = Path(self.results_dir) / "server"
            for r in results:
                # The hash in a plain spec summary covers the flags actually sent (its plan
                # genes); the client checked its full flags against the spec before filtering.
                s = summary_from_result(r)
                summaries[s.meta["client"]] = s
            write_summaries(summaries, server / "received")
            pooled, frames = spec_server.pool_plain_spec(summaries, rows, self.flag,
                                                         self.methods)
            site_n = {c: (s.meta["n_flight"], s.meta["n_ground"]) for c, s in summaries.items()}
            spec_server.write_outputs(server, pooled, methods=self.methods, rows=rows,
                                      site_n=site_n, aggregation={"strategy": "plain"},
                                      provenance=self._provenance(site_n),
                                      run_info=self._run_info(site_n, {"n_rows": len(rows)}),
                                      frames=frames, symbols=self._symbols())
        except ValueError as exc:
            self.panic(f"plain spec run refused: {exc}")
            return
        self.info(f"Finished plain spec run: {len(summaries)} clients, {len(rows)} genes")


class MetaAnalysisRecipe(Recipe):
    """One-round DE meta-analysis job: our controller on the server, a script on each client.

    ``controller`` defaults to the flag-driven ``MetaAnalysisController``; spec runs pass a
    ``SecureSumController`` or ``SpecPlainController``."""

    def __init__(self, *, name: str, clients: list[str], client_script: str,
                 client_args: str, results_dir: str, drop_variants: list[str] | None = None,
                 controller: ModelController | None = None):
        job = FedJob(name=name, min_clients=len(clients))
        job.to_server(controller or MetaAnalysisController(
            results_dir=results_dir, expected_clients=clients, drop_variants=drop_variants))
        job.to_clients(ScriptRunner(script=client_script, script_args=client_args,
                                    framework=FrameworkType.RAW,
                                    server_expected_format=ExchangeFormat.RAW,
                                    params_transfer_type=TransferType.FULL),
                       tasks=[TASK_NAME])
        super().__init__(job)


__all__ = ["GENE_FIELDS", "MASKED_META_KEY", "MetaAnalysisController", "MetaAnalysisRecipe",
           "SUMMARY_META_KEY", "SecureSumController", "SpecPlainController", "TASK_NAME",
           "load_summaries", "summary_from_result", "variants_for", "write_pooled",
           "write_summaries"]
