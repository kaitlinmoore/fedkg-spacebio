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

from fedkg_spacebio.federation import stratum
from fedkg_spacebio.federation.site_de import GENE_FIELDS, SiteSummary, validate_summary

TASK_NAME = "site_de_summary"
SUMMARY_META_KEY = "fedkg_site_summary"


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


class MetaAnalysisController(ModelController):
    def __init__(self, results_dir: str, expected_clients: list[str],
                 drop_variants: list[str] | None = None, task_name: str = TASK_NAME,
                 timeout: int = 0):
        super().__init__(persistor_id="")
        self.results_dir = results_dir
        self.expected_clients = sorted(expected_clients)
        self.drop_variants = list(drop_variants or [])
        self.task_name = task_name
        self.timeout = timeout

    def run(self):
        clients = self.sample_clients(None)
        if sorted(clients) != self.expected_clients:
            self.panic(f"connected clients {sorted(clients)} != expected {self.expected_clients}")
            return
        task = FLModel(params={"task": self.task_name}, current_round=0, total_rounds=1)
        results = self.send_model_and_wait(task_name=self.task_name, targets=clients,
                                           min_responses=len(clients), timeout=self.timeout,
                                           data=task)
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


class MetaAnalysisRecipe(Recipe):
    """One-round DE meta-analysis job: our controller on the server, a script on each client."""

    def __init__(self, *, name: str, clients: list[str], client_script: str,
                 client_args: str, results_dir: str, drop_variants: list[str] | None = None):
        job = FedJob(name=name, min_clients=len(clients))
        job.to_server(MetaAnalysisController(results_dir=results_dir, expected_clients=clients,
                                             drop_variants=drop_variants))
        job.to_clients(ScriptRunner(script=client_script, script_args=client_args,
                                    framework=FrameworkType.RAW,
                                    server_expected_format=ExchangeFormat.RAW,
                                    params_transfer_type=TransferType.FULL),
                       tasks=[TASK_NAME])
        super().__init__(job)


__all__ = ["GENE_FIELDS", "MetaAnalysisController", "MetaAnalysisRecipe", "SUMMARY_META_KEY",
           "TASK_NAME", "load_summaries", "summary_from_result", "variants_for", "write_pooled",
           "write_summaries"]
