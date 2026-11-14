"""Shared helpers for the policy test suite (CONTRACTS §5, §6).

Not a test module (no ``test_`` prefix). The tests import it directly; pytest puts ``tests/`` on
``sys.path`` because the folder has no ``__init__.py``.

What an "environment" is
------------------------
Each test builds a small world under ``tmp_path``:
- **policies:** a profile folder, usually the demo's own ``policies/<profile>/`` (so the suite
  tests exactly the files the demo uses);
- **manifests:** minimal site manifests (CONTRACTS §3), one per client, written from a committed
  client list with the ``policy_id`` the profile gives each site. Only the fields the evaluator
  reads are present. Real manifests also list samples and files, which the policy check
  never needs;
- **a policy root** (``tmp_path / "policy"``), standing in for ``$FEDKG_WORKDIR/policy``. Gene
  plans, decision records and the release log live under it;
- **the profile's active release log.**

Specs are built by ``make_spec`` from a gene → contributing-clients mapping. It writes the gene
plan in canonical form and fills in the real hashes, so a test changes only what it is about.

Fake genes: ``FAKE:0001`` in CONTRACTS §5.5 is ``9000000001`` here (CONTRACTS §1: integer fake
gene IDs start at 9,000,000,001).
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fedkg_spacebio.federation.gene_plan import read_gene_list, write_gene_list, write_gene_plan
from fedkg_spacebio.federation.jobspec import parse_job_spec
from fedkg_spacebio.policy.registry import load_registry
from fedkg_spacebio.policy.release_log import active_log
from fedkg_spacebio.policy.schema import load_profile

REPO = Path(__file__).resolve().parents[1]
POLICIES = REPO / "policies"
FIX = REPO / "tests" / "fixtures" / "policies"
SPECS = REPO / "tests" / "fixtures" / "jobspecs"

G1, G3, G4 = 9000000001, 9000000003, 9000000004  # FAKE:0001, FAKE:0003, FAKE:0004

# Example B's columns (CONTRACTS §5.5): A..D in the ISS stratum.
A, B, C, D = ISS = ["SpaceX-4", "SpaceX-8", "SpaceX-13-ISST", "SpaceX-16-ISST"]
EARTH = ["SpaceX-13-LAR", "SpaceX-16-LAR", "SpaceX-21"]
FEDAVG = ["FAKE-SITE-1", "FAKE-SITE-2", "FAKE-SITE-3"]

FULL_SET = ["fe", "fe_het", "reml_hksj", "v3"]  # the phase 3a method set of ROADMAP §1 step 3


def client_list(name: str = "clients_liver.json") -> list[dict]:
    return json.loads((FIX / name).read_text())["clients"]


def write_manifests(out_dir: Path, clients: list[dict], policy_ids: Mapping[str, str],
                    overrides: Mapping[str, Mapping] | None = None) -> Path:
    """One minimal manifest per client. ``overrides`` maps client → fields to set (None deletes)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for rec in clients:
        m = {k: v for k, v in rec.items()}
        m["policy_id"] = policy_ids[rec["site"]]
        for k, v in (overrides or {}).get(rec["client"], {}).items():
            if v is None:
                m.pop(k, None)
            else:
                m[k] = v
        (out_dir / f"{rec['client']}.json").write_text(json.dumps(m, indent=1))
    return out_dir


@dataclass
class Env:
    root: Path  # the policy root ($FEDKG_WORKDIR/policy stand-in)
    profile_name: str
    profile: object
    registry: dict
    log: object
    manifest_dir: Path


def make_env(tmp_path: Path, profile: str = "restricted", *, policies_dir: Path | None = None,
             clients_file: str = "clients_liver.json",
             overrides: Mapping[str, Mapping] | None = None) -> Env:
    prof = load_profile(policies_dir or POLICIES / profile)
    clients = client_list(clients_file)
    mdir = write_manifests(tmp_path / "manifests" / profile, clients, prof.site_policy_ids(),
                           overrides)
    registry = load_registry(mdir, clients=[c["client"] for c in clients])
    root = tmp_path / "policy"
    return Env(root=root, profile_name=profile, profile=prof, registry=registry,
               log=active_log(profile, root=root), manifest_dir=mdir)


def plan_frame(rows: Mapping[int, Iterable[str]], clients: Iterable[str]) -> pd.DataFrame:
    """Gene plan as a frame: index ncbi_gene, one bool column per client (sorted by name)."""
    cols = sorted(clients)
    data = {c: [c in set(rows[g]) for g in sorted(rows)] for c in cols}
    return pd.DataFrame(data, index=pd.Index(sorted(rows), name="ncbi_gene"))


_counter = {"n": 0}


def make_spec(env: Env, *, clients: list[str], plan: Mapping[int, Iterable[str]],
              task: str = "de_meta", methods: list[str] | None = None,
              outputs: str | Mapping[str, str] = "masked_contribution",
              aggregation: Mapping | None = None, weighting: Mapping | None | str = "default",
              stratum: str | None = None, counts_variant: str = "standard",
              inclusion: Mapping | None = None, genes_exclude: list[int] | None = None,
              exclude_decision: str = "dec:FAKE-0000-1", job_id: str | None = None,
              edit: Mapping | None = None, drop: Iterable[str] = ()):
    """Build, write and parse a spec. ``edit`` overrides top-level fields after building."""
    _counter["n"] += 1
    name = f"t{_counter['n']:04d}"
    methods = methods or (["fedavg"] if task == "fedavg_classifier" else
                          ["clock_plane"] if task == "clock_plane" else ["fe"])
    if isinstance(outputs, str):
        outputs = {c: outputs for c in clients}
    if aggregation is None:
        masked = all(o == "masked_contribution" for o in outputs.values())
        aggregation = ({"strategy": "secure_sum", "fixed_point_bits": 32, "modulus_bits": 64}
                       if masked else {"strategy": "plain"})
    if weighting == "default":
        weighting = ({"scheme": "sample_count"} if task == "fedavg_classifier"
                     else {"scheme": "inverse_variance"})
    if stratum is None:
        stratum = env.registry[clients[0]].stratum if clients[0] in env.registry else "iss"
    plan_ref = f"plans/{name}/gene_plan.csv"
    plan_sha = write_gene_plan(plan_frame(plan, clients), env.root / plan_ref)
    spec = {
        "schema": "job-spec/1",
        "job_id": job_id,
        "task": task,
        "question": "test",
        "stratum": stratum,
        "clients": list(clients),
        "methods": list(methods),
        "counts_variant": counts_variant,
        "inclusion": dict(inclusion or {"rule": "mean10_finite", "min_sites_per_gene": "all"}),
        "gene_plan": {"ref": plan_ref, "sha256": plan_sha},
        "genes_exclude": None,
        "outputs": dict(outputs),
        "aggregation": dict(aggregation),
        "weighting": weighting,
        "rounds": 20 if task == "fedavg_classifier" else 1,
        "split": ({"scheme": "loso", "folds_ref": "splits/FAKE-folds.json",
                   "sha256": "FAKE-sha256-0010"} if task == "fedavg_classifier" else None),
        "contact_windows": None,
        "provenance": {
            "kg_version": "SPOKE-GeneLab v0.3.1+20260311",
            "census_access_date": "2026-11-12",
            "annotation_sha256": "FAKE-sha256-0004",
            "manifest_sha256": {c: (env.registry[c].manifest_sha256 if c in env.registry
                                    else "FAKE-sha256-0099") for c in clients},
            "inclusion_flags_sha256": ({c: f"FAKE-sha256-{1000 + i:04d}"
                                        for i, c in enumerate(clients)}
                                       if task == "de_meta" else {}),
        },
        "policy_decision": None,
    }
    if genes_exclude is not None:
        ref = f"plans/{name}/genes_exclude.csv"
        sha = write_gene_list(genes_exclude, env.root / ref)
        spec["genes_exclude"] = {"decision_id": exclude_decision, "ref": ref, "sha256": sha}
    for k, v in (edit or {}).items():
        spec[k] = copy.deepcopy(v)
    for k in drop:
        spec.pop(k, None)
    return parse_job_spec(spec)


def check(env: Env, spec, **kw):
    """Run the policy check against the env's profile, registry and log."""
    from fedkg_spacebio.policy.evaluate import check_job  # imported here: only these tests need it

    return check_job(spec, env.profile, env.registry, env.log, **kw)


def evals(decision) -> dict:
    """rule_id → evaluation."""
    return {e.rule_id: e for e in decision.evaluations}


def rule_order(decision) -> list[str]:
    return [e.rule_id for e in decision.evaluations]


def failures(decision) -> list[str]:
    return [e.rule_id for e in decision.evaluations if e.result == "fail"]


def failed_genes(env: Env, evaluation) -> list[int]:
    """The gene list a gene-scope failure points to, read with its hash checked."""
    return [int(g) for g in read_gene_list(env.root / evaluation.genes_failed_ref,
                                           evaluation.sha256)]
