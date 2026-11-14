"""Time the policy check on the real phase 3a inclusion flags (full genome, restricted profile).

    # WSL/Ubuntu, after `fedkg`; needs the phase 3a FLARE runs (sent/*.npz) and manifests
    uv run python jobs/policy_timing/time_check.py

Reads only the inclusion flags each client already sent (``tested_*`` columns of
``results/phase3a/<stratum>/standard/flare/sent/<client>.npz``, the site_metadata of CONTRACTS
§5.1) and the open-profile manifests. Never reads estimates. Everything it writes (restricted
manifests, gene plans, the release log, decision records) goes to a temporary folder that is
removed afterwards, so the real ``$FEDKG_WORKDIR/policy`` log is never touched and no existing
run is backfilled into a log.

Runs example B's sequence (CONTRACTS §5.5) at full scale in the ISS stratum, and an Earth
leave-one-out, appending each allowed release, and prints the decision and the time per check.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fedkg_spacebio.federation.gene_plan import build_gene_plan, write_gene_list, write_gene_plan
from fedkg_spacebio.federation.jobspec import parse_job_spec
from fedkg_spacebio.policy.evaluate import check_job
from fedkg_spacebio.policy.registry import load_registry
from fedkg_spacebio.policy.release_log import active_log
from fedkg_spacebio.policy.schema import load_profile
from fedkg_spacebio.sites.inclusion import FLAGS, inclusion_flags_sha256

REPO = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("FEDKG_WORKDIR", str(Path.home() / "fedkg-work")))
ISS = ["SpaceX-4", "SpaceX-8", "SpaceX-13-ISST", "SpaceX-16-ISST"]
EARTH = ["SpaceX-13-LAR", "SpaceX-16-LAR", "SpaceX-21"]


def load_flags(stratum: str, client: str) -> pd.DataFrame:
    z = np.load(WORK / "results" / "phase3a" / stratum / "standard" / "flare" / "sent"
                / f"{client}.npz")
    return pd.DataFrame({"ncbi_gene": z["ncbi_gene"].astype(np.int64),
                         **{f: z[f].astype(bool) for f in FLAGS}})


def restricted_manifests(out: Path, policy_ids: dict[str, str]) -> Path:
    """The open manifests with each site's policy_id set as job.py --profile restricted would."""
    out.mkdir(parents=True)
    for p in sorted((WORK / "phase3a" / "manifests" / "standard").glob("*.json")):
        m = json.loads(p.read_text())
        m["policy_id"] = policy_ids[m["site"]]
        (out / p.name).write_text(json.dumps(m, indent=1))
    return out


def main() -> None:
    profile = load_profile(REPO / "policies" / "restricted")
    flags = {**{c: load_flags("iss", c) for c in ISS}, **{c: load_flags("earth", c)
                                                           for c in EARTH}}
    flag_sha = {c: inclusion_flags_sha256(f) for c, f in flags.items()}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mdir = restricted_manifests(tmp / "manifests", profile.site_policy_ids())
        registry = load_registry(mdir, clients=ISS + EARTH)
        root = tmp / "policy"
        log = active_log("restricted", root=root)
        n = {"i": 0}

        def spec(clients, min_sites, exclude=None, exclude_decision=None):
            n["i"] += 1
            stratum = registry[clients[0]].stratum
            plan = build_gene_plan({c: flags[c] for c in clients}, "mean10_finite", min_sites)
            ref = f"plans/t{n['i']}/gene_plan.csv"
            d = {"schema": "job-spec/1", "task": "de_meta", "question": "timing",
                 "stratum": stratum, "clients": clients, "methods": ["fe"],
                 "counts_variant": "standard",
                 "inclusion": {"rule": "mean10_finite", "min_sites_per_gene": min_sites},
                 "gene_plan": {"ref": ref, "sha256": write_gene_plan(plan, root / ref)},
                 "outputs": {c: "masked_contribution" for c in clients},
                 "aggregation": {"strategy": "secure_sum", "fixed_point_bits": 32,
                                 "modulus_bits": 64},
                 "weighting": {"scheme": "inverse_variance"}, "rounds": 1,
                 "provenance": {"kg_version": "SPOKE-GeneLab v0.3.1+20260311",
                                "census_access_date": "2026-11-12",
                                "annotation_sha256": "FAKE-sha256-0004",
                                "manifest_sha256": {c: registry[c].manifest_sha256
                                                    for c in clients},
                                "inclusion_flags_sha256": {c: flag_sha[c] for c in clients}}}
            if exclude is not None:
                eref = f"plans/t{n['i']}/genes_exclude.csv"
                d["genes_exclude"] = {"decision_id": exclude_decision, "ref": eref,
                                      "sha256": write_gene_list(exclude, root / eref)}
            return parse_job_spec(d), len(plan)

        def step(name, s_and_n):
            s, n_genes = s_and_n
            t0 = time.perf_counter()
            minted, dec = check_job(s, profile, registry, log)
            dt = time.perf_counter() - t0
            if dec.outcome == "allow":
                log.append(minted, dec)
            failed = {e.rule_id: e.genes_failed for e in dec.evaluations if e.result == "fail"}
            print(f"{name:34s} {n_genes:>6} genes  {dt:6.2f} s  {dec.outcome:5s}  "
                  f"{dec.deciding_rule or ''}  {failed or ''}")
            return dec

        print(f"{'step':34s} {'plan':>12}  {'check':>8}  outcome")
        step("R1 ISS complete case", spec(ISS, "all"))
        step("R2 ISS >= 3 sites", spec(ISS, 3))
        step("J3 ISS without SpaceX-8, >= 3", spec([ISS[0], ISS[2], ISS[3]], 3))
        j4 = step("J4 ISS without SpaceX-16, all", spec(ISS[:3], "all"))
        e = next(e for e in j4.evaluations if e.rule_id == "pol:SpaceX-16:2#diff")
        if e.result == "fail":
            failing = pd.read_csv(root / e.genes_failed_ref)["ncbi_gene"].tolist()
            step("J4 resubmitted, failing excluded",
                 spec(ISS[:3], "all", exclude=failing, exclude_decision=j4.decision_id))
        step("Earth complete case", spec(EARTH, "all"))
        step("Earth without SpaceX-16-LAR", spec([EARTH[0], EARTH[2]], "all"))
        print(f"log: {len(log.entries())} releases; verify: {log.verify()} entries OK")


if __name__ == "__main__":
    main()
