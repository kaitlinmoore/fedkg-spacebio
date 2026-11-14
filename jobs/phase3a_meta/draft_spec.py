"""Draft a phase 3a job spec (CONTRACTS §6) and its gene plan from the sites' released flags.

Run from WSL after ``fedkg``:

    uv run python jobs/phase3a_meta/draft_spec.py --stratum iss --profile restricted \\
        --aggregation secure_sum --methods fe fe_het
    uv run python jobs/phase3a_meta/job.py --spec <printed path> --profile restricted

This is the coordinator's step before a job: build the gene plan from each client's
``site_metadata`` (its ``tested_*`` inclusion flags, §5.1) and the spec's inclusion rule, and
write a spec that names the plan and every input by SHA-256. In the agent (WS-15) the planning
step does this itself; this script exists for the demo replays and the tests.

Where the flags come from: by default the flags each client released with the open-profile
phase 3a run (``results/phase3a/<stratum>/<counts>/flare/sent/<client>.npz``, only the
``tested_*`` columns and ``ncbi_gene`` are read, never estimates), as
``jobs/policy_timing/time_check.py`` does. The flags don't depend on the policy profile.

Outputs, under the policy root (``$FEDKG_WORKDIR/policy/``; spec refs resolve against it):
``plans/<job slug>/gene_plan.csv`` and ``plans/<job slug>/job_spec.yaml``. The spec gets a fresh
job ID and ``policy_decision: null``: it hasn't been checked. If the profile's manifests don't
exist yet they are built (with that profile's policy IDs), since the spec records their hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402

from fedkg_spacebio.federation.gene_plan import build_gene_plan, write_gene_plan  # noqa: E402
from fedkg_spacebio.federation.jobspec import parse_job_spec, spec_to_dict  # noqa: E402
from fedkg_spacebio.federation.spec_run import policy_root_default  # noqa: E402
from fedkg_spacebio.harmonize.gene_map import ANNOTATION_SHA256  # noqa: E402
from fedkg_spacebio.ids import mint_job_id  # noqa: E402
from fedkg_spacebio.sites.inclusion import FLAGS, inclusion_flags_sha256  # noqa: E402
from fedkg_spacebio.sites.manifest import clients_in, write_manifests  # noqa: E402

KG_VERSION = "SPOKE-GeneLab v0.3.1+20260311"  # assumed; not exposed live (docs/kg-schema.md)
FIXED_POINT_BITS, MODULUS_BITS = 40, 64       # WS-13 choice (secure_sum.py docstring)


def released_flags(stratum: str, counts: str, clients: list[str],
                   root: Path | None = None) -> dict[str, pd.DataFrame]:
    """Each client's released inclusion flags, read from the open run's sent copies."""
    sent = config.results_dir(stratum, counts, "flare", root) / "sent"
    out = {}
    for c in clients:
        with np.load(sent / f"{c}.npz") as z:
            out[c] = pd.DataFrame({k: z[k] for k in ("ncbi_gene", *FLAGS)})
    return out


def census_access_date(root: Path | None = None) -> str:
    path = (root or config.workdir()) / "census" / "liver_summary.json"
    return json.loads(path.read_text())["access_date"]


def draft(*, stratum: str, counts: str = "standard", profile: str = "open",
          aggregation: str = "secure_sum", methods: list[str] | None = None,
          clients: list[str] | None = None, min_sites_per_gene: int | str = "all",
          rule: str = "mean10_finite", flags: dict[str, pd.DataFrame] | None = None,
          root: Path | None = None, policy_root: Path | None = None,
          question: str | None = None, job_id: str | None = None,
          access_date: str | None = None, annotation_sha256: str = ANNOTATION_SHA256,
          build_manifests: bool = True) -> Path:
    """Write the plan and spec; return the spec's path."""
    clients = sorted(clients or clients_in(stratum))
    methods = methods or (["fe", "fe_het"] if aggregation == "secure_sum" else ["fe"])
    policy_root = Path(policy_root) if policy_root is not None else policy_root_default()
    manifests = config.manifest_dir(counts, root, profile)
    if build_manifests and not all((manifests / f"{c}.json").is_file() for c in clients):
        write_manifests(out_dir=manifests, workdir=root or config.workdir(),
                        counts_variant=counts, policy_ids=config.policy_ids(profile))
    flags = flags if flags is not None else released_flags(stratum, counts, clients, root)
    plan = build_gene_plan({c: flags[c] for c in clients}, rule, min_sites_per_gene)
    job_id = job_id or mint_job_id()
    slug = job_id.replace(":", "-")
    ref = f"plans/{slug}/gene_plan.csv"
    output = "masked_contribution" if aggregation == "secure_sum" else "per_site_summary"
    agg = ({"strategy": "secure_sum", "fixed_point_bits": FIXED_POINT_BITS,
            "modulus_bits": MODULUS_BITS} if aggregation == "secure_sum"
           else {"strategy": "plain"})
    spec = parse_job_spec({
        "schema": "job-spec/1", "job_id": job_id, "task": "de_meta",
        "question": question or ("Which liver genes change with spaceflight across "
                                 f"{'ISS' if stratum == 'iss' else 'Earth-return'} missions?"),
        "stratum": stratum, "clients": clients, "methods": methods, "counts_variant": counts,
        "inclusion": {"rule": rule, "min_sites_per_gene": min_sites_per_gene},
        "gene_plan": {"ref": ref, "sha256": write_gene_plan(plan, policy_root / ref)},
        "outputs": {c: output for c in clients}, "aggregation": agg,
        "weighting": {"scheme": "inverse_variance"}, "rounds": 1,
        "provenance": {
            "kg_version": KG_VERSION,
            "census_access_date": access_date or census_access_date(root),
            "annotation_sha256": annotation_sha256,
            "manifest_sha256": {c: hashlib.sha256((manifests / f"{c}.json").read_bytes())
                                .hexdigest() for c in clients},
            "inclusion_flags_sha256": {c: inclusion_flags_sha256(flags[c]) for c in clients}},
        "policy_decision": None,
    })
    path = policy_root / "plans" / slug / "job_spec.yaml"
    path.write_text("# Drafted by jobs/phase3a_meta/draft_spec.py; not policy-checked "
                    "(policy_decision: null).\n"
                    + yaml.safe_dump(spec_to_dict(spec), sort_keys=False))
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stratum", choices=["iss", "earth"], required=True)
    parser.add_argument("--counts", choices=["standard", "rrnarm"], default="standard")
    parser.add_argument("--profile", choices=config.PROFILES, default="open")
    parser.add_argument("--aggregation", choices=["secure_sum", "plain"], default="secure_sum")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--min_sites_per_gene", default="all",
                        type=lambda v: v if v == "all" else int(v))
    args = parser.parse_args()
    path = draft(stratum=args.stratum, counts=args.counts, profile=args.profile,
                 aggregation=args.aggregation, methods=args.methods,
                 min_sites_per_gene=args.min_sites_per_gene)
    print(path)


if __name__ == "__main__":
    main()
