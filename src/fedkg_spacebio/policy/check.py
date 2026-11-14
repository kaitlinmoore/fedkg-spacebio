"""Check a job spec against a policy profile, from the command line (CONTRACTS §5.4, §7).

    uv run python -m fedkg_spacebio.policy.check SPEC.yaml --profile restricted
    uv run python -m fedkg_spacebio.policy.check SPEC.yaml --profile restricted --dry-run

Prints the decision record (JSON). Exit status: 0 allow, 1 deny, 2 when the spec can't be
evaluated at all (malformed, unreadable policies or manifests, or a manifest whose ``policy_id``
isn't the profile's). Nothing runs: this is the check the agent's ``policy.check`` tool wraps.

Unless ``--dry-run``, it writes, under the policy root (``$FEDKG_WORKDIR/policy``):
- the decision record and any failing-gene lists (``decisions/``);
- the spec as checked, job ID minted, to ``specs/<job suffix>.yaml``. On an allow its
  ``policy_decision`` names this decision (the spec hash ignores that field, so the decision
  still matches); on a deny it is null. A later check of the same job overwrites the file; each
  decision record keeps the hash of the spec it judged.

``--dry-run`` writes nothing at all, not even a new log's start event.

Defaults: ``--policies`` is the repo's ``policies/<profile>/``; ``--manifests`` is where
``jobs/phase3a_meta/job.py`` writes them for the profile and the spec's counts variant
(``phase3a/manifests/[profiles/<profile>/]<counts_variant>/`` under ``$FEDKG_WORKDIR``; the same
rule as ``jobs/phase3a_meta/config.py``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from fedkg_spacebio.federation.jobspec import JobSpecError, load_job_spec, spec_to_dict
from fedkg_spacebio.ids import file_stem, job_suffix
from fedkg_spacebio.policy.evaluate import check_job
from fedkg_spacebio.policy.registry import PolicyMismatchError, load_registry
from fedkg_spacebio.policy.release_log import active_log, default_root
from fedkg_spacebio.policy.schema import PolicyError, load_profile

REPO_POLICIES = Path(__file__).resolve().parents[3] / "policies"


def default_manifests(profile: str, counts_variant: str) -> Path:
    work = Path(os.environ.get("FEDKG_WORKDIR", str(Path.home() / "fedkg-work")))
    extra = () if profile == "open" else ("profiles", profile)
    return work.joinpath("phase3a", "manifests", *extra, counts_variant)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m fedkg_spacebio.policy.check",
                                description="Check a job spec against a policy profile.")
    p.add_argument("spec", type=Path, help="job spec, YAML or JSON (CONTRACTS §6)")
    p.add_argument("--profile", default="open", help="policy profile (default: open)")
    p.add_argument("--policies", type=Path, help="profile folder (default: policies/<profile>/)")
    p.add_argument("--manifests", type=Path, help="site manifest folder")
    p.add_argument("--root", type=Path, help="policy root (default: $FEDKG_WORKDIR/policy)")
    p.add_argument("--decision-id", help="use this decision ID (e.g. an agent trace's)")
    p.add_argument("--dry-run", action="store_true", help="decide, but write nothing")
    return p


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:  # argparse prints its own message
        return int(exc.code or 0)
    root = args.root or default_root()
    try:
        spec = load_job_spec(args.spec)
        profile = load_profile(args.policies or REPO_POLICIES / args.profile)
        registry = load_registry(args.manifests or default_manifests(args.profile,
                                                                     spec.counts_variant))
        log = active_log(args.profile, root=root, create=not args.dry_run)
        minted, decision = check_job(spec, profile, registry, log, decision_id=args.decision_id,
                                     write=not args.dry_run)
    except (JobSpecError, PolicyError, PolicyMismatchError, OSError, ValueError) as exc:
        print(f"policy check not run: {exc}", file=sys.stderr)
        return 2
    if not args.dry_run:
        d = spec_to_dict(minted)
        d["policy_decision"] = decision.decision_id if decision.outcome == "allow" else None
        path = root / "specs" / f"{file_stem(job_suffix(minted.job_id))}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True),
                        encoding="utf-8")
    print(json.dumps(decision.to_dict(), indent=1, ensure_ascii=False))
    return 0 if decision.outcome == "allow" else 1


if __name__ == "__main__":
    sys.exit(main())
