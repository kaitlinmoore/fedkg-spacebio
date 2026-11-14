"""From a job spec (CONTRACTS §6) to what the phase 3a FLARE job runs: the ``de_meta`` subset.

``jobs/phase3a_meta/job.py --spec`` calls ``prepare``, and so will WS-15's ``job.submit``. This
module answers "can this spec be run, and with which settings?", and checks everything it can
**before** any client starts:

- the task is ``de_meta``, one round, inverse-variance weighting;
- every output is the same kind: all ``per_site_summary`` (plain pooling) or all
  ``masked_contribution`` (secure sum). Mixed outputs, and masked outputs without
  ``secure_sum``, are refused: the server could read the plain ones and subtract them from the
  sum. ``secure_sum`` serves only methods computable from totals (``fe``, ``fe_het``);
- ``modulus_bits`` ≤ 64 (an implementation limit; the parser already refuses more, §6.3, and
  ``Params`` checks again);
- the gene plan and the exclusion list exist, are canonical and match their hashes; the plan's
  columns are exactly the job's clients;
- every client's manifest is in the manifest folder, matches the spec's ``manifest_sha256``,
  names the spec's stratum and counts variant; every client has an ``inclusion_flags_sha256``.

**It doesn't gate.** Whether the policies allow the job is the policy check's question
(``fedkg_spacebio.policy``), and refusing to run without an allow is ``job.submit``'s. These
checks overlap with the policy check's ``spec#…`` stage on purpose: a run started some other way
(by hand, in a test) still can't execute an inconsistent spec.

The rows (``SpecRun.rows``) are the plan minus ``genes_exclude``. Excluded genes are left out of
every vector entirely (WS-13 plan, approved), so the server never sums or reads a gene that a
policy refused.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from fedkg_spacebio.federation.gene_plan import (
    INCLUSION_FLAG,
    exclude_genes,
    read_gene_list,
    read_gene_plan,
)
from fedkg_spacebio.federation.jobspec import METHODS, JobSpec, job_spec_sha256, load_job_spec
from fedkg_spacebio.federation.secure_sum import Params, quantities_for

SECURE_SUM_METHODS = frozenset({"fe", "fe_het"})


class SpecRunError(ValueError):
    """A well-formed spec that this job can't run as written. ``problems`` lists them all."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("job spec can't be run:\n  " + "\n  ".join(self.problems))


@dataclass
class SpecRun:
    spec: JobSpec
    spec_path: Path
    spec_sha256: str
    mode: str                      # "plain" or "secure_sum"
    clients: list[str]
    flag: str                      # the tested_* flag of the inclusion rule
    plan_path: Path
    rows: pd.DataFrame             # plan minus exclusions (index ncbi_gene, bool per client)
    exclude_path: Path | None = None
    params: Params | None = None   # secure_sum only
    quantities: tuple[str, ...] = ()
    site_meta: dict = field(default_factory=dict)  # client → manifest fields used

    @property
    def job_id(self) -> str:
        return self.spec.job_id

    @property
    def slug(self) -> str:
        """A path-safe name for the job's folder: ``job:…`` → ``job-…``."""
        return self.spec.job_id.replace(":", "-")

    def aggregation(self) -> dict:
        """The table metadata's ``aggregation`` block."""
        if self.mode == "plain":
            return {"strategy": "plain"}
        return {"strategy": "secure_sum", **self.params.to_dict(),
                "quantities": list(self.quantities)}


def policy_root_default() -> Path:
    from fedkg_spacebio.policy.release_log import default_root  # $FEDKG_WORKDIR/policy
    return default_root()


def prepare(spec_or_path, *, manifest_dir: Path, policy_root: Path | None = None) -> SpecRun:
    """Check a spec against the files it names and return the run settings."""
    if isinstance(spec_or_path, JobSpec):
        spec, spec_path = spec_or_path, Path("<in memory>")
    else:
        spec_path = Path(spec_or_path)
        spec = load_job_spec(spec_path)
    root = Path(policy_root) if policy_root is not None else policy_root_default()
    problems: list[str] = []

    if spec.job_id is None:
        problems.append("job_id is null: mint one first (the run's folder and masks use it)")
    if spec.task != "de_meta":
        problems.append(f"task {spec.task}: this job runs de_meta only")
    if spec.rounds != 1:
        problems.append("de_meta runs in one round")
    if spec.weighting is None or spec.weighting.scheme != "inverse_variance":
        problems.append("de_meta needs weighting {scheme: inverse_variance}")
    for m in spec.methods:
        if m not in METHODS or METHODS[m].task != "de_meta":
            problems.append(f"{m} isn't a de_meta method")

    kinds = {spec.outputs.get(c) for c in spec.clients}
    if set(spec.outputs) != set(spec.clients):
        problems.append("outputs must name exactly the job's clients")
    mode = None
    if kinds == {"per_site_summary"}:
        mode = "plain"
        if spec.aggregation.strategy != "plain":
            problems.append("per_site_summary outputs run with aggregation: plain")
    elif kinds == {"masked_contribution"}:
        mode = "secure_sum"
        if spec.aggregation.strategy != "secure_sum":
            problems.append("masked_contribution needs aggregation: secure_sum (only the sum "
                            "is readable)")
        extra = sorted(set(spec.methods) - SECURE_SUM_METHODS)
        if extra:
            problems.append(f"secure_sum can't serve {extra}: they need each site's estimate")
    else:
        problems.append(f"outputs must all be per_site_summary or all masked_contribution; got "
                        f"{sorted(str(k) for k in kinds)} (mixed outputs would let the server "
                        "subtract the plain ones from the sum)")

    params = None
    if spec.aggregation.strategy == "secure_sum":
        try:
            params = Params(spec.aggregation.fixed_point_bits, spec.aggregation.modulus_bits)
        except ValueError as exc:
            problems.append(str(exc))

    rows = None
    plan_path = root / spec.gene_plan.ref
    try:
        plan = read_gene_plan(plan_path, spec.gene_plan.sha256)
        if sorted(plan.columns) != sorted(spec.clients):
            problems.append(f"plan columns {sorted(plan.columns)} aren't the job's clients")
        else:
            rows = plan
    except ValueError as exc:
        problems.append(f"gene plan: {exc}")
    exclude_path = None
    if spec.genes_exclude is not None:
        exclude_path = root / spec.genes_exclude.ref
        try:
            excluded = read_gene_list(exclude_path, spec.genes_exclude.sha256)
            if rows is not None:
                rows = exclude_genes(rows, excluded)
        except ValueError as exc:
            problems.append(f"genes_exclude: {exc}")
    if rows is not None and len(rows) == 0:
        problems.append("no genes left to pool")

    site_meta = {}
    for c in spec.clients:
        path = Path(manifest_dir) / f"{c}.json"
        if not path.is_file():
            problems.append(f"{c}: no manifest at {path}")
            continue
        data = path.read_bytes()
        if spec.provenance.manifest_sha256.get(c) != hashlib.sha256(data).hexdigest():
            problems.append(f"{c}: manifest_sha256 doesn't match {path}")
        m = json.loads(data)
        if m.get("stratum") != spec.stratum:
            problems.append(f"{c}: manifest stratum {m.get('stratum')!r}, spec {spec.stratum!r}")
        if m.get("counts_variant") != spec.counts_variant:
            problems.append(f"{c}: manifest counts {m.get('counts_variant')!r}, spec "
                            f"{spec.counts_variant!r}")
        if c not in spec.provenance.inclusion_flags_sha256:
            problems.append(f"{c}: no inclusion_flags_sha256 in provenance")
        site_meta[c] = {"site": m.get("site"), "policy_id": m.get("policy_id")}

    if problems:
        raise SpecRunError(problems)
    return SpecRun(
        spec=spec, spec_path=spec_path, spec_sha256=job_spec_sha256(spec), mode=mode,
        clients=sorted(spec.clients), flag=INCLUSION_FLAG[spec.inclusion.rule],
        plan_path=plan_path, rows=rows, exclude_path=exclude_path, params=params,
        quantities=quantities_for(spec.methods) if mode == "secure_sum" else (),
        site_meta=site_meta)


def plan_rows(spec: JobSpec, policy_root: Path | None = None
              ) -> tuple[pd.DataFrame, str, str | None]:
    """What a **client** reads: the plan minus exclusions, checked against the spec's hashes.

    Returns (rows, plan SHA-256, exclusion SHA-256 or None). It touches only the spec's own files,
    not other sites' manifests, so a client still reads nothing about another site.
    """
    root = Path(policy_root) if policy_root is not None else policy_root_default()
    rows = read_gene_plan(root / spec.gene_plan.ref, spec.gene_plan.sha256)
    if sorted(rows.columns) != sorted(spec.clients):
        raise SpecRunError([f"plan columns {sorted(rows.columns)} aren't the job's clients"])
    excl_sha = None
    if spec.genes_exclude is not None:
        excl_sha = spec.genes_exclude.sha256
        rows = exclude_genes(rows, read_gene_list(root / spec.genes_exclude.ref, excl_sha))
    return rows, spec.gene_plan.sha256, excl_sha


__all__ = ["SECURE_SUM_METHODS", "SpecRun", "SpecRunError", "plan_rows", "prepare"]
