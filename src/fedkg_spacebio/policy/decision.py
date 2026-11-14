"""The decision record (CONTRACTS §5.4): what the policy check decided, and why.

One record per check of one exact spec against one exact log state. It names:
- the spec, by ``job_id`` and ``job_spec_sha256``;
- the log, by ``release_log_id`` and its head (release ID and entry hash) at check time;
- the outcome and the deciding rule (the first failure in evaluation order);
- every rule evaluated, with failing genes for gene-scope failures;
- the hash of every policy file used, and the evaluator's version.

So a reader can recompute the decision from the files it names, and anyone releasing a result
can tell whether the decision still applies (``release_log.append`` refuses it if the log has
moved on, or if the spec isn't the one checked).

Files, under the policy root (``$FEDKG_WORKDIR/policy/``):
- ``decisions/<decision id>.json``, with ``:`` and ``/`` in the ID written as ``_``;
- ``decisions/<decision id>.<site or project>.<rule key>.genes.csv``: a gene-scope failure's gene
  list, canonical gene-list format (§6.1), referenced from the evaluation with its SHA-256.

Records are written once and never overwritten: a second check of the same job gets the next
decision number (``next_decision_id``).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from fedkg_spacebio.ids import decision_id_for, file_stem, job_suffix

EVALUATOR_VERSION = "fedkg_spacebio.policy/1"


@dataclass
class Evaluation:
    """One rule (or spec check) evaluated. Gene fields are set for per-gene rules only."""

    rule_id: str
    site: str | None
    scope: str  # "job" | "gene"
    result: str  # "pass" | "fail" | "not_applicable"
    detail: str
    genes_failed: int | None = None
    genes_failed_ref: str | None = None
    sha256: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("genes_failed", "genes_failed_ref", "sha256"):
            if d[k] is None:
                del d[k]
        return d


@dataclass
class Decision:
    decision_id: str
    job_id: str
    job_spec_sha256: str
    release_log_id: str
    release_log_head: str | None
    release_log_head_sha256: str | None
    outcome: str  # "allow" | "deny"
    deciding_rule: str | None
    reason: str
    evaluations: list[Evaluation] = field(default_factory=list)
    policies: list[dict] = field(default_factory=list)  # [{policy_id, sha256}]
    evaluator_version: str = EVALUATOR_VERSION

    def to_dict(self) -> dict:
        """Fields in contract order (§5.4)."""
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["evaluations"] = [e.to_dict() for e in self.evaluations]
        d["policies"] = [dict(p) for p in self.policies]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Decision:
        d = dict(d)
        d["evaluations"] = [Evaluation(**e) for e in d.get("evaluations", [])]
        return cls(**d)


def decision_path(root: str | Path, decision_id: str) -> Path:
    return Path(root) / "decisions" / f"{file_stem(decision_id)}.json"


def genes_failed_ref(decision_id: str, site: str | None, rule_key: str) -> str:
    """Where a gene-scope failure's gene list goes, relative to the policy root."""
    return f"decisions/{file_stem(decision_id)}.{site or 'project'}.{rule_key}.genes.csv"


def write_decision(decision: Decision, root: str | Path) -> Path:
    """Write the record as JSON. Refuses to overwrite an existing record."""
    path = decision_path(root, decision.decision_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(decision.to_dict(), f, indent=1, ensure_ascii=False)
        f.write("\n")
    return path


def read_decision(path: str | Path) -> Decision:
    return Decision.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def next_decision_id(job_id: str, root: str | Path) -> str:
    """``dec:<job suffix>-<n>`` with n one more than any record already written for the job."""
    stem = file_stem(f"dec:{job_suffix(job_id)}")
    pattern = re.compile(rf"{re.escape(stem)}-([1-9][0-9]*)\.json")
    taken = [int(m.group(1)) for p in (Path(root) / "decisions").glob(f"{stem}-*.json")
             if (m := pattern.fullmatch(p.name))]
    return decision_id_for(job_id, max(taken, default=0) + 1)
