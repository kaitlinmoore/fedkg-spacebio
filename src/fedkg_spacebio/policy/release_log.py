"""The append-only release log (CONTRACTS §5.3, D-033).

Why a log at all
----------------
A per-job check can't catch **differencing**: two releases that are each fine alone but give a
site's own values when subtracted (a FE over A, B, C, D minus a FE over A, B, C is D). To refuse
the second, the check has to know what was already released. The log records, for every
release, which clients contributed to which gene. It stores **no output values** (no pooled or
per-site floats; D-036 also found V3's floats differ at the ulp level across processes, so they
couldn't be hashed reproducibly anyway). Only the *pattern* of contributions matters for
differencing.

Why append-only
---------------
If the log could be cleared, the differencing check could be bypassed by clearing it between two
requests. So entries are never edited or removed. A "reset" (between demo rehearsals, say)
starts a **new** log, ``<profile>-<n+1>``, and records the switch, with the old log's head, in
an index. The old log stays on disk.

Layout, under the policy root (``$FEDKG_WORKDIR/policy/``)::

    logs/index.jsonl                      start and switch events, append-only
    logs/<log id>.jsonl                   one entry per release
    logs/<log id>/<release id>.csv        per-gene contributors (canonical gene-plan format)

(``:`` in a release ID is written ``_`` in its file name.)

Integrity (tamper-evident, not tamper-proof)
---------------------------------------------
- Each entry stores the previous entry's hash and its own: ``entry_sha256`` is the SHA-256 of
  the entry's canonical JSON without that field. Editing any entry, or removing one from the
  middle, breaks the chain from that point. Each contributor file is checked against the hash its
  entry stores.
- A hash chain can't see its **newest** entries removed: what remains is still a valid chain.
  Decision records catch that. Each one names the log head it saw; ``cross_check_decisions``
  checks every such head is still in the log.
- Whoever can rewrite every file can rebuild a consistent chain. A real deployment would anchor
  the head somewhere the coordinator can't rewrite (a transparency log, the sites themselves).
  Here the point is that the evaluator's input is checkable, not that it is unforgeable.

Appending
---------
``append`` takes the spec and the decision that allowed it, and refuses unless:
- the decision is ``allow``, for this spec (same ``job_id`` and ``job_spec_sha256``), for this
  log, and this log is still the profile's active one;
- the same decision or spec hasn't been released already;
- **the log head is unchanged since the decision** (else ``StaleDecisionError``). Two checks made
  against the same head could each pass alone and then combine by subtraction, so the second has
  to be checked again against the log that now includes the first.

The contributors recorded are the gene plan minus ``genes_exclude``: exactly what is released.
Appends hold a file lock, so the head comparison and the write can't interleave.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from fedkg_spacebio.federation.gene_plan import (
    GenePlanError,
    canonical_plan_csv,
    exclude_genes,
    read_gene_list,
    read_gene_plan,
)
from fedkg_spacebio.federation.jobspec import JobSpec, job_spec_sha256
from fedkg_spacebio.ids import LOG_ID_RE, PROFILE_RE, file_stem, release_id_for
from fedkg_spacebio.policy.decision import Decision, read_decision


class LogIntegrityError(RuntimeError):
    """The log, or a file it references, isn't what its hashes say."""


class AppendRefused(ValueError):
    """A release that may not be logged under the decision given."""


class StaleDecisionError(AppendRefused):
    """The log has moved on since the decision: check the spec again."""


def default_root() -> Path:
    """``$FEDKG_WORKDIR/policy`` (``~/fedkg-work/policy`` if unset); never inside the repo."""
    return Path(os.environ.get("FEDKG_WORKDIR", str(Path.home() / "fedkg-work"))) / "policy"


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _entry_hash(entry: dict) -> str:
    return hashlib.sha256(_canonical({k: v for k, v in entry.items()
                                      if k != "entry_sha256"})).hexdigest()


def _now(now: datetime | None) -> str:
    return (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve(root: str | Path | None) -> Path:
    return Path(root) if root is not None else default_root()


@contextmanager
def _locked(root: Path):
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / ".lock").open("a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _append_line(path: Path, obj: dict) -> None:
    with path.open("ab") as f:
        f.write(_canonical(obj) + b"\n")
        f.flush()
        os.fsync(f.fileno())


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        raise LogIntegrityError(f"{path}: last line is incomplete")
    out = []
    for i, line in enumerate(text.splitlines()):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            raise LogIntegrityError(f"{path}: line {i + 1} isn't JSON") from None
    return out


# ------------------------------------------------------------------ the index of logs


def read_index(root: str | Path | None = None) -> list[dict]:
    """Every start/switch event, oldest first."""
    return _read_jsonl(_resolve(root) / "logs" / "index.jsonl")


def _active_id(events: list[dict], profile: str) -> str | None:
    ids = [e["log_id"] for e in events if e.get("profile") == profile]
    return ids[-1] if ids else None


def active_log(profile: str, root: str | Path | None = None, create: bool = True) -> ReleaseLog:
    """The profile's current log; starts ``<profile>-1`` if there is none.

    With ``create=False`` nothing is written: a profile without a log gets an empty
    ``<profile>-1`` view.
    """
    if not PROFILE_RE.fullmatch(profile or ""):
        raise ValueError(f"profile names are lowercase letters, digits and _: {profile!r}")
    root = _resolve(root)
    log_id = _active_id(read_index(root), profile)
    if log_id is None:
        log_id = f"{profile}-1"
        if create:
            with _locked(root):
                if _active_id(read_index(root), profile) is None:
                    _append_line(root / "logs" / "index.jsonl", {
                        "event": "start", "ts": _now(None), "profile": profile,
                        "log_id": log_id})
                log_id = _active_id(read_index(root), profile)
    return ReleaseLog(profile, log_id, root)


def reset_log(profile: str, reason: str, root: str | Path | None = None) -> ReleaseLog:
    """Start ``<profile>-<n+1>``, recording the old log's head and the reason. Deletes nothing."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("a reset needs a reason; it is recorded in the index")
    root = _resolve(root)
    old = active_log(profile, root)
    old.verify()  # don't record the head of a log that doesn't check out
    head = old.head()
    n = int(LOG_ID_RE.fullmatch(old.log_id).group(2)) + 1
    new_id = f"{profile}-{n}"
    with _locked(root):
        if _active_id(read_index(root), profile) != old.log_id:
            raise RuntimeError(f"profile {profile!r} was switched concurrently; try again")
        _append_line(root / "logs" / "index.jsonl", {
            "event": "switch", "ts": _now(None), "profile": profile, "log_id": new_id,
            "previous_log_id": old.log_id,
            "previous_head": head[0] if head else None,
            "previous_head_sha256": head[1] if head else None,
            "reason": reason.strip()})
    return ReleaseLog(profile, new_id, root)


# ------------------------------------------------------------------ one log


class ReleaseLog:
    def __init__(self, profile: str, log_id: str, root: str | Path):
        self.profile = profile
        self.log_id = log_id
        self.root = Path(root)
        self.path = self.root / "logs" / f"{log_id}.jsonl"
        self.dir = self.root / "logs" / log_id

    def __repr__(self) -> str:
        return f"ReleaseLog({self.log_id!r}, root={str(self.root)!r})"

    # -------------------------------------------------------------- reading

    def entries(self) -> list[dict]:
        """All entries, oldest first, after checking the hash chain."""
        entries = _read_jsonl(self.path)
        prev, seen = None, set()
        for i, e in enumerate(entries):
            where = f"{self.path.name}: entry {i}"
            if e.get("seq") != i:
                raise LogIntegrityError(f"{where}: seq is {e.get('seq')!r}, expected {i}")
            if e.get("prev_entry_sha256") != prev:
                raise LogIntegrityError(f"{where}: doesn't link to the entry before it")
            if e.get("entry_sha256") != _entry_hash(e):
                raise LogIntegrityError(f"{where}: content doesn't match its hash")
            if e.get("release_id") in seen:
                raise LogIntegrityError(f"{where}: release ID {e.get('release_id')} repeated")
            seen.add(e.get("release_id"))
            prev = e["entry_sha256"]
        return entries

    def contributors(self, entry: dict) -> pd.DataFrame:
        """The entry's per-gene contributor table, hash-checked."""
        ref = entry["contributors"]
        try:
            return read_gene_plan(self.root / ref["ref"], ref["sha256"])
        except GenePlanError as exc:
            raise LogIntegrityError(f"{entry['release_id']}: {exc}") from None

    def verify(self) -> int:
        """Check the chain and every contributor file. Returns the number of entries."""
        entries = self.entries()
        for e in entries:
            self.contributors(e)
        return len(entries)

    def head(self) -> tuple[str, str] | None:
        """(release ID, entry hash) of the newest entry, or None if the log is empty."""
        entries = self.entries()
        return (entries[-1]["release_id"], entries[-1]["entry_sha256"]) if entries else None

    def history(self, task: str, stratum: str) -> list[tuple[dict, pd.DataFrame]]:
        """Every release with data key (task, stratum), with its contributors.

        Not filtered by counts variant, inclusion rule or client set: D-033 compares across them.
        """
        return [(e, self.contributors(e)) for e in self.entries()
                if e["data_key"] == {"task": task, "stratum": stratum}]

    # -------------------------------------------------------------- appending

    def append(self, spec: JobSpec, decision: Decision, now: datetime | None = None) -> dict:
        """Log the release of ``spec`` under ``decision``. Returns the entry written."""
        if decision.outcome != "allow":
            raise AppendRefused(f"{decision.decision_id} is {decision.outcome}, not allow")
        if spec.job_id is None or decision.job_id != spec.job_id:
            raise AppendRefused(f"{decision.decision_id} is for {decision.job_id}, not this "
                                f"spec ({spec.job_id})")
        spec_sha = job_spec_sha256(spec)
        if decision.job_spec_sha256 != spec_sha:
            raise AppendRefused(f"{decision.decision_id} was made for another version of "
                                f"{spec.job_id} (spec hash differs)")
        if decision.release_log_id != self.log_id:
            raise AppendRefused(f"{decision.decision_id} was checked against "
                                f"{decision.release_log_id}, not {self.log_id}")

        with _locked(self.root):
            active = _active_id(read_index(self.root), self.profile)
            if active != self.log_id:
                raise AppendRefused(f"{self.log_id} is no longer the active log of "
                                    f"{self.profile!r} ({active} is)")
            entries = self.entries()
            for e in entries:
                if e["decision_id"] == decision.decision_id or e["job_spec_sha256"] == spec_sha:
                    raise AppendRefused(f"already released as {e['release_id']}")
            head = (entries[-1]["release_id"], entries[-1]["entry_sha256"]) if entries else None
            seen = ((decision.release_log_head, decision.release_log_head_sha256)
                    if decision.release_log_head is not None else None)
            if seen != head:
                raise StaleDecisionError(
                    f"{decision.decision_id} saw head {seen[0] if seen else 'empty'}, but the "
                    f"log is now at {head[0] if head else 'empty'}; check the spec again")

            table = self._contributors_for(spec)
            n = 1 + sum(e["job_id"] == spec.job_id for e in entries)
            release_id = release_id_for(spec.job_id, n)
            ref = self._write_contributors(release_id, table)

            entry = {
                "seq": len(entries),
                "release_id": release_id,
                "ts": _now(now),
                "data_key": {"task": spec.task, "stratum": spec.stratum},
                "clients": list(spec.clients),
                "outputs": dict(spec.outputs),
                "counts_variant": spec.counts_variant,
                "inclusion": spec.inclusion.model_dump(mode="json"),
                "contributors": ref,
                "job_id": spec.job_id,
                "job_spec_sha256": spec_sha,
                "decision_id": decision.decision_id,
                "policies": {p["policy_id"]: p["sha256"] for p in decision.policies},
                "inputs": {
                    "manifest_sha256": dict(spec.provenance.manifest_sha256),
                    "inclusion_flags_sha256": dict(spec.provenance.inclusion_flags_sha256),
                    "gene_plan_sha256": spec.gene_plan.sha256,
                    "genes_exclude_sha256": (spec.genes_exclude.sha256
                                             if spec.genes_exclude else None),
                },
                "prev_entry_sha256": head[1] if head else None,
            }
            entry["entry_sha256"] = _entry_hash(entry)
            _append_line(self.path, entry)
        return entry

    def _contributors_for(self, spec: JobSpec) -> pd.DataFrame:
        """The gene plan minus ``genes_exclude``: what this release actually covers."""
        try:
            plan = read_gene_plan(self.root / spec.gene_plan.ref, spec.gene_plan.sha256)
            if spec.genes_exclude is not None:
                plan = exclude_genes(plan, read_gene_list(self.root / spec.genes_exclude.ref,
                                                          spec.genes_exclude.sha256))
        except GenePlanError as exc:
            raise AppendRefused(f"gene plan: {exc}") from None
        if sorted(plan.columns) != sorted(spec.clients):
            raise AppendRefused("gene plan columns aren't the job's clients")
        return plan

    def _write_contributors(self, release_id: str, table: pd.DataFrame) -> dict:
        data = canonical_plan_csv(table)
        path = self.dir / f"{file_stem(release_id)}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != data:
            # Left by an append that failed before writing its entry, with other content.
            raise AppendRefused(f"{path} exists with other content")
        path.write_bytes(data)
        return {"ref": path.relative_to(self.root).as_posix(),
                "sha256": hashlib.sha256(data).hexdigest()}


# ------------------------------------------------------------------ decisions vs the log


def cross_check_decisions(log: ReleaseLog, decisions_dir: str | Path | None = None) -> int:
    """Every decision made against ``log`` must name a head that is still in it.

    This catches removal of the newest entries, which the hash chain alone can't see. Returns
    the number of decision records checked.
    """
    decisions_dir = Path(decisions_dir) if decisions_dir is not None else log.root / "decisions"
    known = {(e["release_id"], e["entry_sha256"]) for e in log.entries()}
    checked, problems = 0, []
    for path in sorted(decisions_dir.glob("*.json")):
        dec = read_decision(path)
        if dec.release_log_id != log.log_id:
            continue
        checked += 1
        if dec.release_log_head is None:
            continue
        if (dec.release_log_head, dec.release_log_head_sha256) not in known:
            problems.append(f"{dec.decision_id} saw head {dec.release_log_head}, which isn't "
                            f"in {log.log_id} (entries removed from the end?)")
    if problems:
        raise LogIntegrityError("\n".join(problems))
    return checked
