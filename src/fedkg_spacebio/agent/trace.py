"""The agent trace (CONTRACTS §8): one JSON Lines file per run, written append-only.

What a trace is for
-------------------
Every agent run writes down, in order, what it was asked, which tools it called with which
inputs, what each tool returned, every decision it made and why, the report it wrote, and the
grounding checker's verdict on that report. The trace is the provenance record of the run: a
reader can follow every claim in the report back to the tool result it came from. The grounding
checker (``agent/grounding.py``) reads **only** this file, so anything not in the trace can't
support a claim.

Layout: ``$FEDKG_WORKDIR/traces/<trace id with ":" and "/" as "_">.jsonl``. One JSON object per
line. Every record has ``trace_id``, ``seq`` (0, 1, 2, …), ``ts`` (UTC, ISO 8601 with ``Z``),
``type``, the fields of its type (§8), and two hash fields (below).

Append-only, then sealed
------------------------
The writer only ever appends. ``run_end`` is the last record: after it the writer refuses any
append and makes the file read-only. Nothing is rewritten in place, so a crash leaves a valid
prefix of the run rather than a half-edited file.

Hash chain (tamper-evident, not tamper-proof)
---------------------------------------------
Each record carries ``prev_sha256`` (the previous record's ``record_sha256``; null for the first)
and ``record_sha256`` (SHA-256 of the record's canonical JSON without that field). Editing a
record, or removing or reordering one, breaks the chain from that point, and ``read_trace``
refuses the file. This matters because the grounding verdict is only as good as the file it read:
if a tool result could be edited after the fact to contain a cited ID, a fabricated citation
would pass. Someone who can rewrite the whole file can rebuild a consistent chain; a real
deployment would anchor the final hash somewhere the writer can't change. Same design, and same
limit, as the release log (``policy/release_log.py``).

Alternatives considered: making the file read-only only at the end (nothing a reader could check
later); one hash over the whole file at ``run_end`` (an unfinished trace couldn't be checked, and
a crash would leave no verifiable prefix).

Canonical JSON (for the hashes): sorted keys, ``,`` and ``:`` separators, UTF-8, no NaN or
infinity (a tool that has a missing number writes null).

IDs minted here (CONTRACTS §1)
------------------------------
- trace: ``trace:<yyyymmddThhmmssZ>-<8 hex>``;
- tool call: ``<trace id>/call/<n>``, n from 1;
- decision: ``<trace id>/dec/<n>``, n from 1, one counter for every decision in the run (the
  policy checks the agent runs and its own planning decisions).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import stat
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fedkg_spacebio.census.osdr_client import default_workdir
from fedkg_spacebio.ids import TRACE_ID_RE, file_stem

TRACE_VERSION = "agent-trace/1"

# §8 record types and the fields each must carry (beyond trace_id, seq, ts, type and the hashes).
RECORD_FIELDS: dict[str, tuple[str, ...]] = {
    "run_start": ("question", "policy_profile", "agent_version", "model", "kg_version", "tools"),
    "tool_call": ("call_id", "tool", "input"),
    "tool_result": ("call_id", "tool", "tool_version", "ok", "error", "data", "ids", "cache"),
    "decision": ("decision_id", "kind", "choice", "reason", "based_on"),
    "model_call": ("provider", "model", "served_model", "stop", "usage", "request_id",
                   "n_tool_calls", "error"),
    "report": ("claims",),
    "grounding": ("checker_version", "claims", "supported_rate", "fabricated_ids"),
    "run_end": ("status", "error"),
}
DECISION_KINDS = ("compatibility", "policy", "plan", "schedule")
HASH_FIELDS = ("prev_sha256", "record_sha256")
BASE_FIELDS = ("trace_id", "seq", "ts", "type")

CALL_ID_RE = re.compile(rf"({TRACE_ID_RE.pattern})/call/([1-9][0-9]*)")
TRACE_DECISION_RE = re.compile(rf"({TRACE_ID_RE.pattern})/dec/([1-9][0-9]*)")


class TraceError(ValueError):
    """A trace that can't be written as asked, or a file that isn't a valid trace."""


def mint_trace_id(now: datetime | None = None, token: str | None = None) -> str:
    """A new trace ID. ``now`` must be timezone-aware; ``token`` is 8 lowercase hex digits."""
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware (trace IDs are in UTC)")
    token = token if token is not None else secrets.token_hex(4)
    if not re.fullmatch(r"[0-9a-f]{8}", token):
        raise ValueError(f"token must be 8 lowercase hex digits, got {token!r}")
    return f"trace:{now.astimezone(UTC):%Y%m%dT%H%M%SZ}-{token}"


def default_trace_dir() -> Path:
    """``$FEDKG_WORKDIR/traces`` (never inside the repo)."""
    return default_workdir() / "traces"


def trace_path(trace_id: str, directory: str | Path | None = None) -> Path:
    directory = Path(directory) if directory is not None else default_trace_dir()
    return directory / f"{file_stem(trace_id)}.jsonl"


def canonical_json(obj: Any) -> bytes:
    """Sorted keys, compact separators, UTF-8; NaN and infinity are refused."""
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False).encode("utf-8")
    except ValueError as exc:
        raise TraceError(f"record isn't valid JSON (NaN or infinity?): {exc}") from None


def record_sha256(record: dict) -> str:
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    return hashlib.sha256(canonical_json(body)).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class TraceWriter:
    """Appends records to one trace file. Not thread-safe: one writer per run."""

    def __init__(self, trace_id: str, path: Path, clock: Callable[[], str] = _utc_now):
        if not TRACE_ID_RE.fullmatch(trace_id or ""):
            raise TraceError(f"not a trace ID: {trace_id!r}")
        self.trace_id = trace_id
        self.path = Path(path)
        self._clock = clock
        self._seq = 0
        self._prev: str | None = None
        self._n_calls = 0
        self._n_decisions = 0
        self._ended = False
        self._call_ids: set[str] = set()

    @classmethod
    def create(cls, directory: str | Path | None = None, trace_id: str | None = None,
               clock: Callable[[], str] = _utc_now) -> TraceWriter:
        """Start a new, empty trace file. Refuses to reuse an existing file."""
        trace_id = trace_id or mint_trace_id()
        path = trace_path(trace_id, directory)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8"):
            pass
        return cls(trace_id, path, clock)

    # ------------------------------------------------------------------ IDs

    def next_call_id(self) -> str:
        self._n_calls += 1
        return f"{self.trace_id}/call/{self._n_calls}"

    def next_decision_id(self) -> str:
        self._n_decisions += 1
        return f"{self.trace_id}/dec/{self._n_decisions}"

    @property
    def ended(self) -> bool:
        return self._ended

    # ------------------------------------------------------------------ writing

    def append(self, type_: str, **fields: Any) -> dict:
        """Write one record of ``type_`` with ``fields``. Returns the record as written."""
        if self._ended:
            raise TraceError("the trace has ended (run_end written); it can't be appended to")
        if type_ not in RECORD_FIELDS:
            raise TraceError(f"unknown record type {type_!r}")
        missing = [f for f in RECORD_FIELDS[type_] if f not in fields]
        if missing:
            raise TraceError(f"{type_} record lacks {missing}")
        clash = [f for f in fields if f in BASE_FIELDS or f in HASH_FIELDS]
        if clash:
            raise TraceError(f"fields {clash} are set by the writer")
        if (self._seq == 0) != (type_ == "run_start"):
            raise TraceError("run_start must be the first record, and only the first")
        _check_fields(type_, fields, self.trace_id)
        if type_ == "tool_call":
            if fields["call_id"] in self._call_ids:
                raise TraceError(f"call ID {fields['call_id']} already used")
            self._call_ids.add(fields["call_id"])
        record = {"trace_id": self.trace_id, "seq": self._seq, "ts": self._clock(),
                  "type": type_, **fields, "prev_sha256": self._prev}
        record["record_sha256"] = record_sha256(record)
        line = canonical_json(record).decode("utf-8")
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._seq += 1
        self._prev = record["record_sha256"]
        if type_ == "run_end":
            self._ended = True
            mode = self.path.stat().st_mode
            self.path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        return record

    def end(self, status: str, error: Any = None) -> dict:
        return self.append("run_end", status=status, error=error)


def _check_fields(type_: str, fields: dict, trace_id: str) -> None:
    """Per-type checks the writer can make cheaply (IDs belong to this trace, kinds are known)."""
    if type_ in ("tool_call", "tool_result"):
        m = CALL_ID_RE.fullmatch(fields["call_id"] or "")
        if not m or m.group(1) != trace_id:
            raise TraceError(f"call ID {fields['call_id']!r} isn't a call of {trace_id}")
    if type_ == "decision":
        m = TRACE_DECISION_RE.fullmatch(fields["decision_id"] or "")
        if not m or m.group(1) != trace_id:
            raise TraceError(f"decision ID {fields['decision_id']!r} isn't a decision of "
                             f"{trace_id}")
        if fields["kind"] not in DECISION_KINDS:
            raise TraceError(f"decision kind must be one of {DECISION_KINDS}")
        if not isinstance(fields["based_on"], list):
            raise TraceError("based_on must be a list of call IDs")
    if type_ == "tool_result" and not isinstance(fields["ids"], list):
        raise TraceError("ids must be a list")


# ------------------------------------------------------------------ reading


def read_trace(path: str | Path) -> list[dict]:
    """Every record, oldest first, after checking the file is a valid trace.

    Checks: each line is a JSON object; one trace ID throughout; ``seq`` is 0, 1, 2, …; the hash
    chain links and every record matches its hash; types are known and carry their fields;
    ``run_start`` comes first and only first; nothing follows ``run_end``; call IDs aren't reused.
    Raises ``TraceError`` naming the first problem.
    """
    path = Path(path)
    records: list[dict] = []
    prev: str | None = None
    trace_id: str | None = None
    call_ids: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            where = f"{path.name}: line {i + 1}"
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TraceError(f"{where}: not JSON ({exc.msg})") from None
            if not isinstance(rec, dict):
                raise TraceError(f"{where}: not a JSON object")
            if records and records[-1]["type"] == "run_end":
                raise TraceError(f"{where}: a record follows run_end")
            if rec.get("seq") != i:
                raise TraceError(f"{where}: seq is {rec.get('seq')!r}, expected {i}")
            trace_id = trace_id or rec.get("trace_id")
            if rec.get("trace_id") != trace_id or not TRACE_ID_RE.fullmatch(trace_id or ""):
                raise TraceError(f"{where}: trace_id {rec.get('trace_id')!r} doesn't match")
            if rec.get("prev_sha256") != prev:
                raise TraceError(f"{where}: doesn't link to the record before it")
            if rec.get("record_sha256") != record_sha256(rec):
                raise TraceError(f"{where}: content doesn't match its hash")
            type_ = rec.get("type")
            if type_ not in RECORD_FIELDS:
                raise TraceError(f"{where}: unknown record type {type_!r}")
            missing = [k for k in RECORD_FIELDS[type_] if k not in rec]
            if missing:
                raise TraceError(f"{where}: {type_} record lacks {missing}")
            if (i == 0) != (type_ == "run_start"):
                raise TraceError(f"{where}: run_start must be the first record, and only it")
            try:
                _check_fields(type_, rec, trace_id)
            except TraceError as exc:
                raise TraceError(f"{where}: {exc}") from None
            if type_ == "tool_result":
                if rec["call_id"] in call_ids:
                    raise TraceError(f"{where}: second result for {rec['call_id']}")
                call_ids.add(rec["call_id"])
            prev = rec["record_sha256"]
            records.append(rec)
    if not records:
        raise TraceError(f"{path.name}: empty trace")
    return records
