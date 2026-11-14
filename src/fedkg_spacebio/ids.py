"""Run-time identifiers (CONTRACTS §1, "Run-time IDs"): syntax checks and minting.

Static IDs (genes, studies, tissues) come from the data and the KG. These are the IDs the
system makes up while it runs: jobs, decisions, releases, logs. They exist so that a decision,
a released result and a trace can point at each other unambiguously.

Forms (UTC times; mocks put ``FAKE-NNNN`` where the time and random part go):
- job: ``job:<yyyymmddThhmmssZ>-<6 hex>`` or ``job:FAKE-NNNN``. The part after ``job:`` is the
  *job suffix*; decision and release IDs reuse it, so you can tell which job they belong to by
  reading them;
- decision: ``dec:<job suffix>-<n>`` when the policy check runs on its own, or
  ``<trace id>/dec/<n>`` when an agent runs it inside a trace;
- release: ``rel:<job suffix>-<n>``;
- trace: ``trace:<yyyymmddThhmmssZ>-<8 hex>`` (or ``trace:FAKE-NNNN``);
- release log: ``<profile>-<n>``.

Why a timestamp plus random hex rather than a counter or a UUID: a counter needs one shared
place that hands out numbers (and a reset would reuse them); a UUID is unique but tells a reader
nothing. The timestamp sorts and reads naturally; 6 hex digits (16.7 million values) make two
jobs minted in the same second very unlikely to collide.

Hashes: lowercase hex SHA-256 (64 characters). Mock hashes are ``FAKE-sha256-NNNN``; parsers
accept them so the contract examples parse, and any check that compares a hash then fails.
"""

from __future__ import annotations

import re
import secrets
from datetime import UTC, datetime

_STAMP = r"\d{8}T\d{6}Z"
_JOB_SUFFIX = rf"(?:{_STAMP}-[0-9a-f]{{6}}|FAKE-\d{{4}})"
_TRACE = rf"trace:(?:{_STAMP}-[0-9a-f]{{8}}|FAKE-\d{{4}})"
_N = r"[1-9][0-9]*"

JOB_ID_RE = re.compile(rf"job:({_JOB_SUFFIX})")
TRACE_ID_RE = re.compile(_TRACE)
DECISION_ID_RE = re.compile(rf"dec:{_JOB_SUFFIX}-{_N}|{_TRACE}/dec/{_N}")
RELEASE_ID_RE = re.compile(rf"rel:{_JOB_SUFFIX}-{_N}")
PROFILE_RE = re.compile(r"[a-z][a-z0-9_]*")  # no hyphen: the log ID appends "-<n>"
LOG_ID_RE = re.compile(rf"({PROFILE_RE.pattern})-({_N})")
# A client or site name: a mission slug, optionally with a stratum suffix (§1). No spaces,
# commas or colons, so it can be a CSV header cell and part of a file name.
CLIENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

SHA256_RE = re.compile(r"[0-9a-f]{64}")
FAKE_SHA256_RE = re.compile(r"FAKE-sha256-\d{4}")


def mint_job_id(now: datetime | None = None, token: str | None = None) -> str:
    """A new job ID. ``now`` must be timezone-aware (converted to UTC); ``token`` is 6 hex."""
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware (job IDs are in UTC)")
    token = token if token is not None else secrets.token_hex(3)
    if not re.fullmatch(r"[0-9a-f]{6}", token):
        raise ValueError(f"token must be 6 lowercase hex digits, got {token!r}")
    return f"job:{now.astimezone(UTC):%Y%m%dT%H%M%SZ}-{token}"


def job_suffix(job_id: str) -> str:
    """The part after ``job:``, reused by decision and release IDs."""
    m = JOB_ID_RE.fullmatch(job_id or "")
    if not m:
        raise ValueError(f"not a job ID: {job_id!r}")
    return m.group(1)


def decision_id_for(job_id: str, n: int) -> str:
    """``dec:<job suffix>-<n>``: the n-th policy check of this job outside an agent trace."""
    if n < 1:
        raise ValueError("decision numbers start at 1")
    return f"dec:{job_suffix(job_id)}-{n}"


def release_id_for(job_id: str, n: int) -> str:
    """``rel:<job suffix>-<n>``: the n-th release of this job in a log."""
    if n < 1:
        raise ValueError("release numbers start at 1")
    return f"rel:{job_suffix(job_id)}-{n}"


def is_sha256(value: object, allow_fake: bool = True) -> bool:
    """Lowercase 64-hex SHA-256, or (if ``allow_fake``) a ``FAKE-sha256-NNNN`` mock."""
    if not isinstance(value, str):
        return False
    return bool(SHA256_RE.fullmatch(value) or (allow_fake and FAKE_SHA256_RE.fullmatch(value)))


def file_stem(identifier: str) -> str:
    """An ID as a file-name stem: ``:`` and ``/`` become ``_`` (CONTRACTS §5.4)."""
    return identifier.replace(":", "_").replace("/", "_")
