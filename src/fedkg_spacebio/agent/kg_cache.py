"""Disk cache for KG template calls, with an offline (replay-only) mode.

Layout (``$FEDKG_WORKDIR/cache/kg/templates/``, never inside the repo; D-010)::

    manifest.json        the current KG fingerprint, when it was taken, and earlier fingerprints
    <64 hex>.json        one raw KG response per (template, KG params, Cypher, fingerprint)

What is cached, and why
-----------------------
- **The raw KG rows, not the tool's output.** ``kg_query``'s post-processing (sorting, flags, the
  tissue filter, the row cap) is cheap and deterministic, so one entry per (template, gene) serves
  every option, and a change to the post-processing (a new ``tool_version``) doesn't invalidate
  the cache.
- **Entries have the same shape as the part 1 recordings** in ``tests/fixtures/kg/``:
  ``template``, ``kg_params``, ``cypher_sha256``, ``kg_fingerprint_sha256``, ``fetched``,
  ``source``, ``total_rows``, ``rows_canonical_sha256``, ``rows``. So tests load the recordings
  as cache entries directly, and a live pre-warm can be compared with a recording by one hash.

The cache key
-------------
SHA-256 of the canonical JSON of ``{template, kg_params, cypher_sha256, fingerprint_sha256}``:
- ``kg_params`` are the parameters actually sent (``{"gene": "13170"}``), not the tool's input;
- ``cypher_sha256`` makes an edited template miss instead of serving rows from the old Cypher;
- ``fingerprint_sha256`` (the KG version signal, D-038) makes a changed graph miss instead of
  serving stale rows. Its stated limit: it misses a rerun that changes edge values but keeps every
  count and comparison ID.

Offline mode
------------
``offline=True`` (or ``FEDKG_KG_OFFLINE=1``) never touches the network: the fingerprint comes from
``manifest.json``, and a miss is an error, not a live call. This is the demo's replay path, where a
silent live call would make the run depend on the endpoint being up.

Canonical JSON
--------------
Keys sorted, no whitespace, non-ASCII kept, and **numbers formatted the ECMAScript way**
(``1.06e-9``, ``0.0000162``, ``2`` for 2.0). The part 1 hashes were computed in a JavaScript
client; Python's default float format (``1.06e-09``, ``1.62e-05``) gives a different string for
the same number, so ``rows_canonical_sha256`` would not reproduce. The digits themselves agree:
both languages print the shortest string that round-trips.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from fedkg_spacebio.agent.envelope import ToolError

OFFLINE_ENV = "FEDKG_KG_OFFLINE"
MANIFEST = "manifest.json"
_KEY_RE = re.compile(r"[0-9a-f]{64}")


def default_cache_dir() -> Path:
    """``$FEDKG_WORKDIR/cache/kg/templates`` (``~/fedkg-work`` if unset)."""
    root = Path(os.environ.get("FEDKG_WORKDIR", str(Path.home() / "fedkg-work")))
    return root / "cache" / "kg" / "templates"


def _js_number(x: float) -> str:
    """Format a float as ECMAScript's ``Number.prototype.toString`` does (§7.1.12.1 of ES2023)."""
    if not math.isfinite(x):
        raise ValueError(f"non-finite number has no JSON form: {x!r}")
    if x == 0:
        return "0"
    sign = "-" if x < 0 else ""
    # repr gives the shortest digits that round-trip (as JS does); only the layout differs.
    # value = 0.d1d2...dk × 10^n, i.e. the digits string with the decimal point after n digits.
    t = Decimal(repr(abs(x))).normalize().as_tuple()
    digits = "".join(map(str, t.digits))
    k = len(digits)
    n = k + int(t.exponent)
    if k <= n <= 21:
        s = digits + "0" * (n - k)
    elif 0 < n <= 21:
        s = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        s = "0." + "0" * (-n) + digits
    else:
        e = n - 1
        mant = digits if k == 1 else digits[0] + "." + digits[1:]
        s = f"{mant}e{'+' if e >= 0 else '-'}{abs(e)}"
    return sign + s


def canonical_json(obj: Any) -> str:
    """Sorted keys, no whitespace, ECMAScript number format. Used for every hash in this cache."""
    if obj is None:
        return "null"
    if obj is True:
        return "true"
    if obj is False:
        return "false"
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, float):
        return _js_number(obj)
    if isinstance(obj, str):
        return json.dumps(obj, ensure_ascii=False)
    if isinstance(obj, Mapping):
        items = sorted(obj.items())
        return "{" + ",".join(
            f"{json.dumps(str(k), ensure_ascii=False)}:{canonical_json(v)}" for k, v in items
        ) + "}"
    if isinstance(obj, list | tuple):
        return "[" + ",".join(canonical_json(v) for v in obj) + "]"
    raise TypeError(f"not JSON-serializable: {type(obj).__name__}")


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rows_sha256(rows: Any) -> str:
    """``rows_canonical_sha256``: SHA-256 of the canonical JSON of the rows."""
    return sha256_hex(canonical_json(rows))


def cache_key(template: str, kg_params: Mapping[str, Any], cypher_sha256: str,
              fingerprint_sha256: str) -> str:
    """The 64-hex cache key (the envelope shows it as ``sha256:<key>``)."""
    return sha256_hex(canonical_json({
        "template": template,
        "kg_params": dict(kg_params),
        "cypher_sha256": cypher_sha256,
        "fingerprint_sha256": fingerprint_sha256,
    }))


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class CacheEntry:
    key: str
    record: dict[str, Any]

    @property
    def rows(self) -> list[Any]:
        return self.record["rows"]

    @property
    def fetched(self) -> str:
        return self.record["fetched"]


class KgCache:
    """The on-disk store. Doesn't talk to the network; ``KgStore`` (in ``kg_query``) does."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else default_cache_dir()

    # -- manifest (the current fingerprint) --------------------------------------------------
    def read_manifest(self) -> dict[str, Any] | None:
        p = self.root / MANIFEST
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def write_fingerprint(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Record a freshly taken fingerprint as current; keep any earlier one in ``history``.

        ``record`` has the shape of ``tests/fixtures/kg/kg_fingerprint.json``.
        """
        old = self.read_manifest() or {}
        history = list(old.get("history", []))
        current = old.get("current")
        if current and current["fingerprint_sha256"] != record["fingerprint_sha256"]:
            history.append(current)
        manifest = {"current": dict(record), "history": history}
        self._write_json(self.root / MANIFEST, manifest)
        return manifest

    def current_fingerprint(self) -> dict[str, Any]:
        m = self.read_manifest()
        if not m or "current" not in m:
            raise ToolError(
                "cache_miss",
                f"no KG fingerprint recorded in {self.root / MANIFEST}; pre-warm the cache first",
            )
        return m["current"]

    # -- entries -----------------------------------------------------------------------------
    def path(self, key: str) -> Path:
        if not _KEY_RE.fullmatch(key):
            raise ValueError(f"bad cache key {key!r}")
        return self.root / f"{key}.json"

    def get(self, key: str) -> CacheEntry | None:
        p = self.path(key)
        if not p.exists():
            return None
        record = json.loads(p.read_text(encoding="utf-8"))
        if rows_sha256(record["rows"]) != record.get("rows_canonical_sha256"):
            raise ToolError("cache_corrupt", f"rows hash mismatch in cache entry {p.name}")
        return CacheEntry(key, record)

    def put(self, key: str, record: Mapping[str, Any]) -> CacheEntry:
        rec = dict(record)
        rec["rows_canonical_sha256"] = rows_sha256(rec["rows"])
        self._write_json(self.path(key), rec)
        return CacheEntry(key, rec)

    def entries(self) -> list[Path]:
        return sorted(p for p in self.root.glob("*.json") if _KEY_RE.fullmatch(p.stem))

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.glob("*.json"))

    def _write_json(self, path: Path, obj: Any) -> None:
        """Write atomically (temp file, then rename) so a crash can't leave half an entry."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, path)


def offline_from_env() -> bool:
    return os.environ.get(OFFLINE_ENV, "").strip().lower() in {"1", "true", "yes"}
