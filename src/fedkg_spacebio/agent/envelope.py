"""The return envelope every agent tool uses (CONTRACTS §7).

Every tool answers with the same shape::

    {"call_id", "tool", "tool_version", "ok", "error", "data", "ids", "cache"}

Why one shared shape: the trace (§8) stores tool results verbatim, and the grounding checker reads
only the trace. If every tool puts its citable IDs in the same place (``ids``), the checker needs
no per-tool logic: a claim may cite an ID only if that ID is in the ``ids`` of the tool result it
names.

What this module does and doesn't do:
- It **builds and checks** the envelope: ``ok`` agrees with ``error``, ``ids`` are strings, and
  duplicates are dropped while keeping first-seen order (so a trace diff stays stable).
- It does **not** decide which IDs belong in ``ids``. Each tool lists them itself, from the
  values it puts in ``data``. Alternative considered: scan ``data`` with ID regexes and fill
  ``ids`` automatically. Rejected: then ``ids`` could never disagree with ``data``, and the §7
  test that scans ``data`` and compares it with ``ids`` would check nothing. Keeping the two
  separate makes that test meaningful.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


class ToolError(Exception):
    """A tool-level failure with a stable ``code`` (e.g. ``invalid_param``, ``cache_miss``).

    Tools raise it internally and turn it into ``ok: false`` with ``error: {code, message}``.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


def unique_ids(ids: Iterable[str]) -> list[str]:
    """Drop duplicates, keep first-seen order, reject anything that isn't a non-empty string."""
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        if not isinstance(i, str) or not i:
            raise TypeError(f"ids must be non-empty strings, got {i!r}")
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def envelope(
    tool: str,
    tool_version: str,
    *,
    call_id: str | None = None,
    data: Mapping[str, Any] | None = None,
    ids: Iterable[str] = (),
    cache: Mapping[str, Any] | None = None,
    error: ToolError | None = None,
) -> dict[str, Any]:
    """Build a §7 envelope.

    ``call_id`` is minted by the agent runtime (``<trace id>/call/<n>``, §1); a tool called outside
    a trace leaves it ``None``. ``cache`` is ``None`` for tools that make no external call, else
    ``{"hit", "key", "fetched"}``. A failed call (``error`` given) still carries ``cache`` (the
    failure may have come from a cached response) but no ``ids``: nothing in a failed result can be
    cited.
    """
    ok = error is None
    id_list = unique_ids(ids)
    if not ok and id_list:
        raise ValueError("a failed tool result must not list citable ids")
    return {
        "call_id": call_id,
        "tool": tool,
        "tool_version": tool_version,
        "ok": ok,
        "error": None if ok else error.as_dict(),
        "data": dict(data) if data is not None else {},
        "ids": id_list,
        "cache": dict(cache) if cache is not None else None,
    }
