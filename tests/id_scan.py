"""The §7 ID-scan check, as a test helper: every ID-shaped token in a tool's ``data`` must be in
its ``ids``.

It lives in ``tests/`` on purpose. The grounding checker (§8, rule 3) needs its own scanner for
claim text, and which tokens count as ID-shaped there (e.g. whether mission slugs do) is decided
with that checker, not here. Mission slugs are deliberately not matched: they are opaque (§1) and
have no pattern that wouldn't also match ordinary words.

Patterns, tried in this order at each position so the longest form wins:
1. composite edge keys ``kgedge:<TYPE>|<start>|<end>`` (they contain other IDs);
2. comparison IDs ``OSD-<n>-<32 hex>`` (before bare accessions, which are their prefix);
3. OSD accessions ``OSD-<n>`` and mock ``OSD-FAKE-<n>``;
4. CURIEs in the namespaces we use, including ``FAKE:``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

ID_TOKEN_RE = re.compile(
    r"kgedge:[A-Za-z_]+\|[^|\s\"]+\|[^|\s\"]+"
    r"|OSD-\d+-[0-9a-f]{32}"
    r"|OSD-FAKE-\d+(?:-0{32})?"
    r"|OSD-\d+"
    r"|\b(?:NCBIGene|UBERON|CL|MONDO|HP|NCBITaxon|ENSEMBL|FAKE):[A-Za-z0-9_.]+"
)


def strings(obj: Any) -> Iterator[str]:
    """Every string value in a nested structure (dict keys are field names and are skipped)."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from strings(v)
    elif isinstance(obj, list | tuple):
        for v in obj:
            yield from strings(v)


def id_tokens(obj: Any) -> set[str]:
    return {m.group(0) for s in strings(obj) for m in ID_TOKEN_RE.finditer(s)}


def assert_ids_cover_data(result: dict[str, Any]) -> None:
    """§7: every ID-shaped token in ``data`` is in ``ids``; and ``ids`` lists only values that
    occur in ``data`` (a listed ID that isn't in the result could be cited without support)."""
    ids = set(result["ids"])
    missing = id_tokens(result["data"]) - ids
    assert not missing, f"ID-shaped tokens in data but not in ids: {sorted(missing)[:10]}"
    values = set(strings(result["data"]))
    extra = ids - values
    assert not extra, f"ids not present as values in data: {sorted(extra)[:10]}"
