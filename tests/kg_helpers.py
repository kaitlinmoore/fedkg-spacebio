"""Load the recorded KG responses (``tests/fixtures/kg/``) into a ``KgCache`` for offline tests.

The recordings already have the cache-entry shape, so a test cache is the recordings plus a
manifest holding the recorded fingerprint, keyed exactly as a live pre-warm would key them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fedkg_spacebio.agent import kg_query as kq
from fedkg_spacebio.agent.kg_cache import KgCache, cache_key, sha256_hex

KG_FIXTURES = Path(__file__).parent / "fixtures" / "kg"


def load_fixture(name: str) -> dict[str, Any]:
    return json.loads((KG_FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def recorded(template: str | None = None) -> list[dict[str, Any]]:
    """Every recorded template run (not the fingerprint), optionally for one template."""
    out = []
    for p in sorted(KG_FIXTURES.glob("*__*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if template is None or rec["template"] == template:
            out.append(rec)
    return out


def fixture_cache(root: Path, *, with_entries: bool = True) -> KgCache:
    cache = KgCache(root)
    fp = load_fixture("kg_fingerprint")
    cache.write_fingerprint(fp)
    if with_entries:
        for rec in recorded():
            key = cache_key(rec["template"], rec["kg_params"], rec["cypher_sha256"],
                            fp["fingerprint_sha256"])
            cache.put(key, rec)
    return cache


class FakeClient:
    """Answers from the recordings by Cypher hash; records what was sent."""

    def __init__(self):
        self.sent = []
        self.closed = False
        t = kq.load_templates()
        self.by_hash = {v["cypher_sha256"]: k for k, v in t.items()}

    def query(self, cypher, params):
        self.sent.append(dict(params))
        name = self.by_hash[sha256_hex(cypher)]
        if name == "kg_fingerprint":
            return 1, load_fixture("kg_fingerprint")["rows"]
        rec = load_fixture(f"{name}__{params['gene']}")
        return rec["total_rows"], rec["rows"]

    def close(self):
        self.closed = True
