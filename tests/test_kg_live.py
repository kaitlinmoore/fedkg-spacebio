"""Live checks against the hosted GeneLab KG (``-m live``; skipped by default).

They check that the recordings in ``tests/fixtures/kg/`` still describe the live graph, and show
the integer-parameter trap on the real server. A failure here means the KG changed (or is down):
report it; don't edit the recordings to match.
"""

from __future__ import annotations

import pytest

from fedkg_spacebio.agent import kg_query as kq
from fedkg_spacebio.agent.kg_cache import KgCache, canonical_json, rows_sha256, sha256_hex
from fedkg_spacebio.agent.kg_client import KgClient
from kg_helpers import load_fixture

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def client():
    with KgClient() as c:
        yield c


def test_fingerprint_matches_the_recording(client):
    rec = load_fixture("kg_fingerprint")
    total, rows = client.query(kq.load_templates()["kg_fingerprint"]["cypher"], {})
    assert total == 1
    assert rows == rec["rows"], "KG fingerprint changed since the 2026-11-14 recording"
    assert sha256_hex(canonical_json(rows[0])) == rec["fingerprint_sha256"]


@pytest.mark.parametrize("name", ["gene_human_orthologs__99412", "gene_de_comparisons__13170"])
def test_template_rows_match_the_recording(client, name):
    rec = load_fixture(name)
    _, rows = client.query(kq.load_templates()[rec["template"]]["cypher"], rec["kg_params"])
    if rec["template"] == "gene_de_comparisons":
        # collect() order isn't guaranteed; compare after the tool's own sort.
        key = kq._sort_key
        rows[0]["rows"] = sorted(rows[0]["rows"], key=key)
        rec["rows"][0]["rows"] = sorted(rec["rows"][0]["rows"], key=key)
        assert rows == rec["rows"]
    else:
        assert rows_sha256(rows) == rec["rows_canonical_sha256"]


def test_integer_gene_parameter_matches_nothing(client):
    cypher = kq.load_templates()["gene_human_orthologs"]["cypher"]
    assert client.query(cypher, {"gene": 13170})[0] == 0
    assert client.query(cypher, {"gene": "13170"})[0] == 1


def test_tool_end_to_end_live(tmp_path):
    with kq.KgStore(KgCache(tmp_path), offline=False) as store:
        res = kq.kg_query("gene_human_orthologs", {"gene": "NCBIGene:13170"}, store=store)
    assert res["ok"] and res["cache"]["hit"] is False
    assert "kgedge:IS_ORTHOLOG_MGiG|NCBIGene:13170|NCBIGene:1628" in res["ids"]
