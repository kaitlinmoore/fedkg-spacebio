"""The KG cache pre-warm: gene selection from the results table, and the warm-then-offline path."""

from __future__ import annotations

import json

import pandas as pd

from fedkg_spacebio.agent import kg_query as kq
from fedkg_spacebio.agent.kg_cache import KgCache
from fedkg_spacebio.agent.kg_prewarm import compare_with_recordings, genes_from_results, prewarm
from kg_helpers import KG_FIXTURES, FakeClient


def test_genes_come_from_ncbi_gene_of_fe_hits_sorted_by_q(tmp_path):
    # Synthetic table: fake-range gene IDs (CONTRACTS §1), symbols that must be ignored.
    t = pd.DataFrame({
        "ncbi_gene": [9000000003, 9000000001, 9000000002, 9000000004, 9000000005],
        "symbol": ["Fake3", "Fake1", "Fake2", "Fake4", "Fake5"],
        "fe_q": [0.01, 0.01, 0.001, 0.2, 0.0001],
        "fe_hit": [True, True, True, False, True],
    })
    p = tmp_path / "per_gene.csv.gz"
    t.to_csv(p, index=False)
    assert genes_from_results(p, 3) == [
        "NCBIGene:9000000005", "NCBIGene:9000000002", "NCBIGene:9000000001"]
    assert len(genes_from_results(p, 25)) == 4  # non-hits never included


def test_prewarm_then_offline_serves_every_call(tmp_path):
    cache = KgCache(tmp_path)
    genes = ["NCBIGene:13170", "NCBIGene:99412", "NCBIGene:7157"]
    with kq.KgStore(cache, offline=False, client_factory=FakeClient) as store:
        calls = prewarm(genes, store)
    assert len(calls) == 6 and all(c["cache_hit"] is False for c in calls)
    assert [c["error"]["code"] for c in calls if not c["ok"]] == ["wrong_taxon", "wrong_taxon"]
    offline = kq.KgStore(cache, offline=True)
    for g in genes:
        for t in kq.agent_templates():
            assert kq.kg_query(t, {"gene": g}, store=offline)["cache"]["hit"] is True
    assert compare_with_recordings(cache, KG_FIXTURES) == []


def test_compare_reports_a_changed_gene(tmp_path):
    cache = KgCache(tmp_path)
    with kq.KgStore(cache, offline=False, client_factory=FakeClient) as store:
        prewarm(["NCBIGene:13170"], store)
    p = next(p for p in cache.entries()
             if json.loads(p.read_text())["template"] == "gene_human_orthologs")
    e = json.loads(p.read_text())
    e["rows"][0]["gene_symbol"] = "CHANGED"
    p.write_text(json.dumps(e))
    diffs = compare_with_recordings(cache, KG_FIXTURES)
    assert [(d["template"], d["gene"]) for d in diffs] == [("gene_human_orthologs", "13170")]
