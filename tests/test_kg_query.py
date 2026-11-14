"""``kg.query`` (CONTRACTS §7; D-038), offline against the recorded KG responses.

Covers: the allow-list and its hashes; canonical-JSON hashes of the recordings; CURIE conversion;
edge keys; the §7 ID scan for every recording and option; ortholog cases; strict-comparison
labels (relation to inputs, liver, repeats); the row cap; parameter validation; the cache key and
offline misses; and the live path through a fake client.
"""

from __future__ import annotations

import copy
import json

import pytest

from fedkg_spacebio.agent import kg_query as kq
from fedkg_spacebio.agent.envelope import ToolError, envelope, unique_ids
from fedkg_spacebio.agent.kg_cache import (
    KgCache,
    _js_number,
    cache_key,
    canonical_json,
    rows_sha256,
    sha256_hex,
)
from id_scan import assert_ids_cover_data, id_tokens
from kg_helpers import FakeClient, fixture_cache, load_fixture, recorded

FP = load_fixture("kg_fingerprint")


def REC_ID(r):
    return f"{r['template']}-{r['kg_params']['gene']}"


@pytest.fixture
def store(tmp_path):
    return kq.KgStore(fixture_cache(tmp_path), offline=True)


def run(store, template, **params):
    return kq.kg_query(template, params, store=store, call_id="trace:FAKE-0001/call/1")


# -- allow-list and hashes ---------------------------------------------------------------------
def test_templates_load_and_match_the_recordings():
    t = kq.load_templates()
    assert set(t) == {"gene_human_orthologs", "gene_de_comparisons", "kg_fingerprint"}
    assert kq.agent_templates() == ["gene_de_comparisons", "gene_human_orthologs"]
    for rec in recorded() + [FP]:
        assert t[rec["template"]]["cypher_sha256"] == rec["cypher_sha256"]
        assert sha256_hex(t[rec["template"]]["cypher"]) == rec["cypher_sha256"]


@pytest.mark.parametrize("rec", recorded(), ids=REC_ID)
def test_canonical_rows_hash_reproduces_the_recording(rec):
    assert rows_sha256(rec["rows"]) == rec["rows_canonical_sha256"]
    assert rec["kg_fingerprint_sha256"] == FP["fingerprint_sha256"]


def test_fingerprint_hash_reproduces():
    assert sha256_hex(canonical_json(FP["rows"][0])) == FP["fingerprint_sha256"]


@pytest.mark.parametrize("x, s", [
    (2.0, "2"), (0.1, "0.1"), (-0.5, "-0.5"), (100.0, "100"), (123.456, "123.456"),
    (1e-6, "0.000001"), (1.234e-6, "0.000001234"), (1.5e-7, "1.5e-7"),
    (1.0654885962300702e-09, "1.0654885962300702e-9"), (1e21, "1e+21"),
    (1e20, "100000000000000000000"),
    (0.0, "0"),
])
def test_numbers_are_formatted_as_ecmascript_does(x, s):
    assert _js_number(x) == s


def test_canonical_json_keeps_integers_exact():
    # The fingerprint's 48-bit checksum sum exceeds 2**53; it must not pass through a float.
    assert canonical_json({"b": 1862439281406715662, "a": [True, None, "é"]}) == (
        '{"a":[true,null,"é"],"b":1862439281406715662}'
    )


# -- IDs ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("curie, bare", [("NCBIGene:13170", "13170"), ("NCBIGene:1", "1")])
def test_gene_curie_to_kg(curie, bare):
    assert kq.gene_to_kg(curie) == bare
    assert kq.kg_gene_to_curie(bare) == curie


@pytest.mark.parametrize("bad", [
    13170, "13170", "NCBIGene:0123", "NCBIGene:", "ENSEMBL:ENSMUSG00000059824", "Dbp",
    "FAKE:0001", "ncbigene:13170", "NCBIGene:13170 ", "NCBIGene:12345678901", None,
])
def test_gene_curie_rejects(bad):
    with pytest.raises(ToolError) as e:
        kq.gene_to_kg(bad)
    assert e.value.code == "invalid_param"


def test_edge_key_round_trips_and_refuses_pipes():
    k = kq.edge_key("IS_ORTHOLOG_MGiG", "NCBIGene:13170", "NCBIGene:1628")
    assert k == "kgedge:IS_ORTHOLOG_MGiG|NCBIGene:13170|NCBIGene:1628"
    assert kq.parse_edge_key(k) == ("IS_ORTHOLOG_MGiG", "NCBIGene:13170", "NCBIGene:1628")
    for bad in ("Taf9|Ak6", "", "a b"):
        with pytest.raises(ValueError):
            kq.edge_key("IS_ORTHOLOG_MGiG", "NCBIGene:1", bad)


def test_every_edge_key_built_from_the_recordings_round_trips(store):
    n = 0
    for rec in recorded():
        g = f"NCBIGene:{rec['kg_params']['gene']}"
        res = run(store, rec["template"], gene=g, **(
            {"limit": 50} if rec["template"] == "gene_de_comparisons" else {}))
        for i in res["ids"]:
            if i.startswith("kgedge:"):
                rel, a, b = kq.parse_edge_key(i)
                assert kq.edge_key(rel, a, b) == i
                n += 1
    assert n > 300


# -- the §7 ID scan ----------------------------------------------------------------------------
OPTIONS = [{}, {"tissue_group": "liver"}, {"limit": 1}, {"limit": 50},
           {"tissue_group": "liver", "limit": 1}]


@pytest.mark.parametrize("rec", recorded(), ids=REC_ID)
def test_id_scan_every_recording(store, rec):
    g = f"NCBIGene:{rec['kg_params']['gene']}"
    opts = OPTIONS if rec["template"] == "gene_de_comparisons" else [{}]
    for o in opts:
        res = run(store, rec["template"], gene=g, **o)
        if res["ok"]:
            assert_ids_cover_data(res)
        else:
            assert res["ids"] == [] and res["data"] == {}


def test_symbols_never_in_ids(store):
    res = run(store, "gene_human_orthologs", gene="NCBIGene:99412")
    symbols = {o["gene_symbol"] for o in res["data"]["orthologs"]} | {res["data"]["gene_symbol"]}
    assert not symbols & set(res["ids"])
    res = run(store, "gene_de_comparisons", gene="NCBIGene:13170", limit=50)
    assert "Dbp" not in res["ids"]
    assert not {r["material_name"] for r in res["data"]["rows"]} & set(res["ids"])


def test_mission_slugs_are_in_ids(store):
    res = run(store, "gene_de_comparisons", gene="NCBIGene:13170", limit=50)
    missions = {m for r in res["data"]["rows"] for m in r["missions"]}
    assert {"SpaceX-4", "SpaceX-8"} <= missions <= set(res["ids"])


# -- gene_human_orthologs ----------------------------------------------------------------------
def test_one_ortholog(store):
    res = run(store, "gene_human_orthologs", gene="NCBIGene:13170")
    assert res["ok"] and res["tool"] == "kg.query" and res["call_id"] == "trace:FAKE-0001/call/1"
    d = res["data"]
    assert d["orthologs"] == [{
        "gene": "NCBIGene:1628", "gene_symbol": "DBP",
        "edge": "kgedge:IS_ORTHOLOG_MGiG|NCBIGene:13170|NCBIGene:1628",
    }]
    assert (d["n_orthologs"], d["one_to_many"]) == (1, False)
    assert res["ids"] == ["NCBIGene:13170", "NCBIGene:1628",
                          "kgedge:IS_ORTHOLOG_MGiG|NCBIGene:13170|NCBIGene:1628"]
    assert d["cypher_sha256"] == kq.load_templates()["gene_human_orthologs"]["cypher_sha256"]
    assert d["kg_version"] == "SPOKE-GeneLab v0.3.1+20260311 (assumed); fp:f0045ca84ae8"


def test_no_ortholog_is_ok_and_empty(store):
    res = run(store, "gene_human_orthologs", gene="NCBIGene:77059")
    assert res["ok"] and res["data"]["orthologs"] == [] and res["data"]["n_orthologs"] == 0
    assert res["ids"] == ["NCBIGene:77059"]


def test_one_to_many_returns_every_ortholog_sorted(store):
    res = run(store, "gene_human_orthologs", gene="NCBIGene:99412")
    orth = res["data"]["orthologs"]
    assert len(orth) == 19 and res["data"]["one_to_many"]
    nums = [int(o["gene"].split(":")[1]) for o in orth]
    assert nums == sorted(nums)
    assert sum(o["gene_symbol"] is None for o in orth) == 2  # no human MGene for two of them
    assert kq.NOTE_ONE_TO_MANY in res["data"]["notes"]


@pytest.mark.parametrize("template", ["gene_human_orthologs", "gene_de_comparisons"])
@pytest.mark.parametrize("gene, code", [("NCBIGene:999999999", "not_found"),
                                        ("NCBIGene:7157", "wrong_taxon")])
def test_not_found_and_wrong_taxon(store, template, gene, code):
    res = run(store, template, gene=gene)
    assert not res["ok"] and res["error"]["code"] == code
    assert res["ids"] == [] and res["cache"]["hit"] is True


# -- gene_de_comparisons -----------------------------------------------------------------------
# From the WS-14 design note's table (strict rows, all relations counted per row).
DESIGN = {  # gene: (strict, studies, input, same mission, other mission)
    13436: (6, 5, 2, 4, 0), 77059: (9, 6, 2, 7, 0), 99412: (11, 9, 1, 8, 2),
    70024: (12, 9, 1, 5, 6), 12660: (16, 13, 4, 10, 2), 13170: (35, 24, 2, 30, 3),
}


@pytest.mark.parametrize("gene", sorted(DESIGN))
def test_relation_counts_match_the_design_note(store, gene):
    res = run(store, "gene_de_comparisons", gene=f"NCBIGene:{gene}", limit=50)
    d = res["data"]
    rows, s = d["rows"], d["summary"]
    strict, studies, n_in, n_same, n_other = DESIGN[gene]
    assert (s["n_strict"], s["n_studies"], len(rows)) == (strict, studies, strict)
    raw = {k: sum(r["relation_to_inputs"] == k for r in rows) for k in kq.RELATIONS}
    assert (raw["phase3a_input"], raw["same_mission_as_input"], raw["other_mission"]) == (
        n_in, n_same, n_other)
    # Summary counts each repeat group once (approved 2026-11-14).
    uniq = {k: sum(r["relation_to_inputs"] == k and r["repeat_of"] is None for r in rows)
            for k in kq.RELATIONS}
    assert s["n_strict_by_relation"] == uniq
    assert sum(uniq.values()) == s["n_strict"] - s["n_repeats"]


def test_rows_sorted_by_adj_p_then_comparison(store):
    rows = run(store, "gene_de_comparisons", gene="NCBIGene:13170", limit=50)["data"]["rows"]
    keys = [(r["adj_p"], r["comparison"]) for r in rows]
    assert keys == sorted(keys)


def test_relation_to_inputs_all_five():
    assert kq.relation_to_inputs("OSD-379", ["SpaceX-16"]) == "phase3a_input"
    assert kq.relation_to_inputs("OSD-168", ["SpaceX-4"]) == "shares_animals_with_input"
    assert kq.relation_to_inputs("OSD-162", ["SpaceX-9", "SpaceX-8"]) == "same_mission_as_input"
    assert kq.relation_to_inputs("OSD-25", ["STS-135"]) == "other_mission"
    assert kq.relation_to_inputs("OSD-FAKE-1", []) == "no_mission"


def test_study_with_two_missions(store):
    rows = run(store, "gene_de_comparisons", gene="NCBIGene:13170", limit=50)["data"]["rows"]
    r = next(r for r in rows if r["study"] == "OSD-162")
    assert r["missions"] == ["SpaceX-8", "SpaceX-9"]
    assert r["relation_to_inputs"] == "same_mission_as_input"
    assert r["edges"]["conducted"] == ["kgedge:CONDUCTED_MIcS|SpaceX-8|OSD-162",
                                       "kgedge:CONDUCTED_MIcS|SpaceX-9|OSD-162"]


def test_row_edges(store):
    r = run(store, "gene_de_comparisons", gene="NCBIGene:13436")["data"]["rows"][0]
    c, s = r["comparison"], r["study"]
    assert r["edges"]["de"] == f"kgedge:MEASURED_DIFFERENTIAL_EXPRESSION_ASmMG|{c}|NCBIGene:13436"
    assert r["edges"]["performed"] == f"kgedge:PERFORMED_SpAS|{s}|{c}"
    assert r["edges"]["investigated"] == [f"kgedge:INVESTIGATED_ASiA|{c}|{t}" for t in r["tissues"]]


def test_repeats_point_at_a_returned_earlier_row(store):
    for gene in (77059, 99412, 12660, 13170):
        for o in OPTIONS:
            d = run(store, "gene_de_comparisons", gene=f"NCBIGene:{gene}", **o)["data"]
            pos = {r["comparison"]: i for i, r in enumerate(d["rows"])}
            for i, r in enumerate(d["rows"]):
                if r["repeat_of"] is not None:
                    assert r["study"] == "OSD-462"
                    assert pos[r["repeat_of"]] < i


def test_osd462_counted_once(store):
    d = run(store, "gene_de_comparisons", gene="NCBIGene:77059", limit=50)["data"]
    osd462 = [r for r in d["rows"] if r["study"] == "OSD-462"]
    assert len(osd462) == 3
    assert sum(r["repeat_of"] is None for r in osd462) == 1
    assert d["summary"]["n_repeats"] == 2


def test_liver_terms_and_filter(store):
    assert kq.LIVER_TERMS == ("UBERON:0002107", "UBERON:0006727", "UBERON:0001115")
    assert kq.is_liver(["UBERON:0006727"], None)
    assert kq.is_liver([], "UBERON:0001115")
    assert not kq.is_liver(["UBERON:0000955"], "UBERON:0000955")
    for gene in DESIGN:
        res = run(store, "gene_de_comparisons", gene=f"NCBIGene:{gene}", tissue_group="liver")
        d = res["data"]
        assert d["tissue_terms"] == list(kq.LIVER_TERMS)
        assert all(r["is_liver"] for r in d["rows"])
        assert d["summary"]["n_in_tissue_group"] == len(d["rows"])
        assert set(kq.LIVER_TERMS) <= set(res["ids"])


def test_any_tissue_has_no_term_list(store):
    d = run(store, "gene_de_comparisons", gene="NCBIGene:13436")["data"]
    assert d["tissue_group"] == "any" and d["tissue_terms"] is None


def test_truncation_drops_ids_of_cut_rows(store):
    full = run(store, "gene_de_comparisons", gene="NCBIGene:13170", limit=50)
    res = run(store, "gene_de_comparisons", gene="NCBIGene:13170")
    d = res["data"]
    assert d["limit"] == 25 and d["truncated"] and len(d["rows"]) == 25
    assert d["summary"] == {**full["data"]["summary"], "n_returned": 25}
    kept = {r["comparison"] for r in d["rows"]}
    cut = [r["comparison"] for r in full["data"]["rows"] if r["comparison"] not in kept]
    assert len(cut) == 10 and not set(cut) & set(res["ids"])
    assert not full["data"]["truncated"]


def test_gene_without_strict_rows_is_ok(store):
    res = run(store, "gene_de_comparisons", gene="NCBIGene:11470")
    assert res["ok"] and res["data"]["rows"] == [] and res["ids"] == ["NCBIGene:11470"]
    assert res["data"]["summary"]["n_strict"] == 0


def test_cell_type_material_without_anatomy_edge():
    # The recording for 7157 (human) holds an OSD-635 row whose material is a CL term with no
    # Anatomy edge. The tool refuses human genes, so relabel the taxon in memory to exercise it.
    raw = copy.deepcopy(load_fixture("gene_de_comparisons__7157")["rows"])
    raw[0]["mgene_taxon"] = "10090"
    data, ids = kq.comparisons_data(raw, "7157", "any", 50)
    r = next(r for r in data["rows"] if r["material_id"].startswith("CL:"))
    assert r["tissues"] == [] and r["edges"]["investigated"] == []
    assert r["material_id"] in ids
    assert id_tokens(data) <= set(ids)


def test_notes_are_present(store):
    d = run(store, "gene_de_comparisons", gene="NCBIGene:13436")["data"]
    assert kq.NOTE_DE_EDGE in d["notes"] and kq.NOTE_SAME_MISSION in d["notes"]
    assert d["scope"] == "flight_vs_control_strict"


# -- validation --------------------------------------------------------------------------------
@pytest.mark.parametrize("template, params, code", [
    ("free_cypher", {"gene": "NCBIGene:13170"}, "invalid_template"),
    ("kg_fingerprint", {}, "invalid_template"),
    (None, {}, "invalid_template"),
    ("gene_human_orthologs", {"gene": 13170}, "invalid_param"),
    ("gene_human_orthologs", {"gene": "13170"}, "invalid_param"),
    ("gene_human_orthologs", {}, "invalid_param"),
    ("gene_human_orthologs", {"gene": "NCBIGene:13170", "limit": 5}, "invalid_param"),
    ("gene_human_orthologs", ["NCBIGene:13170"], "invalid_param"),
    ("gene_de_comparisons", {"gene": "NCBIGene:13170", "limit": 0}, "invalid_param"),
    ("gene_de_comparisons", {"gene": "NCBIGene:13170", "limit": 51}, "invalid_param"),
    ("gene_de_comparisons", {"gene": "NCBIGene:13170", "limit": True}, "invalid_param"),
    ("gene_de_comparisons", {"gene": "NCBIGene:13170", "limit": "5"}, "invalid_param"),
    ("gene_de_comparisons", {"gene": "NCBIGene:13170", "tissue_group": "brain"}, "invalid_param"),
    ("gene_de_comparisons", {"gene": "NCBIGene:13170", "cypher": "MATCH (n) RETURN n"},
     "invalid_param"),
])
def test_invalid_calls_are_refused_before_any_lookup(store, template, params, code):
    res = kq.kg_query(template, params, store=store)
    assert not res["ok"] and res["error"]["code"] == code
    assert res["cache"] is None and res["ids"] == [] and res["data"] == {}


# -- cache and offline mode --------------------------------------------------------------------
def test_offline_hit_reports_the_cache_entry(store):
    res = run(store, "gene_human_orthologs", gene="NCBIGene:13170")
    rec = load_fixture("gene_human_orthologs__13170")
    key = cache_key("gene_human_orthologs", {"gene": "13170"}, rec["cypher_sha256"],
                    FP["fingerprint_sha256"])
    assert res["cache"] == {"hit": True, "key": f"sha256:{key}", "fetched": rec["fetched"]}


def test_offline_miss_is_an_error(tmp_path):
    store = kq.KgStore(fixture_cache(tmp_path, with_entries=False), offline=True)
    res = run(store, "gene_human_orthologs", gene="NCBIGene:13170")
    assert not res["ok"] and res["error"]["code"] == "cache_miss"


def test_offline_without_a_manifest_is_an_error(tmp_path):
    res = run(kq.KgStore(KgCache(tmp_path), offline=True), "gene_human_orthologs",
              gene="NCBIGene:13170")
    assert res["error"]["code"] == "cache_miss"


def test_offline_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("FEDKG_KG_OFFLINE", "1")
    assert kq.KgStore(KgCache(tmp_path)).offline
    monkeypatch.setenv("FEDKG_KG_OFFLINE", "0")
    assert not kq.KgStore(KgCache(tmp_path)).offline


def test_cache_key_changes_with_each_part():
    base = ("gene_human_orthologs", {"gene": "13170"}, "a" * 64, "b" * 64)
    k = cache_key(*base)
    assert cache_key(base[0], {"gene": "13170"}, *base[2:]) == k  # params order-insensitive
    assert cache_key("gene_de_comparisons", *base[1:]) != k
    assert cache_key(base[0], {"gene": "1628"}, *base[2:]) != k
    assert cache_key(base[0], {"gene": 13170}, *base[2:]) != k  # an int is a different request
    assert cache_key(*base[:2], "c" * 64, base[3]) != k
    assert cache_key(*base[:3], "c" * 64) != k


def test_changed_fingerprint_misses(tmp_path):
    cache = fixture_cache(tmp_path)
    fp2 = {**FP, "fingerprint_sha256": "0" * 64}
    cache.write_fingerprint(fp2)
    m = cache.read_manifest()
    assert m["current"]["fingerprint_sha256"] == "0" * 64
    assert m["history"][0]["fingerprint_sha256"] == FP["fingerprint_sha256"]
    res = run(kq.KgStore(cache, offline=True), "gene_human_orthologs", gene="NCBIGene:13170")
    assert res["error"]["code"] == "cache_miss"


def test_corrupt_entry_is_detected(tmp_path, store):
    path = store.cache.entries()[0]
    rec = json.loads(path.read_text())
    rec["rows"] = rec["rows"] + [{"tampered": True}]
    path.write_text(json.dumps(rec))
    template = rec["template"]
    res = run(store, template, gene=f"NCBIGene:{rec['kg_params']['gene']}")
    assert res["error"]["code"] == "cache_corrupt"


# -- live path, with a fake client -------------------------------------------------------------
def test_live_miss_then_hit(tmp_path):
    fake = FakeClient()
    clock = iter(range(0, 1000, 1)).__next__
    cache = KgCache(tmp_path)
    with kq.KgStore(cache, offline=False, client_factory=lambda: fake,
                    clock=lambda: clock() / 1000) as store:
        a = run(store, "gene_de_comparisons", gene="NCBIGene:13170")
        b = run(store, "gene_de_comparisons", gene="NCBIGene:13170", tissue_group="liver")
    assert fake.closed
    assert (a["cache"]["hit"], b["cache"]["hit"]) == (False, True)
    assert a["cache"]["key"] == b["cache"]["key"]  # one entry serves every option
    # The gene goes out as a string: an integer silently matches nothing in the KG.
    assert fake.sent == [{}, {"gene": "13170"}]
    assert cache.current_fingerprint()["fingerprint_sha256"] == FP["fingerprint_sha256"]
    entry = cache.get(a["cache"]["key"].removeprefix("sha256:"))
    assert entry.record["rows_canonical_sha256"] == load_fixture(
        "gene_de_comparisons__13170")["rows_canonical_sha256"]
    # And the result offline afterwards is the same.
    c = run(kq.KgStore(cache, offline=True), "gene_de_comparisons", gene="NCBIGene:13170")
    assert c["data"] == a["data"] and c["ids"] == a["ids"]


def test_live_transport_failure_becomes_an_error(tmp_path):
    from fedkg_spacebio.agent.kg_client import KgTransportError

    class Down:
        def query(self, *a):
            raise KgTransportError("down")

        def close(self):
            pass

    res = run(kq.KgStore(KgCache(tmp_path), offline=False, client_factory=Down),
              "gene_human_orthologs", gene="NCBIGene:13170")
    assert res["error"]["code"] == "transport_error"


# -- envelope ----------------------------------------------------------------------------------
def test_envelope_shape_and_rules():
    e = envelope("kg.query", "0.1.0", data={"x": 1}, ids=["A:1", "B:2", "A:1"])
    assert list(e) == ["call_id", "tool", "tool_version", "ok", "error", "data", "ids", "cache"]
    assert e["ok"] and e["error"] is None and e["ids"] == ["A:1", "B:2"] and e["cache"] is None
    f = envelope("kg.query", "0.1.0", error=ToolError("not_found", "x"))
    assert not f["ok"] and f["error"] == {"code": "not_found", "message": "x"}
    with pytest.raises(ValueError):
        envelope("t", "0", ids=["A:1"], error=ToolError("x", "y"))
    with pytest.raises(TypeError):
        unique_ids(["A:1", 2])
