"""The grounding checker (CONTRACTS §8 rules 1–5), on synthetic traces with fake IDs."""

from __future__ import annotations

import json

import pytest

from agent_helpers import TRACE, claim, example_trace, report, start, tool
from fedkg_spacebio.agent.grounding import CHECKER_VERSION, GroundingError, check_trace
from fedkg_spacebio.agent.trace import TraceError


def verdicts(g):
    return {c["claim_id"]: c["verdict"] for c in g["claims"]}


def rules(g, cid):
    c = next(c for c in g["claims"] if c["claim_id"] == cid)
    return sorted((f["rule"], f["id"]) for f in c["failures"])


def test_contract_worked_example(tmp_path):
    """§8's example: c1 and c2 supported; c3 fails rules 1 and 3; OSD-FAKE-7 is fabricated."""
    w, c5, c6 = example_trace(tmp_path)
    report(w,
           claim("c1", "result", "FAKE:0001 is lower in flight across these missions (FE).",
                 ("FAKE:0001", c5)),
           claim("c2", "kg_fact", "Its human ortholog is FAKE:9001.",
                 ("kgedge:IS_ORTHOLOG_MGiG|FAKE:0001|FAKE:9001", c6), ("FAKE:9001", c6)),
           claim("c3", "kg_fact", "FAKE:0001 is also changed in OSD-FAKE-7.",
                 ("OSD-FAKE-7", c6)))
    g = check_trace(w.path)
    assert c5 == f"{TRACE}/call/5" and c6 == f"{TRACE}/call/6"
    assert verdicts(g) == {"c1": "supported", "c2": "supported", "c3": "unsupported"}
    assert rules(g, "c3") == [(1, "OSD-FAKE-7"), (3, "FAKE:0001"), (3, "OSD-FAKE-7"),
                              (5, None)]
    assert g["fabricated_ids"] == ["OSD-FAKE-7"]
    assert g["supported_rate"] == pytest.approx(2 / 3)
    assert (g["n_claims"], g["n_supported"]) == (3, 2)
    assert g["checker_version"] == CHECKER_VERSION


def test_uncited_id_in_text(tmp_path):
    w, c5, c6 = example_trace(tmp_path)
    report(w, claim("c", "kg_fact", "FAKE:0001 maps to FAKE:9001.", ("FAKE:9001", c6)))
    g = check_trace(w.path)
    assert rules(g, "c") == [(3, "FAKE:0001")]
    assert g["fabricated_ids"] == []  # uncited, but present in a result: not invented


def test_right_id_wrong_tool_kind(tmp_path):
    """FAKE:0001 is in kg.query's ids, but a result claim needs results.read."""
    w, c5, c6 = example_trace(tmp_path)
    report(w, claim("c", "result", "FAKE:0001 is lower in flight.", ("FAKE:0001", c6)))
    g = check_trace(w.path)
    assert rules(g, "c") == [(2, "FAKE:0001"), (3, "FAKE:0001"), (5, None)]


def test_id_present_in_a_different_call(tmp_path):
    """FAKE:9001 is in call 6, not call 5: citing it from call 5 fails rule 1."""
    w, c5, c6 = example_trace(tmp_path)
    report(w, claim("c", "result", "FAKE:9001 changed.", ("FAKE:9001", c5)))
    g = check_trace(w.path)
    assert (1, "FAKE:9001") in rules(g, "c")
    assert g["fabricated_ids"] == []


def test_failed_call_supports_nothing(tmp_path):
    w = start(tmp_path)
    bad = tool(w, "kg.query", {}, [], ok=False)
    report(w, claim("c", "kg_fact", "FAKE:0001 has no ortholog.", ("FAKE:0001", bad)))
    g = check_trace(w.path)
    c = g["claims"][0]
    assert c["verdict"] == "unsupported"
    assert "failed" in c["failures"][0]["reason"]
    assert g["fabricated_ids"] == ["FAKE:0001"]


def test_fabricated_cited_id(tmp_path):
    w, c5, c6 = example_trace(tmp_path)
    report(w, claim("c", "result", "FAKE:0042 is a hit.", ("FAKE:0042", c5)))
    assert check_trace(w.path)["fabricated_ids"] == ["FAKE:0042"]


def test_fabricated_uncited_id(tmp_path):
    """An invented ID nobody cited still counts as fabricated."""
    w, c5, c6 = example_trace(tmp_path)
    report(w, claim("c", "result", "FAKE:0001 is a hit, like FAKE:0077.", ("FAKE:0001", c5)))
    g = check_trace(w.path)
    assert rules(g, "c") == [(3, "FAKE:0077")]
    assert g["fabricated_ids"] == ["FAKE:0077"]


def test_comparison_id_before_bare_accession(tmp_path):
    cmp_id = "OSD-379-d6b14f9ef8590e45a17dbc53c87dafd8"
    w = start(tmp_path)
    c = tool(w, "kg.query", {"rows": [{"comparison": cmp_id}]}, [cmp_id])
    report(w, claim("c", "kg_fact", f"Reported changed in {cmp_id}.", (cmp_id, c)))
    g = check_trace(w.path)
    assert verdicts(g) == {"c": "supported"}  # OSD-379 isn't a separate uncited token


def test_edge_key_needs_only_the_key(tmp_path):
    w, c5, c6 = example_trace(tmp_path)
    edge = "kgedge:IS_ORTHOLOG_MGiG|FAKE:0001|FAKE:9001"
    report(w, claim("c", "kg_fact", f"Ortholog edge {edge}.", (edge, c6)))
    assert verdicts(check_trace(w.path)) == {"c": "supported"}


def test_client_and_site_slugs(tmp_path):
    w = start(tmp_path)
    s = tool(w, "sites.list", {"clients": [{"client": "SpaceX-16-ISST", "site": "SpaceX-16",
                                            "n_flight": 20}]},
             ["SpaceX-16-ISST", "SpaceX-16"])
    report(w,
           claim("ok", "site", "SpaceX-16-ISST has 20 flight animals.", ("SpaceX-16-ISST", s)),
           claim("site_only", "site", "SpaceX-16-ISST has 20 flight animals.",
                 ("SpaceX-16", s)),
           claim("both", "site", "SpaceX-16 (client SpaceX-16-ISST) has 20 flight animals.",
                 ("SpaceX-16", s), ("SpaceX-16-ISST", s)))
    g = check_trace(w.path)
    assert verdicts(g) == {"ok": "supported", "site_only": "unsupported", "both": "supported"}
    assert rules(g, "site_only") == [(3, "SpaceX-16-ISST")]


def test_fabricated_mission_slug(tmp_path):
    w = start(tmp_path)
    r = tool(w, "results.read", {"rows": [], "release_id": "rel:FAKE-0001-1"},
             ["rel:FAKE-0001-1"])
    report(w, claim("c", "caveat", "The result holds without SpaceX-99.",
                    ("rel:FAKE-0001-1", r)))
    g = check_trace(w.path)
    assert rules(g, "c") == [(3, "SpaceX-99")]
    assert g["fabricated_ids"] == ["SpaceX-99"]


def test_trailing_punctuation(tmp_path):
    w, c5, c6 = example_trace(tmp_path)
    report(w, claim("c", "result", "Lower in flight: FAKE:0001.", ("FAKE:0001", c5)))
    assert verdicts(check_trace(w.path)) == {"c": "supported"}


def test_gene_symbols_are_names_not_ids(tmp_path):
    """Nkx2-1 / NKX2-5 look like slugs, but they are symbols in the trace's results."""
    w = start(tmp_path)
    k = tool(w, "kg.query",
             {"gene": "FAKE:0001", "gene_symbol": "Nkx2-1",
              "orthologs": [{"gene": "FAKE:9001", "gene_symbol": "NKX2-5",
                             "edge": "kgedge:IS_ORTHOLOG_MGiG|FAKE:0001|FAKE:9001"}]},
             ["FAKE:0001", "FAKE:9001", "kgedge:IS_ORTHOLOG_MGiG|FAKE:0001|FAKE:9001"])
    report(w, claim("c", "kg_fact", "FAKE:0001 (Nkx2-1) has human ortholog FAKE:9001 (NKX2-5).",
                    ("FAKE:0001", k), ("FAKE:9001", k)))
    g = check_trace(w.path)
    assert verdicts(g) == {"c": "supported"}
    assert g["fabricated_ids"] == []


def test_symbol_not_in_any_result_is_slug_shaped(tmp_path):
    w, c5, c6 = example_trace(tmp_path)
    report(w, claim("c", "result", "FAKE:0001 behaves like Nkx2-1.", ("FAKE:0001", c5)))
    g = check_trace(w.path)
    assert rules(g, "c") == [(3, "Nkx2-1")]
    assert g["fabricated_ids"] == ["Nkx2-1"]


def test_citing_a_symbol_fails(tmp_path):
    w, c5, c6 = example_trace(tmp_path)
    report(w, claim("c", "result", "Fake1 is lower in flight.", ("Fake1", c5)))
    g = check_trace(w.path)
    assert rules(g, "c") == [(1, "Fake1"), (5, None)]
    assert g["fabricated_ids"] == []  # a symbol is never "fabricated"


def test_result_after_the_report_does_not_count(tmp_path):
    w = start(tmp_path)
    call = w.next_call_id()
    w.append("tool_call", call_id=call, tool="results.read", input={})
    report(w, claim("c", "result", "FAKE:0001 is a hit.", ("FAKE:0001", call)))
    w.append("tool_result", call_id=call, tool="results.read", tool_version="0.1.0", ok=True,
             error=None, data={"rows": [{"ncbi_gene": "FAKE:0001"}]}, ids=["FAKE:0001"],
             cache=None)
    g = check_trace(w.path)
    assert rules(g, "c")[0] == (1, "FAKE:0001")
    assert g["fabricated_ids"] == ["FAKE:0001"]


def test_call_id_from_another_trace(tmp_path):
    w, c5, c6 = example_trace(tmp_path)
    report(w, claim("c", "result", "FAKE:0001 is a hit.",
                    ("FAKE:0001", "trace:FAKE-0002/call/5")))
    assert rules(check_trace(w.path), "c")[0] == (1, "FAKE:0001")


def test_claim_without_citations(tmp_path):
    w, c5, c6 = example_trace(tmp_path)
    report(w, claim("c", "caveat", "Policies are invented for the testbed."))
    assert rules(check_trace(w.path), "c") == [(5, None)]


def test_unknown_kind(tmp_path):
    w, c5, c6 = example_trace(tmp_path)
    report(w, claim("c", "opinion", "FAKE:0001 matters.", ("FAKE:0001", c5)))
    assert rules(check_trace(w.path), "c") == [("kind", None)]


@pytest.mark.parametrize("kind, tool_name, ok", [
    ("policy", "policy.check", True), ("policy", "policy.check_plan", True),
    ("caveat", "policy.check", True), ("caveat", "results.read", True),
    ("site", "sites.list", True), ("site", "results.read", False),
    ("caveat", "kg.query", False), ("policy", "results.read", False),
])
def test_kind_to_tool_table(tmp_path, kind, tool_name, ok):
    w = start(tmp_path)
    c = tool(w, tool_name, {"x": "rel:FAKE-0001-1"}, ["rel:FAKE-0001-1"])
    report(w, claim("c", kind, "Released.", ("rel:FAKE-0001-1", c)))
    assert verdicts(check_trace(w.path))["c"] == ("supported" if ok else "unsupported")


def test_empty_report(tmp_path):
    w, c5, c6 = example_trace(tmp_path)
    report(w)
    g = check_trace(w.path)
    assert g["supported_rate"] is None and g["claims"] == [] and g["fabricated_ids"] == []


def test_needs_exactly_one_report(tmp_path):
    w, c5, c6 = example_trace(tmp_path)
    with pytest.raises(GroundingError):
        check_trace(w.path)
    report(w)
    report(w)
    with pytest.raises(GroundingError):
        check_trace(w.path)


def test_edited_trace_refused(tmp_path):
    """Adding the cited ID to a result after the fact doesn't make a claim pass."""
    w, c5, c6 = example_trace(tmp_path)
    report(w, claim("c", "result", "FAKE:0042 is a hit.", ("FAKE:0042", c5)))
    lines = w.path.read_text().splitlines()
    i = next(i for i, ln in enumerate(lines) if json.loads(ln).get("call_id") == c5
             and json.loads(ln)["type"] == "tool_result")
    rec = json.loads(lines[i])
    rec["ids"].append("FAKE:0042")
    lines[i] = json.dumps(rec)
    w.path.write_text("\n".join(lines) + "\n")
    with pytest.raises(TraceError):
        check_trace(w.path)
