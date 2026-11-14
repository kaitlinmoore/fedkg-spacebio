"""Helpers for the agent tests: build small synthetic traces (CONTRACTS §8).

Not a test module. IDs are fake (``trace:FAKE-0001``, ``FAKE:0001``, ``OSD-FAKE-7``) except where a
test is about the shape of a real form (client and mission slugs, comparison IDs).
"""

from __future__ import annotations

from pathlib import Path

from fedkg_spacebio.agent.trace import TraceWriter

TRACE = "trace:FAKE-0001"
QUESTION = "Which liver genes change with spaceflight across ISS missions?"


def fixed_clock():
    n = {"i": 0}

    def clock():
        n["i"] += 1
        return f"2026-11-20T15:{n['i'] // 60:02d}:{n['i'] % 60:02d}Z"
    return clock


def start(tmp_path: Path, trace_id: str = TRACE) -> TraceWriter:
    w = TraceWriter.create(tmp_path / "traces", trace_id=trace_id, clock=fixed_clock())
    w.append("run_start", question=QUESTION, policy_profile="restricted",
             agent_version="0.0.0-test", model=None,
             kg_version="SPOKE-GeneLab v0.3.1+20260311 (assumed); fp:FAKE",
             tools={"results.read": "0.1.0", "kg.query": "0.1.0"})
    return w


def tool(w: TraceWriter, name: str, data: dict, ids: list[str], *, ok: bool = True,
         inp: dict | None = None, call_id: str | None = None) -> str:
    """Record a tool call and its result; returns the call ID."""
    call_id = call_id or w.next_call_id()
    w.append("tool_call", call_id=call_id, tool=name, input=inp or {})
    w.append("tool_result", call_id=call_id, tool=name, tool_version="0.1.0", ok=ok,
             error=None if ok else {"code": "test_error", "message": "failed"},
             data=data if ok else {}, ids=ids if ok else [], cache=None)
    return call_id


def claim(cid: str, kind: str, text: str, *cites: tuple[str, str]) -> dict:
    return {"claim_id": cid, "kind": kind, "text": text,
            "citations": [{"id": i, "call_id": c} for i, c in cites]}


def report(w: TraceWriter, *claims: dict) -> None:
    w.append("report", claims=list(claims))


def example_trace(tmp_path: Path) -> tuple[TraceWriter, str, str]:
    """CONTRACTS §8's worked example: a results.read call (5) and a kg.query call (6)."""
    w = start(tmp_path)
    for _ in range(4):
        w.next_call_id()  # calls 1-4 happened earlier in the example; not recorded here
    c5 = tool(w, "results.read",
              {"method": "fe", "rows": [{"ncbi_gene": "FAKE:0001", "symbol": "Fake1",
                                         "estimate": -1.2, "q": 1e-6}]},
              ["FAKE:0001", "rel:FAKE-0001-1"])
    c6 = tool(w, "kg.query",
              {"rows": [{"mgene": "FAKE:0001", "gene": "FAKE:9001",
                         "edge": "kgedge:IS_ORTHOLOG_MGiG|FAKE:0001|FAKE:9001"}]},
              ["FAKE:0001", "FAKE:9001", "kgedge:IS_ORTHOLOG_MGiG|FAKE:0001|FAKE:9001"])
    return w, c5, c6


def assert_ids_cover_data(result: dict) -> None:
    """CONTRACTS §7: every ID-shaped token in ``data`` is in ``ids`` (so a claim can cite it),
    and every ``ids`` entry occurs in ``data`` (an ID listed but not returned would be citable
    without support). Uses the grounding checker's own scanner, slugs included."""
    from fedkg_spacebio.agent.idscan import data_tokens, is_pattern_id, strings

    ids = set(result["ids"])
    vocab = [i for i in ids if not is_pattern_id(i)]
    tokens = data_tokens(result["data"], vocab)
    missing = tokens - ids
    assert not missing, f"ID-shaped tokens in data but not in ids: {sorted(missing)[:10]}"
    extra = ids - tokens - set(strings(result["data"]))
    assert not extra, f"ids not present in data: {sorted(extra)[:10]}"
