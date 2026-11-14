"""The trace writer and reader (CONTRACTS §8): IDs, order, append-only, sealing, hash chain."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime

import pytest

from agent_helpers import TRACE, start, tool
from fedkg_spacebio.agent.trace import (
    TraceError,
    TraceWriter,
    mint_trace_id,
    read_trace,
    trace_path,
)
from fedkg_spacebio.ids import TRACE_ID_RE


def test_mint_trace_id_form():
    tid = mint_trace_id(datetime(2026, 11, 20, 15, 0, tzinfo=UTC), "0f3c9a12")
    assert tid == "trace:20261120T150000Z-0f3c9a12"
    assert TRACE_ID_RE.fullmatch(mint_trace_id())
    with pytest.raises(ValueError):
        mint_trace_id(datetime(2026, 11, 20))  # naive
    with pytest.raises(ValueError):
        mint_trace_id(token="XYZ")


def test_file_name_and_records(tmp_path):
    w = start(tmp_path)
    assert w.path == trace_path(TRACE, tmp_path / "traces")
    assert w.path.name == "trace_FAKE-0001.jsonl"
    tool(w, "sites.list", {"clients": []}, [])
    w.end("ok")
    recs = read_trace(w.path)
    assert [r["seq"] for r in recs] == [0, 1, 2, 3]
    assert [r["type"] for r in recs] == ["run_start", "tool_call", "tool_result", "run_end"]
    assert all(r["trace_id"] == TRACE for r in recs)
    assert recs[0]["prev_sha256"] is None
    assert all(b["prev_sha256"] == a["record_sha256"] for a, b in zip(recs, recs[1:], strict=False))
    assert recs[1]["call_id"] == f"{TRACE}/call/1"


def test_ids_are_minted_per_trace(tmp_path):
    w = start(tmp_path)
    assert w.next_call_id() == f"{TRACE}/call/1"
    assert w.next_call_id() == f"{TRACE}/call/2"
    assert w.next_decision_id() == f"{TRACE}/dec/1"
    assert w.next_decision_id() == f"{TRACE}/dec/2"


def test_existing_file_is_not_reused(tmp_path):
    start(tmp_path)
    with pytest.raises(FileExistsError):
        TraceWriter.create(tmp_path / "traces", trace_id=TRACE)


def test_run_start_first_and_only_first(tmp_path):
    w = TraceWriter.create(tmp_path, trace_id=TRACE)
    with pytest.raises(TraceError, match="first"):
        w.append("run_end", status="ok", error=None)
    w = start(tmp_path / "b")
    with pytest.raises(TraceError, match="first"):
        w.append("run_start", question="q", policy_profile="open", agent_version="0",
                 model=None, kg_version="x", tools={})


def test_unknown_type_and_missing_fields_refused(tmp_path):
    w = start(tmp_path)
    with pytest.raises(TraceError, match="unknown record type"):
        w.append("note", text="x")
    with pytest.raises(TraceError, match="lacks"):
        w.append("decision", decision_id=w.next_decision_id(), kind="plan", choice="a")
    with pytest.raises(TraceError, match="set by the writer"):
        w.append("run_end", status="ok", error=None, seq=99)


def test_ids_must_belong_to_this_trace(tmp_path):
    w = start(tmp_path)
    with pytest.raises(TraceError, match="isn't a call of"):
        w.append("tool_call", call_id="trace:FAKE-0002/call/1", tool="x", input={})
    with pytest.raises(TraceError, match="isn't a decision of"):
        w.append("decision", decision_id="dec:FAKE-0001-1", kind="plan", choice="a",
                 reason="r", based_on=[])
    with pytest.raises(TraceError, match="kind"):
        w.append("decision", decision_id=w.next_decision_id(), kind="whim", choice="a",
                 reason="r", based_on=[])


def test_call_id_not_reused(tmp_path):
    w = start(tmp_path)
    cid = tool(w, "sites.list", {}, [])
    with pytest.raises(TraceError, match="already used"):
        w.append("tool_call", call_id=cid, tool="sites.list", input={})


def test_nan_refused(tmp_path):
    w = start(tmp_path)
    cid = w.next_call_id()
    w.append("tool_call", call_id=cid, tool="results.read", input={})
    with pytest.raises(TraceError, match="NaN"):
        w.append("tool_result", call_id=cid, tool="results.read", tool_version="0.1.0",
                 ok=True, error=None, data={"p": float("nan")}, ids=[], cache=None)


def test_sealed_after_run_end(tmp_path):
    w = start(tmp_path)
    w.end("ok")
    assert w.ended
    with pytest.raises(TraceError, match="ended"):
        w.end("ok")
    assert not os.access(w.path, os.W_OK) or os.geteuid() == 0
    assert read_trace(w.path)[-1]["type"] == "run_end"


def _rewrite(path, fn):
    path.chmod(0o644)
    lines = path.read_text().splitlines()
    path.write_text("\n".join(fn(lines)) + "\n")


@pytest.mark.parametrize("tamper, match", [
    (lambda ls: ls[:1] + ls[2:], "seq"),                         # a record removed
    (lambda ls: [ls[0], ls[2], ls[1], *ls[3:]], "seq"),          # reordered
    (lambda ls: ls + [ls[-1]], "follows run_end"),               # appended after the end
])
def test_structure_tampering_caught(tmp_path, tamper, match):
    w = start(tmp_path)
    tool(w, "sites.list", {"clients": ["SpaceX-4"]}, ["SpaceX-4"])
    w.end("ok")
    _rewrite(w.path, tamper)
    with pytest.raises(TraceError, match=match):
        read_trace(w.path)


def test_edited_record_caught(tmp_path):
    """Adding an ID to a tool result after the fact (to make a citation pass) breaks the hash."""
    w = start(tmp_path)
    tool(w, "results.read", {"rows": []}, ["rel:FAKE-0001-1"])
    w.end("ok")

    def edit(lines):
        rec = json.loads(lines[2])
        rec["ids"].append("FAKE:0001")
        lines[2] = json.dumps(rec)
        return lines
    _rewrite(w.path, edit)
    with pytest.raises(TraceError, match="hash"):
        read_trace(w.path)


def test_recomputed_hash_breaks_the_chain(tmp_path):
    from fedkg_spacebio.agent.trace import record_sha256
    w = start(tmp_path)
    tool(w, "results.read", {"rows": []}, ["rel:FAKE-0001-1"])
    w.end("ok")

    def edit(lines):
        rec = json.loads(lines[2])
        rec["ids"].append("FAKE:0001")
        rec["record_sha256"] = record_sha256(rec)
        lines[2] = json.dumps(rec)
        return lines
    _rewrite(w.path, edit)
    with pytest.raises(TraceError, match="link"):
        read_trace(w.path)


def test_unfinished_trace_reads(tmp_path):
    """A run that crashed leaves a valid prefix."""
    w = start(tmp_path)
    tool(w, "sites.list", {}, [])
    assert len(read_trace(w.path)) == 3


def test_empty_file_refused(tmp_path):
    w = TraceWriter.create(tmp_path, trace_id=TRACE)
    with pytest.raises(TraceError, match="empty"):
        read_trace(w.path)
