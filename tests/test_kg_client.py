"""The MCP HTTP client for the hosted KG, against a fake HTTP session (no network).

The reply texts below are copied from real server replies (2026-11-14), with the session token
replaced by a fake one.
"""

from __future__ import annotations

import json

import pytest
import requests

from fedkg_spacebio.agent.kg_client import (
    KgClient,
    KgQueryError,
    KgSessionError,
    KgTransportError,
    parse_query_text,
    parse_sse,
)

SESSION_TEXT = (
    "session_id: FAKE-session-0001\n\nPass this value as the `session_id` argument on EVERY "
    "subsequent tool call."
)
CYPHER_ERROR = (
    "The query could not be executed (CypherSyntaxError). Check the Cypher syntax and property "
    "names against the schema (get_neo4j_schema / get_node_metadata)."
)
EXPIRED = "Error (expired session): Session expired. Sessions end after 60 minutes of inactivity."
ROWS = 'total_rows: 1\nrows:\n[{"g": "13170", "t": "10090"}]'


class FakeResponse:
    def __init__(self, status=200, text="", ctype="text/event-stream", headers=None):
        self.status_code = status
        self.text = text
        self.headers = {"Content-Type": ctype, **(headers or {})}

    def json(self):
        return json.loads(self.text)


def sse(rid, result):
    msg = json.dumps({"jsonrpc": "2.0", "id": rid, "result": result})
    return f"event: message\ndata: {msg}\n\n"


def tool_result(text, is_error=False):
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


class FakeHttp:
    """Answers by JSON-RPC method; ``tool_texts`` gives the text of successive ``query`` calls."""

    def __init__(self, tool_texts=(ROWS,), failures=(), session_texts=None):
        self.posts = []
        self.tool_texts = list(tool_texts)
        self.failures = list(failures)  # consumed before each POST: a status code or exception
        self.session_texts = list(session_texts or [SESSION_TEXT])

    def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append({"json": json, "headers": headers, "timeout": timeout})
        if self.failures:
            f = self.failures.pop(0)
            if isinstance(f, Exception):
                raise f
            return FakeResponse(status=f, text="")
        method = json["method"]
        rid = json.get("id")
        if method == "initialize":
            return FakeResponse(text=sse(rid, {"capabilities": {}}))
        if method == "notifications/initialized":
            return FakeResponse(status=202, text="", ctype="")
        name = json["params"]["name"]
        if name == "create_session":
            texts = self.session_texts
            text = texts.pop(0) if len(texts) > 1 else texts[0]
            return FakeResponse(text=sse(rid, tool_result(text)))
        if name == "end_session":
            return FakeResponse(text=sse(rid, tool_result("Session ended.")))
        return FakeResponse(text=sse(rid, tool_result(self.tool_texts.pop(0))))

    def methods(self):
        out = []
        for p in self.posts:
            j = p["json"]
            out.append(j["params"]["name"] if j["method"] == "tools/call" else j["method"])
        return out


def client(http, **kw):
    sleeps = []
    c = KgClient(http=http, sleep=sleeps.append, **kw)
    return c, sleeps


# -- parsing -----------------------------------------------------------------------------------
def test_parse_sse_single_and_multiple_events():
    body = ('event: message\ndata: {"id": 1}\n\n'
            ': comment\nevent: message\ndata: {"id":\ndata: 2}\n\n')
    assert parse_sse(body) == [{"id": 1}, {"id": 2}]


def test_parse_query_text_rows():
    assert parse_query_text(ROWS) == (1, [{"g": "13170", "t": "10090"}])
    assert parse_query_text("total_rows: 0\nrows:\n[]") == (0, [])


def test_parse_splits_on_newline_rows_only():
    # "total_rows:" contains "rows:"; a value containing "\nrows:" must not split either.
    text = 'total_rows: 1\nrows:\n[{"note": "x\\nrows: y"}]'
    assert parse_query_text(text) == (1, [{"note": "x\nrows: y"}])


@pytest.mark.parametrize("text", [CYPHER_ERROR, "", "rows:\n[]", "total_rows: 1\n[]",
                                  "total_rows: x\nrows:\n[]", "total_rows: 1\nrows:\n{}",
                                  "total_rows: 1\nrows:\n[not json"])
def test_anything_but_rows_is_an_error(text):
    with pytest.raises(KgQueryError):
        parse_query_text(text)


def test_session_errors_are_recognised():
    with pytest.raises(KgSessionError):
        parse_query_text(EXPIRED)
    with pytest.raises(KgSessionError):
        parse_query_text("Error (unknown session): Unknown session_id.")


def test_row_count_mismatch_is_an_error():
    with pytest.raises(KgQueryError, match="truncated"):
        parse_query_text('total_rows: 1001\nrows:\n[{"a": 1}]')


# -- the conversation --------------------------------------------------------------------------
def test_handshake_then_query_with_session_id():
    http = FakeHttp()
    c, _ = client(http)
    assert c.query("MATCH (m) RETURN m", {"gene": "13170"}) == (1, [{"g": "13170", "t": "10090"}])
    assert http.methods() == ["initialize", "notifications/initialized", "create_session", "query"]
    q = http.posts[-1]["json"]["params"]["arguments"]
    assert q == {"query": "MATCH (m) RETURN m", "params": {"gene": "13170"},
                 "session_id": "FAKE-session-0001"}
    h = http.posts[-1]["headers"]
    assert h["Accept"] == "application/json, text/event-stream"
    assert "fedkg" in h["User-Agent"] and "@" not in h["User-Agent"]
    assert http.posts[-1]["timeout"] == (10.0, 75.0)


def test_params_are_sent_unchanged():
    # The client never coerces: the KG matches an integer gene to nothing, so kg_query must
    # validate and send strings (tested there). Here: what goes in is what goes out.
    http = FakeHttp(tool_texts=["total_rows: 0\nrows:\n[]"])
    c, _ = client(http)
    c.query("MATCH (m) RETURN m", {"gene": 13170})
    assert http.posts[-1]["json"]["params"]["arguments"]["params"] == {"gene": 13170}


def test_second_query_reuses_the_session():
    http = FakeHttp(tool_texts=[ROWS, ROWS])
    c, _ = client(http)
    c.query("A", {})
    c.query("B", {})
    assert http.methods().count("create_session") == 1 and http.methods().count("initialize") == 1


def test_cypher_error_is_raised_and_not_retried():
    http = FakeHttp(tool_texts=[CYPHER_ERROR])
    c, sleeps = client(http)
    with pytest.raises(KgQueryError, match="CypherSyntaxError"):
        c.query("bad", {})
    assert http.methods().count("query") == 1 and sleeps == []


def test_is_error_flag_is_an_error():
    class ErrHttp(FakeHttp):
        def post(self, url, json=None, headers=None, timeout=None):
            if json.get("method") == "tools/call" and json["params"]["name"] == "query":
                self.posts.append({"json": json})
                return FakeResponse(text=sse(json["id"], tool_result("boom", is_error=True)))
            return super().post(url, json=json, headers=headers, timeout=timeout)

    c, _ = client(ErrHttp())
    with pytest.raises(KgQueryError, match="boom"):
        c.query("A", {})


def test_expired_session_is_renewed_once():
    http = FakeHttp(tool_texts=[EXPIRED, ROWS],
                    session_texts=["session_id: FAKE-s1\n", "session_id: FAKE-s2\n"])
    c, _ = client(http)
    assert c.query("A", {})[0] == 1
    assert http.methods().count("create_session") == 2
    assert http.posts[-1]["json"]["params"]["arguments"]["session_id"] == "FAKE-s2"


def test_session_still_failing_after_renewal_raises():
    http = FakeHttp(tool_texts=[EXPIRED, EXPIRED])
    c, _ = client(http)
    with pytest.raises(KgSessionError):
        c.query("A", {})


def test_create_session_without_id_is_an_error():
    c, _ = client(FakeHttp(session_texts=["no id here"]))
    with pytest.raises(KgQueryError, match="session_id"):
        c.query("A", {})


def test_close_ends_the_session():
    http = FakeHttp()
    with KgClient(http=http, sleep=lambda s: None) as c:
        c.query("A", {})
    assert http.methods()[-1] == "end_session"
    assert http.posts[-1]["json"]["params"]["arguments"] == {"session_id": "FAKE-session-0001"}
    assert c.session_id is None


def test_plain_json_reply_is_accepted():
    class JsonHttp(FakeHttp):
        def post(self, url, json=None, headers=None, timeout=None):
            r = super().post(url, json=json, headers=headers, timeout=timeout)
            if r.text.startswith("event:"):
                payload = r.text.split("data: ", 1)[1].strip()
                return FakeResponse(text=payload, ctype="application/json")
            return r

    c, _ = client(JsonHttp())
    assert c.query("A", {})[0] == 1


def test_mcp_session_header_is_echoed():
    class HeaderHttp(FakeHttp):
        def post(self, url, json=None, headers=None, timeout=None):
            r = super().post(url, json=json, headers=headers, timeout=timeout)
            r.headers["Mcp-Session-Id"] = "FAKE-transport-1"
            return r

    http = HeaderHttp()
    c, _ = client(http)
    c.query("A", {})
    assert "Mcp-Session-Id" not in http.posts[0]["headers"]
    assert http.posts[-1]["headers"]["Mcp-Session-Id"] == "FAKE-transport-1"


# -- bounded retry -----------------------------------------------------------------------------
def test_retries_transient_failures_with_backoff():
    http = FakeHttp(failures=[503, requests.ConnectionError("reset")])
    c, sleeps = client(http, max_attempts=3, backoff_s=1.0)
    assert c.query("A", {})[0] == 1
    assert sleeps == [1.0, 2.0]


def test_gives_up_after_max_attempts():
    http = FakeHttp(failures=[503, 502, requests.Timeout("slow")])
    c, sleeps = client(http, max_attempts=3, backoff_s=0.5)
    with pytest.raises(KgTransportError, match="3 attempts"):
        c.query("A", {})
    assert sleeps == [0.5, 1.0]


def test_client_errors_are_not_retried():
    http = FakeHttp(failures=[400])
    c, sleeps = client(http)
    with pytest.raises(KgTransportError, match="HTTP 400"):
        c.query("A", {})
    assert sleeps == [] and len(http.posts) == 1


def test_reply_with_wrong_id_is_an_error():
    class WrongId(FakeHttp):
        def post(self, url, json=None, headers=None, timeout=None):
            if json.get("method") == "initialize":
                self.posts.append({"json": json})
                return FakeResponse(text=sse(999, {}))
            return super().post(url, json=json, headers=headers, timeout=timeout)

    c, _ = client(WrongId())
    with pytest.raises(KgTransportError, match="no JSON-RPC reply"):
        c.query("A", {})
