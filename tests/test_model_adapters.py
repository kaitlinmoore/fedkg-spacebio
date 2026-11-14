"""Model interface and both adapters, against recorded-shape fake responses (no network).

The fake session records every request and answers from a queue. Response shapes follow each
provider's documented format (Anthropic Messages API; OpenAI-compatible chat completions).
"""

from __future__ import annotations

import json

import pytest
import requests

from fedkg_spacebio.agent import model_claude, model_openai
from fedkg_spacebio.agent.model import (
    AssistantTurn,
    ModelError,
    NameMap,
    ToolResult,
    ToolResults,
    ToolSpec,
    UserText,
    describe,
    inline_refs,
    model_call_fields,
    strict_compatible,
    tool_specs,
)

FAKE_KEY = "sk-FAKE-do-not-log-0123456789"


class FakeResponse:
    def __init__(self, status=200, payload=None, headers=None, text=None):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.text = text if text is not None else json.dumps(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


class FakeSession:
    def __init__(self, *responses):
        self.queue = list(responses)
        self.requests = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.requests.append({"url": url, "json": json, "headers": headers})
        r = self.queue.pop(0)
        if isinstance(r, Exception):
            raise r
        return r if isinstance(r, FakeResponse) else FakeResponse(200, r)


TOOLS = [
    ToolSpec("sites.list", "List clients.",
             {"type": "object", "properties": {"stratum": {"type": "string"}},
              "additionalProperties": False}),
    ToolSpec("policy.check", "Check a spec.",
             {"type": "object", "properties": {"spec": {"type": "object"}},
              "required": ["spec"], "additionalProperties": False}),
]


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    monkeypatch.setenv("NVIDIA_API_KEY", FAKE_KEY)


def claude(*responses, **kw):
    s = FakeSession(*responses)
    return model_claude.ClaudeClient(model_claude.settings(**kw), session=s,
                                     sleep=lambda _: None), s


def nim(*responses, **kw):
    s = FakeSession(*responses)
    return model_openai.OpenAICompatClient(
        model_openai.settings("meta/llama-3.3-70b-instruct", **kw), session=s,
        sleep=lambda _: None), s


CLAUDE_TOOL_USE = {
    "id": "msg_FAKE", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
    "content": [
        {"type": "thinking", "thinking": "", "signature": "FAKE-SIG"},
        {"type": "text", "text": "Checking sites."},
        {"type": "tool_use", "id": "toolu_FAKE1", "name": "sites_list",
         "input": {"stratum": "iss"}},
        {"type": "tool_use", "id": "toolu_FAKE2", "name": "policy_check",
         "input": {"spec": {"schema": "job-spec/1"}}},
    ],
    "stop_reason": "tool_use", "usage": {"input_tokens": 100, "output_tokens": 20},
}
CLAUDE_END = {**CLAUDE_TOOL_USE, "content": [{"type": "text", "text": "Done."}],
              "stop_reason": "end_turn"}

OPENAI_TOOL_CALLS = {
    "id": "chatcmpl-FAKE", "model": "meta/llama-3.3-70b-instruct",
    "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None,
        "tool_calls": [
            {"id": "call_FAKE1", "type": "function",
             "function": {"name": "sites_list", "arguments": "{\"stratum\": \"iss\"}"}},
            {"id": "call_FAKE2", "type": "function",
             "function": {"name": "policy_check", "arguments": "{\"spec\": {"}},
        ]}}],
    "usage": {"prompt_tokens": 100, "completion_tokens": 20},
}


# ------------------------------------------------------------------ shared pieces


def test_name_map_round_trip_and_collisions():
    m = NameMap(["sites.list", "policy.check_plan"])
    assert m.to_wire("policy.check_plan") == "policy_check_plan"
    assert m.from_wire("sites_list") == "sites.list"
    assert m.from_wire("made_up") == "made_up"
    with pytest.raises(ValueError, match="both map"):
        NameMap(["a.b", "a_b"])
    with pytest.raises(ValueError):
        NameMap(["bad name"])


def test_schema_helpers_on_real_tools():
    from fedkg_spacebio.agent.tool_policy import TOOLS as POLICY
    from fedkg_spacebio.agent.tool_results import TOOL as RESULTS
    from fedkg_spacebio.agent.tool_sites import TOOL as SITES
    from fedkg_spacebio.agent.tools import Registry
    specs = {t.name: t for t in tool_specs(Registry([SITES, RESULTS, *POLICY]))}
    flat = json.dumps(specs["results.read"].input_schema)
    assert "$ref" not in flat and "$defs" not in flat and '"title"' not in flat
    assert strict_compatible(specs["sites.list"].input_schema)
    assert not strict_compatible(specs["policy.check"].input_schema)  # a free-form spec
    assert inline_refs({"$defs": {"X": {"type": "integer", "title": "X"}},
                        "properties": {"a": {"$ref": "#/$defs/X"}}}) == \
        {"properties": {"a": {"type": "integer"}}}


# ------------------------------------------------------------------ Claude


def test_claude_request_shape():
    c, s = claude(CLAUDE_END)
    c.complete("You are careful.", [UserText("Which liver genes?")], TOOLS)
    req = s.requests[0]
    assert req["url"] == "https://api.anthropic.com/v1/messages"
    assert req["headers"]["x-api-key"] == FAKE_KEY
    assert req["headers"]["anthropic-version"] == "2026-10-21"
    assert "anthropic-beta" not in req["headers"]  # fallbacks off by default
    body = req["json"]
    assert body["model"] == "claude-opus-5-5" and body["system"] == "You are careful."
    assert body["tool_choice"] == {"type": "auto"}
    assert body["output_config"] == {"effort": "high"}
    assert "fallbacks" not in body
    assert not {"temperature", "top_p", "top_k"} & set(body)
    tools = {t["name"]: t for t in body["tools"]}
    assert tools["sites_list"]["strict"] is True
    assert "strict" not in tools["policy_check"]  # free-form object: not strict-compatible
    assert body["messages"] == [{"role": "user", "content": "Which liver genes?"}]


def test_claude_fallbacks_only_when_asked():
    c, s = claude(CLAUDE_END, fallbacks=True)
    c.complete("s", [UserText("q")], TOOLS)
    assert s.requests[0]["json"]["fallbacks"] == "default"
    assert s.requests[0]["headers"]["anthropic-beta"] == "server-side-fallback-2026-07-01"


def test_claude_parses_parallel_tool_calls():
    c, _ = claude(CLAUDE_TOOL_USE)
    r = c.complete("s", [UserText("q")], TOOLS)
    assert r.stop == "tool_calls" and r.served_model == "claude-opus-5-5"
    assert [(t.id, t.name, t.arguments) for t in r.turn.tool_calls] == [
        ("toolu_FAKE1", "sites.list", {"stratum": "iss"}),
        ("toolu_FAKE2", "policy.check", {"spec": {"schema": "job-spec/1"}})]
    assert r.turn.text == "Checking sites."
    assert r.usage["input_tokens"] == 100


def test_claude_sends_history_back_unchanged_and_results_in_one_message():
    c, s = claude(CLAUDE_TOOL_USE, CLAUDE_END)
    first = c.complete("s", [UserText("q")], TOOLS)
    results = ToolResults([ToolResult("toolu_FAKE1", '{"ok": true}'),
                           ToolResult("toolu_FAKE2", '{"ok": false}', is_error=True)])
    r = c.complete("s", [UserText("q"), first.turn, results], TOOLS)
    msgs = s.requests[1]["json"]["messages"]
    assert msgs[1] == {"role": "assistant", "content": CLAUDE_TOOL_USE["content"]}  # thinking kept
    assert msgs[2]["role"] == "user" and len(msgs[2]["content"]) == 2
    assert msgs[2]["content"][1] == {"type": "tool_result", "tool_use_id": "toolu_FAKE2",
                                     "content": '{"ok": false}', "is_error": True}
    assert "is_error" not in msgs[2]["content"][0]
    assert r.stop == "end" and r.turn.text == "Done."


def test_claude_refusal_is_a_stop_not_an_exception():
    c, _ = claude({**CLAUDE_END, "content": [], "stop_reason": "refusal",
                   "stop_details": {"type": "refusal", "category": "cyber",
                                    "explanation": "FAKE"}})
    r = c.complete("s", [UserText("q")], TOOLS)
    assert r.stop == "refusal" and r.detail == {"category": "cyber", "explanation": "FAKE"}
    rec = model_call_fields(c.settings, response=r)
    assert rec["error"]["code"] == "refusal" and rec["served_model"] == "claude-opus-5-5"


@pytest.mark.parametrize("reason, stop", [("max_tokens", "max_tokens"), ("pause_turn", "pause"),
                                          ("stop_sequence", "end")])
def test_claude_stop_reasons(reason, stop):
    c, _ = claude({**CLAUDE_END, "stop_reason": reason})
    assert c.complete("s", [UserText("q")], TOOLS).stop == stop


def test_claude_unknown_stop_reason_is_an_error():
    c, _ = claude({**CLAUDE_END, "stop_reason": "mystery"})
    with pytest.raises(ModelError) as e:
        c.complete("s", [UserText("q")], TOOLS)
    assert e.value.code == "bad_response"


# ------------------------------------------------------------------ OpenAI-compatible


def test_openai_request_shape():
    c, s = nim(OPENAI_TOOL_CALLS)
    c.complete("You are careful.", [UserText("q")], TOOLS)
    req = s.requests[0]
    assert req["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert req["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
    body = req["json"]
    assert body["messages"][0] == {"role": "system", "content": "You are careful."}
    assert body["tool_choice"] == "auto" and "parallel_tool_calls" not in body
    fn = body["tools"][0]
    assert fn["type"] == "function" and fn["function"]["name"] == "sites_list"
    assert "strict" not in fn["function"]  # off by default for compatible servers


def test_openai_parses_calls_and_flags_bad_json():
    c, _ = nim(OPENAI_TOOL_CALLS)
    r = c.complete("s", [UserText("q")], TOOLS)
    a, b = r.turn.tool_calls
    assert (a.name, a.arguments, a.parse_error) == ("sites.list", {"stratum": "iss"}, None)
    assert b.name == "policy.check" and b.arguments is None and "JSON" in b.parse_error
    assert r.stop == "tool_calls" and r.served_model == "meta/llama-3.3-70b-instruct"
    assert r.request_id == "chatcmpl-FAKE"


def test_openai_tool_calls_with_finish_reason_stop():
    payload = json.loads(json.dumps(OPENAI_TOOL_CALLS))
    payload["choices"][0]["finish_reason"] = "stop"
    c, _ = nim(payload)
    assert c.complete("s", [UserText("q")], TOOLS).stop == "tool_calls"


def test_openai_history_and_tool_messages():
    end = {"model": "m", "choices": [{"finish_reason": "stop",
                                      "message": {"role": "assistant", "content": "Done."}}]}
    c, s = nim(OPENAI_TOOL_CALLS, end)
    first = c.complete("s", [UserText("q")], TOOLS)
    results = ToolResults([ToolResult("call_FAKE1", '{"ok": true}'),
                           ToolResult("call_FAKE2", '{"ok": false}', is_error=True)])
    r = c.complete("s", [UserText("q"), first.turn, results], TOOLS)
    msgs = s.requests[1]["json"]["messages"]
    assert msgs[2] == OPENAI_TOOL_CALLS["choices"][0]["message"]
    assert msgs[3] == {"role": "tool", "tool_call_id": "call_FAKE1", "content": '{"ok": true}'}
    assert msgs[4]["tool_call_id"] == "call_FAKE2"
    assert r.stop == "end" and r.turn.text == "Done."


@pytest.mark.parametrize("reason, stop", [("length", "max_tokens"),
                                          ("content_filter", "refusal")])
def test_openai_stop_reasons(reason, stop):
    c, _ = nim({"model": "m", "choices": [{"finish_reason": reason,
                                           "message": {"role": "assistant", "content": ""}}]})
    assert c.complete("s", [UserText("q")], TOOLS).stop == stop


def test_turns_from_another_provider_are_translated():
    turn = AssistantTurn(text="", tool_calls=[], provider="anthropic", raw=None)
    from fedkg_spacebio.agent.model import ToolCall
    turn.tool_calls.append(ToolCall("toolu_X", "sites.list", {"stratum": "iss"},
                                    raw_arguments='{"stratum": "iss"}'))
    c, s = nim({"model": "m", "choices": [{"finish_reason": "stop",
                                           "message": {"role": "assistant", "content": "x"}}]})
    c.complete("s", [UserText("q"), turn], TOOLS)
    m = s.requests[0]["json"]["messages"][2]
    assert m["tool_calls"][0]["function"] == {"name": "sites_list",
                                              "arguments": '{"stratum": "iss"}'}


# ------------------------------------------------------------------ errors and keys


@pytest.mark.parametrize("make", [claude, nim])
def test_retries_then_succeeds(make):
    ok = CLAUDE_END if make is claude else OPENAI_TOOL_CALLS
    c, s = make(FakeResponse(529, {"error": {"message": "overloaded"}}),
                FakeResponse(429, {"error": {"message": "slow down"}}, {"retry-after": "3"}),
                requests.ConnectionError("reset"), FakeResponse(200, ok),
                max_retries=3)
    c.complete("s", [UserText("q")], TOOLS)
    assert len(s.requests) == 4


@pytest.mark.parametrize("make", [claude, nim])
@pytest.mark.parametrize("status, code, tries", [(400, "bad_request", 1), (401, "auth", 1),
                                                  (500, "server_error", 3)])
def test_errors_surface_with_codes(make, status, code, tries):
    body = {"error": {"message": f"bad thing; key was {FAKE_KEY}"}}
    c, s = make(*[FakeResponse(status, body, {"request-id": "req_FAKE",
                                               "x-request-id": "req_FAKE"})] * 3)
    with pytest.raises(ModelError) as e:
        c.complete("s", [UserText("q")], TOOLS)
    assert e.value.code == code and e.value.http_status == status
    assert len(s.requests) == tries
    assert FAKE_KEY not in str(e.value) and FAKE_KEY not in json.dumps(e.value.as_dict())
    assert e.value.request_id == "req_FAKE"


def test_missing_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    c, s = claude(CLAUDE_END)
    with pytest.raises(ModelError) as e:
        c.complete("s", [UserText("q")], TOOLS)
    assert e.value.code == "missing_api_key" and "ANTHROPIC_API_KEY" in e.value.message
    assert s.requests == []


def test_non_json_reply():
    c, _ = claude(FakeResponse(200, None, text="<html>"))
    with pytest.raises(ModelError) as e:
        c.complete("s", [UserText("q")], TOOLS)
    assert e.value.code == "bad_response"


def test_key_never_in_descriptions_or_records(tmp_path):
    from agent_helpers import start
    c, _ = claude(CLAUDE_TOOL_USE)
    r = c.complete("s", [UserText("q")], TOOLS)
    w = start(tmp_path)
    w.append("model_call", **model_call_fields(c.settings, response=r))
    w.append("model_call", **model_call_fields(
        c.settings, error=ModelError("auth", "denied", http_status=401)))
    text = w.path.read_text() + json.dumps(describe(c.settings)) + repr(c.settings)
    assert FAKE_KEY not in text
    assert describe(c.settings)["endpoint_host"] == "api.anthropic.com"
