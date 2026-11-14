"""Claude through Anthropic's native Messages API (``POST /v1/messages``), over plain HTTP.

Translation (neutral types in ``model.py``)
-------------------------------------------
- **Tools** → ``{name, description, input_schema}``, plus ``strict: true`` when the tool's schema
  closes every object (``model.strict_compatible``); strict guarantees the arguments validate
  against the schema. ``tool_choice`` is ``auto``: current Claude models refuse a forced tool
  choice, and the loop checks whether a call was made instead.
- **Tool calls** come back as ``tool_use`` content blocks ``{id, name, input}``; ``input`` is
  already a JSON object.
- **Tool results** go back in **one** user message, one ``tool_result`` block per call
  (``tool_use_id``, ``content``, ``is_error``). Splitting them across messages teaches the model
  to stop making parallel calls.
- **The assistant's own turn is sent back unchanged** (``AssistantTurn.raw``: every content block,
  thinking blocks included). Current Claude models think on every request and check that earlier
  thinking blocks come back exactly as produced, so the history is append-only: nothing in an
  earlier turn is edited or dropped.

Stop reasons: ``tool_use`` → ``tool_calls``; ``end_turn`` / ``stop_sequence`` → ``end``;
``max_tokens``; ``refusal`` (the model declined; ``stop_details`` kept in ``detail``);
``pause_turn`` → ``pause`` (send the conversation again to continue).

Settings: ``effort`` goes in ``output_config``. No ``temperature``/``top_p``/``top_k``: the API
version used here (``2026-10-21``) rejects them. **Refusal fallbacks** (the API answering with
another model when a request is declined) are off unless ``fallbacks`` is set; a refusal is then
recorded as a model error and the deterministic planner finishes the run. The model that
answered is recorded in every ``model_call`` either way.
"""

from __future__ import annotations

from typing import Any

from fedkg_spacebio.agent.model import (
    AssistantTurn,
    Message,
    ModelError,
    ModelResponse,
    ModelSettings,
    NameMap,
    ToolCall,
    ToolResults,
    ToolSpec,
    UserText,
    api_key,
    default_session,
    post_with_retries,
    strict_compatible,
)

PROVIDER = "anthropic"
API_VERSION = "2026-10-21"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
DEFAULT_BASE_URL = "https://api.anthropic.com"
STOP_MAP = {"tool_use": "tool_calls", "end_turn": "end", "stop_sequence": "end",
            "max_tokens": "max_tokens", "refusal": "refusal", "pause_turn": "pause"}


def settings(model: str = "claude-opus-5-5", **kw) -> ModelSettings:
    kw.setdefault("effort", "high")
    return ModelSettings(provider=PROVIDER, model=model,
                         base_url=kw.pop("base_url", DEFAULT_BASE_URL),
                         api_key_env=kw.pop("api_key_env", "ANTHROPIC_API_KEY"), **kw)


class ClaudeClient:
    def __init__(self, settings: ModelSettings, session=None, sleep=None):
        if settings.provider != PROVIDER:
            raise ValueError(f"ClaudeClient needs provider {PROVIDER!r}")
        self.settings = settings
        self.session = session or default_session()
        self._sleep = sleep

    # -------------------------------------------------------------- request

    def build_request(self, system: str, messages: list[Message], tools: list[ToolSpec],
                      names: NameMap) -> dict:
        s = self.settings
        body: dict[str, Any] = {
            "model": s.model, "max_tokens": s.max_tokens, "system": system,
            "messages": [m for m in (self._message(x, names) for x in messages) if m],
            "tools": [self._tool(t, names) for t in tools],
            "tool_choice": {"type": "auto"},
        }
        if s.effort:
            body["output_config"] = {"effort": s.effort}
        if s.fallbacks:
            body["fallbacks"] = "default"
        if not tools:
            del body["tools"], body["tool_choice"]
        return body

    def _tool(self, t: ToolSpec, names: NameMap) -> dict:
        out = {"name": names.to_wire(t.name), "description": t.description,
               "input_schema": t.input_schema}
        if self.settings.strict_tools and strict_compatible(t.input_schema):
            out["strict"] = True
        return out

    def _message(self, m: Message, names: NameMap) -> dict:
        if isinstance(m, UserText):
            return {"role": "user", "content": m.text}
        if isinstance(m, ToolResults):
            return {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": r.call_id, "content": r.content,
                 **({"is_error": True} if r.is_error else {})} for r in m.results]}
        if isinstance(m, AssistantTurn):
            if m.provider == PROVIDER and m.raw is not None:
                return {"role": "assistant", "content": m.raw}
            blocks: list[dict] = [{"type": "text", "text": m.text}] if m.text else []
            blocks += [{"type": "tool_use", "id": c.id, "name": names.to_wire(c.name),
                        "input": c.arguments or {}} for c in m.tool_calls]
            return {"role": "assistant", "content": blocks}
        raise TypeError(f"unknown message type {type(m).__name__}")

    def headers(self, key: str) -> dict:
        h = {"x-api-key": key, "anthropic-version": API_VERSION,
             "content-type": "application/json"}
        if self.settings.fallbacks:
            h["anthropic-beta"] = FALLBACK_BETA
        return h

    # -------------------------------------------------------------- response

    def parse_response(self, payload: dict, request_id: str | None,
                       names: NameMap) -> ModelResponse:
        content = payload.get("content")
        if not isinstance(content, list):
            raise ModelError("bad_response", "reply has no content list", request_id=request_id)
        texts, calls = [], []
        for block in content:
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                args = block.get("input")
                ok = isinstance(args, dict)
                calls.append(ToolCall(id=block.get("id", ""),
                                      name=names.from_wire(block.get("name", "")),
                                      arguments=args if ok else None,
                                      parse_error=None if ok else "input isn't a JSON object"))
        reason = payload.get("stop_reason")
        stop = STOP_MAP.get(reason)
        if stop is None:
            raise ModelError("bad_response", f"unknown stop_reason {reason!r}",
                             request_id=request_id)
        detail = {}
        if stop == "refusal" and isinstance(payload.get("stop_details"), dict):
            sd = payload["stop_details"]
            detail = {"category": sd.get("category"), "explanation": sd.get("explanation")}
        return ModelResponse(
            turn=AssistantTurn(text="".join(texts), tool_calls=calls, provider=PROVIDER,
                               raw=content),
            stop=stop, usage=dict(payload.get("usage") or {}),
            served_model=payload.get("model"), request_id=request_id, detail=detail)

    # -------------------------------------------------------------- the call

    def complete(self, system: str, messages: list[Message],
                 tools: list[ToolSpec]) -> ModelResponse:
        names = NameMap(t.name for t in tools)
        key = api_key(self.settings)
        body = self.build_request(system, messages, tools, names)
        kw = {"sleep": self._sleep} if self._sleep else {}
        payload, headers = post_with_retries(
            self.session, self.settings.base_url.rstrip("/") + "/v1/messages",
            headers=self.headers(key), body=body, settings=self.settings, key=key,
            error_message=lambda p: (p.get("error") or {}).get("message", str(p)),
            request_id_header="request-id", **kw)
        return self.parse_response(payload, headers.get("request-id"), names)
