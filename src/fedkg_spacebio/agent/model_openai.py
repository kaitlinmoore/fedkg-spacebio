"""Any OpenAI-compatible chat endpoint (``POST {base_url}/chat/completions``), e.g. NVIDIA NIM.

Translation (neutral types in ``model.py``)
-------------------------------------------
- **Tools** → ``{"type": "function", "function": {name, description, parameters}}``, plus
  ``strict: true`` only when the settings ask for it and the tool's schema allows it. Many
  compatible servers ignore or reject ``strict``, so it is off by default here, and our own code
  validates every argument either way.
- **Tool calls** come back in ``choices[0].message.tool_calls`` as ``{id, function: {name,
  arguments}}``, where ``arguments`` is a JSON **string**. A string that doesn't parse becomes a
  ``ToolCall`` with ``parse_error``; the tool isn't run.
- **Tool results** go back as one ``{"role": "tool", "tool_call_id", "content"}`` message per call.
  The format has no error flag: the content is the §7 envelope, whose ``ok: false`` says so.
- **The assistant's turn** is sent back as the provider returned it (``AssistantTurn.raw``).
- The system prompt is the first message, ``role: system``.

Stop reasons (``finish_reason``): ``tool_calls`` → ``tool_calls`` (also when a server reports
``stop`` but the message carries tool calls, which some do); ``stop`` → ``end``; ``length`` →
``max_tokens``; ``content_filter`` → ``refusal``.

Defaults for NIM: ``https://integrate.api.nvidia.com/v1``, key in ``NVIDIA_API_KEY``. Model names
are the provider's (e.g. ``meta/llama-3.3-70b-instruct``); the comparison records them as given.
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
    parse_arguments,
    post_with_retries,
    strict_compatible,
)

PROVIDER = "openai_compat"
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
STOP_MAP = {"tool_calls": "tool_calls", "function_call": "tool_calls", "stop": "end",
            "length": "max_tokens", "content_filter": "refusal"}


def settings(model: str, **kw) -> ModelSettings:
    kw.setdefault("strict_tools", False)
    return ModelSettings(provider=PROVIDER, model=model,
                         base_url=kw.pop("base_url", NIM_BASE_URL),
                         api_key_env=kw.pop("api_key_env", "NVIDIA_API_KEY"), **kw)


class OpenAICompatClient:
    def __init__(self, settings: ModelSettings, session=None, sleep=None):
        if settings.provider != PROVIDER:
            raise ValueError(f"OpenAICompatClient needs provider {PROVIDER!r}")
        self.settings = settings
        self.session = session or default_session()
        self._sleep = sleep

    # -------------------------------------------------------------- request

    def build_request(self, system: str, messages: list[Message], tools: list[ToolSpec],
                      names: NameMap) -> dict:
        s = self.settings
        msgs: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            msgs.extend(self._messages(m, names))
        body: dict[str, Any] = {"model": s.model, "max_tokens": s.max_tokens, "messages": msgs}
        if tools:
            body["tools"] = [self._tool(t, names) for t in tools]
            body["tool_choice"] = "auto"
            if s.parallel_tool_calls is not None:
                body["parallel_tool_calls"] = s.parallel_tool_calls
        return body

    def _tool(self, t: ToolSpec, names: NameMap) -> dict:
        fn = {"name": names.to_wire(t.name), "description": t.description,
              "parameters": t.input_schema}
        if self.settings.strict_tools and strict_compatible(t.input_schema):
            fn["strict"] = True
        return {"type": "function", "function": fn}

    def _messages(self, m: Message, names: NameMap) -> list[dict]:
        if isinstance(m, UserText):
            return [{"role": "user", "content": m.text}]
        if isinstance(m, ToolResults):
            return [{"role": "tool", "tool_call_id": r.call_id, "content": r.content}
                    for r in m.results]
        if isinstance(m, AssistantTurn):
            if m.provider == PROVIDER and m.raw is not None:
                return [m.raw]
            out: dict[str, Any] = {"role": "assistant", "content": m.text or None}
            if m.tool_calls:
                out["tool_calls"] = [
                    {"id": c.id, "type": "function",
                     "function": {"name": names.to_wire(c.name),
                                  "arguments": c.raw_arguments or "{}"}}
                    for c in m.tool_calls]
            return [out]
        raise TypeError(f"unknown message type {type(m).__name__}")

    # -------------------------------------------------------------- response

    def parse_response(self, payload: dict, request_id: str | None,
                       names: NameMap) -> ModelResponse:
        try:
            choice = payload["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError):
            raise ModelError("bad_response", "reply has no choices[0].message",
                             request_id=request_id) from None
        calls = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            raw = fn.get("arguments")
            if isinstance(raw, dict):  # some servers send an object, not a string
                args, err, raw_s = raw, None, None
            else:
                args, err = parse_arguments(raw)
                raw_s = raw
            calls.append(ToolCall(id=tc.get("id", ""), name=names.from_wire(fn.get("name", "")),
                                  arguments=args, raw_arguments=raw_s, parse_error=err))
        reason = choice.get("finish_reason")
        stop = "tool_calls" if calls else STOP_MAP.get(reason)
        if stop is None:
            raise ModelError("bad_response", f"unknown finish_reason {reason!r}",
                             request_id=request_id)
        return ModelResponse(
            turn=AssistantTurn(text=message.get("content") or "", tool_calls=calls,
                               provider=PROVIDER, raw=message),
            stop=stop, usage=dict(payload.get("usage") or {}),
            served_model=payload.get("model"), request_id=request_id or payload.get("id"))

    # -------------------------------------------------------------- the call

    def complete(self, system: str, messages: list[Message],
                 tools: list[ToolSpec]) -> ModelResponse:
        names = NameMap(t.name for t in tools)
        key = api_key(self.settings)
        body = self.build_request(system, messages, tools, names)
        kw = {"sleep": self._sleep} if self._sleep else {}
        payload, headers = post_with_retries(
            self.session, self.settings.base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                     "Accept": "application/json"},
            body=body, settings=self.settings, key=key,
            error_message=lambda p: ((p.get("error") or {}).get("message")
                                     if isinstance(p.get("error"), dict)
                                     else p.get("detail") or str(p)),
            request_id_header="x-request-id", **kw)
        return self.parse_response(payload, headers.get("x-request-id"), names)
