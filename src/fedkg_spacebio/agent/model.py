"""The agent's model interface: one internal shape, two provider adapters.

Why an interface at all
-----------------------
The agent loop shouldn't know which company's model it talks to. Two wire formats matter here:
the OpenAI-compatible chat format (NVIDIA NIM and most hosted providers speak it) and Anthropic's
Messages API (Claude's native format). They express the same three things differently:
- **tool definitions** (name, description, JSON Schema of the input);
- **tool calls** the model makes (an ID, a tool name, arguments);
- **tool results** sent back (the call ID and the result).

So the loop works with the small neutral types below, and each adapter translates them to and
from its wire format (``model_claude.py``, ``model_openai.py``). Provider and model are settings,
recorded in the trace's ``run_start`` (``describe``), and every request gets a ``model_call``
trace record (``model_call_fields``), including which model actually answered.

Alternatives considered: one OpenAI-compatible client for everything, using Anthropic's
OpenAI-compatibility layer for Claude (Anthropic describes that layer as for quick checks; it
ignores ``strict``, so tool arguments aren't guaranteed to match the schema); the official
``openai`` and ``anthropic`` SDKs (the usual route; left out here so the runtime adds no
dependency: both adapters are plain ``requests``).

Tool names
----------
Our tools are named with dots (``sites.list``); both APIs allow only ``[a-zA-Z0-9_-]`` in tool
names. ``NameMap`` maps ``.`` to ``_`` on the way out and back on the way in, and refuses a tool
set where two names would collide.

Tool arguments are always checked by our own code (the registry validates every input against
the tool's model before running it), even where the provider promises schema-valid arguments:
an OpenAI-compatible server may ignore ``strict``, and an argument string may not even be JSON.
A call whose arguments don't parse comes back as a ``ToolCall`` with ``parse_error`` set; the loop
returns an error result to the model and never runs the tool.

API keys
--------
Read from the environment variable named in the settings, at request time, and sent only in the
request header. They are never stored on an object, written to a trace, put in an exception or
logged; ``scrub`` removes the key from any provider error text before it is kept.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse

ADAPTER_VERSION = "0.1.0"
STOPS = ("tool_calls", "end", "max_tokens", "refusal", "pause")
RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504, 529})


# ------------------------------------------------------------------ neutral types


@dataclass(frozen=True)
class ToolSpec:
    name: str              # our name, e.g. "sites.list"
    description: str
    input_schema: dict


@dataclass
class ToolCall:
    id: str
    name: str                       # our name (mapped back), or the wire name if unknown
    arguments: dict | None          # None if the arguments didn't parse
    raw_arguments: str | None = None
    parse_error: str | None = None


@dataclass
class UserText:
    text: str


@dataclass
class AssistantTurn:
    text: str
    tool_calls: list[ToolCall]
    provider: str
    raw: Any = None   # the provider's own message content, sent back unchanged on replay


@dataclass
class ToolResult:
    call_id: str
    content: str      # JSON text of the §7 envelope
    is_error: bool = False


@dataclass
class ToolResults:
    results: list[ToolResult]


Message = UserText | AssistantTurn | ToolResults


@dataclass
class ModelResponse:
    turn: AssistantTurn
    stop: str                       # one of STOPS
    usage: dict
    served_model: str | None
    request_id: str | None
    detail: dict = field(default_factory=dict)  # e.g. refusal category


class ModelError(Exception):
    """A request that failed. ``message`` never contains the API key."""

    def __init__(self, code: str, message: str, *, http_status: int | None = None,
                 retryable: bool = False, request_id: str | None = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.http_status = http_status
        self.retryable = retryable
        self.request_id = request_id

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "http_status": self.http_status,
                "retryable": self.retryable, "request_id": self.request_id}


@dataclass(frozen=True)
class ModelSettings:
    provider: str                     # "anthropic" | "openai_compat"
    model: str
    base_url: str
    api_key_env: str
    max_tokens: int = 16000
    effort: str | None = None         # Claude: low | medium | high | xhigh | max
    fallbacks: bool = False           # Claude server-side refusal fallbacks; off by default
    strict_tools: bool = True         # ask for schema-exact arguments where a tool allows it
    parallel_tool_calls: bool | None = None   # OpenAI-compatible: send only if set
    timeout_s: float = 120.0
    max_retries: int = 2


class ModelClient(Protocol):
    settings: ModelSettings

    def complete(self, system: str, messages: list[Message],
                 tools: list[ToolSpec]) -> ModelResponse: ...


# ------------------------------------------------------------------ names and schemas


class NameMap:
    """Our tool names ↔ wire names (``.`` → ``_``)."""

    def __init__(self, names: Iterable[str]):
        self.to_wire_map: dict[str, str] = {}
        self.from_wire_map: dict[str, str] = {}
        for n in names:
            w = n.replace(".", "_")
            if not w or len(w) > 64 or not all(c.isalnum() or c in "_-" for c in w):
                raise ValueError(f"tool name {n!r} can't be sent as {w!r}")
            if w in self.from_wire_map:
                raise ValueError(f"tool names {self.from_wire_map[w]!r} and {n!r} both map to "
                                 f"{w!r}")
            self.to_wire_map[n] = w
            self.from_wire_map[w] = n

    def to_wire(self, name: str) -> str:
        return self.to_wire_map[name]

    def from_wire(self, wire: str) -> str:
        return self.from_wire_map.get(wire, wire)


def inline_refs(schema: dict) -> dict:
    """The schema with ``$ref``/``$defs`` resolved inline and ``title`` keys dropped.

    pydantic emits ``$defs`` for nested models; inlining keeps the schema within what every
    provider accepts.
    """
    defs = schema.get("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].rsplit("/", 1)[-1]
                return walk(defs[name])
            return {k: walk(v) for k, v in node.items() if k not in ("$defs", "title")}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node
    return walk(schema)


def strict_compatible(schema: Any) -> bool:
    """True if every object in the schema closes its properties (``additionalProperties: false``).

    A tool whose input includes a free-form object (``policy.check`` takes a whole job spec) can't
    be described in strict mode; it is sent without ``strict`` and checked by our own code.
    """
    if isinstance(schema, dict):
        if schema.get("type") == "object" and schema.get("additionalProperties", True) is not False:
            return False
        return all(strict_compatible(v) for k, v in schema.items() if k != "default")
    if isinstance(schema, list):
        return all(strict_compatible(v) for v in schema)
    return True


def tool_specs(registry) -> list[ToolSpec]:
    """``ToolSpec``s for every tool of an ``agent.tools.Registry``."""
    return [ToolSpec(t.name, t.description, inline_refs(t.input_schema()))
            for t in registry.tools.values()]


# ------------------------------------------------------------------ keys, HTTP, retries


def api_key(settings: ModelSettings) -> str:
    key = os.environ.get(settings.api_key_env)
    if not key:
        raise ModelError("missing_api_key", f"environment variable {settings.api_key_env} is "
                                            "not set")
    return key


def scrub(text: str, key: str | None) -> str:
    text = str(text)
    if key:
        text = text.replace(key, "[redacted]")
    return text[:500]


def _code_for(status: int) -> str:
    return {400: "bad_request", 401: "auth", 403: "auth", 404: "not_found", 408: "timeout",
            413: "too_large", 429: "rate_limited", 529: "overloaded"}.get(
        status, "server_error" if status >= 500 else "http_error")


def post_with_retries(session, url: str, *, headers: dict, body: dict, settings: ModelSettings,
                      key: str, error_message: Callable[[Any], str],
                      request_id_header: str, sleep: Callable[[float], None] = time.sleep):
    """POST, retrying connection errors, timeouts, 408/429/5xx/529 with backoff.

    Waits ``retry-after`` seconds when the provider sends it (capped at 60), else 1 s, 2 s, 4 s.
    Returns (parsed JSON, response headers). Raises ``ModelError``.
    """
    import requests

    attempt = 0
    while True:
        wait = None
        try:
            resp = session.post(url, json=body, headers=headers, timeout=settings.timeout_s)
        except requests.Timeout as exc:
            err = ModelError("timeout", scrub(exc, key), retryable=True)
        except requests.ConnectionError as exc:
            err = ModelError("connection_error", scrub(exc, key), retryable=True)
        else:
            rid = resp.headers.get(request_id_header)
            if resp.status_code == 200:
                try:
                    return resp.json(), resp.headers
                except ValueError:
                    raise ModelError("bad_response", "the provider's reply isn't JSON",
                                     http_status=200, request_id=rid) from None
            try:
                msg = error_message(resp.json())
            except (ValueError, AttributeError, TypeError):
                msg = resp.text
            err = ModelError(_code_for(resp.status_code), scrub(msg, key),
                             http_status=resp.status_code,
                             retryable=resp.status_code in RETRY_STATUS, request_id=rid)
            wait = resp.headers.get("retry-after")
        if not err.retryable or attempt >= settings.max_retries:
            raise err
        try:
            delay = min(float(wait), 60.0) if wait else 2.0 ** attempt
        except ValueError:
            delay = 2.0 ** attempt
        sleep(delay)
        attempt += 1


def default_session():
    import requests
    return requests.Session()


# ------------------------------------------------------------------ trace helpers


def describe(settings: ModelSettings) -> dict:
    """What ``run_start.model`` records: provider, model, endpoint host and settings. No key."""
    return {"provider": settings.provider, "model": settings.model,
            "endpoint_host": urlparse(settings.base_url).netloc,
            "adapter_version": ADAPTER_VERSION, "max_tokens": settings.max_tokens,
            "effort": settings.effort, "fallbacks": settings.fallbacks,
            "strict_tools": settings.strict_tools,
            "parallel_tool_calls": settings.parallel_tool_calls}


def model_call_fields(settings: ModelSettings, response: ModelResponse | None = None,
                      error: ModelError | None = None) -> dict:
    """Fields of a ``model_call`` trace record (no prompts, no keys)."""
    if response is not None:
        return {"provider": settings.provider, "model": settings.model,
                "served_model": response.served_model, "stop": response.stop,
                "usage": response.usage, "request_id": response.request_id,
                "n_tool_calls": len(response.turn.tool_calls),
                "error": ({"code": "refusal", **response.detail}
                          if response.stop == "refusal" else None)}
    return {"provider": settings.provider, "model": settings.model, "served_model": None,
            "stop": None, "usage": {}, "request_id": error.request_id if error else None,
            "n_tool_calls": 0, "error": error.as_dict() if error else None}


def parse_arguments(raw: str | None) -> tuple[dict | None, str | None]:
    """An OpenAI-style argument string → (dict, None) or (None, why it didn't parse)."""
    if raw is None or raw == "":
        return {}, None
    try:
        args = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"arguments aren't valid JSON ({exc.msg} at {exc.pos})"
    if not isinstance(args, dict):
        return None, "arguments must be a JSON object"
    return args, None
