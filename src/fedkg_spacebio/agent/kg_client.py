"""A small client for the hosted GeneLab KG MCP server (``mcp-genelab``), over plain HTTP.

What it talks to (checked from WSL on 2026-11-14)
-------------------------------------------------
``POST https://visualization.osdr.nasa.gov/kg/mcp``, the MCP "streamable HTTP" transport. Every
message is one JSON-RPC 2.0 request in the body; the reply comes back as server-sent events
(``event: message`` / ``data: {...}``), one event holding the JSON-RPC response.

The conversation, in order:
1. ``initialize`` (protocol handshake), then the ``notifications/initialized`` notification;
2. ``tools/call create_session``: the server's own application-level session. Its reply is text,
   ``session_id: <token>\\n\\n...``. The token goes in the ``session_id`` argument of every later
   call. It expires after 60 minutes idle or 8 hours in all;
3. ``tools/call query`` with ``{query, params, session_id}``, as many times as needed;
4. ``tools/call end_session`` when done.

The server sent no ``Mcp-Session-Id`` header, so the transport itself is stateless. If one ever
appears, this client echoes it back, as the MCP spec asks.

Traps found on the wire (each has a test)
-----------------------------------------
- **Errors are prose with ``isError: false``.** A Cypher error reads "The query could not be
  executed (CypherSyntaxError)..."; a session problem reads "Error (expired session): ...". So a
  ``query`` reply is accepted only if its text starts with ``total_rows:``. Anything else is an
  error, whatever ``isError`` says.
- **Parse after ``\\nrows:``**, not after ``rows:``: the first line, ``total_rows: N``, also
  contains ``rows:``.
- **Parameter types matter.** ``{"gene": 13170}`` (an integer) returns 0 rows and no error,
  because KG identifiers are strings. The client doesn't coerce; ``kg_query`` validates and always
  sends strings.
- The server caps a query at 1,000 rows and 60 s. A reply whose ``total_rows`` differs from the
  number of rows it carries is treated as an error rather than silently truncated.

Design choices (and the alternatives)
-------------------------------------
- **Plain ``requests``, not the ``mcp`` Python SDK.** The SDK would add a dependency (and an async
  stack) for four request types. The transport is simple enough to read in one file, and the
  project already depends on ``requests``.
- **Bounded retry:** up to ``max_attempts`` tries with exponential backoff, only for failures that
  might pass on their own (connection errors, timeouts, HTTP 429 and 5xx). A Cypher error or a
  bad parameter is never retried.
- **One session recovery:** if a call reports a missing, unknown or expired session, the client
  opens a new session and retries that call once.
- **Timeouts:** 10 s to connect, 75 s to read (the server's own query limit is 60 s).
- **Testability:** the HTTP session and the sleep function are injectable; tests pass fakes and
  never touch the network.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping
from typing import Any

import requests

KG_MCP_URL = "https://visualization.osdr.nasa.gov/kg/mcp"
MCP_PROTOCOL_VERSION = "2025-03-26"
USER_AGENT = "fedkg-spacebio-kg/0.1.0 (research testbed; python-requests)"
CLIENT_INFO = {"name": "fedkg-spacebio", "version": "0.1.0"}

RETRY_STATUS = {429, 500, 502, 503, 504}
_SESSION_RE = re.compile(r"session_id:\s*([A-Za-z0-9_-]+)")
_SESSION_ERROR_RE = re.compile(r"^Error \((missing|unknown|expired) session\)")


class KgTransportError(RuntimeError):
    """The server couldn't be reached, or answered with something that isn't a JSON-RPC reply."""


class KgQueryError(RuntimeError):
    """The server answered, but not with rows (Cypher error, session error, malformed text)."""


class KgSessionError(KgQueryError):
    """The server says our session is missing, unknown or expired."""


def parse_sse(body: str) -> list[dict[str, Any]]:
    """The JSON payloads of every ``data:`` event in an SSE body.

    Multi-line ``data:`` fields of one event are joined with newlines (SSE rules); events are
    separated by a blank line.
    """
    messages: list[dict[str, Any]] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            messages.append(json.loads("\n".join(buf)))
            buf.clear()

    for line in body.splitlines():
        if not line.strip():
            flush()
        elif line.startswith("data:"):
            buf.append(line[5:].removeprefix(" "))
        # "event:", "id:", "retry:" and ":" comments carry nothing we use.
    flush()
    return messages


def parse_query_text(text: str) -> tuple[int, list[Any]]:
    """Parse the ``query`` tool's text reply: ``total_rows: N\\nrows:\\n<JSON array>``.

    Raises ``KgSessionError`` for a session problem and ``KgQueryError`` for anything else that
    isn't rows (the server reports Cypher errors as prose, with ``isError: false``).
    """
    if _SESSION_ERROR_RE.match(text):
        raise KgSessionError(text)
    if not text.startswith("total_rows:"):
        raise KgQueryError(f"KG query failed: {text[:300]}")
    head, sep, tail = text.partition("\nrows:")
    if not sep:
        raise KgQueryError(f"KG reply has no rows section: {text[:300]}")
    try:
        total = int(head.removeprefix("total_rows:").strip())
        rows = json.loads(tail)
    except ValueError as exc:  # json.JSONDecodeError is a ValueError
        raise KgQueryError(f"KG reply could not be parsed: {exc}") from exc
    if not isinstance(rows, list):
        raise KgQueryError("KG reply rows are not a JSON array")
    if total != len(rows):
        raise KgQueryError(
            f"KG reply says total_rows={total} but carries {len(rows)} rows "
            "(server row cap reached?); refusing a truncated answer"
        )
    return total, rows


def _tool_text(result: Mapping[str, Any]) -> str:
    """Concatenate the text parts of an MCP ``tools/call`` result."""
    parts = [c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"]
    text = "".join(parts)
    if result.get("isError"):
        raise KgQueryError(f"KG tool reported an error: {text[:300]}")
    return text


class KgClient:
    """A session with the hosted KG. Use as a context manager so the session is ended.

    Only ``query`` (read-only Cypher) is exposed; which Cypher may run is decided by the caller's
    allow-list (``kg_query``), not here.
    """

    def __init__(
        self,
        url: str = KG_MCP_URL,
        *,
        http: requests.Session | None = None,
        connect_timeout_s: float = 10.0,
        read_timeout_s: float = 75.0,
        max_attempts: int = 3,
        backoff_s: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.url = url
        self.http = http or requests.Session()
        self.timeout = (connect_timeout_s, read_timeout_s)
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s
        self.sleep = sleep
        self._next_id = 0
        self._initialized = False
        self._mcp_session_header: str | None = None
        self.session_id: str | None = None

    # -- context manager ---------------------------------------------------------------------
    def __enter__(self) -> KgClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """End the server session (best effort) and drop it."""
        if self.session_id is not None:
            try:
                self._call_tool("end_session", {"session_id": self.session_id})
            except (KgTransportError, KgQueryError):
                pass
            self.session_id = None

    # -- JSON-RPC over HTTP ------------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": USER_AGENT,
        }
        if self._mcp_session_header:
            h["Mcp-Session-Id"] = self._mcp_session_header
        return h

    def _post(self, payload: Mapping[str, Any]) -> requests.Response:
        """POST with the bounded retry. Returns a 2xx response or raises ``KgTransportError``."""
        last: str = ""
        for attempt in range(1, self.max_attempts + 1):
            try:
                resp = self.http.post(
                    self.url, json=payload, headers=self._headers(), timeout=self.timeout
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                last = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code in RETRY_STATUS:
                    last = f"HTTP {resp.status_code}"
                elif resp.status_code >= 400:
                    raise KgTransportError(f"HTTP {resp.status_code} from {self.url}")
                else:
                    sid = resp.headers.get("Mcp-Session-Id")
                    if sid:
                        self._mcp_session_header = sid
                    return resp
            if attempt < self.max_attempts:
                self.sleep(self.backoff_s * 2 ** (attempt - 1))
        raise KgTransportError(f"{self.url} failed after {self.max_attempts} attempts ({last})")

    def _request(self, method: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        self._next_id += 1
        rid = self._next_id
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            payload["params"] = dict(params)
        resp = self._post(payload)
        ctype = resp.headers.get("Content-Type", "")
        try:
            if ctype.startswith("text/event-stream"):
                messages = parse_sse(resp.text)
            else:
                messages = [resp.json()]
        except ValueError as exc:
            raise KgTransportError(f"unparseable reply to {method}: {exc}") from exc
        for msg in messages:
            if isinstance(msg, dict) and msg.get("id") == rid:
                if "error" in msg:
                    raise KgTransportError(f"JSON-RPC error for {method}: {msg['error']}")
                return msg.get("result", {})
        raise KgTransportError(f"no JSON-RPC reply with id {rid} to {method}")

    def _notify(self, method: str) -> None:
        self._post({"jsonrpc": "2.0", "method": method})

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": CLIENT_INFO,
            },
        )
        self._notify("notifications/initialized")
        self._initialized = True

    def _call_tool(self, name: str, arguments: Mapping[str, Any]) -> str:
        self._ensure_initialized()
        result = self._request("tools/call", {"name": name, "arguments": dict(arguments)})
        return _tool_text(result)

    # -- sessions and queries ----------------------------------------------------------------
    def open_session(self) -> str:
        """Create a new server session and remember its ID."""
        text = self._call_tool("create_session", {})
        m = _SESSION_RE.search(text)
        if not m:
            raise KgQueryError(f"create_session gave no session_id: {text[:200]}")
        self.session_id = m.group(1)
        return self.session_id

    def query(self, cypher: str, params: Mapping[str, Any] | None = None) -> tuple[int, list[Any]]:
        """Run one read-only Cypher query; return ``(total_rows, rows)``.

        ``params`` are passed through unchanged (no type coercion: see the integer trap above).
        """
        for attempt in (1, 2):
            if self.session_id is None:
                self.open_session()
            args = {"query": cypher, "params": dict(params or {}), "session_id": self.session_id}
            text = self._call_tool("query", args)
            try:
                return parse_query_text(text)
            except KgSessionError:
                self.session_id = None
                if attempt == 2:
                    raise
        raise AssertionError("unreachable")  # pragma: no cover
