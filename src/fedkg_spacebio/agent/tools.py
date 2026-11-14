"""The agent's tool registry: names, versions, input checks, and the trace records of every call.

Every tool the agent can call is registered here with a name (``sites.list``), a version, a
one-paragraph description (what a model sees), a pydantic model for its input, and a function
that returns the CONTRACTS §7 envelope (``agent/envelope.py``).

``Registry.call`` is the only way a tool runs inside an agent run. In order, it:
1. mints the call ID (``<trace id>/call/<n>``) and writes a ``tool_call`` record with the input
   exactly as given;
2. validates the input against the tool's model (unknown fields refused). A bad input becomes
   ``ok: false`` with code ``invalid_input``; the tool never runs on it;
3. runs the tool. A ``ToolError`` becomes ``ok: false`` with its code. Any other exception becomes
   ``internal_error`` with the exception type and message, so the run records it instead of
   stopping half-written;
4. writes the envelope as the ``tool_result`` record and returns it.

Why the registry and not each tool writes its own records: then no tool can forget to record a
call, and the input recorded is the one the model sent, before any tool-side normalisation.

The run-level settings (policy profile, counts variant, where the work directory and policy root
are) live in ``ToolContext``. The model never chooses them: a run has one profile, and a tool
input that could switch profile mid-run would let one run mix two release logs.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from fedkg_spacebio.agent.envelope import ToolError, envelope
from fedkg_spacebio.agent.trace import TraceWriter

REPO = Path(__file__).resolve().parents[3]
PROFILES = ("open", "restricted")


def _workdir() -> Path:
    return Path(os.environ.get("FEDKG_WORKDIR", str(Path.home() / "fedkg-work")))


@dataclass
class ToolContext:
    """Run-level settings every tool reads. Paths default to the work directory layout."""

    profile: str
    counts_variant: str = "standard"
    workdir: Path = field(default_factory=_workdir)
    policy_root: Path | None = None       # default: <workdir>/policy
    policies_dir: Path | None = None      # default: <repo>/policies/<profile>
    manifest_dir: Path | None = None      # default: the profile's manifest folder (§3)
    flags_dir: Callable[[str], Path] | None = None  # stratum → folder of released flags
    releases_dir: Path | None = None      # default: <workdir>/agent/releases
    trace: TraceWriter | None = None

    def __post_init__(self):
        if self.profile not in PROFILES:
            raise ValueError(f"unknown policy profile {self.profile!r}; expected {PROFILES}")
        self.workdir = Path(self.workdir)
        if self.policy_root is None:
            self.policy_root = self.workdir / "policy"
        if self.policies_dir is None:
            self.policies_dir = REPO / "policies" / self.profile
        if self.manifest_dir is None:
            part = () if self.profile == "open" else ("profiles", self.profile)
            self.manifest_dir = self.workdir.joinpath("phase3a", "manifests", *part,
                                                      self.counts_variant)
        if self.releases_dir is None:
            self.releases_dir = self.workdir / "agent" / "releases"

    def released_flags_dir(self, stratum: str) -> Path:
        """Where each client's released inclusion flags are (``sent/<client>.npz`` + meta).

        The flags released before any job don't depend on the policy profile (CONTRACTS §5.1
        ``site_metadata``), so every profile reads the open run's copies, as
        ``jobs/phase3a_meta/draft_spec.py`` does.
        """
        if self.flags_dir is not None:
            return self.flags_dir(stratum)
        return self.workdir.joinpath("results", "phase3a", stratum, self.counts_variant,
                                     "flare", "sent")


ToolFn = Callable[[Any, ToolContext, str | None], dict]


@dataclass(frozen=True)
class Tool:
    name: str
    version: str
    description: str
    input_model: type[BaseModel]
    fn: ToolFn

    def input_schema(self) -> dict:
        """JSON Schema of the input, as sent to a model (unknown fields refused)."""
        schema = self.input_model.model_json_schema()
        schema.setdefault("additionalProperties", False)
        return schema


class Registry:
    def __init__(self, tools: Iterable[Tool]):
        self.tools: dict[str, Tool] = {}
        for t in tools:
            if t.name in self.tools:
                raise ValueError(f"tool {t.name} registered twice")
            self.tools[t.name] = t

    def versions(self) -> dict[str, str]:
        """name → version, for the trace's ``run_start``."""
        return {n: t.version for n, t in sorted(self.tools.items())}

    def call(self, name: str, raw_input: Any, ctx: ToolContext) -> dict:
        trace = ctx.trace
        call_id = trace.next_call_id() if trace is not None else None
        if trace is not None:
            trace.append("tool_call", call_id=call_id, tool=name, input=raw_input)
        result = self._run(name, raw_input, ctx, call_id)
        if trace is not None:
            trace.append("tool_result", **result)
        return result

    def _run(self, name: str, raw_input: Any, ctx: ToolContext, call_id: str | None) -> dict:
        tool = self.tools.get(name)
        if tool is None:
            return envelope(name, "unknown", call_id=call_id,
                            error=ToolError("unknown_tool", f"no tool named {name!r}"))
        try:
            parsed = tool.input_model.model_validate(raw_input)
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'input'}: {e['msg']}"
                                 for e in exc.errors())
            return envelope(name, tool.version, call_id=call_id,
                            error=ToolError("invalid_input", problems))
        try:
            return tool.fn(parsed, ctx, call_id)
        except ToolError as exc:
            return envelope(name, tool.version, call_id=call_id, error=exc)
        except Exception as exc:  # recorded, not swallowed: the trace shows the failure
            return envelope(name, tool.version, call_id=call_id,
                            error=ToolError("internal_error", f"{type(exc).__name__}: {exc}"))
