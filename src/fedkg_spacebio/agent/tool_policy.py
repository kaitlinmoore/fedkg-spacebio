"""``policy.check`` and ``policy.check_plan`` (CONTRACTS §7): the policy evaluator as agent tools.

Both wrap ``policy.evaluate.check_job`` without changing what it decides. The rules, their order
and the release log are the policy layer's (CONTRACTS §5); these tools only call it with the run's
profile, site registry and active release log, and put the answer in the §7 envelope.

``policy.check``: one spec, against the real log
-------------------------------------------------
Input: a job spec (§6). The decision ID is the trace's next ``<trace id>/dec/<n>``. The decision
record and any failing-gene lists are written under the policy root, as for a check made from
the command line. ``data`` is the decision record (§5.4) plus ``spec``: the spec as checked, with
its minted ``job_id`` and, on an allow, ``policy_decision`` set. The agent submits exactly that
spec; any other spec has a different hash and no decision. A malformed spec (it can't be parsed)
is ``ok: false`` with ``invalid_spec``; a well-formed but wrong spec is *denied* by a
``spec#…`` rule and gets a record like any other deny.

``policy.check_plan``: every intended release, together, before any is made
---------------------------------------------------------------------------
Why: the release log is first come, first served. Under the restricted profile the main
fixed-effect result over all four ISS clients and the same analysis without SpaceX-16 can't both
be released: whichever is released second is refused, because together they reveal SpaceX-16's
own values by subtraction. If the agent just submitted in some order, which result the federation
got would depend on that order. Checking the whole plan first surfaces the conflict, and the agent
then records which release it keeps as a ``plan`` decision, with the reason.

How: the active log (index, entries, contributor files) and every file the specs reference are
copied into a temporary folder. The specs are checked in the order given, and each allowed one
is appended to the **copy** as if it had been released. The real log and decision folder are
never written. Then, for each spec refused in the plan:
- **alone**: checked against a fresh copy of the real log. Refused alone means refused whatever
  the plan; allowed alone means the plan itself causes the refusal;
- **conflicts_with**: which earlier allowed specs cause it, each tried on its own (copy + that
  spec released, then this one checked). If no single one does, ``combination`` is true: only
  several together cause it;
- **mutual_with**: for each conflicting spec, whether the reverse order would refuse *it*
  (copy + this spec released, then that one checked). A mutual conflict can't be fixed by
  reordering; one of the two has to go.

Gene lists of failing rules are copied to ``<policy root>/plan_checks/<call>/`` (outside the log
and decision folders) and referenced with their SHA-256; ``data`` carries counts, not gene IDs.
Cost: one check per spec, plus one per (refused spec, earlier allowed spec) pair and one per
reverse check, at about 0.1 s each on the real ISS genome.

Alternatives considered for the plan check: (1) leave the log first come, first served and state
it as a limitation (the decision would be made by arrival order, silently); (2) a dry-run flag on
``check_job`` that keeps an in-memory list of hypothetical releases (it would duplicate how the
log builds contributor rows, which ``ReleaseLog.append`` already does exactly).
"""

from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fedkg_spacebio.agent.envelope import ToolError, envelope
from fedkg_spacebio.agent.idscan import scan, strings
from fedkg_spacebio.agent.tools import Tool, ToolContext
from fedkg_spacebio.federation.jobspec import (
    JobSpec,
    JobSpecError,
    parse_job_spec,
    spec_to_dict,
)
from fedkg_spacebio.ids import file_stem
from fedkg_spacebio.policy.evaluate import check_job
from fedkg_spacebio.policy.registry import PolicyMismatchError, load_registry
from fedkg_spacebio.policy.release_log import ReleaseLog, active_log
from fedkg_spacebio.policy.schema import load_profile

CHECK = "policy.check"
PLAN = "policy.check_plan"
VERSION = "0.1.0"
MAX_PLAN = 12


class PolicyCheckInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spec: dict


class PlannedSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = Field(pattern=r"^[A-Za-z0-9_-]{1,40}$")
    spec: dict


class PolicyCheckPlanInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    specs: list[PlannedSpec] = Field(min_length=1, max_length=MAX_PLAN)

    @field_validator("specs")
    @classmethod
    def _unique_labels(cls, v):
        labels = [s.label for s in v]
        dup = sorted({x for x in labels if labels.count(x) > 1})
        if dup:
            raise ValueError(f"labels must be unique: {dup}")
        return v


# ------------------------------------------------------------------ shared


def _parse(spec: dict, what: str = "spec") -> JobSpec:
    try:
        return parse_job_spec(spec)
    except JobSpecError as exc:
        raise ToolError("invalid_spec", f"{what}: " + "; ".join(exc.problems)) from None


def _world(ctx: ToolContext):
    profile = load_profile(ctx.policies_dir)
    registry = load_registry(ctx.manifest_dir)
    return profile, registry


def _slug_ids(data: dict, registry) -> list[str]:
    """Client and site names of the registry that appear anywhere in ``data``.

    Decision details and rule reasons are free text that can name clients and sites; those
    names must be in ``ids`` so a claim that repeats the text can cite them (§8 rule 3).
    """
    names = {r.client for r in registry.values()} | {r.site for r in registry.values()}
    found = set()
    for s in strings(data):
        found.update(t for t in scan(s, sorted(names)) if t in names)
    return sorted(found)


def _decision_ids(record: dict, registry) -> list[str]:
    ids = [record["decision_id"], record["job_id"], record["release_log_id"]]
    if record.get("release_log_head"):
        ids.append(record["release_log_head"])
    ids += [e["rule_id"] for e in record["evaluations"]]
    ids += [p["policy_id"] for p in record["policies"]]
    return ids


# ------------------------------------------------------------------ policy.check


def policy_check(inp: PolicyCheckInput, ctx: ToolContext, call_id: str | None) -> dict:
    spec = _parse(inp.spec)
    try:
        profile, registry = _world(ctx)
        log = active_log(ctx.profile, root=ctx.policy_root)
        decision_id = ctx.trace.next_decision_id() if ctx.trace is not None else None
        spec, decision = check_job(spec, profile, registry, log, decision_id=decision_id)
    except PolicyMismatchError as exc:
        raise ToolError("profile_mismatch", str(exc)) from None
    if decision.outcome == "allow":
        spec = parse_job_spec({**spec_to_dict(spec), "policy_decision": decision.decision_id})
    record = decision.to_dict()
    data = {**record, "spec": spec_to_dict(spec)}
    ids = _decision_ids(record, registry) + list(spec.clients)
    if spec.genes_exclude is not None:
        ids.append(spec.genes_exclude.decision_id)
    ids += _slug_ids(data, registry)
    return envelope(CHECK, VERSION, call_id=call_id, data=data, ids=ids)


# ------------------------------------------------------------------ policy.check_plan


@contextmanager
def _sandbox(ctx: ToolContext, refs: list[str]):
    """A temporary policy root holding a copy of the active log and the specs' files."""
    real = active_log(ctx.profile, root=ctx.policy_root, create=False)
    with tempfile.TemporaryDirectory(prefix="fedkg-plan-") as tmp:
        root = Path(tmp)
        logs = Path(ctx.policy_root) / "logs"
        if (logs / "index.jsonl").is_file():
            (root / "logs").mkdir(parents=True)
            shutil.copy2(logs / "index.jsonl", root / "logs" / "index.jsonl")
            if real.path.is_file():
                shutil.copy2(real.path, root / "logs" / real.path.name)
            if real.dir.is_dir():
                shutil.copytree(real.dir, root / "logs" / real.dir.name)
        for ref in refs:
            src = Path(ctx.policy_root) / ref
            if src.is_file():
                dst = root / ref
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        log = active_log(ctx.profile, root=root)
        if log.log_id != real.log_id:
            raise RuntimeError("sandbox log doesn't match the active log")
        yield root, log


def _refs(spec: JobSpec) -> list[str]:
    refs = [spec.gene_plan.ref]
    if spec.genes_exclude is not None:
        refs.append(spec.genes_exclude.ref)
    return refs


class _Checker:
    """Runs check/append in sandboxes; keeps each spec's job ID stable across sandboxes."""

    def __init__(self, ctx: ToolContext, specs: dict[str, JobSpec]):
        self.ctx = ctx
        self.profile, self.registry = _world(ctx)
        self.specs = specs
        self.refs = sorted({r for s in specs.values() for r in _refs(s)})

    def check(self, log: ReleaseLog, label: str):
        spec, decision = check_job(self.specs[label], self.profile, self.registry, log)
        self.specs[label] = spec  # job ID minted once, reused
        return spec, decision

    def run(self, released_first: list[str], label: str):
        """Release ``released_first`` (each only if allowed) in a fresh sandbox, then check
        ``label``. Returns (decision for label, labels that were actually released first)."""
        with _sandbox(self.ctx, self.refs) as (root, log):
            done = []
            for first in released_first:
                spec, dec = self.check(log, first)
                if dec.outcome == "allow":
                    log.append(spec, dec)
                    done.append(first)
            return self.check(log, label)[1], done


def _failing(decision, root: Path, out_dir: Path | None, label: str) -> list[dict]:
    out = []
    for e in decision.evaluations:
        if e.result != "fail":
            continue
        item = {"rule_id": e.rule_id, "site": e.site, "scope": e.scope, "detail": e.detail}
        if e.genes_failed is not None:
            item["genes_failed"] = e.genes_failed
        if e.sha256:
            item["genes_sha256"] = e.sha256
            if out_dir is not None and e.genes_failed_ref:
                site = e.site or "project"
                key = e.rule_id.split("#", 1)[1]
                dst = out_dir / f"{label}.{site}.{key}.genes.csv"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / e.genes_failed_ref, dst)
                item["genes_ref"] = dst.relative_to(out_dir.parents[1]).as_posix()
        out.append(item)
    return out


def policy_check_plan(inp: PolicyCheckPlanInput, ctx: ToolContext, call_id: str | None) -> dict:
    labels = [s.label for s in inp.specs]
    specs = {s.label: _parse(s.spec, f"spec {s.label!r}") for s in inp.specs}
    for label, spec in specs.items():
        if spec.policy_decision is not None:
            raise ToolError("invalid_spec", f"spec {label!r} already names a decision; plan "
                                            "drafts, not checked specs")
    try:
        chk = _Checker(ctx, specs)
    except PolicyMismatchError as exc:
        raise ToolError("profile_mismatch", str(exc)) from None
    real = active_log(ctx.profile, root=ctx.policy_root, create=False)
    head = real.head() if real.path.is_file() else None
    out_dir = (Path(ctx.policy_root) / "plan_checks" / file_stem(call_id or "untraced"))

    # 1. The plan, in order, on one sandbox.
    results: dict[str, dict] = {}
    allowed: list[str] = []
    with _sandbox(ctx, chk.refs) as (root, log):
        for label in labels:
            spec, dec = chk.check(log, label)
            results[label] = {"label": label, "outcome_in_plan": dec.outcome,
                              "deciding_rule": dec.deciding_rule,
                              "failing": _failing(dec, root, out_dir, label),
                              "outcome_alone": None, "deciding_rule_alone": None,
                              "conflicts_with": [], "combination": False, "mutual_with": []}
            if dec.outcome == "allow":
                log.append(spec, dec)
                allowed.append(label)

    # 2. For each refusal: alone? which earlier specs? mutual?
    for i, label in enumerate(labels):
        r = results[label]
        if r["outcome_in_plan"] == "allow":
            continue
        with _sandbox(ctx, chk.refs) as (_, log):
            alone = chk.check(log, label)[1]
        r["outcome_alone"], r["deciding_rule_alone"] = alone.outcome, alone.deciding_rule
        if alone.outcome != "allow":
            continue
        earlier = [x for x in labels[:i] if x in allowed]
        for other in earlier:
            dec, done = chk.run([other], label)
            if done and dec.outcome == "deny":
                r["conflicts_with"].append(other)
                rev, done_rev = chk.run([label], other)
                if done_rev and rev.outcome == "deny":
                    r["mutual_with"].append(other)
        r["combination"] = not r["conflicts_with"]

    rows = [results[x] for x in labels]
    data = {
        "profile": ctx.profile, "log_id": real.log_id, "log_head": head[0] if head else None,
        "order": labels, "specs": rows,
        "summary": {"n_specs": len(labels), "n_allowed_in_plan": len(allowed),
                    "n_refused_in_plan": len(labels) - len(allowed),
                    "n_refused_by_the_plan": sum(r["outcome_alone"] == "allow" for r in rows)},
    }
    ids = [real.log_id] + ([head[0]] if head else [])
    ids += [f["rule_id"] for r in rows for f in r["failing"]]
    ids += [r["deciding_rule"] for r in rows if r["deciding_rule"]]
    ids += [r["deciding_rule_alone"] for r in rows if r["deciding_rule_alone"]]
    ids += _slug_ids(data, chk.registry)
    return envelope(PLAN, VERSION, call_id=call_id, data=data, ids=ids)


TOOLS = [
    Tool(name=CHECK, version=VERSION, input_model=PolicyCheckInput, fn=policy_check,
         description=(
             "Check one job spec against every participating site's policy and the release "
             "log. Returns the decision record (allow or deny, the deciding rule, every rule "
             "evaluated) and the spec with its job ID; on allow the spec names the decision "
             "and can be submitted as is.")),
    Tool(name=PLAN, version=VERSION, input_model=PolicyCheckPlanInput, fn=policy_check_plan,
         description=(
             "Check a whole plan of intended releases before any is made: the specs are "
             "evaluated in order as if each allowed one had been released (on a copy of the "
             "log; nothing is released). For each refusal: the rule and gene counts, whether it "
             "would be allowed alone, which earlier specs conflict with it, and whether the "
             "conflict holds in both orders.")),
]
