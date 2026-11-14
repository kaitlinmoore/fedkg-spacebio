"""The grounding checker (CONTRACTS §8): is every claim in the report backed by this trace?

The question it answers
-----------------------
The agent's report is a list of claims. Each claim has a ``kind`` (``result``, ``kg_fact``,
``policy``, ``caveat``, ``site``) and ``citations``, each a pair ``{id, call_id}``: "this ID, as
returned by that tool call". The checker decides, for every claim, whether the trace itself
supports it. It reads **only the trace file**: not the release log, the KG or the results
tables. Grounding means "this run's own tool results said so", nothing more. Whether the tool
results are right is the tools' job and their tests'.

The rules (one function each, below)
------------------------------------
1. **Cited from the named call.** A citation's ``id`` must be in the ``ids`` of the
   ``tool_result`` with that ``call_id`` in this trace, recorded before the report. A failed
   tool result lists no IDs, so nothing in it can be cited. A result recorded after the report
   doesn't count: the report can't have been written from it.
2. **The right kind of tool.** The citing call must come from a tool that can support the
   claim's kind (``KIND_TOOLS``): results from ``results.read``, KG facts from ``kg.query``,
   policy outcomes from ``policy.check`` or ``policy.check_plan``, caveats from ``results.read``
   or ``policy.check``, site facts from ``sites.list``. A gene ID that ``kg.query`` returned
   can't back a statement about the federated result, even though the ID is the same.
3. **No uncited IDs in the text.** Every ID-shaped token in the claim's text (``idscan.scan``)
   must have a citation that passed rules 1 and 2. Otherwise a claim could cite one ID and slip
   in others.
4. **Fabricated IDs.** IDs that appear in no tool result of the trace at all: every cited ID and
   every ID-shaped token of the claim texts. The target is zero. (§8 lists cited IDs; uncited
   tokens are included too, because an invented ID that nobody cited is the worst case and
   would otherwise never be counted.)
5. **At least one citation.** A claim with no citation that passes rules 1 and 2 is unsupported,
   even if its text has no IDs: there's nothing in the trace to check it against.

A claim is **supported** only if every citation passes and no rule fails. One bad citation makes
the whole claim unsupported: the report put that citation forward as evidence.

Gene symbols are display labels: a symbol never needs citing and is never counted as fabricated
(``idscan`` handles this), and citing one fails rule 1 because no tool lists symbols in ``ids``.

What the checker doesn't do: judge whether the text says what the cited result says (that a
cited estimate is negative when the text says "lower"). That needs per-kind semantics. The
report builder guards it by writing claims from tool data; the checker guards the IDs.

Output (the ``grounding`` record's body): ``checker_version``, ``report_seq``, per claim its
``verdict`` and ``failures`` (each with the ``rule``, the ``id``, the ``call_id`` if any and a
reason), ``n_claims``, ``n_supported``, ``supported_rate`` (null for an empty report) and
``fabricated_ids`` (sorted).
"""

from __future__ import annotations

from pathlib import Path

from fedkg_spacebio.agent import idscan
from fedkg_spacebio.agent.trace import read_trace

CHECKER_VERSION = "fedkg_spacebio.grounding/1"

KIND_TOOLS: dict[str, frozenset[str]] = {
    "result": frozenset({"results.read"}),
    "kg_fact": frozenset({"kg.query"}),
    "policy": frozenset({"policy.check", "policy.check_plan"}),
    "caveat": frozenset({"results.read", "policy.check"}),
    "site": frozenset({"sites.list"}),
}


class GroundingError(ValueError):
    """The trace can't be graded (no report, or more than one)."""


def check_trace(path: str | Path) -> dict:
    """Grade the report in the trace at ``path``. The trace must verify (``read_trace``)."""
    return grade(read_trace(path))


def grade(records: list[dict]) -> dict:
    """Grade the single ``report`` record among ``records`` (already verified)."""
    reports = [r for r in records if r["type"] == "report"]
    if len(reports) != 1:
        raise GroundingError(f"expected one report record, found {len(reports)}")
    report = reports[0]
    results = {r["call_id"]: r for r in records
               if r["type"] == "tool_result" and r["seq"] < report["seq"]}
    evidence = Evidence(results)

    claims_out = []
    fabricated: set[str] = set()
    for claim in report["claims"]:
        failures = check_claim(claim, evidence)
        claims_out.append({"claim_id": claim.get("claim_id"),
                           "verdict": "unsupported" if failures else "supported",
                           "failures": failures})
        fabricated |= fabricated_ids(claim, evidence)
    n = len(claims_out)
    n_ok = sum(c["verdict"] == "supported" for c in claims_out)
    return {
        "checker_version": CHECKER_VERSION,
        "report_seq": report["seq"],
        "claims": claims_out,
        "n_claims": n,
        "n_supported": n_ok,
        "supported_rate": (n_ok / n) if n else None,
        "fabricated_ids": sorted(fabricated),
    }


class Evidence:
    """What the trace's tool results (before the report) make citable."""

    def __init__(self, results: dict[str, dict]):
        self.results = results
        self.all_ids: set[str] = set()
        symbols: set[str] = set()
        for r in results.values():
            self.all_ids.update(r["ids"])
            symbols |= idscan.symbol_values(r["data"])
        # Listed IDs without a fixed pattern (clients, sites, missions, log IDs): idscan's
        # class 2. A listed ID spelled like a symbol stays an ID.
        self.vocabulary = sorted(i for i in self.all_ids if not idscan.is_pattern_id(i))
        self.symbols = symbols - self.all_ids

    def tokens(self, text: str) -> list[str]:
        return idscan.scan(text, self.vocabulary, self.symbols)


def check_claim(claim: dict, evidence: Evidence) -> list[dict]:
    """Every rule failure for one claim (empty list: supported)."""
    failures: list[dict] = []
    kind = claim.get("kind")
    if kind not in KIND_TOOLS:
        failures.append(_fail("kind", None, None, f"unknown claim kind {kind!r}"))
        return failures
    valid: set[str] = set()
    for cit in claim.get("citations") or []:
        problem = check_citation(cit, kind, evidence)
        if problem is None:
            valid.add(cit["id"])
        else:
            failures.append(problem)
    for token in evidence.tokens(claim.get("text") or ""):
        if token not in valid:
            failures.append(_fail(3, token, None, "ID in the claim text has no valid citation"))
    if not valid:
        failures.append(_fail(5, None, None, "no citation passes rules 1 and 2"))
    return failures


def check_citation(cit: dict, kind: str, evidence: Evidence) -> dict | None:
    """Rules 1 and 2 for one citation. None if it passes, else the failure."""
    cid, call_id = cit.get("id"), cit.get("call_id")
    result = evidence.results.get(call_id)
    if result is None:
        return _fail(1, cid, call_id, "no tool result with this call ID in this trace before "
                                      "the report")
    if cid not in result["ids"]:
        reason = ("the call failed; a failed result lists no IDs" if not result["ok"]
                  else "the ID isn't in that call's ids")
        return _fail(1, cid, call_id, reason)
    if result["tool"] not in KIND_TOOLS[kind]:
        return _fail(2, cid, call_id, f"a {kind} claim can't be supported by {result['tool']} "
                                      f"(allowed: {sorted(KIND_TOOLS[kind])})")
    return None


def fabricated_ids(claim: dict, evidence: Evidence) -> set[str]:
    """Rule 4: IDs the claim cites or mentions that no tool result of the trace returned."""
    cited = {c.get("id") for c in claim.get("citations") or [] if isinstance(c.get("id"), str)}
    mentioned = set(evidence.tokens(claim.get("text") or ""))
    return (cited | mentioned) - evidence.all_ids - evidence.symbols


def _fail(rule, id_, call_id, reason: str) -> dict:
    return {"rule": rule, "id": id_, "call_id": call_id, "reason": reason}
