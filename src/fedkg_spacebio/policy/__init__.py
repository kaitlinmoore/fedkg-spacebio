"""Governance as code (phase 4). OWNED CORE.

Each site declares a policy (allowed analyses, outputs that may leave, minimum aggregation size,
DP requirement). Given a job spec and all participating sites' policies, decide allow/deny and
return the rule that decided it. Schema lives in CONTRACTS.md.
"""
