"""CONTRACTS §5.5, worked examples A, B and C, as executable checks.

These are the demo's own policy files (``policies/<profile>/``). Example B runs R1 → J5 **in
order on one log**: allowed steps are released (appended), refused steps add nothing.
"""

import pytest

from policy_helpers import (
    EARTH,
    FULL_SET,
    G1,
    G3,
    G4,
    ISS,
    A,
    B,
    C,
    D,
    check,
    evals,
    failed_genes,
    failures,
    make_env,
    make_spec,
    rule_order,
)

SPACEX16_RELEASE = "pol:SpaceX-16:2#release"
SPACEX16_DIFF = "pol:SpaceX-16:2#diff"


def full_set_spec(env):
    return make_spec(env, clients=ISS, plan={G1: ISS}, methods=FULL_SET,
                     outputs="per_site_summary")


def fe_masked_spec(env, clients=ISS, plan=None, **kw):
    return make_spec(env, clients=clients, plan=plan or {G1: clients}, methods=["fe"],
                     outputs="masked_contribution", **kw)


# ------------------------------------------------------------------ example A


def test_a_full_set_denied_by_spacex16_release(tmp_path):
    env = make_env(tmp_path, "restricted")
    _, dec = check(env, full_set_spec(env))
    assert dec.outcome == "deny"
    assert dec.deciding_rule == SPACEX16_RELEASE
    p16 = env.profile.sites["SpaceX-16"].policy
    assert dec.reason.startswith(p16.rules[1].reason)
    for m in ("reml_hksj", "v3"):
        assert m in dec.reason
    assert "SpaceX-16-ISST" in dec.reason
    assert "fe," not in dec.reason  # fe can be served by a masked contribution
    # So can fe_het since WS-13 (D-039): Q and I² across all clients need only three masked
    # totals (Σw·y², Σw·y, Σw), so SpaceX-16's release rule doesn't block them.
    assert "fe_het" not in dec.reason
    e = evals(dec)
    assert e[SPACEX16_RELEASE].detail == "per_site_summary requested from SpaceX-16-ISST"
    assert e["pol:project:1#no-per-sample"].result == "pass"
    assert e["pol:project:1#fedavg-weighting"].result == "not_applicable"
    assert e["pol:project:1#fedavg-weighting"].detail == "task de_meta"
    assert e["pol:SpaceX-16:2#analyses"].result == "pass"
    assert e["pol:SpaceX-16:2#min-k"].result == "pass"
    assert e["pol:SpaceX-16:2#min-k"].scope == "gene"
    # C2: SpaceX-16-ISST's own per_site_summary is a release over it alone
    assert e[SPACEX16_DIFF].result == "fail"
    assert failures(dec) == [SPACEX16_RELEASE, SPACEX16_DIFF]


def test_a_replan_to_masked_fe_allowed(tmp_path):
    env = make_env(tmp_path, "restricted")
    _, dec = check(env, fe_masked_spec(env))
    assert dec.outcome == "allow"
    assert dec.deciding_rule is None
    assert failures(dec) == []


def test_a_full_set_allowed_under_open(tmp_path):
    env = make_env(tmp_path, "open")
    _, dec = check(env, full_set_spec(env))
    assert dec.outcome == "allow", failures(dec)


def test_a_evaluation_order(tmp_path):
    """Spec checks, project, then sites in the job's client order, rules in file order."""
    env = make_env(tmp_path, "restricted")
    _, dec = check(env, full_set_spec(env))
    order = [r for r in rule_order(dec) if not r.startswith("spec#")]
    expected = ["pol:project:1#no-per-sample", "pol:project:1#fedavg-weighting"]
    for pid in ("pol:SpaceX-4:1", "pol:SpaceX-8:1", "pol:SpaceX-13:1"):
        expected += [f"{pid}#{k}" for k in ("analyses", "release", "min-k", "dp")]
    expected += [f"pol:SpaceX-16:2#{k}" for k in ("analyses", "release", "min-k", "diff", "dp")]
    assert order == expected
    assert all(r.startswith("spec#") for r in rule_order(dec)[:len(rule_order(dec))
                                                                 - len(order)])
    assert {p["policy_id"] for p in dec.policies} == {
        "pol:project:1", "pol:SpaceX-4:1", "pol:SpaceX-8:1", "pol:SpaceX-13:1",
        "pol:SpaceX-16:2"}


# ------------------------------------------------------------------ example B, in order


def step(env, spec):
    """Check; release if allowed. Returns the decision. Checks the log grows only on allow."""
    before = len(env.log.entries())
    spec, dec = check(env, spec)
    if dec.outcome == "allow":
        env.log.append(spec, dec)
    assert len(env.log.entries()) == before + (dec.outcome == "allow")
    return dec


def test_example_b_sequence(tmp_path):
    env = make_env(tmp_path, "restricted")
    b = lambda **kw: fe_masked_spec(env, **kw)  # noqa: E731

    # R1: FE secure sum, complete case. k = 4; the log is empty.
    r1 = step(env, b(plan={G1: ISS}))
    assert r1.outcome == "allow" and r1.release_log_head is None
    assert evals(r1)[SPACEX16_DIFF].result == "pass"

    # R2: FE secure sum, >= 3 sites.
    r2 = step(env, b(plan={G1: ISS, G3: [A, C, D], G4: [A, B, C]},
                     inclusion={"rule": "mean10_finite", "min_sites_per_gene": 3}))
    assert r2.outcome == "allow", failures(r2)
    assert r2.release_log_head == env.log.entries()[0]["release_id"]

    # J3: FE without SpaceX-8, >= 3 sites. 1111 − 1011 isolates B, whose policy allows it.
    j3 = step(env, b(clients=[A, C, D], plan={G1: [A, C, D], G3: [A, C, D]},
                     inclusion={"rule": "mean10_finite", "min_sites_per_gene": 3}))
    assert j3.outcome == "allow", failures(j3)
    assert len(env.log.entries()) == 3

    # J4: FE without SpaceX-16, complete case over A, B, C. 1111 − 1110 isolates D.
    j4 = step(env, b(clients=[A, B, C], plan={G1: [A, B, C], G4: [A, B, C]}))
    assert j4.outcome == "deny"
    assert j4.deciding_rule == SPACEX16_DIFF
    e = evals(j4)[SPACEX16_DIFF]
    assert e.site == "SpaceX-16" and e.scope == "gene"
    assert e.genes_failed == 1 and failed_genes(env, e) == [G1]  # FAKE:0004 passes
    assert failures(j4) == [SPACEX16_DIFF]
    assert rule_order(j4)[-1] == SPACEX16_DIFF  # a site outside the job comes last
    assert "pol:SpaceX-16:2#min-k" not in rule_order(j4)  # only its differencing rule

    # J5: per_site_summary from SpaceX-4 only. C1: SpaceX-4's own min-k decides.
    j5 = step(env, make_spec(env, clients=[A], plan={G1: [A], G3: [A], G4: [A]},
                             methods=["fe"], outputs="per_site_summary"))
    assert j5.outcome == "deny"
    assert j5.deciding_rule == "pol:SpaceX-4:1#min-k"
    mk = evals(j5)["pol:SpaceX-4:1#min-k"]
    assert mk.genes_failed == 3 and failed_genes(env, mk) == [G1, G3, G4]
    diff = evals(j5)[SPACEX16_DIFF]
    assert diff.result == "fail"
    assert failed_genes(env, diff) == [G1, G3]  # 1011 − 1000 = 0011; FAKE:0004 passes
    assert failures(j5) == ["pol:SpaceX-4:1#min-k", SPACEX16_DIFF]

    # Refused steps added nothing.
    assert len(env.log.entries()) == 3
    env.log.verify()


def test_j4_resubmitted_without_the_failing_genes(tmp_path):
    env = make_env(tmp_path, "restricted")
    step(env, fe_masked_spec(env, plan={G1: ISS}))
    step(env, fe_masked_spec(env, plan={G1: ISS, G3: [A, C, D], G4: [A, B, C]},
                             inclusion={"rule": "mean10_finite", "min_sites_per_gene": 3}))
    j4_plan = {G1: [A, B, C], G4: [A, B, C]}
    _, j4 = check(env, fe_masked_spec(env, clients=[A, B, C], plan=j4_plan))
    assert j4.outcome == "deny"
    failing = failed_genes(env, evals(j4)[SPACEX16_DIFF])
    assert failing == [G1]

    # Excluding exactly the failing genes: allowed. Only FAKE:0004, which D never touched.
    _, ok = check(env, fe_masked_spec(env, clients=[A, B, C], plan=j4_plan,
                                      genes_exclude=failing, exclude_decision=j4.decision_id))
    assert ok.outcome == "allow", failures(ok)

    # Excluding fewer: still refused.
    _, still = check(env, fe_masked_spec(env, clients=[A, B, C], plan=j4_plan,
                                         genes_exclude=[], exclude_decision=j4.decision_id))
    assert still.deciding_rule == SPACEX16_DIFF
    _, wrong = check(env, fe_masked_spec(env, clients=[A, B, C], plan=j4_plan,
                                         genes_exclude=[G4], exclude_decision=j4.decision_id))
    assert wrong.deciding_rule == SPACEX16_DIFF


def test_example_b_under_open_profile_allows_j4(tmp_path):
    """No site in the open profile has a differencing rule."""
    env = make_env(tmp_path, "open")
    step(env, fe_masked_spec(env, plan={G1: ISS}))
    _, j4 = check(env, fe_masked_spec(env, clients=[A, B, C], plan={G1: [A, B, C]}))
    assert j4.outcome == "allow", failures(j4)


# ------------------------------------------------------------------ example C


LOO = [(EARTH[1], EARTH[2], EARTH[0]), (EARTH[0], EARTH[2], EARTH[1]),
       (EARTH[0], EARTH[1], EARTH[2])]  # (kept, kept, dropped)


def site_of(env, client):
    return env.registry[client].site


@pytest.mark.parametrize("profile", ["open", "restricted"])
@pytest.mark.parametrize("kept1,kept2,dropped", LOO)
def test_c_leave_one_out_fails_min_k_at_both_remaining_sites(tmp_path, profile, kept1, kept2,
                                                              dropped):
    env = make_env(tmp_path, profile)
    clients = [kept1, kept2]
    _, dec = check(env, fe_masked_spec(env, clients=clients, plan={G1: clients, G3: clients}))
    assert dec.outcome == "deny"
    min_k = {f"{env.registry[c].policy_id}#min-k" for c in clients}
    assert min_k <= set(failures(dec))
    assert dec.deciding_rule == f"{env.registry[kept1].policy_id}#min-k"
    for r in min_k:
        assert evals(dec)[r].genes_failed == 2


@pytest.mark.parametrize("kept1,kept2", [(EARTH[1], EARTH[2]), (EARTH[0], EARTH[1])])
def test_c_restricted_keeping_spacex16_also_fails_diff_on_an_empty_log(tmp_path, kept1, kept2):
    env = make_env(tmp_path, "restricted")
    _, dec = check(env, fe_masked_spec(env, clients=[kept1, kept2], plan={G1: [kept1, kept2]}))
    assert SPACEX16_DIFF in failures(dec)


def test_c_restricted_dropping_spacex16_fails_diff_after_a_full_release(tmp_path):
    env = make_env(tmp_path, "restricted")
    drop16 = [EARTH[0], EARTH[2]]
    _, before = check(env, fe_masked_spec(env, clients=drop16, plan={G1: drop16}))
    assert SPACEX16_DIFF not in failures(before)  # empty log: nothing to subtract from

    step(env, fe_masked_spec(env, clients=EARTH, plan={G1: EARTH}))  # complete case, allowed
    _, after = check(env, fe_masked_spec(env, clients=drop16, plan={G1: drop16}))
    assert SPACEX16_DIFF in failures(after)  # 111 − 101 = 010
    assert after.deciding_rule == "pol:SpaceX-13:1#min-k"


@pytest.mark.parametrize("profile", ["open", "restricted"])
def test_c_complete_case_passes(tmp_path, profile):
    env = make_env(tmp_path, profile)
    _, dec = check(env, fe_masked_spec(env, clients=EARTH, plan={G1: EARTH, G3: EARTH}))
    assert dec.outcome == "allow", failures(dec)
