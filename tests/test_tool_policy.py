"""``policy.check`` and ``policy.check_plan`` (CONTRACTS §7): the evaluator as agent tools.

Built on the policy suite's helpers: the demo's own ``policies/<profile>/`` files, minimal
manifests for the seven liver clients, fake genes (G1 = FAKE:0001 = 9000000001). Example B's
columns: A..D = SpaceX-4, SpaceX-8, SpaceX-13-ISST, SpaceX-16-ISST.
"""

from __future__ import annotations

import pytest

from agent_helpers import assert_ids_cover_data, start
from fedkg_spacebio.agent.tool_policy import TOOLS
from fedkg_spacebio.agent.tools import Registry, ToolContext
from fedkg_spacebio.federation.jobspec import parse_job_spec, spec_to_dict
from fedkg_spacebio.policy.release_log import active_log
from policy_helpers import FULL_SET, G1, G3, G4, ISS, POLICIES, A, B, C, D, make_env, make_spec

REG = Registry(TOOLS)


def ctx_for(env, tmp_path, trace=True):
    return ToolContext(profile=env.profile_name, workdir=tmp_path, policy_root=env.root,
                       policies_dir=POLICIES / env.profile_name, manifest_dir=env.manifest_dir,
                       trace=start(tmp_path / "t") if trace else None)


def d(spec):
    return spec_to_dict(spec)


# ------------------------------------------------------------------ policy.check


def test_allow_returns_record_and_submittable_spec(tmp_path):
    env = make_env(tmp_path, "restricted")
    spec = make_spec(env, clients=ISS, plan={G1: ISS})
    ctx = ctx_for(env, tmp_path)
    r = REG.call("policy.check", {"spec": d(spec)}, ctx)
    assert r["ok"], r["error"]
    data = r["data"]
    assert data["outcome"] == "allow" and data["deciding_rule"] is None
    assert data["decision_id"] == "trace:FAKE-0001/dec/1"
    checked = parse_job_spec(data["spec"])
    assert checked.job_id == data["job_id"] and checked.policy_decision == data["decision_id"]
    assert (env.root / "decisions" / "trace_FAKE-0001_dec_1.json").is_file()
    for want in (data["decision_id"], data["job_id"], "pol:SpaceX-16:2", "pol:SpaceX-16:2#diff",
                 "restricted-1", *ISS):
        assert want in r["ids"]
    assert_ids_cover_data(r)


def test_full_method_set_denied_by_the_release_rule(tmp_path):
    env = make_env(tmp_path, "restricted")
    spec = make_spec(env, clients=ISS, plan={G1: ISS}, methods=FULL_SET,
                     outputs="per_site_summary")
    r = REG.call("policy.check", {"spec": d(spec)}, ctx_for(env, tmp_path))
    assert r["data"]["outcome"] == "deny"
    assert r["data"]["deciding_rule"] == "pol:SpaceX-16:2#release"
    assert r["data"]["spec"]["policy_decision"] is None
    assert "pol:SpaceX-16:2#release" in r["ids"]
    assert_ids_cover_data(r)


def test_decision_ids_follow_the_trace_counter(tmp_path):
    env = make_env(tmp_path, "restricted")
    ctx = ctx_for(env, tmp_path)
    ctx.trace.next_decision_id()  # an agent decision first
    r = REG.call("policy.check", {"spec": d(make_spec(env, clients=ISS, plan={G1: ISS}))}, ctx)
    assert r["data"]["decision_id"] == "trace:FAKE-0001/dec/2"


def test_without_a_trace_the_cli_form_is_used(tmp_path):
    env = make_env(tmp_path, "restricted")
    r = REG.call("policy.check", {"spec": d(make_spec(env, clients=ISS, plan={G1: ISS}))},
                 ctx_for(env, tmp_path, trace=False))
    assert r["data"]["decision_id"].startswith("dec:") and r["call_id"] is None


def test_malformed_spec_is_not_a_decision(tmp_path):
    env = make_env(tmp_path, "restricted")
    bad = d(make_spec(env, clients=ISS, plan={G1: ISS}))
    bad["rounds"] = "1"
    r = REG.call("policy.check", {"spec": bad}, ctx_for(env, tmp_path))
    assert not r["ok"] and r["error"]["code"] == "invalid_spec" and "rounds" in \
        r["error"]["message"]
    assert not (env.root / "decisions").exists()


def test_profile_mismatch(tmp_path):
    env = make_env(tmp_path, "open")  # manifests name pol:SpaceX-16:1
    spec = make_spec(env, clients=ISS, plan={G1: ISS})
    ctx = ctx_for(env, tmp_path)
    ctx.policies_dir = POLICIES / "restricted"
    r = REG.call("policy.check", {"spec": d(spec)}, ctx)
    assert r["error"]["code"] == "profile_mismatch"


# ------------------------------------------------------------------ policy.check_plan


def plan_specs(env):
    main = make_spec(env, clients=ISS, plan={G1: ISS, G4: [A, B, C]})
    without8 = make_spec(env, clients=[A, C, D], plan={G1: [A, C, D]})
    without16 = make_spec(env, clients=[A, B, C], plan={G1: [A, B, C], G4: [A, B, C]})
    return {"main": main, "without8": without8, "without16": without16}


def plan_call(env, tmp_path, order, specs=None):
    specs = specs or plan_specs(env)
    ctx = ctx_for(env, tmp_path)
    r = REG.call("policy.check_plan",
                 {"specs": [{"label": x, "spec": d(specs[x])} for x in order]}, ctx)
    assert r["ok"], r["error"]
    return r, {s["label"]: s for s in r["data"]["specs"]}


def _log_state(env):
    log = active_log(env.profile_name, root=env.root)
    files = sorted(p.relative_to(env.root) for p in env.root.rglob("*") if p.is_file())
    return len(log.entries()), files


def test_without_spacex16_conflicts_with_both_earlier_releases(tmp_path):
    env = make_env(tmp_path, "restricted")
    specs = plan_specs(env)
    before = _log_state(env)
    r, s = plan_call(env, tmp_path, ["main", "without8", "without16"], specs)
    assert s["main"]["outcome_in_plan"] == "allow"
    assert s["without8"]["outcome_in_plan"] == "allow"
    w = s["without16"]
    assert w["outcome_in_plan"] == "deny" and w["deciding_rule"] == "pol:SpaceX-16:2#diff"
    assert w["outcome_alone"] == "allow"
    # Both earlier releases conflict with it, each on its own: main - without16 = D alone, and
    # without8 - without16 = D - B, a group of 2 sites that includes SpaceX-16 (its rule's
    # min_sites is 3). Both conflicts hold in either order.
    assert w["conflicts_with"] == ["main", "without8"]
    assert w["mutual_with"] == ["main", "without8"]
    assert not w["combination"]
    diff = next(f for f in w["failing"] if f["rule_id"] == "pol:SpaceX-16:2#diff")
    assert diff["genes_failed"] == 1  # G1; G4 passes (SpaceX-16 never tested it)
    assert (env.root / diff["genes_ref"]).read_text().strip().splitlines()[-1] == str(G1)
    assert r["data"]["summary"] == {"n_specs": 3, "n_allowed_in_plan": 2,
                                    "n_refused_in_plan": 1, "n_refused_by_the_plan": 1}
    # The real log and decisions are untouched; only the plan_checks gene lists are new.
    n, files = _log_state(env)
    assert n == before[0] == 0
    new = set(files) - set(before[1])
    assert new and all(str(p).startswith("plan_checks") for p in new)
    assert_ids_cover_data(r)


def test_reverse_order_refuses_the_main_result(tmp_path):
    env = make_env(tmp_path, "restricted")
    _, s = plan_call(env, tmp_path, ["without16", "main"])
    assert s["without16"]["outcome_in_plan"] == "allow"
    assert s["main"]["outcome_in_plan"] == "deny"
    assert s["main"]["conflicts_with"] == ["without16"]
    assert s["main"]["mutual_with"] == ["without16"]


def test_open_profile_has_no_conflict(tmp_path):
    env = make_env(tmp_path, "open")
    _, s = plan_call(env, tmp_path, ["main", "without8", "without16"])
    assert all(x["outcome_in_plan"] == "allow" for x in s.values())


def test_refused_alone_is_not_a_plan_conflict(tmp_path):
    """Earth leave-one-out: refused by minimum k whatever else is planned."""
    env = make_env(tmp_path, "restricted")
    earth = make_spec(env, clients=["SpaceX-13-LAR", "SpaceX-21"],
                      plan={G3: ["SpaceX-13-LAR", "SpaceX-21"]})
    _, s = plan_call(env, tmp_path, ["earth_loo"], {"earth_loo": earth})
    e = s["earth_loo"]
    assert e["outcome_in_plan"] == e["outcome_alone"] == "deny"
    assert e["deciding_rule"].endswith("#min-k") and e["conflicts_with"] == []


def test_existing_releases_count(tmp_path):
    """With the main result already in the real log, without-SpaceX-16 is refused alone too."""
    from policy_helpers import check
    env = make_env(tmp_path, "restricted")
    specs = plan_specs(env)
    spec, dec = check(env, specs["main"])
    env.log.append(spec, dec)
    r, s = plan_call(env, tmp_path, ["without16"], specs)
    assert s["without16"]["outcome_alone"] == "deny"
    assert r["data"]["log_head"] is not None and r["data"]["log_head"] in r["ids"]
    assert len(env.log.entries()) == 1


def test_combination_only(tmp_path):
    """A refusal no single earlier spec causes: ACD (1011) and ABD (1101) together with ABCD
    (1111) isolate D (1011 + 1101 - 1111 = 0001); either one alone isolates only B or C."""
    env = make_env(tmp_path, "restricted")
    specs = {"acd": make_spec(env, clients=[A, C, D], plan={G1: [A, C, D]}),
             "abd": make_spec(env, clients=[A, B, D], plan={G1: [A, B, D]}),
             "all4": make_spec(env, clients=ISS, plan={G1: ISS})}
    _, s = plan_call(env, tmp_path, ["acd", "abd", "all4"], specs)
    assert s["acd"]["outcome_in_plan"] == s["abd"]["outcome_in_plan"] == "allow"
    x = s["all4"]
    assert x["outcome_in_plan"] == "deny" and x["outcome_alone"] == "allow"
    assert x["conflicts_with"] == [] and x["combination"] is True


@pytest.mark.parametrize("specs, code", [
    ([], "invalid_input"),
    ([{"label": "a b", "spec": {}}], "invalid_input"),
    ([{"label": "x", "spec": {}}, {"label": "x", "spec": {}}], "invalid_input"),
    ([{"label": "x", "spec": {"schema": "job-spec/1"}}], "invalid_spec"),
])
def test_bad_plans_refused(tmp_path, specs, code):
    env = make_env(tmp_path, "restricted")
    r = REG.call("policy.check_plan", {"specs": specs}, ctx_for(env, tmp_path))
    assert not r["ok"] and r["error"]["code"] == code


def test_checked_specs_are_not_plan_drafts(tmp_path):
    env = make_env(tmp_path, "restricted")
    spec = d(make_spec(env, clients=ISS, plan={G1: ISS}))
    spec["policy_decision"] = "dec:FAKE-0001-1"
    r = REG.call("policy.check_plan", {"specs": [{"label": "x", "spec": spec}]},
                 ctx_for(env, tmp_path))
    assert r["error"]["code"] == "invalid_spec"
