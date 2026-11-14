"""Contract edges of the policy check (CONTRACTS §5.1–§5.4, §6.1).

Spec checks (``spec#…``), the project policy, DP, profile mismatch, evaluation order, the
decision record's own fields, and the release-log cases that §5.5 doesn't spell out.
"""

import json

import pytest
import yaml

from fedkg_spacebio.federation.jobspec import (
    job_spec_sha256,
    parse_job_spec,
    spec_to_dict,
    with_job_id,
)
from fedkg_spacebio.ids import JOB_ID_RE
from fedkg_spacebio.policy.decision import EVALUATOR_VERSION, read_decision
from fedkg_spacebio.policy.evaluate import check_job
from fedkg_spacebio.policy.registry import PolicyMismatchError, load_registry
from fedkg_spacebio.policy.release_log import StaleDecisionError
from fedkg_spacebio.policy.schema import load_profile
from policy_helpers import (
    EARTH,
    FULL_SET,
    G1,
    G3,
    ISS,
    POLICIES,
    A,
    B,
    C,
    D,
    check,
    evals,
    failures,
    make_env,
    make_spec,
    rule_order,
)

DIFF16 = "pol:SpaceX-16:2#diff"


def fe(env, clients=ISS, plan=None, **kw):
    return make_spec(env, clients=clients, plan=plan or {G1: clients}, **kw)


def spec_checks(decision):
    return [r for r in rule_order(decision) if r.startswith("spec#")]


# ------------------------------------------------------------------ project policy and DP


def test_per_sample_is_denied_by_the_project_policy(tmp_path):
    """per_sample covers every method's need (§5.2), so the rule that forbids it decides."""
    env = make_env(tmp_path, "open")
    _, dec = check(env, fe(env, outputs={A: "per_sample", B: "per_site_summary",
                                         C: "per_site_summary", D: "per_site_summary"}))
    assert dec.deciding_rule == "pol:project:1#no-per-sample"
    assert "per_sample" in evals(dec)["pol:project:1#no-per-sample"].detail
    assert "SpaceX-4" in dec.reason


def profile_with(tmp_path, base, site, edit):
    """Copy a demo profile, editing one site's YAML (policy ID unchanged)."""
    d = tmp_path / f"policies-{base}-edited"
    d.mkdir()
    for f in (POLICIES / base).glob("*.yaml"):
        text = f.read_text()
        if f.stem == site:
            head, body = text.split("\n", 1)
            data = yaml.safe_load(body)
            edit(data)
            text = head + "\n" + yaml.safe_dump(data, sort_keys=False)
        (d / f.name).write_text(text)
    return d


def set_dp_required(data):
    for r in data["rules"]:
        if r["kind"] == "dp":
            r["required"] = True


def test_dp_required_denies(tmp_path):
    pdir = profile_with(tmp_path, "open", "SpaceX-8", set_dp_required)
    env = make_env(tmp_path, "open", policies_dir=pdir)
    _, dec = check(env, fe(env))
    assert dec.deciding_rule == "pol:SpaceX-8:1#dp"
    assert evals(dec)["pol:SpaceX-8:1#dp"].result == "fail"


def test_first_failure_decides_and_every_failure_is_listed(tmp_path):
    """SpaceX-4 requires DP (first site), and the restricted full set fails at SpaceX-16."""
    pdir = profile_with(tmp_path, "restricted", "SpaceX-4", set_dp_required)
    env = make_env(tmp_path, "restricted", policies_dir=pdir)
    _, dec = check(env, make_spec(env, clients=ISS, plan={G1: ISS}, methods=FULL_SET,
                                  outputs="per_site_summary"))
    assert dec.deciding_rule == "pol:SpaceX-4:1#dp"
    assert failures(dec) == ["pol:SpaceX-4:1#dp", "pol:SpaceX-16:2#release", DIFF16]
    dp_rule = env.profile.sites["SpaceX-4"].policy.rules[-1]
    assert dp_rule.kind == "dp" and dec.reason.startswith(dp_rule.reason)


def test_min_n_per_arm(tmp_path):
    env = make_env(tmp_path, "open", overrides={"SpaceX-8": {"n_ground": 4}})
    _, dec = check(env, fe(env))
    assert dec.deciding_rule == "pol:SpaceX-8:1#min-k"
    detail = evals(dec)["pol:SpaceX-8:1#min-k"].detail
    assert "SpaceX-8" in detail and "n_ground" in detail
    assert evals(dec)["pol:SpaceX-8:1#min-k"].genes_failed == 0  # every gene has k = 4


@pytest.mark.parametrize("arm", ["n_flight", "n_ground"])
def test_undeclared_n_per_arm_fails_min_k_by_name(tmp_path, arm):
    """A manifest without a per-arm count fails min_aggregation, naming the client and the
    count. It isn't skipped: an undeclared count can't show the client meets the minimum."""
    env = make_env(tmp_path, "open", overrides={"SpaceX-8": {arm: None}})  # None deletes it
    assert getattr(env.registry["SpaceX-8"], arm) is None
    _, dec = check(env, fe(env))
    assert dec.outcome == "deny"
    assert dec.deciding_rule == "pol:SpaceX-8:1#min-k"
    e = evals(dec)["pol:SpaceX-8:1#min-k"]
    assert "SpaceX-8" in e.detail and arm in e.detail and "doesn't declare" in e.detail
    assert e.genes_failed == 0  # every gene still pools 4 sites; the count alone decides
    assert "SpaceX-8" in dec.reason
    assert failures(dec) == ["pol:SpaceX-8:1#min-k"]


def test_clock_plane_restricted_denied_by_analyses(tmp_path):
    env = make_env(tmp_path, "restricted")
    spec = make_spec(env, clients=ISS, plan={G1: ISS}, task="clock_plane",
                     outputs="per_site_summary")
    _, dec = check(env, spec)
    assert dec.deciding_rule == "pol:SpaceX-16:2#analyses"
    e = evals(dec)
    assert e["pol:SpaceX-16:2#min-k"].scope == "job"
    assert e[DIFF16].scope == "job"


def test_clock_plane_open_allowed(tmp_path):
    env = make_env(tmp_path, "open")
    spec = make_spec(env, clients=ISS, plan={G1: ISS}, task="clock_plane",
                     outputs="per_site_summary")
    _, dec = check(env, spec)
    assert dec.outcome == "allow", failures(dec)


# ------------------------------------------------------------------ spec checks


@pytest.mark.parametrize("clients,needle", [
    ([A, B, C, "SpaceX-99"], "SpaceX-99"),       # not in the registry
    ([A, B, C, "SpaceX-21"], "SpaceX-21"),       # an Earth client in an ISS job
])
def test_unknown_or_wrong_stratum_client(tmp_path, clients, needle):
    env = make_env(tmp_path, "open")
    _, dec = check(env, make_spec(env, clients=clients, plan={G1: clients}, stratum="iss"))
    assert dec.outcome == "deny"
    assert dec.deciding_rule == "spec#clients-known"
    assert needle in dec.reason
    assert "not evaluated" in dec.reason
    assert all(r.startswith("spec#") for r in rule_order(dec))
    assert dec.policies == []


@pytest.mark.parametrize("methods,needle", [(["fe", "fe_magic"], "fe_magic"),
                                            (["fe", "fedavg"], "fedavg")])
def test_unknown_or_wrong_task_method(tmp_path, methods, needle):
    env = make_env(tmp_path, "open")
    _, dec = check(env, fe(env, methods=methods, outputs="per_site_summary"))
    assert dec.deciding_rule == "spec#methods-known"
    assert needle in dec.reason


def test_output_that_doesnt_cover_a_method(tmp_path):
    env = make_env(tmp_path, "open")
    outputs = {A: "masked_contribution", B: "per_site_summary", C: "per_site_summary",
               D: "per_site_summary"}
    # v3, not fe_het: since WS-13 (D-039) a masked contribution serves fe_het (Q and I² come
    # from three masked totals), but V3 still needs each site's estimate and SE.
    _, dec = check(env, fe(env, methods=["fe", "v3"], outputs=outputs,
                           aggregation={"strategy": "plain"}))
    assert dec.deciding_rule == "spec#outputs-cover"
    assert "v3" in dec.reason and "SpaceX-4" in dec.reason


@pytest.mark.parametrize("outputs", [
    {A: "masked_contribution", B: "masked_contribution", C: "masked_contribution"},  # D missing
    {A: "masked_contribution", B: "masked_contribution", C: "masked_contribution",
     D: "masked_contribution", "SpaceX-21": "masked_contribution"},                  # extra
    {A: "raw_counts", B: "masked_contribution", C: "masked_contribution",
     D: "masked_contribution"},                                                       # unknown
])
def test_outputs_must_name_exactly_the_clients_with_known_kinds(tmp_path, outputs):
    env = make_env(tmp_path, "open")
    _, dec = check(env, fe(env, outputs=outputs,
                           aggregation={"strategy": "secure_sum", "fixed_point_bits": 32,
                                        "modulus_bits": 64}))
    assert dec.deciding_rule == "spec#outputs-cover"


def test_masked_under_plain_and_plain_under_secure_sum(tmp_path):
    env = make_env(tmp_path, "open")
    _, d1 = check(env, fe(env, outputs="masked_contribution", aggregation={"strategy": "plain"}))
    assert d1.deciding_rule == "spec#aggregation"
    _, d2 = check(env, fe(env, outputs="per_site_summary",
                          aggregation={"strategy": "secure_sum", "fixed_point_bits": 32,
                                       "modulus_bits": 64}))
    assert d2.deciding_rule == "spec#aggregation"


def test_de_meta_weighting_must_be_inverse_variance(tmp_path):
    env = make_env(tmp_path, "open")
    _, dec = check(env, fe(env, weighting={"scheme": "sample_count"}))
    assert dec.deciding_rule == "spec#weighting"


def test_gene_plan_hash_mismatch(tmp_path):
    env = make_env(tmp_path, "open")
    spec = fe(env)
    (env.root / spec.gene_plan.ref).write_text("ncbi_gene,SpaceX-13-ISST,SpaceX-16-ISST,"
                                                "SpaceX-4,SpaceX-8\n9000000001,1,1,1,0\n")
    _, dec = check(env, spec)
    assert dec.deciding_rule == "spec#gene-plan"


def test_gene_plan_missing_or_with_other_clients(tmp_path):
    env = make_env(tmp_path, "open")
    spec = fe(env)
    (env.root / spec.gene_plan.ref).unlink()
    _, dec = check(env, spec)
    assert dec.deciding_rule == "spec#gene-plan"

    other = make_spec(env, clients=ISS, plan={G1: [A, B, C]})  # D's column is all zeros: fine
    _, ok = check(env, other)
    assert "spec#gene-plan" not in failures(ok)
    wrong_cols = make_spec(env, clients=[A, B, C], plan={G1: [A, B, C]},
                           edit={"clients": ISS,
                                 "outputs": {c: "masked_contribution" for c in ISS}})
    _, dec2 = check(env, wrong_cols)
    assert dec2.deciding_rule == "spec#gene-plan"


def test_genes_exclude_hash_mismatch(tmp_path):
    env = make_env(tmp_path, "open")
    spec = fe(env, genes_exclude=[G1])
    (env.root / spec.genes_exclude.ref).write_text("ncbi_gene\n9000000003\n")
    _, dec = check(env, spec)
    assert dec.deciding_rule == "spec#genes-exclude"


def test_provenance_manifest_hash_must_match(tmp_path):
    env = make_env(tmp_path, "open")
    spec = fe(env)
    (env.manifest_dir / "SpaceX-8.json").write_text(
        (env.manifest_dir / "SpaceX-8.json").read_text().replace('"n_flight": 6',
                                                                   '"n_flight": 7'))
    _, dec = check(env, spec)  # the registry in env was read before the edit
    assert dec.outcome == "allow"
    env.registry = load_registry(env.manifest_dir, clients=list(env.registry))
    _, dec = check(env, spec)
    assert dec.deciding_rule == "spec#provenance"
    assert "SpaceX-8" in dec.reason


def test_provenance_needs_inclusion_flag_hashes_for_de_meta(tmp_path):
    env = make_env(tmp_path, "open")
    d = json.loads(json.dumps(spec_to_dict(fe(env)), default=str))
    d["provenance"]["inclusion_flags_sha256"].pop("SpaceX-8")
    _, dec = check(env, parse_job_spec(d))
    assert dec.deciding_rule == "spec#provenance"


def test_spec_checks_listed_on_allow(tmp_path):
    env = make_env(tmp_path, "open")
    _, dec = check(env, fe(env))
    assert spec_checks(dec) == ["spec#clients-known", "spec#methods-known", "spec#outputs-cover",
                                "spec#aggregation", "spec#weighting", "spec#gene-plan",
                                "spec#genes-exclude", "spec#provenance"]
    assert all(evals(dec)[r].result in ("pass", "not_applicable") for r in spec_checks(dec))
    assert all(evals(dec)[r].site is None and evals(dec)[r].scope == "job"
               for r in spec_checks(dec))


# ------------------------------------------------------------------ profile mismatch


def test_manifest_policy_id_mismatch_raises(tmp_path):
    env = make_env(tmp_path, "open")  # manifests say pol:SpaceX-16:1
    with pytest.raises(PolicyMismatchError):
        check_job(fe(env), load_profile(POLICIES / "restricted"), env.registry, env.log)


# ------------------------------------------------------------------ the decision record


def test_record_fields_and_ids(tmp_path):
    env = make_env(tmp_path, "restricted")
    spec = fe(env)
    assert spec.job_id is None
    minted, dec = check(env, spec)
    assert JOB_ID_RE.fullmatch(minted.job_id)
    assert dec.job_id == minted.job_id
    suffix = minted.job_id.split(":", 1)[1]
    assert dec.decision_id == f"dec:{suffix}-1"
    assert dec.job_spec_sha256 == job_spec_sha256(minted)
    assert dec.release_log_id == "restricted-1"
    assert dec.release_log_head is None and dec.release_log_head_sha256 is None
    assert dec.evaluator_version == EVALUATOR_VERSION
    pol = {p["policy_id"]: p["sha256"] for p in dec.policies}
    assert pol["pol:SpaceX-16:2"] == env.profile.sites["SpaceX-16"].sha256
    d = dec.to_dict()
    assert list(d) == ["decision_id", "job_id", "job_spec_sha256", "release_log_id",
                       "release_log_head", "release_log_head_sha256", "outcome",
                       "deciding_rule", "reason", "evaluations", "policies",
                       "evaluator_version"]


def test_second_check_of_a_job_gets_the_next_decision_id(tmp_path):
    env = make_env(tmp_path, "restricted")
    minted, d1 = check(env, fe(env))
    again, d2 = check(env, minted)
    assert again.job_id == minted.job_id
    assert d2.decision_id.endswith("-2") and d1.decision_id.endswith("-1")


def test_given_decision_id_is_used(tmp_path):
    env = make_env(tmp_path, "restricted")
    _, dec = check(env, fe(env), decision_id="trace:FAKE-0001/dec/2")
    assert dec.decision_id == "trace:FAKE-0001/dec/2"


def test_record_is_written_and_reads_back(tmp_path):
    env = make_env(tmp_path, "restricted")
    _, dec = check(env, make_spec(env, clients=ISS, plan={G1: ISS}, methods=FULL_SET,
                                  outputs="per_site_summary"))
    path = env.root / "decisions" / (dec.decision_id.replace(":", "_").replace("/", "_")
                                     + ".json")
    assert read_decision(path).to_dict() == dec.to_dict()
    e = evals(dec)[DIFF16]
    assert e.genes_failed_ref == f"decisions/{path.stem}.SpaceX-16.diff.genes.csv"


def test_dry_run_writes_nothing(tmp_path):
    env = make_env(tmp_path, "restricted")
    spec = make_spec(env, clients=ISS, plan={G1: ISS}, methods=FULL_SET,
                     outputs="per_site_summary")
    _, dec = check(env, spec, write=False)
    assert not (env.root / "decisions").exists()
    e = evals(dec)[DIFF16]
    assert e.genes_failed == 1 and e.genes_failed_ref is None and len(e.sha256) == 64


def test_hash_ignores_policy_decision_in_the_record(tmp_path):
    env = make_env(tmp_path, "restricted")
    minted, d1 = check(env, fe(env))
    d = spec_to_dict(minted)
    d["policy_decision"] = d1.decision_id
    _, d2 = check(env, parse_job_spec(json.loads(json.dumps(d, default=str))))
    assert d2.job_spec_sha256 == d1.job_spec_sha256


def test_release_log_head_recorded(tmp_path):
    env = make_env(tmp_path, "restricted")
    spec, dec = check(env, fe(env))
    entry = env.log.append(spec, dec)
    _, dec2 = check(env, fe(env, clients=ISS, plan={G3: ISS}))
    assert dec2.release_log_head == entry["release_id"]
    assert dec2.release_log_head_sha256 == entry["entry_sha256"]


# ------------------------------------------------------------------ the release log in decisions


def test_counts_variants_share_one_data_key(tmp_path):
    """A standard release and an rrnarm request are compared (D-033, §5.3)."""
    env = make_env(tmp_path, "restricted")
    spec, dec = check(env, fe(env, counts_variant="standard"))
    env.log.append(spec, dec)
    _, j4 = check(env, fe(env, clients=[A, B, C], counts_variant="rrnarm",
                          inclusion={"rule": "finite_p", "min_sites_per_gene": "all"}))
    assert j4.deciding_rule == DIFF16


def test_other_strata_and_tasks_are_not_compared(tmp_path):
    env = make_env(tmp_path, "restricted")
    spec, dec = check(env, fe(env, clients=EARTH, plan={G1: EARTH}))
    env.log.append(spec, dec)  # an Earth release
    _, iss = check(env, fe(env, clients=[A, B, C], plan={G1: [A, B, C]}))
    assert DIFF16 not in failures(iss)  # nothing logged in the ISS stratum


@pytest.mark.parametrize("plain_first", [True, False])
def test_open_sites_per_site_summaries_and_a_masked_fe_combine(tmp_path, plain_first):
    """Per-site summaries from A, B, C plus a masked FE over A–D isolate D, in either order."""
    env = make_env(tmp_path, "restricted")
    plain = lambda: fe(env, clients=[A, B, C], plan={G1: [A, B, C]},  # noqa: E731
                       outputs="per_site_summary")
    masked = lambda: fe(env)  # noqa: E731
    first, second = (plain, masked) if plain_first else (masked, plain)
    spec, dec = check(env, first())
    assert dec.outcome == "allow", failures(dec)
    env.log.append(spec, dec)
    _, dec2 = check(env, second())
    assert dec2.deciding_rule == DIFF16


def test_stale_decision_cannot_be_released(tmp_path):
    env = make_env(tmp_path, "restricted")
    s1, d1 = check(env, fe(env))
    s2, d2 = check(env, fe(env, clients=[A, B, C], plan={G1: [A, B, C]}))
    assert d2.outcome == "allow"  # checked against the empty log
    env.log.append(s1, d1)
    with pytest.raises(StaleDecisionError):
        env.log.append(s2, d2)
    _, d3 = check(env, s2)  # a re-check sees R1
    assert d3.deciding_rule == DIFF16


def test_minted_spec_is_returned_unchanged_otherwise(tmp_path):
    env = make_env(tmp_path, "open")
    spec = fe(env)
    minted, _ = check(env, spec)
    assert job_spec_sha256(with_job_id(spec, minted.job_id)) == job_spec_sha256(minted)
