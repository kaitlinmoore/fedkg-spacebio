"""The append-only release log (CONTRACTS §5.3 "Log layout" and "Integrity", D-033).

The log is what makes differencing checkable: every release records which clients contributed to
which gene. These tests build decisions by hand, so they test the store on its own, without the
evaluator.
"""

import json

import pytest
from fedkg_spacebio.federation.gene_plan import read_gene_plan
from fedkg_spacebio.federation.jobspec import job_spec_sha256, with_job_id
from fedkg_spacebio.policy.decision import EVALUATOR_VERSION, Decision, write_decision
from fedkg_spacebio.policy.release_log import (
    AppendRefused,
    LogIntegrityError,
    StaleDecisionError,
    active_log,
    cross_check_decisions,
    read_index,
    reset_log,
)

from policy_helpers import G1, G3, G4, ISS, A, B, C, D, make_env, make_spec

_n = {"i": 0}


def allow(env, spec, *, outcome="allow", head="current", job_spec_sha=None):
    """A decision record for ``spec`` against the log as it stands (or as given)."""
    _n["i"] += 1
    if spec.job_id is None:
        spec = with_job_id(spec, f"job:FAKE-{_n['i']:04d}")
    h = env.log.head() if head == "current" else head
    dec = Decision(
        decision_id=f"dec:{spec.job_id.split(':', 1)[1]}-1", job_id=spec.job_id,
        job_spec_sha256=job_spec_sha or job_spec_sha256(spec),
        release_log_id=env.log.log_id,
        release_log_head=h[0] if h else None, release_log_head_sha256=h[1] if h else None,
        outcome=outcome, deciding_rule=None if outcome == "allow" else "pol:X:1#y",
        reason="test", evaluations=[], policies=[], evaluator_version=EVALUATOR_VERSION)
    return spec, dec


def release(env, plan, clients=ISS, **kw):
    spec = make_spec(env, clients=list(clients), plan=plan, **kw)
    spec, dec = allow(env, spec)
    return env.log.append(spec, dec), spec, dec


def test_new_log_is_empty_and_named_by_profile(tmp_path):
    env = make_env(tmp_path, "restricted")
    assert env.log.log_id == "restricted-1"
    assert env.log.entries() == []
    assert env.log.head() is None
    events = read_index(env.root)
    assert [(e["event"], e["log_id"]) for e in events] == [("start", "restricted-1")]


def test_append_and_read_back(tmp_path):
    env = make_env(tmp_path, "restricted")
    entry, spec, dec = release(env, {G1: ISS, G3: [A, C, D]})
    assert entry["seq"] == 0
    assert entry["release_id"] == f"rel:{spec.job_id.split(':', 1)[1]}-1"
    assert entry["data_key"] == {"task": "de_meta", "stratum": "iss"}
    assert entry["clients"] == ISS
    assert entry["outputs"] == {c: "masked_contribution" for c in ISS}
    assert entry["counts_variant"] == "standard"
    assert entry["job_id"] == spec.job_id
    assert entry["job_spec_sha256"] == job_spec_sha256(spec)
    assert entry["decision_id"] == dec.decision_id
    assert entry["prev_entry_sha256"] is None
    assert set(entry["inputs"]) >= {"manifest_sha256", "inclusion_flags_sha256",
                                    "gene_plan_sha256"}
    contributors = read_gene_plan(env.root / entry["contributors"]["ref"],
                                  entry["contributors"]["sha256"])
    assert {int(g) for g in contributors.index} == {G1, G3}
    assert env.log.head() == (entry["release_id"], entry["entry_sha256"])

    # a fresh object on the same files sees the same log
    again = active_log("restricted", root=env.root)
    assert again.entries() == env.log.entries()


def test_entries_chain(tmp_path):
    env = make_env(tmp_path, "restricted")
    e1, *_ = release(env, {G1: ISS})
    e2, *_ = release(env, {G1: ISS})
    assert e2["seq"] == 1 and e2["prev_entry_sha256"] == e1["entry_sha256"]
    env.log.verify()


def test_contributors_are_the_plan_minus_exclusions(tmp_path):
    env = make_env(tmp_path, "restricted")
    entry, *_ = release(env, {G1: ISS, G3: [A, C, D], G4: [A, B, C]}, genes_exclude=[G3])
    c = read_gene_plan(env.root / entry["contributors"]["ref"], entry["contributors"]["sha256"])
    assert {int(g) for g in c.index} == {G1, G4}


def test_no_output_values_in_entries(tmp_path):
    """D-036: inputs and settings are hashed, never pooled or per-site floats."""
    env = make_env(tmp_path, "restricted")
    release(env, {G1: ISS})
    text = env.log.path.read_text()
    for word in ("v3_", "fe_est", "log2fc", "estimate", "pvalue", "\"se\""):
        assert word not in text


def test_history_filters_by_task_and_stratum_not_variant(tmp_path):
    env = make_env(tmp_path, "restricted")
    release(env, {G1: ISS}, counts_variant="standard")
    release(env, {G1: ISS}, counts_variant="rrnarm",
            inclusion={"rule": "finite_p", "min_sites_per_gene": 3})
    release(env, {G1: ["SpaceX-13-LAR", "SpaceX-16-LAR", "SpaceX-21"]},
            clients=["SpaceX-13-LAR", "SpaceX-16-LAR", "SpaceX-21"])
    assert len(env.log.history("de_meta", "iss")) == 2
    assert len(env.log.history("de_meta", "earth")) == 1
    assert env.log.history("clock_plane", "iss") == []


# ------------------------------------------------------------------ append refuses


def test_append_refuses_a_deny(tmp_path):
    env = make_env(tmp_path, "restricted")
    spec, dec = allow(env, make_spec(env, clients=ISS, plan={G1: ISS}), outcome="deny")
    with pytest.raises(AppendRefused):
        env.log.append(spec, dec)
    assert env.log.entries() == []


def test_append_refuses_another_spec(tmp_path):
    env = make_env(tmp_path, "restricted")
    spec, dec = allow(env, make_spec(env, clients=ISS, plan={G1: ISS}), job_spec_sha="0" * 64)
    with pytest.raises(AppendRefused):
        env.log.append(spec, dec)


def test_append_refuses_a_decision_for_another_log(tmp_path):
    env = make_env(tmp_path, "restricted")
    spec, dec = allow(env, make_spec(env, clients=ISS, plan={G1: ISS}))
    dec.release_log_id = "restricted-7"
    with pytest.raises(AppendRefused):
        env.log.append(spec, dec)


def test_append_refuses_a_stale_decision(tmp_path):
    """Two checks against the same head can't both be released: the second must be re-checked."""
    env = make_env(tmp_path, "restricted")
    s1, d1 = allow(env, make_spec(env, clients=ISS, plan={G1: ISS}))
    s2, d2 = allow(env, make_spec(env, clients=[A, B, C], plan={G1: [A, B, C]}))
    env.log.append(s1, d1)
    with pytest.raises(StaleDecisionError):
        env.log.append(s2, d2)
    assert len(env.log.entries()) == 1


def test_append_refuses_twice(tmp_path):
    env = make_env(tmp_path, "restricted")
    spec, dec = allow(env, make_spec(env, clients=ISS, plan={G1: ISS}))
    env.log.append(spec, dec)
    with pytest.raises(AppendRefused):
        env.log.append(spec, dec)


# ------------------------------------------------------------------ tampering is detected


def _rewrite(path, fn):
    lines = path.read_text().splitlines()
    path.write_text("\n".join(fn(lines)) + "\n")


def test_edited_entry_is_detected(tmp_path):
    env = make_env(tmp_path, "restricted")
    for _ in range(3):
        release(env, {G1: ISS})

    def edit(lines):
        e = json.loads(lines[1])
        e["clients"] = [A, B, C]
        lines[1] = json.dumps(e)
        return lines
    _rewrite(env.log.path, edit)
    with pytest.raises(LogIntegrityError):
        env.log.verify()
    with pytest.raises(LogIntegrityError):
        env.log.entries()


def test_removed_middle_entry_is_detected(tmp_path):
    env = make_env(tmp_path, "restricted")
    for _ in range(3):
        release(env, {G1: ISS})
    _rewrite(env.log.path, lambda lines: [lines[0], lines[2]])
    with pytest.raises(LogIntegrityError):
        env.log.verify()


def test_edited_contributor_file_is_detected(tmp_path):
    env = make_env(tmp_path, "restricted")
    entry, *_ = release(env, {G1: ISS})
    p = env.root / entry["contributors"]["ref"]
    p.write_text(p.read_text().replace("9000000001,1,1,1,1", "9000000001,1,1,1,0"))
    with pytest.raises(LogIntegrityError):
        env.log.verify()


def test_truncated_tail_is_caught_by_decision_records(tmp_path):
    """A hash chain can't see its newest entry removed; the decision records can."""
    env = make_env(tmp_path, "restricted")
    release(env, {G1: ISS})
    release(env, {G1: ISS})
    spec, dec = allow(env, make_spec(env, clients=ISS, plan={G1: ISS}))  # head = entry 2
    write_decision(dec, env.root)
    cross_check_decisions(env.log)  # consistent
    _rewrite(env.log.path, lambda lines: lines[:1])
    env.log.verify()  # the shortened chain is still valid on its own...
    with pytest.raises(LogIntegrityError):
        cross_check_decisions(env.log)  # ...but a decision saw a head that is gone


# ------------------------------------------------------------------ reset = a new log, recorded


def test_reset_starts_a_new_log_and_keeps_the_old(tmp_path):
    env = make_env(tmp_path, "restricted")
    entry, *_ = release(env, {G1: ISS})
    old_path = env.log.path
    new = reset_log("restricted", reason="demo rehearsal 2", root=env.root)
    assert new.log_id == "restricted-2"
    assert new.entries() == []
    assert old_path.exists()
    assert active_log("restricted", root=env.root).log_id == "restricted-2"
    switch = read_index(env.root)[-1]
    assert switch["event"] == "switch"
    assert switch["log_id"] == "restricted-2"
    assert switch["previous_log_id"] == "restricted-1"
    assert switch["previous_head"] == entry["release_id"]
    assert switch["previous_head_sha256"] == entry["entry_sha256"]
    assert switch["reason"] == "demo rehearsal 2"


def test_reset_needs_a_reason(tmp_path):
    env = make_env(tmp_path, "restricted")
    with pytest.raises(ValueError):
        reset_log("restricted", reason="", root=env.root)


def test_profiles_have_separate_logs(tmp_path):
    env = make_env(tmp_path, "restricted")
    release(env, {G1: ISS})
    other = active_log("open", root=env.root)
    assert other.log_id == "open-1" and other.entries() == []


def test_opening_without_create_writes_nothing(tmp_path):
    log = active_log("open", root=tmp_path / "policy", create=False)
    assert log.log_id == "open-1" and log.entries() == [] and log.head() is None
    assert not (tmp_path / "policy").exists()
