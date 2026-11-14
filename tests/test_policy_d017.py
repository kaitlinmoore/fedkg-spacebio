"""D-017, phase 4 stage: the policy check refuses FedAvg jobs that would fall into FLARE's silent
equal weighting (CONTRACTS §5.1 ``weighting``, §6.1 ``weighting``).

FLARE 2.9.1 silently uses 1.0 for a missing or unusable ``NUM_STEPS_CURRENT_ROUND``. The client
helper already refuses to send one (WS-11). Here the job spec itself is refused unless it
declares sample-count weighting and every client declares n_k, or declares equal/custom
weighting with a reason. WS-02's synthetic sites (FAKE-SITE-1..3, n_k 400/150/50).
"""

import pytest

from policy_helpers import FEDAVG, FIX, check, evals, failures, make_env, make_spec

WEIGHTING = "pol:project:1#fedavg-weighting"


def env_for(tmp_path, overrides=None):
    return make_env(tmp_path, "synthetic", policies_dir=FIX / "synthetic",
                    clients_file="clients_fedavg.json", overrides=overrides)


def fedavg_spec(env, weighting="default", **kw):
    return make_spec(env, clients=FEDAVG, plan={9000000001: FEDAVG, 9000000002: FEDAVG},
                     task="fedavg_classifier", outputs="model_update", weighting=weighting,
                     **kw)


def test_sample_count_with_every_n_k_is_allowed(tmp_path):
    env = env_for(tmp_path)
    _, dec = check(env, fedavg_spec(env))
    assert dec.outcome == "allow", failures(dec)
    assert evals(dec)[WEIGHTING].result == "pass"


def test_no_weighting_is_denied(tmp_path):
    env = env_for(tmp_path)
    _, dec = check(env, fedavg_spec(env, drop=["weighting"]))
    assert dec.outcome == "deny"
    assert dec.deciding_rule == WEIGHTING
    assert "weighting" in evals(dec)[WEIGHTING].detail


def test_null_weighting_is_denied(tmp_path):
    env = env_for(tmp_path)
    _, dec = check(env, fedavg_spec(env, weighting=None))
    assert dec.deciding_rule == WEIGHTING


@pytest.mark.parametrize("bad", [None, 0, -5, "400", float("nan"), float("inf"), True])
def test_sample_count_with_a_client_lacking_usable_n_k_is_denied(tmp_path, bad):
    """None removes n_k from the manifest; every other value is one FLARE would turn into 1.0."""
    env = env_for(tmp_path, overrides={"FAKE-SITE-2": {"n_k": bad}})
    _, dec = check(env, fedavg_spec(env))
    assert dec.outcome == "deny"
    assert dec.deciding_rule == WEIGHTING
    assert "FAKE-SITE-2" in evals(dec)[WEIGHTING].detail
    assert "FAKE-SITE-2" in dec.reason


@pytest.mark.parametrize("scheme", ["equal", "custom"])
@pytest.mark.parametrize("reason", [None, "", "   "])
def test_equal_or_custom_without_reason_is_denied(tmp_path, scheme, reason):
    env = env_for(tmp_path)
    w = {"scheme": scheme}
    if scheme == "custom":
        w["weights"] = {"FAKE-SITE-1": 1, "FAKE-SITE-2": 1, "FAKE-SITE-3": 2}
    if reason is not None:
        w["reason"] = reason
    _, dec = check(env, fedavg_spec(env, weighting=w))
    assert dec.deciding_rule == WEIGHTING


@pytest.mark.parametrize("scheme", ["equal", "custom"])
def test_equal_or_custom_with_reason_is_allowed(tmp_path, scheme):
    env = env_for(tmp_path, overrides={"FAKE-SITE-2": {"n_k": None}})  # n_k not needed here
    w = {"scheme": scheme, "reason": "sites differ in label quality; see methodology"}
    if scheme == "custom":
        w["weights"] = {"FAKE-SITE-1": 1, "FAKE-SITE-2": 1, "FAKE-SITE-3": 2}
    _, dec = check(env, fedavg_spec(env, weighting=w))
    assert dec.outcome == "allow", failures(dec)


@pytest.mark.parametrize("weights", [
    {"FAKE-SITE-1": 1, "FAKE-SITE-2": 1},                          # misses a client
    {"FAKE-SITE-1": 1, "FAKE-SITE-2": 1, "FAKE-SITE-3": 0},        # not positive
    {"FAKE-SITE-1": 1, "FAKE-SITE-2": 1, "FAKE-SITE-3": 1, "FAKE-SITE-9": 1},  # extra client
])
def test_custom_weights_must_cover_exactly_the_clients_and_be_positive(tmp_path, weights):
    env = env_for(tmp_path)
    w = {"scheme": "custom", "weights": weights, "reason": "test"}
    _, dec = check(env, fedavg_spec(env, weighting=w))
    assert dec.deciding_rule == WEIGHTING


def test_weighting_rule_is_not_applicable_to_de_meta(tmp_path):
    from policy_helpers import G1, ISS
    env = make_env(tmp_path, "open")
    _, dec = check(env, make_spec(env, clients=ISS, plan={G1: ISS}))
    assert evals(dec)[WEIGHTING].result == "not_applicable"


def test_inverse_variance_is_not_a_fedavg_scheme(tmp_path):
    env = env_for(tmp_path)
    _, dec = check(env, fedavg_spec(env, weighting={"scheme": "inverse_variance"}))
    assert dec.deciding_rule == "spec#weighting"
