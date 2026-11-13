"""Tests for the WS-08 calibrated simulation (``fedkg_spacebio.evaluation.meta_sim``) and the
method registry it uses (``meta_methods``). Synthetic data only: the calibration is built from a
small fake per-gene table in the WS-07 schema."""

import numpy as np
import pandas as pd
import pytest

from fedkg_spacebio.evaluation import meta_methods, meta_sim
from fedkg_spacebio.federation import meta, results

CLIENTS = ["SiteA", "SiteB", "SiteC"]


def fake_table(rng, G=1500):
    """A per-gene table shaped like WS-07's (schema phase3a-per-gene/1), synthetic values."""
    se = rng.uniform(0.08, 0.3, size=(G, 3))
    mu = np.where(rng.random(G) < 0.8, 0.0, rng.normal(0, 0.4, G))
    tau2 = rng.exponential(0.01, G)
    dev = rng.normal(size=(G, 3)) * np.sqrt(tau2)[:, None]
    y = mu[:, None] + dev + rng.normal(size=(G, 3)) * se
    t = pd.DataFrame({"ncbi_gene": np.arange(G)})
    for i, c in enumerate(CLIENTS):
        t[f"log2fc__{c}"] = y[:, i]
        t[f"se__{c}"] = se[:, i]
        t[f"n_flight__{c}"] = 5 + i
        t[f"n_ground__{c}"] = 5
        t[f"included__{c}"] = True
    return t


@pytest.fixture(scope="module")
def cal():
    rng = np.random.default_rng(0)
    t = fake_table(rng)
    alt = t.copy()
    alt.loc[:199, "included__SiteC"] = False
    return meta_sim.calibrate(t, "fake", alt_table=alt)


def test_matrices_from_table_masks_excluded_cells():
    rng = np.random.default_rng(1)
    t = fake_table(rng, 20)
    t.loc[3, "included__SiteB"] = False
    y, v, clients = meta_methods.matrices_from_table(t)
    assert clients == CLIENTS
    assert np.isnan(y[3, 1]) and np.isnan(v[3, 1])
    assert v[0, 0] == pytest.approx(t.loc[0, "se__SiteA"] ** 2)


def test_run_methods_fe_and_flag_match_ws07_definitions():
    rng = np.random.default_rng(2)
    t = fake_table(rng, 800)
    y, v, _ = meta_methods.matrices_from_table(t)
    frames, info = meta_methods.run_methods(y, v, ("fe", "fe_flag", "hksj_trunc"))
    pooled = results.classify(meta.pool(y, v))
    np.testing.assert_array_equal(frames["fe"]["hit"], pooled["fe_hit"])
    np.testing.assert_array_equal(frames["fe_flag"]["hit"], pooled["fe_hit_consistent"])
    np.testing.assert_array_equal(frames["hksj_trunc"]["hit"], pooled["re_hit"])
    np.testing.assert_allclose(frames["fe"]["q"], pooled["fe_q"])


def test_run_methods_returns_every_method_with_the_same_columns():
    rng = np.random.default_rng(3)
    y, v, _ = meta_methods.matrices_from_table(fake_table(rng, 600))
    frames, info = meta_methods.run_methods(y, v)
    assert list(frames) == list(meta_methods.METHODS)
    for f in frames.values():
        assert list(f.columns) == meta_methods.OUT_COLUMNS and len(f) == 600
    assert {"v2_prior", "v3_prior", "v1_trunc_prior"} <= set(info)


def test_calibration_shapes(cal):
    s = cal.summary()
    assert s["G"] == 1500 and s["K"] == 3
    assert cal.resid_df.tolist() == [8.0, 9.0, 10.0]          # n_flight + n_ground − 2
    assert cal.tau2_pi.sum() == pytest.approx(1.0)
    assert set(cal.sigma_mu) >= {0.80, 0.90, 0.95}
    assert np.isnan(cal.se_alt[:200, 2]).all() and np.isfinite(cal.se_alt[200:]).all()


def test_same_seed_same_replicate(cal):
    s = meta_sim.Setting("x")
    a = meta_sim.run_replicate(cal, s, meta_sim.seed_for(7, 0, 0, 3))
    b = meta_sim.run_replicate(cal, s, meta_sim.seed_for(7, 0, 0, 3))
    c = meta_sim.run_replicate(cal, s, meta_sim.seed_for(7, 0, 0, 4))
    pd.testing.assert_frame_equal(a, b)
    assert not a["calls"].equals(c["calls"])


@pytest.mark.parametrize("gen", meta_sim.GENERATORS)
def test_generate_kinds_and_se_rows(cal, gen):
    rng = np.random.default_rng(11)
    s = meta_sim.Setting("x", gen, frac_mu0=0.9)
    y, v, truth = meta_sim.generate(rng, cal, s)
    kind, mu, theta = truth["kind"], truth["mu"], truth["theta"]
    present = np.isfinite(y)
    # Kind bookkeeping.
    assert np.all(mu[kind <= 2] == 0) and np.all(mu[kind == 3] != 0)
    dev = np.where(present, theta - mu[:, None], 0.0)
    assert np.all(dev[kind == 1] == 0)
    assert np.all(np.abs(dev[kind == 2]).sum(axis=1) > 0)
    assert 0.85 < np.mean(mu == 0) < 0.95
    # SE rows come whole from the calibration pool (before any SE noise).
    pool = cal.se_alt if gen == "min3_pattern" else cal.se
    # (se_noise perturbs v on purpose; ``test_se_noise_changes_reported_variance_only`` covers it.)
    if gen != "se_noise":
        rows = {tuple(np.round(r, 12)) for r in np.where(np.isfinite(pool), pool, -1.0)}
        got = np.where(present, np.sqrt(v), -1.0)
        assert all(tuple(np.round(r, 12)) in rows for r in got[:200])
    if gen == "min3_pattern":
        assert set(np.unique(present.sum(axis=1))) <= {2, 3}


def test_se_noise_changes_reported_variance_only(cal):
    s_clean = meta_sim.Setting("x", "additive_eb")
    s_noisy = meta_sim.Setting("x", "se_noise")
    y1, v1, t1 = meta_sim.generate(np.random.default_rng(5), cal, s_clean)
    y2, v2, t2 = meta_sim.generate(np.random.default_rng(5), cal, s_noisy)
    np.testing.assert_array_equal(y1, y2)
    np.testing.assert_array_equal(t1["mu"], t2["mu"])
    ratio = (v2 / v1).ravel()
    assert not np.allclose(ratio, 1.0)
    assert np.mean(ratio) == pytest.approx(1.0, abs=0.05)       # E[χ²_df/df] = 1


def test_score_hand_case():
    """Five genes, two methods, everything counted by hand."""
    truth = {"kind": np.array([1, 2, 3, 3, 1]), "mu": np.array([0, 0, 0.5, -0.4, 0]),
             "theta_fe": np.array([0.0, 0.3, 0.5, -0.4, 0.0])}
    f = pd.DataFrame({"est": [0.1, 0.3, 0.6, 0.2, 0.0], "se": 0.1,
                      "ci_low": [-0.1, 0.1, 0.4, 0.0, -0.2], "ci_high": [0.3, 0.5, 0.8, 0.4, 0.2],
                      "p": 0.01, "q": 0.01, "hit": [True, True, True, True, False]})
    out = meta_sim.score({"m": f, "none": f.assign(hit=False)}, truth).set_index("method")
    m = out.loc["m"]
    assert m["calls"] == 4
    assert m["fdr_A"] == pytest.approx(1 / 4)                 # gene 0 is the only kind-1 call
    assert m["fdr_B"] == pytest.approx(2 / 4)                 # genes 0 and 1
    assert m["power_A"] == pytest.approx(3 / 3)               # kinds 2–3: genes 1, 2, 3
    assert m["power_B"] == pytest.approx(2 / 2)
    assert m["kind2_call_rate"] == pytest.approx(1.0)
    assert m["wrong_sign"] == 1                               # gene 3: est > 0, μ < 0
    assert m["coverage_mu"] == pytest.approx(3 / 5)           # genes 0, 2, 4
    assert m["coverage_theta_fe"] == pytest.approx(4 / 5)     # genes 0, 1, 2, 4
    assert out.loc["none", "calls"] == 0 and out.loc["none", "fdr_B"] == 0


def test_summarize_groups_by_setting_and_method(cal):
    reps = pd.concat([meta_sim.run_replicate(cal, meta_sim.Setting("x"),
                                             meta_sim.seed_for(1, 0, 0, r),
                                             methods=("fe", "v2")) for r in range(3)])
    summ = meta_sim.summarize(reps)
    assert len(summ) == 2 and set(summ["n_rep"]) == {3}
    assert {"fdr_B_mean", "fdr_B_sd", "power_A_mean"} <= set(summ.columns)
