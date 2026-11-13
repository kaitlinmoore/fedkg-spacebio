"""Tests for the WS-08 real-data comparison helpers (``fedkg_spacebio.evaluation.meta_compare``).
Tiny hand-computable inputs and synthetic data only."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fedkg_spacebio.evaluation import meta_compare, meta_methods


def frame(est, p, hit):
    est = np.asarray(est, dtype=float)
    return pd.DataFrame({"est": est, "se": 0.1, "ci_low": est - 0.2, "ci_high": est + 0.2,
                         "p": p, "q": p, "hit": hit})


def test_hit_counts_and_jaccard():
    a = frame([1, -1, 1, 0.5], [0.01] * 4, [True, True, False, False])
    b = frame([1, -1, 1, 0.5], [0.01] * 4, [True, False, True, False])
    c = frame([1, -1, 1, 0.5], [0.01] * 4, [False] * 4)
    hc = meta_compare.hit_counts({"a": a}).iloc[0]
    assert (hc["hits"], hc["up"], hc["down"]) == (2, 1, 1)
    j = meta_compare.jaccard_matrix({"a": a, "b": b, "c": c})
    assert j.loc["a", "b"] == pytest.approx(1 / 3)
    assert j.loc["a", "a"] == 1.0
    assert np.isnan(j.loc["c", "c"])
    assert j.loc["a", "c"] == 0.0


def test_rank_agreement_identical_and_reversed():
    rng = np.random.default_rng(0)
    p = rng.uniform(1e-6, 1, 200)
    est = rng.normal(size=200)
    a = frame(est, p, p < 0.01)
    b = frame(-est, p, p < 0.01)
    r = meta_compare.rank_agreement({"a": a, "a2": a, "b": b}, top_n=20).set_index(["a", "b"])
    assert r.loc[("a", "a2"), "spearman_signed_logp"] == pytest.approx(1.0)
    assert r.loc[("a", "b"), "spearman_signed_logp"] == pytest.approx(-1.0)
    assert r.loc[("a", "b"), "top20_overlap"] == 1.0


def test_flag_breakdown():
    f = frame([1, 1, 1, 1], [0.01] * 4, [True, True, True, False])
    out = meta_compare.flag_breakdown({"m": f}, np.array([0.05, 0.5, np.nan, 0.01])).iloc[0]
    assert (out["hits"], out["flagged"], out["unflagged"]) == (3, 1, 2)


def test_leave_one_out_sign_agreement_and_survival():
    """Three sites agree on direction for strong genes; site C is noise. Dropping A or B should
    keep most FE hits; LOMO agreement for A and B should be high."""
    rng = np.random.default_rng(1)
    G = 3000
    mu = np.where(rng.random(G) < 0.8, 0.0, rng.choice([-0.8, 0.8], G))
    v = np.full((G, 3), 0.02)
    y = mu[:, None] + rng.normal(size=(G, 3)) * np.sqrt(v)
    full, _ = meta_methods.run_methods(y, v, ("fe", "v2"))
    lomo, infl = meta_compare.leave_one_out(y, v, ["A", "B", "C"], full, ("fe", "v2"))
    assert set(lomo["dropped"]) == {"A", "B", "C"}
    fe = lomo[lomo.method == "fe"].set_index("dropped")
    assert (fe["sign_agreement"] > 0.95).all()
    assert (fe["baseline_all_genes"] < fe["sign_agreement"]).all()
    s = infl[infl.method == "fe"].set_index("dropped")
    assert (s["full_hits_surviving"] <= s["full_hits"]).all()
    assert (s["survival_share"] > 0.8).all()


def test_leave_one_out_by_hand():
    """Two genes, three sites, equal variances: the FE estimate without site i is the mean of
    the other two, and agreement is counted only over full-data hits."""
    y = np.array([[1.0, 2.0, -3.0], [0.5, 0.4, 0.6]])
    v = np.full((2, 3), 0.01)
    full = {"fe": frame([0.0, 0.5], [0.5, 0.001], [False, True])}
    lomo, _ = meta_compare.leave_one_out(y, v, ["A", "B", "C"], full, ("fe",))
    # Gene 1 only (the hit): without A, mean(0.4, 0.6) > 0 and y_A = 0.5 > 0 → agree; etc.
    assert lomo["n_hits_scored"].tolist() == [1, 1, 1]
    assert lomo["sign_agreement"].tolist() == [1.0, 1.0, 1.0]
    # Baseline over both genes: without C, gene 0's pool mean is 1.5 but y_C = −3 → disagree.
    assert lomo.set_index("dropped").loc["C", "baseline_all_genes"] == 0.5


def test_scaled_f_quantiles():
    q = meta_compare.scaled_f_quantiles(4, 3, 10.0, 1.5)
    np.testing.assert_allclose(q, 1.5 * stats.f.ppf([0.125, 0.375, 0.625, 0.875], 3, 10))
    qi = meta_compare.scaled_f_quantiles(2, 3, np.inf, 2.0)
    np.testing.assert_allclose(qi, 2.0 * stats.chi2.ppf([0.25, 0.75], 3) / 3)
