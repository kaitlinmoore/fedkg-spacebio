"""Tests for the phase 3a pooling module (``fedkg_spacebio.federation.meta``).

They define what "correct" means for fixed effect, Q, I², BH, REML τ² and HKSJ.

Reference values come from published output, checked against the primary source on 2026-11-12:
- BCG vaccine trials (Colditz et al. 1994), the standard metafor example.
  - Data and REML output: metadat `dat.bcg` reference page,
    https://wviechtb.github.io/metadat/reference/dat.bcg.html.
  - Knapp–Hartung output (`rma(yi, vi, data=dat, test="knha")`): metafor README,
    https://github.com/wviechtb/metafor (README.md, "Model Results" with tval/df).
  - metafor prints 4 decimals, so the tolerance is 5e-5 (half a unit in the last printed place)
    on values we compute from the exact counts.

The BCG table is a published textbook dataset, not OSDR data, so it lives here as constants
rather than as a fixture file (the fixtures folder is synthetic-only).
"""

import numpy as np
import pytest
from scipy import optimize, stats
from statsmodels.stats.meta_analysis import combine_effects
from statsmodels.stats.multitest import multipletests

from fedkg_spacebio.federation import meta

PRINT_TOL = 5e-5  # half a unit in metafor's 4th printed decimal

# dat.bcg: tpos, tneg, cpos, cneg for the 13 trials (metadat reference page).
BCG = np.array([
    [4, 119, 11, 128], [6, 300, 29, 274], [3, 228, 11, 209], [62, 13536, 248, 12619],
    [33, 5036, 47, 5761], [180, 1361, 372, 1079], [8, 2537, 10, 619], [505, 87886, 499, 87892],
    [29, 7470, 45, 7232], [17, 1699, 65, 1600], [186, 50448, 141, 27197], [5, 2493, 3, 2338],
    [27, 16886, 29, 17825],
], dtype=float)
# escalc(measure="RR") output as printed by metafor.
BCG_YI_PRINTED = [-0.8893, -1.5854, -1.3481, -1.4416, -0.2175, -0.7861, -1.6209, 0.0120,
                  -0.4694, -1.3713, -0.3394, 0.4459, -0.0173]
BCG_VI_PRINTED = [0.3256, 0.1946, 0.4154, 0.0200, 0.0512, 0.0069, 0.2230, 0.0040, 0.0564,
                  0.0730, 0.0124, 0.5325, 0.0714]


@pytest.fixture(scope="module")
def bcg():
    """Log risk ratios and their variances, computed from the raw 2×2 counts."""
    tpos, tneg, cpos, cneg = BCG.T
    yi = np.log((tpos / (tpos + tneg)) / (cpos / (cpos + cneg)))
    vi = 1 / tpos - 1 / (tpos + tneg) + 1 / cpos - 1 / (cpos + cneg)
    return yi, vi


def simulate(rng, n_genes, k, tau2, v_low, v_high):
    """y_ij = μ_j + u_ij + e_ij with u ~ N(0, τ²), e ~ N(0, v_ij), μ_j ~ N(0, 1)."""
    mu = rng.normal(0, 1, size=n_genes)
    v = rng.uniform(v_low, v_high, size=(n_genes, k))
    y = mu[:, None] + rng.normal(0, np.sqrt(tau2), size=(n_genes, k)) + rng.normal(0, np.sqrt(v))
    return y, v, mu


# ---------------------------------------------------------------------------------------------
# Inputs, fixed effect, Q, I², BH (pass without REML/HKSJ)
# ---------------------------------------------------------------------------------------------


def test_bcg_effect_sizes_match_metafor_print(bcg):
    yi, vi = bcg
    np.testing.assert_allclose(yi, BCG_YI_PRINTED, atol=PRINT_TOL)
    np.testing.assert_allclose(vi, BCG_VI_PRINTED, atol=PRINT_TOL)


def test_bcg_fixed_effect_and_q(bcg):
    yi, vi = bcg
    fe = meta.fixed_effect(yi, vi)
    # metafor: Q(df = 12) = 152.2330
    assert fe["q"][0] == pytest.approx(152.2330, abs=PRINT_TOL)
    assert fe["q_df"][0] == 12
    # Independent implementation: statsmodels.
    sm = combine_effects(yi, vi, method_re="pm")
    assert fe["est"][0] == pytest.approx(sm.mean_effect_fe, rel=1e-12)
    assert fe["se"][0] == pytest.approx(np.sqrt(sm.var_eff_w_fe), rel=1e-12)
    assert fe["q"][0] == pytest.approx(sm.q, rel=1e-12)
    # Q-based I² = (Q − df) / Q
    assert fe["i2"][0] == pytest.approx((152.2330 - 12) / 152.2330, abs=1e-6)


def test_fixed_effect_formula_by_hand():
    y, v = np.array([1.0, 3.0]), np.array([1.0, 3.0])
    fe = meta.fixed_effect(y, v)
    # w = (1, 1/3); μ = (1 + 1) / (4/3) = 1.5; SE = sqrt(3/4)
    assert fe["est"][0] == pytest.approx(1.5)
    assert fe["se"][0] == pytest.approx(np.sqrt(0.75))
    assert fe["q"][0] == pytest.approx(1 * 0.25 + (1 / 3) * 2.25)


def test_fixed_effect_k1_and_k0():
    fe = meta.fixed_effect([[0.7, np.nan], [np.nan, np.nan]], [[0.04, np.nan], [np.nan, np.nan]])
    assert list(fe["k"]) == [1, 0]
    assert fe["est"][0] == pytest.approx(0.7) and fe["se"][0] == pytest.approx(0.2)
    assert np.isnan(fe["q_p"][0]) and np.isnan(fe["i2"][0])
    assert np.isnan(fe["est"][1]) and np.isnan(fe["se"][1])


def test_identical_estimates_give_q_zero_and_i2_zero():
    fe = meta.fixed_effect([0.5, 0.5, 0.5], [0.1, 0.2, 0.3])
    assert fe["q"][0] == pytest.approx(0.0, abs=1e-15)
    assert fe["i2"][0] == 0.0


def test_missing_cells_equal_dropping_the_site_fe():
    y = np.array([[0.2, np.nan, -0.1, 0.4]])
    v = np.array([[0.05, np.nan, 0.08, 0.02]])
    full = meta.fixed_effect(y, v)
    dropped = meta.fixed_effect(y[:, [0, 2, 3]], v[:, [0, 2, 3]])
    for key in ("k", "est", "se", "q", "i2"):
        np.testing.assert_allclose(full[key], dropped[key], rtol=1e-14)


def test_as_matrix_rejects_bad_input():
    with pytest.raises(ValueError, match="shapes"):
        meta.as_matrix([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="> 0"):
        meta.as_matrix([1.0, 2.0], [1.0, 0.0])
    # A NaN estimate makes its variance irrelevant, even if that variance is 0.
    _, _, present = meta.as_matrix([1.0, np.nan], [1.0, 0.0])
    assert present.tolist() == [[True, False]]


def test_bh_matches_statsmodels_and_ignores_nan():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=200) ** 3
    p[[3, 50, 120]] = np.nan
    q = meta.bh_qvalues(p)
    ok = np.isfinite(p)
    expected = multipletests(p[ok], method="fdr_bh")[1]
    np.testing.assert_allclose(q[ok], expected, rtol=1e-12)
    assert np.all(np.isnan(q[~ok]))
    assert np.all(np.isnan(meta.bh_qvalues([np.nan, np.nan])))


# ---------------------------------------------------------------------------------------------
# REML τ²
# ---------------------------------------------------------------------------------------------


def test_bcg_reml_matches_metafor(bcg):
    yi, vi = bcg
    tau2, converged = meta.tau2_reml(yi, vi)
    assert converged[0]
    # metafor: tau^2 = 0.3132; estimate −0.7145, se 0.1798, CI −1.0669 to −0.3622 (normal-based)
    assert tau2[0] == pytest.approx(0.3132, abs=PRINT_TOL)
    w = 1 / (vi + tau2[0])
    mu = np.sum(w * yi) / np.sum(w)
    se = 1 / np.sqrt(np.sum(w))
    assert mu == pytest.approx(-0.7145, abs=PRINT_TOL)
    assert se == pytest.approx(0.1798, abs=PRINT_TOL)
    assert mu - stats.norm.ppf(0.975) * se == pytest.approx(-1.0669, abs=PRINT_TOL)
    assert mu + stats.norm.ppf(0.975) * se == pytest.approx(-0.3622, abs=PRINT_TOL)
    # metafor's τ²-based I² = 92.22% (H² = 12.86)
    assert meta.i2_from_tau2(vi, tau2)[0] == pytest.approx(0.9222, abs=PRINT_TOL)


def test_reml_identical_estimates_give_zero():
    tau2, converged = meta.tau2_reml([[0.3, 0.3, 0.3, 0.3]], [[0.1, 0.05, 0.2, 0.3]])
    assert tau2[0] == 0.0
    assert converged[0]


def test_reml_never_negative_and_zero_when_less_spread_than_sampling():
    # Spread far smaller than the sampling SDs (≈ 0.3–0.5): the untruncated root would be < 0.
    y = np.array([[0.10, 0.11, 0.09, 0.105]])
    v = np.array([[0.1, 0.2, 0.15, 0.25]])
    tau2, _ = meta.tau2_reml(y, v)
    assert tau2[0] == 0.0
    rng = np.random.default_rng(2)
    ys, vs, _ = simulate(rng, 500, 4, tau2=0.0, v_low=0.05, v_high=0.5)
    t, _ = meta.tau2_reml(ys, vs)
    assert np.all(t >= 0)


def test_reml_k_below_two_is_nan():
    y = np.array([[0.5, np.nan, np.nan], [np.nan, np.nan, np.nan], [0.1, 0.4, np.nan]])
    v = np.array([[0.1, np.nan, np.nan], [np.nan, np.nan, np.nan], [0.1, 0.2, np.nan]])
    tau2, converged = meta.tau2_reml(y, v)
    assert np.isnan(tau2[0]) and np.isnan(tau2[1]) and not converged[0] and not converged[1]
    assert np.isfinite(tau2[2]) and tau2[2] >= 0


def test_reml_equals_pm_when_variances_are_equal():
    """With equal v, REML and Paule–Mandel both reduce to max(0, s² − v)."""
    rng = np.random.default_rng(3)
    y, _, _ = simulate(rng, 200, 5, tau2=0.3, v_low=0.1, v_high=0.1)
    v = np.full_like(y, 0.1)
    tau2, converged = meta.tau2_reml(y, v)
    assert converged.all()
    closed_form = np.maximum(0.0, y.var(axis=1, ddof=1) - 0.1)
    np.testing.assert_allclose(tau2, closed_form, atol=1e-8)
    for g in range(20):
        sm = combine_effects(y[g], v[g], method_re="pm")
        assert tau2[g] == pytest.approx(sm.tau2, abs=1e-6)


def test_reml_close_to_pm_when_variances_are_similar():
    """Near-equal variances (±10%): PM ≈ REML but not identical. The two estimators weight sites
    differently, so the gap grows with the spread of v. It's 0 at equal v (test above), and
    here (true τ² = 0.2) it was max ≈ 0.009 and median ≈ 0.0013 when the test was written.
    Stated tolerance: max |Δτ²| < 0.015 and median < 0.003, about 7% of τ². A wrong REML
    (e.g. the ML estimator, which is biased down by roughly a factor (k − 1)/k = 0.75 at k = 4)
    misses it by an order of magnitude."""
    rng = np.random.default_rng(4)
    y, v, _ = simulate(rng, 200, 4, tau2=0.2, v_low=0.09, v_high=0.11)
    tau2, _ = meta.tau2_reml(y, v)
    pm = np.array([combine_effects(y[g], v[g], method_re="pm").tau2 for g in range(len(y))])
    diff = np.abs(tau2 - pm)
    assert diff.max() < 0.015
    assert np.median(diff) < 0.003


def test_reml_recovers_known_tau2_on_simulated_data():
    """REML is close to unbiased. Over 4,000 genes × 20 sites, the mean τ̂² is within 0.01 of the
    truth (the Monte Carlo SE of the mean is about 0.002)."""
    rng = np.random.default_rng(5)
    y, v, _ = simulate(rng, 4000, 20, tau2=0.2, v_low=0.05, v_high=0.4)
    tau2, converged = meta.tau2_reml(y, v)
    assert converged.all()
    assert tau2.mean() == pytest.approx(0.2, abs=0.01)


def test_reml_reports_non_convergence():
    rng = np.random.default_rng(6)
    y, v, _ = simulate(rng, 50, 6, tau2=0.5, v_low=0.01, v_high=1.0)
    _, converged = meta.tau2_reml(y, v, tol=1e-300, max_iter=1)
    assert not converged.all()


def test_reml_root_maximizes_the_restricted_likelihood():
    """Independent of metafor and PM: τ̂² must be the maximizer of the REML log-likelihood
    over τ² ≥ 0. For one gene, up to a constant,

        −2 ℓ_R(τ²) = Σ log(v_i + τ²) + log Σ w_i + Σ w_i (y_i − μ̂)²,   w_i = 1/(v_i + τ²)

    A bounded scalar optimizer finds the maximum directly; the boundary τ² = 0 is checked too,
    because when the maximum is at 0 the derivative there isn't zero. Unequal variances, so this
    isn't the equal-v closed form, and k = 3–6 like the real strata."""
    def neg2ll(t2, y, v):
        w = 1 / (v + t2)
        mu = np.sum(w * y) / np.sum(w)
        return np.sum(np.log(v + t2)) + np.log(np.sum(w)) + np.sum(w * (y - mu) ** 2)

    rng = np.random.default_rng(9)
    for k in (3, 4, 6):
        y, v, _ = simulate(rng, 40, k, tau2=0.15, v_low=0.02, v_high=0.5)
        tau2, converged = meta.tau2_reml(y, v)
        assert converged.all()
        for g in range(len(y)):
            r = optimize.minimize_scalar(neg2ll, bounds=(0, 20), args=(y[g], v[g]),
                                         method="bounded", options={"xatol": 1e-12})
            best = r.x if neg2ll(r.x, y[g], v[g]) < neg2ll(0.0, y[g], v[g]) else 0.0
            assert tau2[g] == pytest.approx(best, abs=1e-6)


def test_reml_missing_cells_equal_dropping_the_site():
    y = np.array([[0.2, np.nan, -0.6, 0.9, 0.1]])
    v = np.array([[0.05, np.nan, 0.08, 0.02, 0.1]])
    a, _ = meta.tau2_reml(y, v)
    b, _ = meta.tau2_reml(y[:, [0, 2, 3, 4]], v[:, [0, 2, 3, 4]])
    assert a[0] == pytest.approx(b[0], rel=1e-9)


# ---------------------------------------------------------------------------------------------
# HKSJ
# ---------------------------------------------------------------------------------------------


def test_bcg_hksj_matches_metafor_knha(bcg):
    yi, vi = bcg
    tau2, _ = meta.tau2_reml(yi, vi)
    re = meta.random_effects_hksj(yi, vi, tau2, truncate=False)
    # metafor test="knha": estimate −0.7145, se 0.1808, tval −3.9522, df 12, pval 0.0019,
    # CI −1.1084 to −0.3206
    assert re["est"][0] == pytest.approx(-0.7145, abs=PRINT_TOL)
    assert re["se"][0] == pytest.approx(0.1808, abs=PRINT_TOL)
    assert re["t"][0] == pytest.approx(-3.9522, abs=PRINT_TOL)
    assert re["df"][0] == 12
    assert re["p"][0] == pytest.approx(0.0019, abs=PRINT_TOL)
    assert re["ci_low"][0] == pytest.approx(-1.1084, abs=PRINT_TOL)
    assert re["ci_high"][0] == pytest.approx(-0.3206, abs=PRINT_TOL)
    # Here q > 1, so truncation changes nothing.
    assert re["multiplier"][0] > 1
    tr = meta.random_effects_hksj(yi, vi, tau2, truncate=True)
    for key in ("est", "se", "p", "ci_low", "ci_high"):
        assert tr[key][0] == pytest.approx(re[key][0], rel=1e-12)


def test_hksj_matches_statsmodels_given_the_same_tau2(bcg):
    """An independent check of the HKSJ arithmetic, separate from the τ² estimator: give it
    statsmodels' PM τ² and compare with statsmodels' own (untruncated) HKSJ variance."""
    yi, vi = bcg
    sm = combine_effects(yi, vi, method_re="pm")
    re = meta.random_effects_hksj(yi, vi, np.array([sm.tau2]), truncate=False)
    assert re["est"][0] == pytest.approx(sm.mean_effect_re, rel=1e-12)
    assert re["se"][0] ** 2 == pytest.approx(sm.var_hksj_re, rel=1e-10)


def test_tau2_zero_reduces_to_fixed_effect():
    y = np.array([[0.4, 0.4, 0.4]])
    v = np.array([[0.1, 0.2, 0.4]])
    tau2, _ = meta.tau2_reml(y, v)
    assert tau2[0] == 0.0
    fe = meta.fixed_effect(y, v)
    tr = meta.random_effects_hksj(y, v, tau2, truncate=True)
    un = meta.random_effects_hksj(y, v, tau2, truncate=False)
    assert tr["est"][0] == pytest.approx(fe["est"][0], rel=1e-14)
    assert tr["se"][0] == pytest.approx(fe["se"][0], rel=1e-14)  # max(1, 0) = 1
    assert un["se"][0] == pytest.approx(0.0, abs=1e-15)  # q = 0
    assert tr["multiplier"][0] == pytest.approx(0.0, abs=1e-15)


def test_truncation_binds_when_sites_agree_more_than_expected():
    y = np.array([[0.10, 0.11, 0.09, 0.105]])
    v = np.array([[0.1, 0.2, 0.15, 0.25]])
    tau2, _ = meta.tau2_reml(y, v)
    fe = meta.fixed_effect(y, v)
    tr = meta.random_effects_hksj(y, v, tau2, truncate=True)
    un = meta.random_effects_hksj(y, v, tau2, truncate=False)
    assert tr["multiplier"][0] < 1
    assert un["se"][0] < tr["se"][0]
    assert tr["se"][0] == pytest.approx(fe["se"][0], rel=1e-12)  # τ² = 0 here, so w* = w


def test_hksj_k2_uses_t_with_one_df():
    y, v = np.array([[0.2, 0.8]]), np.array([[0.04, 0.09]])
    tau2, _ = meta.tau2_reml(y, v)
    re = meta.random_effects_hksj(y, v, tau2)
    assert re["df"][0] == 1
    half_width = re["ci_high"][0] - re["est"][0]
    assert half_width / re["se"][0] == pytest.approx(stats.t.ppf(0.975, 1), rel=1e-12)  # 12.706
    assert re["est"][0] - re["ci_low"][0] == pytest.approx(half_width, rel=1e-12)
    assert re["p"][0] == pytest.approx(2 * stats.t.sf(abs(re["t"][0]), 1), rel=1e-12)


def test_hksj_k_below_two_is_nan():
    y = np.array([[0.5, np.nan], [np.nan, np.nan]])
    v = np.array([[0.1, np.nan], [np.nan, np.nan]])
    re = meta.random_effects_hksj(y, v, np.array([np.nan, np.nan]))
    for key in ("est", "se", "t", "p", "ci_low", "ci_high", "multiplier"):
        assert np.all(np.isnan(re[key])), key


def test_hksj_coverage_with_four_sites():
    """Truncated HKSJ should cover the true μ at least about 95% of the time at k = 4 (it's
    conservative). The Monte Carlo SE at 4,000 genes is about 0.0035."""
    rng = np.random.default_rng(7)
    y, v, mu = simulate(rng, 4000, 4, tau2=0.1, v_low=0.02, v_high=0.2)
    tau2, _ = meta.tau2_reml(y, v)
    re = meta.random_effects_hksj(y, v, tau2, truncate=True)
    covered = (re["ci_low"] <= mu) & (mu <= re["ci_high"])
    assert 0.94 <= covered.mean() <= 0.995


def test_sign_flip_and_site_permutation():
    rng = np.random.default_rng(8)
    y, v, _ = simulate(rng, 30, 5, tau2=0.2, v_low=0.05, v_high=0.3)
    base = meta.pool(y, v)
    flipped = meta.pool(-y, v)
    np.testing.assert_allclose(flipped["re_est"], -base["re_est"], rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(flipped["tau2"], base["tau2"], rtol=1e-10, atol=1e-14)
    perm = rng.permutation(5)
    permuted = meta.pool(y[:, perm], v[:, perm])
    for col in ("re_est", "hksj_se", "tau2", "fe_est", "q"):
        np.testing.assert_allclose(permuted[col], base[col], rtol=1e-9, atol=1e-14)


# ---------------------------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------------------------


def test_pool_vectorized_equals_per_gene_loop():
    rng = np.random.default_rng(9)
    y, v, _ = simulate(rng, 40, 5, tau2=0.15, v_low=0.02, v_high=0.4)
    y[rng.uniform(size=y.shape) < 0.2] = np.nan  # some genes untested at some sites
    v[np.isnan(y)] = np.nan
    # eb=False: FE, REML and HKSJ are computed gene by gene, so pooling 40 genes at once must
    # equal 40 one-gene calls. V2 and V3 fit a prior across genes, so for them a one-gene call
    # is a different analysis by design; their own tests are below.
    whole = meta.pool(y, v, eb=False)
    assert list(whole.columns) == meta.CLASSIC_COLUMNS
    for g in range(len(y)):
        one = meta.pool(y[g], v[g], eb=False)
        for col in ("k", "fe_est", "fe_se", "re_est", "hksj_se", "hksj_ci_low", "re_p", "tau2",
                    "i2", "q"):
            a, b = whole[col].iloc[g], one[col].iloc[0]
            assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b, rel=1e-9, abs=1e-14), (
                g, col)


def test_pool_columns_min_k_and_bh_scope():
    rng = np.random.default_rng(10)
    y, v, _ = simulate(rng, 10, 3, tau2=0.1, v_low=0.05, v_high=0.2)
    y[0, 1:] = np.nan  # k = 1
    v[0, 1:] = np.nan
    out = meta.pool(y, v, min_k=2)
    assert list(out.columns) == meta.POOL_COLUMNS
    assert out["k"].iloc[0] == 1
    assert np.isnan(out["re_est"].iloc[0]) and np.isnan(out["re_q"].iloc[0])
    assert not out["reml_converged"].iloc[0]
    # BH runs over the 9 usable genes only.
    np.testing.assert_allclose(out["re_q"].iloc[1:], meta.bh_qvalues(out["re_p"].iloc[1:]))


# run_methods (WS-08) keeps k < min_k genes as all-NaN rows, and eb_tau2's restricted
# likelihood then takes log(0) for them before marking them NaN. Harmless; pool passes only the
# usable rows, so it never hits this.
@pytest.mark.filterwarnings("ignore:divide by zero encountered in log:RuntimeWarning")
def test_pool_eb_columns_equal_the_methods_called_directly():
    """meta.pool's V2/V3 columns are exactly moderated_fe / eb_tau2 on the k >= min_k genes,
    and exactly what WS-08's comparison (meta_methods.run_methods) computes."""
    from fedkg_spacebio.evaluation import meta_methods
    from fedkg_spacebio.federation import eb_tau2, moderation

    rng = np.random.default_rng(11)
    y, v, _ = simulate(rng, 300, 4, tau2=0.1, v_low=0.02, v_high=0.3)
    y[:5, 1:] = np.nan  # k = 1: must stay out of both prior fits
    v[np.isnan(y)] = np.nan
    out = meta.pool(y, v, min_k=2)
    use = np.arange(5, 300)
    v2 = moderation.moderated_fe(y[use], v[use])
    v3 = eb_tau2.eb_tau2(y[use], v[use])
    for col, ref in (("v2_se", v2["se"]), ("v2_p", v2["p"]), ("v2_df", v2["df"]),
                     ("v2_phi_moderated", v2["scale_moderated"]), ("v3_est", v3["est"]),
                     ("v3_sd", v3["sd"]), ("v3_p", v3["p"]), ("v3_ci_low", v3["ci_low"]),
                     ("v3_p_tau2_zero", v3["p_tau2_zero"])):
        np.testing.assert_array_equal(out[col].to_numpy()[use], ref, err_msg=col)
        assert out[col].iloc[:5].isna().all(), col
    np.testing.assert_array_equal(out["v3_q"].to_numpy()[use], meta.bh_qvalues(v3["p"]))
    pri = out.attrs["eb_priors"]
    assert pri["n_genes"] == 295
    assert pri["v2"]["d0"] == v2["d0"] or pri["v2"]["d0"] == "inf"
    assert pri["v3"]["pi"] == v3["pi"].tolist() and pri["v3"]["em_converged"]

    frames, _ = meta_methods.run_methods(y, v, ("v2", "v3"))
    for m in ("v2", "v3"):
        for a, b in (("p", "p"), ("q", "q"), ("ci_low", "ci_low")):
            np.testing.assert_array_equal(out[f"{m}_{b}"].to_numpy(), frames[m][a].to_numpy(),
                                          err_msg=(m, a))
    np.testing.assert_array_equal(out["v3_est"].to_numpy(), frames["v3"]["est"].to_numpy())


def test_pool_eb_too_few_genes_gives_nan_and_a_reason():
    # Two genes: V3 can fit a (meaningless) prior, V2's scaled-F fit needs >= 2 usable phi
    # values. With one usable gene neither can; FE/REML must still be returned.
    y = np.array([[0.1, 0.3, -0.2], [np.nan, 0.2, np.nan]])
    v = np.where(np.isnan(y), np.nan, 0.05)
    out = meta.pool(y, v)
    assert np.isfinite(out["fe_est"].iloc[0])
    assert out["v2_p"].isna().all()
    assert out.attrs["eb_priors"]["v2"]["fitted"] is False
    assert "at least 2" in out.attrs["eb_priors"]["v2"]["reason"]
