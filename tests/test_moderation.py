"""Tests for the empirical-Bayes moderation module (``fedkg_spacebio.federation.moderation``).

Reference: Smyth GK (2004), Stat Appl Genet Mol Biol 3:Article 3. The equations are in §3
(model, posterior variance), §6.2 (hyperparameter estimator) and the Appendix (trigamma
inverse), read from the PDF on 2026-11-13. R/limma isn't installed here, so there are no
hard-coded limma outputs: the tests check the paper's equations computed independently (with
scipy root finding) and recovery of known (d₀, s₀²) from simulated scaled-F data.

All data are synthetic.
"""

import warnings

import numpy as np
import pytest
from scipy import optimize, special, stats
from statsmodels.stats.meta_analysis import combine_effects

from fedkg_spacebio.federation import meta, moderation


def scaled_f_sample(rng, n, d, d0, s02):
    """s²_g with σ²_g ~ d₀s₀²/χ²(d₀) and s²_g | σ²_g ~ σ²_g χ²(d_g)/d_g (Smyth 2004 §3)."""
    d = np.broadcast_to(np.asarray(d, dtype=float), (n,))
    sigma2 = d0 * s02 / rng.chisquare(d0, size=n)
    return sigma2 * rng.chisquare(d) / d, d


# ---------------------------------------------------------------------------------------------
# trigamma inverse and the prior fit
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("x", [1e-8, 1e-3, 0.05, 0.5, 1.0, 3.0, 50.0, 1e4, 1e8])
def test_trigamma_inverse_round_trip(x):
    y = moderation.trigamma_inverse(x)[0]
    assert special.polygamma(1, y) == pytest.approx(x, rel=1e-6)


def test_trigamma_inverse_matches_root_finding():
    xs = np.array([0.01, 0.2, 0.935, 1.645, 7.0])
    got = moderation.trigamma_inverse(xs)
    want = [optimize.brentq(lambda y, x=x: special.polygamma(1, y) - x, 1e-9, 1e9) for x in xs]
    np.testing.assert_allclose(got, want, rtol=1e-8)


def test_trigamma_inverse_nonpositive_is_nan():
    assert np.all(np.isnan(moderation.trigamma_inverse([0.0, -1.0])))


def test_fit_f_dist_matches_section_6_2_equations():
    """The estimator, written out from §6.2 by hand, gives the same (d₀, s₀²)."""
    rng = np.random.default_rng(1)
    x, d = scaled_f_sample(rng, 5000, rng.choice([2.0, 3.0], size=5000), d0=6.0, s02=1.4)
    e = np.log(x) - special.digamma(d / 2) + np.log(d / 2)
    rhs = np.mean((e - e.mean()) ** 2 * len(e) / (len(e) - 1) - special.polygamma(1, d / 2))
    d0 = 2 * optimize.brentq(lambda y: special.polygamma(1, y) - rhs, 1e-9, 1e9)
    s02 = np.exp(e.mean() + special.digamma(d0 / 2) - np.log(d0 / 2))
    got = moderation.fit_f_dist(x, d)
    assert got[0] == pytest.approx(d0, rel=1e-7)
    assert got[1] == pytest.approx(s02, rel=1e-7)


@pytest.mark.parametrize("d0_true,s02_true", [(4.0, 1.5), (10.0, 0.6), (25.0, 1.35)])
def test_fit_f_dist_recovers_known_prior_equal_df(d0_true, s02_true):
    rng = np.random.default_rng(int(d0_true * 10))
    x, d = scaled_f_sample(rng, 200_000, 3.0, d0_true, s02_true)
    d0, s02 = moderation.fit_f_dist(x, d)
    assert d0 == pytest.approx(d0_true, rel=0.15)
    assert s02 == pytest.approx(s02_true, rel=0.03)


def test_fit_f_dist_recovers_known_prior_mixed_df():
    """d_g differs by gene (k = 3 and 4 mixed, as in the ≥3-site sensitivity set)."""
    rng = np.random.default_rng(7)
    d = rng.choice([2.0, 3.0], size=200_000, p=[0.1, 0.9])
    x, d = scaled_f_sample(rng, d.size, d, d0=6.0, s02=1.5)
    d0, s02 = moderation.fit_f_dist(x, d)
    assert d0 == pytest.approx(6.0, rel=0.15)
    assert s02 == pytest.approx(1.5, rel=0.03)


def test_fit_f_dist_common_variance_gives_infinite_d0():
    """Pure χ² sampling around one variance: no excess scatter, so d₀ = ∞ (or very large)."""
    rng = np.random.default_rng(3)
    x = 2.0 * rng.chisquare(3, size=200_000) / 3
    d0, s02 = moderation.fit_f_dist(x, 3)
    assert d0 > 200
    assert s02 == pytest.approx(2.0, rel=0.02)


def test_fit_f_dist_all_equal_is_infinite_and_finite_outputs():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d0, s02 = moderation.fit_f_dist(np.full(100, 0.8), 3)
        assert np.isinf(d0) and s02 == pytest.approx(0.8)
        res = moderation.moderated_t(np.linspace(-1, 1, 100), np.full(100, 0.1),
                                     np.full(100, 0.8), np.full(100, 3.0))
    assert np.isinf(res["d0"])
    for key in ("se", "t", "p", "ci_low", "ci_high"):
        assert np.all(np.isfinite(res[key]))
    # d₀ = ∞: every gene gets s₀², and the reference distribution is normal.
    np.testing.assert_allclose(res["se"], np.sqrt(0.8 * 0.1))
    np.testing.assert_allclose(res["p"], 2 * stats.norm.sf(np.abs(res["t"])))


def test_fit_f_dist_ignores_unusable_entries_and_floors_zeros():
    rng = np.random.default_rng(4)
    x, d = scaled_f_sample(rng, 1000, 3.0, 8.0, 1.0)
    clean = moderation.fit_f_dist(x, d)
    dirty = moderation.fit_f_dist(np.r_[x, np.nan, np.inf, 1.0], np.r_[d, 3.0, 3.0, 0.0])
    assert dirty == pytest.approx(clean)
    with_zero = moderation.fit_f_dist(np.r_[x, 0.0], np.r_[d, 3.0])
    assert np.all(np.isfinite(with_zero))
    with pytest.raises(ValueError):
        moderation.fit_f_dist([1.0], [3.0])


# ---------------------------------------------------------------------------------------------
# squeeze_var and moderated_t: the limits
# ---------------------------------------------------------------------------------------------


def test_squeeze_var_formula_and_limits():
    x, d = np.array([0.2, 1.0, 5.0]), np.array([3.0, 3.0, 2.0])
    np.testing.assert_allclose(moderation.squeeze_var(x, d, 6.0, 1.5),
                               (6.0 * 1.5 + d * x) / (6.0 + d))
    np.testing.assert_array_equal(moderation.squeeze_var(x, d, 0.0, 1.5), x)
    np.testing.assert_array_equal(moderation.squeeze_var(x, d, np.inf, 1.5), np.full(3, 1.5))
    # A gene with no df of its own gets the prior value.
    assert moderation.squeeze_var(np.array([9.0]), np.array([0.0]), 6.0, 1.5)[0] == 1.5


def _sim_genes(rng, n, k, tau2=0.05, mu_sd=0.3):
    v = rng.uniform(0.01, 0.2, size=(n, k))
    mu = rng.normal(0, mu_sd, size=n)
    y = mu[:, None] + rng.normal(0, np.sqrt(tau2), size=(n, k)) + rng.normal(0, np.sqrt(v))
    return y, v


def test_d0_zero_reduces_to_unmoderated_hksj():
    rng = np.random.default_rng(10)
    y, v = _sim_genes(rng, 300, 4)
    tau2, _ = meta.tau2_reml(y, v)
    plain = meta.random_effects_hksj(y, v, tau2, truncate=False)
    mod = moderation.moderated_hksj(y, v, tau2, truncate=False, d0=0.0, s02=1.0)
    np.testing.assert_allclose(mod["se"], plain["se"], rtol=1e-12)
    np.testing.assert_allclose(mod["p"], plain["p"], rtol=1e-10)
    np.testing.assert_allclose(mod["ci_low"], plain["ci_low"], rtol=1e-10)
    # And truncated, against meta's truncated HKSJ.
    plain_t = meta.random_effects_hksj(y, v, tau2, truncate=True)
    mod_t = moderation.moderated_hksj(y, v, tau2, truncate=True, d0=0.0, s02=1.0)
    np.testing.assert_allclose(mod_t["p"], plain_t["p"], rtol=1e-10)


def test_d0_zero_v2_is_unrestricted_wls():
    """V2 with d₀ = 0: FE estimate, variance (Q/(k − 1))/Σw, t with k − 1 df."""
    rng = np.random.default_rng(11)
    y, v = _sim_genes(rng, 200, 3)
    fe = meta.fixed_effect(y, v)
    mod = moderation.moderated_fe(y, v, d0=0.0, s02=1.0)
    se = np.sqrt(fe["q"] / 2 * fe["se"] ** 2)
    np.testing.assert_allclose(mod["est"], fe["est"])
    np.testing.assert_allclose(mod["se"], se, rtol=1e-12)
    np.testing.assert_allclose(mod["p"], 2 * stats.t.sf(np.abs(fe["est"] / se), 2), rtol=1e-10)
    # Also equals HKSJ with τ² = 0 plugged in.
    hk0 = meta.random_effects_hksj(y, v, np.zeros(len(y)), truncate=False)
    np.testing.assert_allclose(mod["p"], hk0["p"], rtol=1e-10)


def test_small_d0_approaches_unmoderated():
    rng = np.random.default_rng(12)
    y, v = _sim_genes(rng, 200, 4)
    a = moderation.moderated_fe(y, v, d0=1e-9, s02=1.3)
    b = moderation.moderated_fe(y, v, d0=0.0, s02=1.3)
    np.testing.assert_allclose(a["p"], b["p"], rtol=1e-6)


def test_d0_infinite_gives_common_scale_and_normal_reference():
    rng = np.random.default_rng(13)
    y, v = _sim_genes(rng, 200, 4)
    fe = meta.fixed_effect(y, v)
    mod = moderation.moderated_fe(y, v, d0=np.inf, s02=1.7)
    np.testing.assert_allclose(mod["scale_moderated"], 1.7)
    np.testing.assert_allclose(mod["se"], np.sqrt(1.7) * fe["se"], rtol=1e-12)
    np.testing.assert_allclose(mod["p"], 2 * stats.norm.sf(np.abs(mod["t"])), rtol=1e-10)


def test_moderated_df_is_d_plus_d0():
    rng = np.random.default_rng(14)
    y, v = _sim_genes(rng, 50, 4)
    mod = moderation.moderated_fe(y, v, d0=6.5, s02=1.2)
    np.testing.assert_allclose(mod["df"], 3 + 6.5)


def test_truncation_applies_only_when_moderated_scale_below_one():
    """The case where max(1, ·) binds: a prior with s₀² < 1 (sites scatter less than their SEs
    predict). It doesn't bind on the real data (s₀² > 1), but the code path must be right."""
    est, uv = np.array([0.5, 0.5]), np.array([0.04, 0.04])
    scale, df = np.array([0.1, 3.0]), np.array([3.0, 3.0])
    off = moderation.moderated_t(est, uv, scale, df, d0=10.0, s02=0.5)
    on = moderation.moderated_t(est, uv, scale, df, d0=10.0, s02=0.5, truncate=True)
    m = (10 * 0.5 + 3 * scale) / 13
    np.testing.assert_allclose(off["scale_moderated"], m)
    assert m[0] < 1 < m[1]
    assert on["se"][0] == pytest.approx(np.sqrt(uv[0]))          # truncated to 1
    assert on["se"][1] == pytest.approx(off["se"][1])            # untouched
    assert on["truncation_applied"] == 1 and off["truncation_applied"] == 0


def test_genes_with_fewer_than_two_sites_are_nan_and_skip_the_fit():
    rng = np.random.default_rng(15)
    y, v = _sim_genes(rng, 500, 4)
    y[0, 1:] = np.nan                                            # k = 1
    full = moderation.moderated_fe(y, v)
    assert np.isnan(full["p"][0]) and np.isnan(full["se"][0])
    rest = moderation.moderated_fe(y[1:], v[1:])
    assert full["d0"] == pytest.approx(rest["d0"])
    np.testing.assert_allclose(full["p"][1:], rest["p"])


def test_mixed_k_uses_per_gene_df():
    rng = np.random.default_rng(16)
    y, v = _sim_genes(rng, 400, 4)
    y[:100, 3] = np.nan                                          # 100 genes with k = 3
    mod = moderation.moderated_fe(y, v, d0=5.0, s02=1.2)
    np.testing.assert_allclose(mod["df"][:100], 2 + 5.0)
    np.testing.assert_allclose(mod["df"][100:], 3 + 5.0)


# ---------------------------------------------------------------------------------------------
# Behaviour that motivates the method choice
# ---------------------------------------------------------------------------------------------


def test_v2_null_p_values_are_calibrated_under_its_own_model():
    """Var(y_i) = φ_g v_i with φ_g from a scaled inverse-χ² prior and μ = 0 for every gene: the
    moderated test should reject at about its nominal rate. The unmoderated FE test, which
    ignores φ, should reject far too often."""
    rng = np.random.default_rng(20)
    G, k = 40_000, 4
    phi = 5.0 * 1.4 / rng.chisquare(5.0, size=G)
    v = rng.uniform(0.01, 0.2, size=(G, k))
    y = rng.normal(0, np.sqrt(phi[:, None] * v))
    mod = moderation.moderated_fe(y, v)
    assert mod["d0"] == pytest.approx(5.0, rel=0.25)
    assert np.mean(mod["p"] < 0.05) == pytest.approx(0.05, abs=0.006)
    assert np.mean(meta.fixed_effect(y, v)["p"] < 0.05) > 0.10


def test_paule_mandel_makes_the_hksj_multiplier_one():
    """Röver, Knapp & Friede (2015, p. 3): PM τ² is defined by setting the HKSJ q to 1, so q = 1
    wherever τ̂²_PM > 0. statsmodels' "iterated" estimator is Paule–Mandel. This is why q is a
    poor candidate for moderation (V1)."""
    rng = np.random.default_rng(21)
    y, v = _sim_genes(rng, 60, 4, tau2=0.1)
    tau2 = np.array([combine_effects(y[g], v[g], method_re="iterated").tau2 for g in range(60)])
    q = meta.random_effects_hksj(y, v, tau2, truncate=False)["multiplier"]
    pos = tau2 > 1e-8
    assert pos.sum() > 20
    np.testing.assert_allclose(q[pos], 1.0, atol=1e-4)
    assert np.all(q[~pos] <= 1.0 + 1e-6)


def test_v1_collapses_to_re_wald_when_q_has_no_excess_scatter():
    """If every gene's q is 1, the prior fit gives d₀ = ∞ and s₀² = 1, and V1 is the RE Wald z."""
    rng = np.random.default_rng(22)
    y, v = _sim_genes(rng, 300, 4)
    tau2, _ = meta.tau2_reml(y, v)
    est = meta.random_effects_hksj(y, v, tau2, truncate=False)["est"]
    uv = 1.0 / moderation.re_weight_sum(y, v, tau2)
    res = moderation.moderated_t(est, uv, np.ones(300), np.full(300, 3.0))
    assert np.isinf(res["d0"]) and res["s02"] == pytest.approx(1.0)
    np.testing.assert_allclose(res["p"], 2 * stats.norm.sf(np.abs(est) / np.sqrt(uv)),
                               rtol=1e-10)
