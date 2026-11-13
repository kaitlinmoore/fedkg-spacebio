"""Tests for V3, the empirical-Bayes τ² method (``fedkg_spacebio.federation.eb_tau2``).

The anchors are exact reductions: with a point-mass prior at τ² = 0 the method must give the
fixed-effect p-value, and at τ² = c the random-effects Wald p-value at c. The rest checks the
likelihood, the EM fit and the posterior summaries on synthetic data.
"""

import numpy as np
import pytest
from scipy import optimize, stats

from fedkg_spacebio.federation import eb_tau2, meta, moderation


def simulate(rng, G, k, tau2, mu_sd=0.0):
    tau2 = np.broadcast_to(np.asarray(tau2, dtype=float), (G,))
    v = rng.uniform(0.01, 0.06, size=(G, k))
    mu = rng.normal(0, mu_sd, size=G) if mu_sd > 0 else np.zeros(G)
    theta = mu[:, None] + rng.normal(size=(G, k)) * np.sqrt(tau2)[:, None]
    return theta + rng.normal(size=(G, k)) * np.sqrt(v), v, mu


def point_mass(grid, value):
    pi = np.zeros(len(grid))
    pi[int(np.argmin(np.abs(grid - value)))] = 1.0
    return pi


def test_point_mass_at_zero_gives_fixed_effect():
    rng = np.random.default_rng(1)
    y, v, _ = simulate(rng, 200, 4, 0.02, mu_sd=0.2)
    grid = np.r_[0.0, np.geomspace(1e-4, 1.0, 20)]
    res = eb_tau2.eb_tau2(y, v, grid=grid, pi=point_mass(grid, 0.0))
    fe = meta.fixed_effect(y, v)
    np.testing.assert_allclose(res["est"], fe["est"], rtol=1e-12)
    np.testing.assert_allclose(res["p"], fe["p"], rtol=1e-9)
    np.testing.assert_allclose(res["ci_low"], fe["est"] - 1.959963984540054 * fe["se"],
                               rtol=1e-7)


def test_point_mass_at_c_gives_random_effects_wald():
    rng = np.random.default_rng(2)
    y, v, _ = simulate(rng, 200, 3, 0.05, mu_sd=0.3)
    grid = np.r_[0.0, 0.01, 0.05, 0.2]
    res = eb_tau2.eb_tau2(y, v, grid=grid, pi=point_mass(grid, 0.05))
    sw = moderation.re_weight_sum(y, v, np.full(200, 0.05))
    est = meta.random_effects_hksj(y, v, np.full(200, 0.05), truncate=False)["est"]
    np.testing.assert_allclose(res["est"], est, rtol=1e-12)
    np.testing.assert_allclose(res["p"], 2 * stats.norm.sf(np.abs(est) * np.sqrt(sw)),
                               rtol=1e-9)


def test_restricted_loglik_maximizer_matches_reml():
    """The grid value with the highest restricted likelihood is (near) REML's τ̂²."""
    rng = np.random.default_rng(3)
    y, v, _ = simulate(rng, 40, 4, 0.1)
    grid = np.r_[0.0, np.linspace(1e-4, 1.0, 20001)]
    ll, _, _ = eb_tau2.restricted_loglik(y, v, grid)
    tau2, _ = meta.tau2_reml(y, v)
    np.testing.assert_allclose(grid[np.argmax(ll, axis=1)], tau2, atol=1e-4)


def test_restricted_loglik_matches_direct_formula():
    """For one gene: −½[Σlog(v+τ²) + log Σw + Σw(y − μ̂)²], written out by hand."""
    y, v, t = np.array([0.3, -0.1, 0.5]), np.array([0.02, 0.05, 0.03]), 0.07
    w = 1 / (v + t)
    m = (w * y).sum() / w.sum()
    want = -0.5 * (np.log(v + t).sum() + np.log(w.sum()) + (w * (y - m) ** 2).sum())
    ll, mu, sw = eb_tau2.restricted_loglik(y, v, [t])
    assert ll[0, 0] == pytest.approx(want, rel=1e-12)
    assert mu[0, 0] == pytest.approx(m) and sw[0, 0] == pytest.approx(w.sum())


def test_posterior_normalizes_and_k_below_two_is_nan():
    rng = np.random.default_rng(4)
    y, v, _ = simulate(rng, 300, 4, 0.03)
    y[0, 1:] = np.nan
    res = eb_tau2.eb_tau2(y, v, n_grid=30)
    ll, _, _ = eb_tau2.restricted_loglik(y, v, res["grid"])
    post = eb_tau2.grid_posterior(ll, res["pi"])
    np.testing.assert_allclose(post[1:].sum(axis=1), 1.0)
    assert np.all(np.isnan(post[0]))
    for key in ("est", "p", "ci_low", "ci_high"):
        assert np.isnan(res[key][0]) and np.all(np.isfinite(res[key][1:]))
    assert res["pi"].sum() == pytest.approx(1.0)


def test_em_recovers_a_two_point_tau2_distribution():
    """60% of genes with τ² = 0 and 40% with τ² = 0.2: the fitted prior's mass splits the same
    way. NPMLE mass can spread to neighbouring grid points, so compare mass on each side of a
    cut halfway (on the log scale) between the two values."""
    rng = np.random.default_rng(5)
    G = 30_000
    tau2 = np.where(rng.random(G) < 0.6, 0.0, 0.2)
    y, v, _ = simulate(rng, G, 4, tau2)
    res = eb_tau2.eb_tau2(y, v, n_grid=60)
    g, pi = res["grid"], res["pi"]
    assert res["fit"]["converged"]
    assert pi[g < 0.02].sum() == pytest.approx(0.6, abs=0.06)
    assert pi[(g > 0.1) & (g < 0.4)].sum() == pytest.approx(0.4, abs=0.06)


def test_em_never_decreases_the_objective():
    rng = np.random.default_rng(6)
    y, v, _ = simulate(rng, 2000, 3, rng.exponential(0.05, 2000))
    ll, _, _ = eb_tau2.restricted_loglik(y, v, eb_tau2.default_grid(y, v, n=25))
    objs = [eb_tau2.fit_grid_prior(ll, max_iter=n, tol=0.0)[1]["loglik"]
            for n in (1, 2, 5, 20, 100)]
    assert all(b >= a - 1e-9 for a, b in zip(objs, objs[1:], strict=False))


def test_results_stable_under_grid_refinement():
    rng = np.random.default_rng(7)
    y, v, _ = simulate(rng, 5000, 4, rng.exponential(0.03, 5000), mu_sd=0.15)
    a = eb_tau2.eb_tau2(y, v, n_grid=40)
    b = eb_tau2.eb_tau2(y, v, n_grid=120)
    assert np.max(np.abs(np.log10(a["p"]) - np.log10(b["p"]))) < 0.05
    np.testing.assert_allclose(a["ci_low"], b["ci_low"], atol=0.01)


def test_credible_interval_is_the_mixture_quantile():
    rng = np.random.default_rng(8)
    y, v, _ = simulate(rng, 50, 4, 0.05, mu_sd=0.3)
    res = eb_tau2.eb_tau2(y, v, n_grid=30)
    ll, mu, sw = eb_tau2.restricted_loglik(y, v, res["grid"])
    post = eb_tau2.grid_posterior(ll, res["pi"])
    sd = 1 / np.sqrt(sw)
    for g in range(50):
        def root(prob, g=g):
            def f(x):
                return (post[g] * stats.norm.cdf((x - mu[g]) / sd[g])).sum() - prob
            return optimize.brentq(f, -20, 20, xtol=1e-12)
        lo, hi, med = root(0.025), root(0.975), root(0.5)
        assert res["ci_low"][g] == pytest.approx(lo, abs=1e-8)
        assert res["ci_high"][g] == pytest.approx(hi, abs=1e-8)
        assert lo < med < hi


def test_tail_probabilities_are_complementary_and_precise():
    rng = np.random.default_rng(9)
    y, v, _ = simulate(rng, 500, 4, 0.01, mu_sd=1.0)
    res = eb_tau2.eb_tau2(y, v, n_grid=30)
    np.testing.assert_allclose(res["p_pos"] + res["p_neg"], 1.0, atol=1e-12)
    assert np.min(res["p"]) < 1e-12 and np.all(res["p"] > 0)       # no rounding to 0


def test_null_p_values_roughly_calibrated_under_its_own_model():
    """μ = 0 for every gene, τ² drawn from a spread-out distribution: the rejection rate at 0.05
    is near nominal once the prior is learned. Not exact theory, so the tolerance is loose."""
    rng = np.random.default_rng(10)
    G = 30_000
    tau2 = np.where(rng.random(G) < 0.3, 0.0, rng.exponential(0.05, G))
    y, v, _ = simulate(rng, G, 4, tau2)
    res = eb_tau2.eb_tau2(y, v)
    assert np.mean(res["p"] < 0.05) == pytest.approx(0.05, abs=0.01)
    # REML plug-in Wald (the method V1 collapses to) is anti-conservative here.
    t2, _ = meta.tau2_reml(y, v)
    re = meta.random_effects_hksj(y, v, t2, truncate=False)
    wald_p = 2 * stats.norm.sf(np.abs(re["est"]) * np.sqrt(moderation.re_weight_sum(y, v, t2)))
    assert np.mean(wald_p < 0.05) > np.mean(res["p"] < 0.05) + 0.01
