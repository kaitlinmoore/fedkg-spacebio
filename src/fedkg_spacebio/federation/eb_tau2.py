"""V3: empirical-Bayes between-mission variance τ², shared across genes (phase 3a, WS-08).

The problem it addresses
------------------------
Random effects asks "what would a *new* mission show?" The answer depends on τ²_g, the
between-mission variance of gene g's true effect. With k = 3–4 missions, τ̂²_g is barely
estimable: REML returns exactly 0 for 42–47% of genes, which then get fixed-effect-sized
intervals, and HKSJ's t with 2–3 df makes the remaining genes untestable.

The idea
--------
Thousands of genes each give a little information about τ². Pool it:

1. **A grid of candidate τ² values** τ²_1 = 0 < τ²_2 < … < τ²_J (``default_grid``).
2. **Each gene's likelihood at each grid value** (``restricted_loglik``). Integrating μ out
   under a flat prior gives the restricted (REML) likelihood:

       log L_g(τ²) = −½ [ Σ log(v_i + τ²) + log Σ w_i + Σ w_i (y_i − μ̂)² ] + const,
       w_i = 1/(v_i + τ²),   μ̂ = Σ w_i y_i / Σ w_i

   REML's τ̂²_g is the grid value that maximizes this for one gene. Here every value is kept.
3. **A prior π over the grid, fitted to all genes at once** (``fit_grid_prior``). It maximizes
   Σ_g log Σ_j π_j L_g(τ²_j), the likelihood of all genes' data if each gene's τ² were drawn
   from π. That's a nonparametric maximum likelihood estimate (NPMLE) of the τ² distribution,
   computed by EM. No shape is imposed: a point mass at 0 is allowed, not forced.
4. **Each gene's posterior over the grid**: p_gj ∝ π_j L_g(τ²_j). A gene's own data move it
   away from the prior only as far as they can. With k = 4 that isn't far, which is the point.
5. **μ given the data**, averaging over τ²: given τ²_j, μ | y ~ N(μ̂_j, 1/Σ w_ij) (flat prior on
   μ), so μ | y is a mixture of normals with weights p_gj. Report:
   - the posterior mean of μ and an equal-tailed 95% credible interval;
   - p_EB = 2·min{P(μ > 0 | y), P(μ < 0 | y)}, the two-sided tail probability. BH runs on
     p_EB like any p-value.

   With a point-mass prior at τ² = c, p_EB is exactly the random-effects Wald p at c; at
   c = 0, the fixed-effect p. (The tests check both.) So p_EB is a Wald p averaged over how
   uncertain τ² is, with that uncertainty informed by every gene.

What it answers: "is the mean effect across missions, the one a new mission would be drawn
around, different from 0?" The question is the same as RE-HKSJ's. The difference is how τ²'s
uncertainty is handled.

Assumptions and known weaknesses (the simulation measures their cost)
---------------------------------------------------------------------
- τ²_g is exchangeable across genes and independent of μ_g and of the SEs. Genes with real
  effects are probably more heterogeneous, so their τ² is shrunk down, toward the bulk of
  near-null genes. The genes where that matters most are μ = 0 with τ² > 0.
- v_i (each site's SE²) is treated as known, as in every method here.
- The prior is estimated, then treated as known (as in limma: it uses ~13,000 genes).
- No exact small-sample theory: whether BH on p_EB controls FDR is an empirical question.

Federation: runs at the server on the per-site (log2FC, SE) summaries it already receives. No
new data leaves a site; like REML/HKSJ it needs per-site estimates, not aggregated sums.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from fedkg_spacebio.federation import meta


def default_grid(y, v, n: int = 60) -> np.ndarray:
    """τ² grid: 0 plus n − 1 log-spaced values.

    The range is set from the data's scale so the same code works for any effect-size unit:
    - lower end: 1e-3 × the median sampling variance (a τ² this small is indistinguishable
      from 0 next to any site's v);
    - upper end: 10 × the larger of the median sampling variance and the 99.9th percentile of
      genes' unweighted between-site variance of y (beyond that, the likelihood is negligible
      for every gene).
    The tests check that results don't depend on the grid's fineness.
    """
    Y, V, present = meta.as_matrix(y, v)
    k = meta.n_sites(present)
    medv = float(np.median(V[present]))
    with np.errstate(invalid="ignore", divide="ignore"):
        ybar = np.where(k > 0, Y.sum(axis=1) / k, np.nan)
        s2 = np.where(present, (Y - ybar[:, None]) ** 2, 0.0).sum(axis=1) / (k - 1)
    s2 = s2[k >= 2]
    hi = 10.0 * max(medv, float(np.quantile(s2, 0.999)) if s2.size else medv)
    lo = 1e-3 * medv
    return np.r_[0.0, np.geomspace(lo, hi, n - 1)]


def restricted_loglik(y, v, grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per gene and grid value: (loglik, mu_hat, sum_w), each of shape (G, J).

    loglik is the restricted log-likelihood above, up to a constant that is the same for every
    grid value (so it cancels in the posterior). Genes with k < 2 get NaN rows: one site
    carries no information about τ².
    """
    Y, V, present = meta.as_matrix(y, v)
    k = meta.n_sites(present)
    grid = np.asarray(grid, dtype=np.float64)
    G, J = Y.shape[0], grid.size
    ll, mu, sw = np.empty((G, J)), np.empty((G, J)), np.empty((G, J))
    for j, t in enumerate(grid):
        tot = np.where(present, V + t, 1.0)
        w = np.where(present, 1.0 / tot, 0.0)
        s = w.sum(axis=1)
        m = (w * Y).sum(axis=1) / np.where(s > 0, s, 1.0)
        r2 = np.where(present, (Y - m[:, None]) ** 2, 0.0)
        ll[:, j] = -0.5 * (np.where(present, np.log(tot), 0.0).sum(axis=1) + np.log(s)
                           + (w * r2).sum(axis=1))
        mu[:, j], sw[:, j] = m, s
    bad = k < 2
    ll[bad], mu[bad], sw[bad] = np.nan, np.nan, np.nan
    return ll, mu, sw


def fit_grid_prior(loglik, *, max_iter: int = 20_000, tol: float = 1e-8,
                   init=None) -> tuple[np.ndarray, dict]:
    """NPMLE of the τ² distribution on the grid, by EM.

    Maximizes Σ_g log Σ_j π_j L_gj over the probability simplex. The EM step is

        p_gj = π_j L_gj / Σ_l π_l L_gl        # each gene's posterior over the grid
        π_j  ← mean_g p_gj                     # new prior = average posterior

    Each step can't decrease the objective, and the objective is concave in π, so EM reaches
    the global maximum. It may need many cheap steps. Rows with NaN (k < 2) are skipped.
    Stops when the objective's increase per step is < tol × G, or at max_iter.

    Returns (pi, info) with info = {"iterations", "converged", "loglik"}.
    """
    ll = np.asarray(loglik, dtype=np.float64)
    ll = ll[np.all(np.isfinite(ll), axis=1)]
    G, J = ll.shape
    if G == 0:
        raise ValueError("no gene with k >= 2 to fit the prior")
    L = np.exp(ll - ll.max(axis=1, keepdims=True))         # rescaled per gene; cancels in p_gj
    pi = np.full(J, 1.0 / J) if init is None else np.asarray(init, dtype=np.float64)
    obj_old, converged, iterations = -np.inf, False, 0
    for iterations in range(1, max_iter + 1):  # noqa: B007 (read after the loop)
        mix = L @ pi                                        # Σ_j π_j L_gj, per gene
        obj = float(np.log(mix).sum())
        # mean_g p_gj = π_j · mean_g (L_gj / mix_g), written as one matrix-vector product.
        pi = pi * (L.T @ (1.0 / mix)) / G
        pi /= pi.sum()
        if obj - obj_old < tol * G:
            converged = True
            break
        obj_old = obj
    total = float(np.log(L @ pi).sum() + ll.max(axis=1).sum())
    return pi, {"iterations": iterations, "converged": converged, "loglik": total}


def grid_posterior(loglik, pi) -> np.ndarray:
    """p_gj ∝ π_j L_gj, normalized per gene. NaN rows stay NaN."""
    ll = np.asarray(loglik, dtype=np.float64)
    with np.errstate(divide="ignore"):
        a = ll + np.log(np.asarray(pi, dtype=np.float64))[None, :]
    a = a - np.nanmax(np.where(np.isfinite(a), a, -np.inf), axis=1, keepdims=True)
    p = np.exp(a)
    return p / p.sum(axis=1, keepdims=True)


def _mixture_cdf(x, post, mu, sd):
    return (post * stats.norm.cdf((x[:, None] - mu) / sd)).sum(axis=1)


def _mixture_quantile(prob, post, mu, sd, start, *, iters: int = 60, tol: float = 1e-10):
    """The prob-quantile of each gene's normal mixture, by safeguarded Newton.

    The mixture CDF F is smooth and strictly increasing, so F(x) = prob has one root. Each step
    tries Newton (x − (F(x) − prob)/f(x)) from ``start``, keeping a bracket [lo, hi] that always
    contains the root; a step that leaves the bracket is replaced by bisection. Converges in a
    handful of steps from a normal-approximation start.
    """
    lo = np.nanmin(mu - 12 * sd, axis=1)
    hi = np.nanmax(mu + 12 * sd, axis=1)
    x = np.clip(start, lo, hi)
    active = np.arange(len(x))                  # genes still moving; converged ones are frozen
    for _ in range(iters):
        a = active
        zz = (x[a, None] - mu[a]) / sd[a]
        F = (post[a] * stats.norm.cdf(zz)).sum(axis=1)
        f = (post[a] * stats.norm.pdf(zz) / sd[a]).sum(axis=1)
        below = F < prob
        lo[a] = np.where(below, x[a], lo[a])
        hi[a] = np.where(below, hi[a], x[a])
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            newton = x[a] - (F - prob) / f      # f ≈ 0 far in a tail: the bracket catches it
        x_new = np.where((newton > lo[a]) & (newton < hi[a]), newton, 0.5 * (lo[a] + hi[a]))
        moved = np.abs(x_new - x[a]) >= tol
        x[a] = x_new
        active = a[moved]
        if active.size == 0:
            break
    return x


def posterior_mu(post, mu, sw, *, level: float = 0.95) -> dict[str, np.ndarray]:
    """Summaries of μ | y, the normal mixture Σ_j p_gj N(μ̂_gj, 1/Σw_gj).

    Tail probabilities use the normal survival function directly on each side, so a tiny
    P(μ < 0) isn't lost to rounding in 1 − P(μ > 0).
    """
    sd = 1.0 / np.sqrt(sw)
    z = mu / sd
    p_pos = (post * stats.norm.sf(-z)).sum(axis=1)          # P(μ > 0 | y)
    p_neg = (post * stats.norm.sf(z)).sum(axis=1)           # P(μ < 0 | y)
    mean = (post * mu).sum(axis=1)
    var = (post * (sd**2 + mu**2)).sum(axis=1) - mean**2
    a = (1.0 - level) / 2.0
    zc = stats.norm.ppf(1.0 - a)
    psd = np.sqrt(np.maximum(var, 0.0))
    ok = np.all(np.isfinite(post), axis=1)
    lo, hi = np.full(len(post), np.nan), np.full(len(post), np.nan)
    if ok.any():
        lo[ok] = _mixture_quantile(a, post[ok], mu[ok], sd[ok], mean[ok] - zc * psd[ok])
        hi[ok] = _mixture_quantile(1.0 - a, post[ok], mu[ok], sd[ok], mean[ok] + zc * psd[ok])
    return {"est": mean, "sd": np.sqrt(np.maximum(var, 0.0)),
            "p": np.minimum(1.0, 2.0 * np.minimum(p_pos, p_neg)),
            "p_pos": p_pos, "p_neg": p_neg, "ci_low": lo, "ci_high": hi}


def eb_tau2(y, v, *, grid=None, pi=None, level: float = 0.95, n_grid: int = 60,
            max_iter: int = 20_000) -> dict:
    """Run V3 end to end: grid, likelihoods, prior (fitted unless ``pi`` is given), posterior.

    Returns a dict of (G,) arrays: ``k``, ``est`` (posterior mean of μ), ``sd``, ``p`` (p_EB),
    ``ci_low``, ``ci_high``, ``tau2_post_mean``, ``p_tau2_zero`` (posterior mass at τ² = 0);
    plus ``grid``, ``pi`` and ``fit`` (EM info, or None if ``pi`` was given). Genes with k < 2
    get NaN.
    """
    if grid is None:
        grid = default_grid(y, v, n=n_grid)
    grid = np.asarray(grid, dtype=np.float64)
    ll, mu, sw = restricted_loglik(y, v, grid)
    info = None
    if pi is None:
        pi, info = fit_grid_prior(ll, max_iter=max_iter)
    post = grid_posterior(ll, pi)
    out = posterior_mu(post, mu, sw, level=level)
    out["k"] = meta.n_sites(meta.as_matrix(y, v)[2])
    out["tau2_post_mean"] = (post * grid[None, :]).sum(axis=1)
    out["p_tau2_zero"] = post[:, 0] if grid[0] == 0 else np.zeros(len(post))
    out.update(grid=grid, pi=np.asarray(pi), fit=info)
    return out
