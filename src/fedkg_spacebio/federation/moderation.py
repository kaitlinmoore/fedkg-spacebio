"""Empirical-Bayes moderation of per-gene scale factors (Smyth 2004), for phase 3a pooling.

What this module does
---------------------
Each gene's meta-analysis yields a *scale factor*: a number that says how much more (or less)
the sites scatter around the pooled estimate than their standard errors predict. With 3–4
missions, it's estimated from only k − 1 = 2–3 degrees of freedom, so it's very noisy. A noisy
scale factor makes the test either reckless (when the scale happens to come out small) or
powerless (a t with 2–3 df needs huge |t|).

limma's idea (Smyth 2004) is to stop treating each gene alone. Fit a prior distribution for the
scale factors across *all* genes, then replace each gene's estimate with a weighted average of
its own value and the prior's typical value:

    s̃²_g = (d₀·s₀² + d_g·s²_g) / (d₀ + d_g)

d₀ ("prior degrees of freedom") says how much the genes' true scale factors agree with each
other. d₀ = 0: they could be anything, so each gene keeps its own value (no moderation).
d₀ = ∞: they're all the same, so every gene gets s₀². In between, a gene's d_g df of its own
evidence are combined with d₀ df of "what genes are typically like," and the test uses a t with
d_g + d₀ df. The test gains df because it borrows information, not because it assumes anything
about the gene's effect.

Where moderation runs (federation)
----------------------------------
At the server, on the per-site (log2FC, SE) summaries it already receives for REML/HKSJ. No
new data leaves a site. Like REML and HKSJ, it needs each site's estimate separately, so it
can't run on securely aggregated sums.

Two scale factors (WS-08)
-------------------------
- **V2, multiplicative heterogeneity** (``moderated_fe``): φ_g = Q_g / (k − 1), Cochran's Q
  with fixed-effect weights. Under the model Var(y_i) = φ_g·v_i, (k − 1)·φ̂_g/φ_g ~ χ²(k − 1)
  exactly. That's precisely limma's assumption for s²_g, so the theory holds as written.
- **V1, the HKSJ multiplier** (``moderated_hksj``): q_g with REML random-effects weights. Kept
  only to document why it fails here. REML's τ̂² is chosen to fit each gene's scatter, so q_g
  sits near 1 whenever τ̂² > 0 (for the closely related Paule–Mandel estimator it is exactly 1;
  Röver, Knapp & Friede 2015, p. 3). q_g then varies *less* than a χ² variance estimate would,
  the prior fit returns d₀ = ∞, and every gene gets the same q̃. The test collapses to the
  random-effects Wald z with the plug-in τ̂², which HKSJ was designed to replace.

References: Smyth GK (2004) Stat Appl Genet Mol Biol 3:Article 3, §3 (model), §6.2
(hyperparameters), Appendix (trigamma inverse). Röver C, Knapp G, Friede T (2015) BMC Med Res
Methodol 15:99.
"""

from __future__ import annotations

import numpy as np
from scipy import special, stats

from fedkg_spacebio.federation import meta

# ---------------------------------------------------------------------------------------------
# The prior fit: Smyth (2004) §6.2 and Appendix
# ---------------------------------------------------------------------------------------------


def trigamma_inverse(x) -> np.ndarray:
    """Solve ψ′(y) = x for y > 0, elementwise (Smyth 2004, Appendix).

    ψ′ (trigamma) is strictly decreasing on y > 0, so the solution is unique for every x > 0.
    Smyth's Newton iteration works on f(y) = 1/ψ′(y), which is nearly linear and convex, so
    Newton's method converges monotonically from y₀ = 0.5 + 1/x:

        δ = ψ′(y)·(1 − ψ′(y)/x) / ψ″(y),   y ← y + δ,   stop when −δ/y < 1e-8

    For extreme x the asymptotes are used directly, as in the paper: y = 1/√x when x > 1e7 and
    y = 1/x when x < 1e-6. x ≤ 0 has no solution and returns NaN.
    """
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    y = np.full(x.shape, np.nan)
    big, small = x > 1e7, (x > 0) & (x < 1e-6)
    y[big] = 1.0 / np.sqrt(x[big])
    y[small] = 1.0 / x[small]
    mid = (x > 0) & ~big & ~small
    if mid.any():
        xm = x[mid]
        ym = 0.5 + 1.0 / xm
        for _ in range(50):
            tri = special.polygamma(1, ym)
            delta = tri * (1.0 - tri / xm) / special.polygamma(2, ym)
            ym = ym + delta
            if np.max(-delta / ym) < 1e-8:
                break
        y[mid] = ym
    return y


def fit_f_dist(x, df) -> tuple[float, float]:
    """Fit the scaled-F prior (d₀, s₀²) to variance estimates x with df degrees of freedom.

    The model (Smyth 2004 §3–4): each gene has a true variance σ²_g, drawn from a scaled
    inverse-χ² prior with d₀ df and scale s₀²; its estimate is s²_g ~ σ²_g·χ²(d_g)/d_g. Then,
    marginally, s²_g ~ s₀²·F(d_g, d₀). Fit by matching the first two moments of log s²_g (§6.2):

        e_g = log s²_g − ψ(d_g/2) + log(d_g/2)        # removes the χ² part's mean
        E(e_g)   = log s₀² − ψ(d₀/2) + log(d₀/2)
        var(e_g) = ψ′(d_g/2) + ψ′(d₀/2)

    So the scatter of e_g *beyond* what χ² sampling alone gives (ψ′(d_g/2)) measures how much
    the true variances differ between genes:

        ψ′(d₀/2) = var(e) − mean(ψ′(d_g/2))           # solved with ``trigamma_inverse``
        s₀²      = exp(ē + ψ(d₀/2) − log(d₀/2))

    Logs are used because log s² has finite moments at any df and is closer to normal. d_g may
    differ by gene (the ≥3-site sensitivity set has k = 3 and 4 mixed); the paper's estimator
    allows it, via mean(ψ′(d_g/2)).

    If the excess scatter is ≤ 0, the estimates vary no more than sampling alone explains: no
    evidence the true variances differ, so d₀ = ∞. Then s₀² is the plain mean of x (as limma's
    current ``fitFDist`` does; the paper uses exp(ē). Both are consistent, and the mean is
    unbiased because E(s²_g) = s₀² when every σ²_g equals s₀²).

    Input handling (as limma's code): entries with non-finite x or df ≤ 0 are ignored. x is
    floored at 1e-5 × median(x) before taking logs, so an exact 0 (e.g. sites that agree to
    machine precision) can't send log x to −∞ and dominate the variance.

    Returns (d0, s02). d0 may be ``np.inf``. Raises ValueError if fewer than 2 usable entries.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    df = np.broadcast_to(np.asarray(df, dtype=np.float64), x.shape).ravel()
    ok = np.isfinite(x) & np.isfinite(df) & (df > 0) & (x >= 0)
    x, df = x[ok], df[ok]
    if x.size < 2:
        raise ValueError("fit_f_dist needs at least 2 usable variance estimates")
    med = np.median(x)
    if med <= 0:
        raise ValueError("median variance estimate is 0; the prior can't be fitted")
    xf = np.maximum(x, 1e-5 * med)
    e = np.log(xf) - special.digamma(df / 2.0) + np.log(df / 2.0)
    emean = e.mean()
    evar = e.var(ddof=1) - special.polygamma(1, df / 2.0).mean()
    if evar <= 0:
        return np.inf, float(x.mean())
    d0 = float(2.0 * trigamma_inverse(evar)[0])
    s02 = float(np.exp(emean + special.digamma(d0 / 2.0) - np.log(d0 / 2.0)))
    return d0, s02


def squeeze_var(x, df, d0: float, s02: float) -> np.ndarray:
    """Posterior (moderated) variance: s̃² = (d₀·s₀² + d·s²) / (d₀ + d)   (Smyth 2004 §3).

    d₀ = 0 returns x unchanged; d₀ = ∞ returns s₀² for every gene. Entries with df = 0 (no
    information of their own) get s₀². NaN in x stays NaN when df > 0.
    """
    x = np.asarray(x, dtype=np.float64)
    df = np.broadcast_to(np.asarray(df, dtype=np.float64), x.shape)
    if np.isinf(d0):
        return np.where(np.isnan(x) & (df > 0), np.nan, s02)
    if d0 == 0:
        return x.copy()
    with np.errstate(invalid="ignore"):
        return np.where(df > 0, (d0 * s02 + df * x) / (d0 + df), s02)


# ---------------------------------------------------------------------------------------------
# The moderated test (shared by V1 and V2)
# ---------------------------------------------------------------------------------------------


def moderated_t(est, unscaled_var, scale, df, *, d0: float | None = None,
                s02: float | None = None, truncate: bool = False,
                level: float = 0.95) -> dict[str, np.ndarray | float]:
    """Moderate a per-gene scale factor and test the estimate with a t on d + d₀ df.

    Inputs, per gene: ``est`` (pooled estimate), ``unscaled_var`` (its variance if the scale
    factor were 1, e.g. 1/Σw), ``scale`` (the gene's own scale-factor estimate, d df) and
    ``df``. Genes with a NaN scale or df < 1 get NaN outputs and don't enter the prior fit.

    If ``d0`` and ``s02`` are given, they're used as the prior (tests, or a prior fitted
    elsewhere); otherwise ``fit_f_dist`` fits them to the usable genes.

        scale~ = squeeze_var(scale, df, d₀, s₀²)
        if truncate:  scale~ = max(1, scale~)       # never narrower than the unscaled SE
        SE  = sqrt(scale~ · unscaled_var)
        t   = est / SE,  df_total = df + d₀  (∞ → normal)
        CI  = est ± t_{df_total, 1−(1−level)/2} · SE

    Returns a dict with arrays ``est``, ``se``, ``t``, ``df``, ``p``, ``ci_low``, ``ci_high``,
    ``scale`` (raw), ``scale_moderated`` (before truncation), the scalars ``d0`` and ``s02``,
    and ``truncation_applied`` (count of genes where max(1, ·) changed the answer).
    """
    est = np.asarray(est, dtype=np.float64)
    uv = np.asarray(unscaled_var, dtype=np.float64)
    s2 = np.asarray(scale, dtype=np.float64)
    d = np.asarray(df, dtype=np.float64)
    ok = np.isfinite(s2) & np.isfinite(d) & (d >= 1) & np.isfinite(est) & np.isfinite(uv)
    if d0 is None or s02 is None:
        d0, s02 = fit_f_dist(s2[ok], d[ok])
    mod = np.full(est.shape, np.nan)
    mod[ok] = squeeze_var(s2[ok], d[ok], d0, s02)
    used = np.maximum(1.0, mod) if truncate else mod
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.sqrt(used * uv)
        t = est / se
        df_total = d + d0
        p = 2.0 * stats.t.sf(np.abs(t), df_total)
        crit = stats.t.ppf(1.0 - (1.0 - level) / 2.0, df_total)
    out = {"est": est.copy(), "se": se, "t": t, "df": df_total, "p": p,
           "ci_low": est - crit * se, "ci_high": est + crit * se,
           "scale": s2.copy(), "scale_moderated": mod}
    for key in ("est", "se", "t", "df", "p", "ci_low", "ci_high", "scale"):
        out[key][~ok] = np.nan
    out["d0"], out["s02"] = float(d0), float(s02)
    out["truncation_applied"] = int(np.sum(ok & (mod < 1.0))) if truncate else 0
    return out


# ---------------------------------------------------------------------------------------------
# V2 and V1
# ---------------------------------------------------------------------------------------------


def moderated_fe(y, v, *, truncate: bool = False, level: float = 0.95,
                 d0: float | None = None, s02: float | None = None) -> dict:
    """V2: fixed-effect estimate with a moderated multiplicative heterogeneity factor.

    Model: y_i = μ + e_i with Var(e_i) = φ·v_i. φ = 1 is fixed effect; φ > 1 means the sites
    scatter more than their SEs predict, whatever the reason (true between-mission differences,
    or SEs that are too small). For one gene:

        w_i = 1/v_i,  μ̂ = Σw_i y_i / Σw_i           # the FE estimate, unchanged
        φ̂  = Q / (k − 1)                            # Cochran's Q over its df; d = k − 1
        φ̃  = (d₀s₀² + d·φ̂) / (d₀ + d)               # moderated across genes
        SE² = φ̃ / Σw_i,   t on d + d₀ df

    With d₀ → 0 this is unrestricted weighted-least-squares meta-analysis: the FE estimate with
    variance (Q/(k − 1))/Σw. That's the HKSJ variance with τ² = 0 plugged in. With d₀ → ∞
    every gene gets the same inflation s₀².

    What it answers: "is there a consistent change across *these* missions?", like FE, but with
    a variance that admits the missions scatter more than their SEs say. It isn't a
    random-effects (new mission) estimate; the weights stay 1/v_i, so a dominant site stays
    dominant.

    Truncation: max(1, φ̃) is available but off by default (WS-08 decision). With the real
    data's priors, φ̃ ≥ d₀s₀²/(d₀ + d) > 1 for every gene, so it would never bind.
    """
    fe = meta.fixed_effect(y, v)
    k = fe["k"].astype(np.float64)
    d = k - 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        phi = np.where(d >= 1, fe["q"] / d, np.nan)
        unscaled = fe["se"] ** 2                    # 1/Σw
    out = moderated_t(fe["est"], unscaled, phi, d, d0=d0, s02=s02, truncate=truncate,
                      level=level)
    out["k"] = fe["k"]
    return out


def re_weight_sum(y, v, tau2) -> np.ndarray:
    """Σ w*_i with w*_i = 1/(v_i + τ²) per gene; missing cells (NaN in y or v) count 0. The RE
    Wald SE is 1/sqrt of this."""
    _, V, present = meta.as_matrix(y, v)
    tau2 = np.atleast_1d(np.asarray(tau2, dtype=np.float64))
    w = np.where(present, 1.0 / np.where(present, V + tau2[:, None], 1.0), 0.0)
    return w.sum(axis=1)


def moderated_hksj(y, v, tau2, *, truncate: bool, level: float = 0.95,
                   d0: float | None = None, s02: float | None = None) -> dict:
    """V1, the original sketch: moderate the HKSJ multiplier q across genes.

        q   = Σ w*_i (y_i − μ_RE)² / (k − 1),  w*_i = 1/(v_i + τ̂²)   (meta.random_effects_hksj)
        q̃   = (d₀s₀² + (k − 1)q) / (d₀ + k − 1)
        SE² = q̃ / Σw*_i  (or max(1, q̃)/Σw*_i if truncated),  t on k − 1 + d₀ df

    Kept so the comparison can show what happens. On the real data, REML's τ̂² has already
    absorbed each gene's scatter, so q varies less than χ² sampling would imply. The prior fit
    gives d₀ = ∞, every gene gets q̃ = s₀², and the test becomes the RE Wald z with the plug-in
    τ̂² (scaled by s₀² if untruncated).
    """
    re = meta.random_effects_hksj(y, v, tau2, truncate=False, level=level)
    unscaled = 1.0 / re_weight_sum(y, v, tau2)
    out = moderated_t(re["est"], unscaled, re["multiplier"], re["df"], d0=d0, s02=s02,
                      truncate=truncate, level=level)
    out["k"] = meta.n_sites(meta.as_matrix(y, v)[2])
    return out
