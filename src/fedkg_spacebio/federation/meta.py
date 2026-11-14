"""Per-gene meta-analysis pooling for phase 3a (D-021).

Contents: input handling, ``fixed_effect`` (with Cochran's Q and I²), ``tau2_reml``,
``random_effects_hksj``, ``bh_qvalues`` and the ``pool`` driver. ``tests/test_meta.py`` checks
them against published reference values.

Everything here is plain numpy/scipy with no FLARE import, so it can be tested on its own. The
FLARE server calls ``pool`` on the summaries it receives, and the central check calls the same
function in-process.

Shapes and missing values
-------------------------
Every function takes ``y`` (effect estimates; here log2 fold changes) and ``v`` (their sampling
variances, i.e. SE²) as arrays of shape (G, K): G genes × K sites. A 1-D input of shape (K,)
is treated as a single gene. **NaN marks "this site did not test this gene"**, and a NaN in
either y or v removes that (gene, site) cell from every sum. So k, the number of contributing
sites, can differ by gene (used in the ≥3-site sensitivity run). Missing cells are handled with
weights of 0, not by dropping columns, so the whole computation stays vectorized over genes.

The model
---------
For one gene, site i reports y_i with variance v_i (treated as known):

    fixed effect:    y_i = μ + e_i,              e_i ~ N(0, v_i)
    random effects:  y_i = μ + u_i + e_i,        u_i ~ N(0, τ²)

τ² is the between-site variance of the true effects. Fixed effect assumes τ² = 0, so every site
estimates the same μ. Random effects lets each site have its own true effect around a mean μ.
With 3–4 sites, τ² is very imprecise. Always report it with that warning, and name the
site-level confounders (strain, sex, age, duration) whenever you talk about heterogeneity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------------------------


def as_matrix(y, v) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Y, V, present) as float64 arrays of shape (G, K).

    ``present`` is True where both y and v are finite. Y and V are set to 0 where a cell is
    missing, so weighted sums can use plain ``.sum(axis=1)`` with a weight of 0 there. Callers
    must use ``present`` (or weights that are 0 when missing), never Y itself, to tell a missing
    cell from a real 0.

    Raises ValueError on mismatched shapes, or on a variance ≤ 0 in a present cell. A
    zero variance would mean an infinitely precise site, and the inverse-variance weight would
    be infinite.
    """
    Y = np.asarray(y, dtype=np.float64)
    V = np.asarray(v, dtype=np.float64)
    if Y.shape != V.shape:
        raise ValueError(f"y and v have different shapes: {Y.shape} vs {V.shape}")
    if Y.ndim == 1:
        Y, V = Y[None, :], V[None, :]
    if Y.ndim != 2:
        raise ValueError(f"expected 1-D (K,) or 2-D (G, K) input, got {Y.ndim}-D")
    present = np.isfinite(Y) & np.isfinite(V)
    if np.any(V[present] <= 0):
        raise ValueError("sampling variances must be > 0 where an estimate is present")
    return np.where(present, Y, 0.0), np.where(present, V, 0.0), present


def n_sites(present: np.ndarray) -> np.ndarray:
    """k per gene: the number of sites with a usable estimate."""
    return present.sum(axis=1)


# ---------------------------------------------------------------------------------------------
# Fixed effect, Cochran's Q, I²
# ---------------------------------------------------------------------------------------------


def fixed_effect(y, v) -> dict[str, np.ndarray]:
    """Inverse-variance fixed-effect pooling, per gene.

        w_i  = 1 / v_i
        μ_FE = Σ w_i y_i / Σ w_i
        SE   = 1 / sqrt(Σ w_i)
        z    = μ_FE / SE,   p = 2·(1 − Φ(|z|))

    The weights are the precision of each site's estimate. A site with half the variance counts
    twice as much. This is the minimum-variance unbiased combination **if** every site estimates
    the same μ.

    Cochran's Q is the weighted sum of squared deviations from μ_FE:

        Q = Σ w_i (y_i − μ_FE)²,   df = k − 1

    Under homogeneity (τ² = 0), Q ~ χ²(k−1). Q's p-value is a weak test with few sites: failing
    to reject is **not** evidence of homogeneity.

    I² (Higgins & Thompson 2002) is the share of total variation attributed to between-site
    heterogeneity rather than sampling error:

        I² = max(0, (Q − df) / Q)

    This Q-based form doesn't depend on which τ² estimator is used. metafor prints a τ²-based
    I² instead (τ² / (τ² + s²)), so for REML its number differs slightly from this one. The
    tests check both.

    Genes with k = 0 get NaN everywhere. With k = 1, Q = 0 and df = 0, so q_p and i2 are NaN.
    """
    Y, V, present = as_matrix(y, v)
    k = n_sites(present)
    w = np.where(present, 1.0 / np.where(present, V, 1.0), 0.0)
    sw = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        est = (w * Y).sum(axis=1) / sw
        se = 1.0 / np.sqrt(sw)
        z = est / se
        p = 2.0 * stats.norm.sf(np.abs(z))
        resid = np.where(present, Y - est[:, None], 0.0)
        q = (w * resid**2).sum(axis=1)
        df = k - 1
        q_p = np.where(df >= 1, stats.chi2.sf(q, np.maximum(df, 1)), np.nan)
        i2 = np.where((df >= 1) & (q > 0), np.maximum(0.0, (q - df) / q), np.nan)
        # Q = 0 with df ≥ 1 means every estimate is identical, so there's no heterogeneity.
        i2 = np.where((df >= 1) & (q == 0), 0.0, i2)
    none = k == 0
    for arr in (est, se, z, p, q):
        arr[none] = np.nan
    return {"k": k, "est": est, "se": se, "z": z, "p": p, "q": q, "q_df": df, "q_p": q_p,
            "i2": i2}


def i2_from_tau2(v, tau2) -> np.ndarray:
    """metafor's τ²-based I²: τ² / (τ² + s²), with the "typical" within-site variance

        s² = (k − 1) Σw / ((Σw)² − Σw²),   w = 1/v.

    It's here only so a test can reproduce metafor's printed I² for the BCG data (92.22%). The
    pipeline reports the Q-based I² from ``fixed_effect``.
    """
    V = np.atleast_2d(np.asarray(v, dtype=np.float64))
    present = np.isfinite(V)
    k = n_sites(present)
    w = np.where(present, 1.0 / np.where(present, V, 1.0), 0.0)
    sw, sw2 = w.sum(axis=1), (w**2).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        s2 = (k - 1) * sw / (sw**2 - sw2)
        return np.asarray(tau2, dtype=np.float64) / (np.asarray(tau2, dtype=np.float64) + s2)


# ---------------------------------------------------------------------------------------------
# Random effects: REML τ² and Hartung–Knapp–Sidik–Jonkman
# ---------------------------------------------------------------------------------------------


def tau2_reml(y, v, *, tol: float = 1e-10, max_iter: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Between-site variance τ² by restricted maximum likelihood, per gene.

    Returns ``(tau2, converged)``, both of shape (G,).

    Contract (the tests check each point):
    - τ² ≥ 0 always. Truncate at 0; a negative variance isn't meaningful.
    - Genes with k < 2 sites: τ² = NaN and converged = False. One site carries no information
      about between-site spread.
    - If every estimate in a gene is identical, τ² = 0.
    - Missing cells (``present`` False) contribute nothing.
    - It's vectorized over genes: no Python loop over G. A loop over iterations is fine.

    The math, for one gene (w_i depends on τ², so this is solved iteratively):

        w_i(τ²) = 1 / (v_i + τ²)
        μ̂(τ²)  = Σ w_i y_i / Σ w_i

    REML maximizes the likelihood of the data *after* μ has been estimated. That's why it
    corrects ML's downward bias in τ², the same way dividing by n − 1 instead of n corrects a
    sample variance. Setting the derivative of the REML log-likelihood to zero gives

        Σ w_i² (y_i − μ̂)²  −  Σ w_i  +  Σ w_i² / Σ w_i  =  0

    Rearranged (using Σ w_i = Σ w_i² (v_i + τ²)), this is a fixed-point update:

        τ²_new = Σ w_i² ((y_i − μ̂)² − v_i) / Σ w_i²  +  1 / Σ w_i

    Iterate: start from a value (0, or the DerSimonian–Laird estimate), compute w and μ̂, update
    τ², truncate at 0, and stop when |τ²_new − τ²| < tol for every gene still running (or at
    max_iter, with converged = False for the genes that didn't settle).

    An alternative is Fisher scoring, which is faster near the optimum. With expected
    information I(τ²) = ½[Σw² − 2Σw³/Σw + (Σw²/Σw)²] and score
    S(τ²) = ½[Σw²(y−μ̂)² − Σw + Σw²/Σw], the step is τ² ← τ² + S/I. Either works, since
    it converges to the same root. Reference: Viechtbauer (2005), J. Educ. Behav. Stat. 30:261–293.

    Implementation choice: the fixed-point update, because each step is the formula above and
    nothing else, so it's easy to check by hand. Starting value τ² = 0. Why 0 and not
    DerSimonian–Laird: when the REML root is 0 (common with 3–4 sites), the first update is
    already ≤ 0, so truncation stops it at exactly 0 after one step. When the root is positive,
    the update climbs to it from below. Fixed-point steps converge linearly, so some genes need
    tens to a few hundred iterations; each is cheap and vectorized, and genes are frozen once
    converged, so a slow gene doesn't keep the others iterating. A gene that hasn't settled by
    ``max_iter`` keeps its last value and is flagged (``reml_converged`` = False in ``pool``).
    """
    Y, V, present = as_matrix(y, v)
    k = n_sites(present)
    G = Y.shape[0]
    tau2 = np.zeros(G)
    converged = np.zeros(G, dtype=bool)
    estimable = k >= 2                     # one site says nothing about between-site spread
    running = estimable.copy()             # genes still iterating

    for _ in range(max_iter):
        if not running.any():
            break
        Yr, Vr, Pr, t = Y[running], V[running], present[running], tau2[running]
        # RE weights at the current τ². Missing cells get weight 0, so they drop out of every
        # sum below; the inner np.where avoids dividing by the 0 placeholder in V.
        w = np.where(Pr, 1.0 / np.where(Pr, Vr + t[:, None], 1.0), 0.0)
        sw = w.sum(axis=1)                 # Σ w_i
        sw2 = (w**2).sum(axis=1)           # Σ w_i²
        mu = (w * Yr).sum(axis=1) / sw     # μ̂(τ²), the RE-weighted mean
        resid2 = np.where(Pr, (Yr - mu[:, None]) ** 2, 0.0)
        # Fixed-point step:  τ² ← Σ w² ((y − μ̂)² − v) / Σ w²  +  1 / Σ w
        # The "+ 1/Σw" term is what makes this REML rather than ML (it accounts for μ having
        # been estimated from the same data). Then truncate: a variance can't be negative.
        new = (w**2 * (resid2 - np.where(Pr, Vr, 0.0))).sum(axis=1) / sw2 + 1.0 / sw
        new = np.maximum(new, 0.0)
        done = np.abs(new - t) < tol
        idx = np.flatnonzero(running)
        tau2[idx] = new
        converged[idx[done]] = True
        running[idx[done]] = False

    tau2[~estimable] = np.nan
    return tau2, converged


def random_effects_hksj(y, v, tau2, *, truncate: bool = True,
                        level: float = 0.95) -> dict[str, np.ndarray]:
    """Random-effects mean with Hartung–Knapp–Sidik–Jonkman inference.

    Given τ² per gene (from ``tau2_reml``), return a dict of (G,) arrays with keys:
    ``est``, ``se``, ``t``, ``df``, ``p``, ``ci_low``, ``ci_high``, ``multiplier``.

    The math, for one gene:

        w*_i = 1 / (v_i + τ²)                       # RE weights: every site's variance grows by τ²
        μ_RE = Σ w*_i y_i / Σ w*_i
        q    = Σ w*_i (y_i − μ_RE)² / (k − 1)       # HKSJ multiplier

    Careful: q uses the **RE weights and μ_RE**. It is not Cochran's Q (which uses 1/v_i and
    μ_FE) divided by k − 1.

        untruncated:  SE² = q / Σ w*_i
        truncated:    SE² = max(1, q) / Σ w*_i      # ← used (D-025, Kaitlin)

        t  = μ_RE / SE,  df = k − 1
        p  = 2 · P(T_df > |t|)                      # scipy.stats.t.sf
        CI = μ_RE ± t_{df, 1−(1−level)/2} · SE      # scipy.stats.t.ppf

    Why HKSJ: the plain random-effects ("Wald") SE, 1/sqrt(Σw*), treats τ² as known and uses a
    normal quantile. With 3–4 sites, τ² is anything but known, and those intervals are too
    narrow. HKSJ rescales the variance by how scattered the sites actually are (q), and uses a t
    distribution with k − 1 df, whose heavy tails at small df make the interval honest.

    Why truncate: when sites agree more closely than their SEs predict, q < 1, and plain HKSJ
    then gives an interval *narrower* than the RE Wald one. That's a known flaw at small k
    (Röver, Knapp & Friede 2015, BMC Med Res Methodol 15:99). max(1, q) stops HKSJ from ever
    being more confident than the model-based SE.

    Contract (the tests check each point):
    - Genes with k < 2: every output is NaN (df would be 0).
    - ``multiplier`` is q itself (before truncation), so callers can count where truncation
      applied.
    - With τ² = 0 and identical estimates, q = 0. Then the truncated SE equals the
      fixed-effect SE, and the untruncated SE is 0.
    """
    Y, V, present = as_matrix(y, v)
    k = n_sites(present)
    tau2 = np.atleast_1d(np.asarray(tau2, dtype=np.float64))
    ok = k >= 2                                        # df = k − 1 must be ≥ 1
    with np.errstate(invalid="ignore", divide="ignore"):
        # RE weights: each site's variance is its sampling variance plus τ².
        w = np.where(present, 1.0 / np.where(present, V + tau2[:, None], 1.0), 0.0)
        sw = w.sum(axis=1)
        est = (w * Y).sum(axis=1) / sw                 # μ_RE
        df = (k - 1).astype(np.float64)
        # HKSJ multiplier: the RE-weighted scatter of the sites around μ_RE, per degree of
        # freedom. ≈ 1 when the sites scatter as much as v + τ² predicts.
        resid2 = np.where(present, (Y - est[:, None]) ** 2, 0.0)
        q = (w * resid2).sum(axis=1) / df
        scale = np.maximum(1.0, q) if truncate else q  # truncation: never below the Wald SE
        se = np.sqrt(scale / sw)
        t = est / se
        p = 2.0 * stats.t.sf(np.abs(t), df)
        crit = stats.t.ppf(1.0 - (1.0 - level) / 2.0, df)  # e.g. 4.30 at df = 2, 3.18 at df = 3
        ci_low, ci_high = est - crit * se, est + crit * se
    out = {"est": est, "se": se, "t": t, "df": df, "p": p, "ci_low": ci_low,
           "ci_high": ci_high, "multiplier": q}
    for arr in out.values():
        arr[~ok] = np.nan
    return out


# ---------------------------------------------------------------------------------------------
# Multiple testing and the driver
# ---------------------------------------------------------------------------------------------


def bh_qvalues(p) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values (q-values), ignoring NaNs.

    For m finite p-values sorted ascending, p_(1) ≤ … ≤ p_(m):

        q_(i) = min_{j ≥ i} ( m · p_(j) / j ),  capped at 1

    The running minimum from the top keeps q monotone in p. NaN p-values stay NaN and don't
    count toward m. Same result as ``statsmodels.stats.multitest.multipletests(method="fdr_bh")``
    (checked in the tests).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv, kind="mergesort")
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


CLASSIC_COLUMNS = [
    "k", "fe_est", "fe_se", "fe_p", "fe_q", "re_est", "hksj_se", "hksj_ci_low", "hksj_ci_high",
    "re_p", "re_q", "hksj_multiplier", "re_p_untruncated", "tau2", "reml_converged", "i2", "q",
    "q_p",
]
# V2 (moderated multiplicative heterogeneity, D-028) and V3 (empirical-Bayes τ², D-028/D-029).
# V2's estimate is the FE estimate (it keeps FE weights), so it has no estimate column.
V2_COLUMNS = ["v2_se", "v2_ci_low", "v2_ci_high", "v2_df", "v2_p", "v2_q", "v2_phi",
              "v2_phi_moderated"]
V3_COLUMNS = ["v3_est", "v3_sd", "v3_ci_low", "v3_ci_high", "v3_p", "v3_q",
              "v3_tau2_post_mean", "v3_p_tau2_zero"]
EB_COLUMNS = V2_COLUMNS + V3_COLUMNS
POOL_COLUMNS = CLASSIC_COLUMNS + EB_COLUMNS


def pool(y, v, *, level: float = 0.95, truncate: bool = True, min_k: int = 2,
         eb: bool = True) -> pd.DataFrame:
    """Pool every gene: FE, REML random effects with HKSJ, Q, I², V2 and V3; BH within the call.

    One row per gene (same order as the input rows). Columns: ``POOL_COLUMNS`` when ``eb`` is
    True (the default, and what the FLARE server and the central check run), otherwise
    ``CLASSIC_COLUMNS``. Genes with k < min_k get NaN estimates and are left out of the BH
    correction **and of the V2/V3 prior fits**. The caller decides which genes enter (complete
    case vs ≥3 sites) and passes only those rows, so BH and the priors cover exactly the genes
    in that analysis.

    Which column answers which question (D-029):
    - ``fe_*``: "these missions". Q p and I² are annotations next to it, not filters.
    - ``v3_*``: "a new mission". It replaced truncated RE-HKSJ in that role.
    - ``v2_*``: sensitivity for "these missions" if the site SEs are misestimated.
    - ``re_*`` / ``hksj_*``: REML + truncated HKSJ (D-025), kept for comparison.
      ``re_p_untruncated`` lets the write-up count how often truncation changed the answer.

    V2 and V3 are empirical-Bayes methods: each fits a prior **across all genes in the call**
    (V2: the scaled-F prior (d₀, s₀²) for φ = Q/(k−1); V3: a τ² distribution on a grid) and
    then uses it gene by gene. So, unlike FE and REML, a gene's V2/V3 result depends on which
    other genes were pooled with it. That's the point (borrowing strength), and it's why the
    priors are recorded in ``df.attrs["eb_priors"]`` next to every table. Same calls and
    defaults as WS-08's comparison (``evaluation.meta_methods.run_methods``): V2 untruncated,
    V3 on a 60-point grid.
    """
    fe = fixed_effect(y, v)
    tau2, converged = tau2_reml(y, v)
    re = random_effects_hksj(y, v, tau2, truncate=truncate, level=level)
    re_u = random_effects_hksj(y, v, tau2, truncate=False, level=level)
    k = fe["k"]
    use = k >= min_k
    df = pd.DataFrame({
        "k": k,
        "fe_est": fe["est"], "fe_se": fe["se"], "fe_p": fe["p"],
        "re_est": re["est"], "hksj_se": re["se"], "hksj_ci_low": re["ci_low"],
        "hksj_ci_high": re["ci_high"], "re_p": re["p"], "hksj_multiplier": re["multiplier"],
        "re_p_untruncated": re_u["p"],
        "tau2": tau2, "reml_converged": converged,
        "i2": fe["i2"], "q": fe["q"], "q_p": fe["q_p"],
    })
    estimate_cols = [c for c in df.columns if c not in ("k", "reml_converged")]
    df.loc[~use, estimate_cols] = np.nan
    df.loc[~use, "reml_converged"] = False
    df["fe_q"] = bh_qvalues(df["fe_p"].to_numpy())
    df["re_q"] = bh_qvalues(df["re_p"].to_numpy())
    if not eb:
        return df[CLASSIC_COLUMNS]
    eb_cols, priors = _eb_columns(y, v, use, level=level)
    for c in EB_COLUMNS:
        df[c] = eb_cols[c]
    out = df[POOL_COLUMNS]
    out.attrs["eb_priors"] = priors
    return out


def _eb_columns(y, v, use: np.ndarray, *, level: float) -> tuple[dict, dict]:
    """V2 and V3 columns for the genes in ``use`` (k ≥ min_k); NaN for the rest.

    Only the ``use`` rows are passed to the two methods, so genes below min_k can't influence
    either prior. If no prior can be fitted (too few usable genes; a tiny test input, never a
    real run with ~10k genes), that method's columns are NaN and the reason is recorded in the
    returned priors dict instead of raising: the FE and REML columns are still valid.
    """
    # Imported here, not at the top: both modules import this one (they reuse as_matrix and
    # fixed_effect), so a top-level import would be circular.
    from fedkg_spacebio.federation import eb_tau2, moderation

    Y, V, present = as_matrix(y, v)
    Y = np.where(present, Y, np.nan)[use]
    V = np.where(present, V, np.nan)[use]
    n = len(use)
    cols = {c: np.full(n, np.nan) for c in EB_COLUMNS}
    priors: dict = {"n_genes": int(use.sum())}

    try:
        r = moderation.moderated_fe(Y, V, level=level)  # V2, untruncated (WS-08, D-028)
    except ValueError as err:
        priors["v2"] = {"fitted": False, "reason": str(err)}
    else:
        for col, key in (("v2_se", "se"), ("v2_ci_low", "ci_low"), ("v2_ci_high", "ci_high"),
                         ("v2_df", "df"), ("v2_p", "p"), ("v2_phi", "scale"),
                         ("v2_phi_moderated", "scale_moderated")):
            cols[col][use] = r[key]
        cols["v2_q"][use] = bh_qvalues(r["p"])
        # d₀ can be ∞ (no excess spread in φ̂ across genes); JSON has no ∞, so store a string.
        priors["v2"] = {"fitted": True, "d0": r["d0"] if np.isfinite(r["d0"]) else "inf",
                        "s02": r["s02"], "truncate": False}

    try:
        r = eb_tau2.eb_tau2(Y, V, level=level)  # V3, default 60-point grid
    except ValueError as err:
        priors["v3"] = {"fitted": False, "reason": str(err)}
    else:
        for col, key in (("v3_est", "est"), ("v3_sd", "sd"), ("v3_ci_low", "ci_low"),
                         ("v3_ci_high", "ci_high"), ("v3_p", "p"),
                         ("v3_tau2_post_mean", "tau2_post_mean"),
                         ("v3_p_tau2_zero", "p_tau2_zero")):
            cols[col][use] = r[key]
        cols["v3_q"][use] = bh_qvalues(r["p"])
        priors["v3"] = {"fitted": True, "grid": r["grid"].tolist(), "pi": r["pi"].tolist(),
                        "em_iterations": int(r["fit"]["iterations"]),
                        "em_converged": bool(r["fit"]["converged"]),
                        "em_loglik": float(r["fit"]["loglik"])}
    return cols, priors
