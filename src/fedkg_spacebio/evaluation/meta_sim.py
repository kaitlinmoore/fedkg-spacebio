"""Calibrated simulation for the phase 3a method comparison (WS-08).

Purpose
-------
On real data we can count hits but not tell true from false ones. Here we generate data whose
truth we know and whose shape copies the real liver data, run every method
(``meta_methods.run_methods``), and score them. The question is "which method keeps its
promise at k = 3–4?", answered separately for the two things a hit can mean.

Two estimands
-------------
Each gene has a mean effect μ and mission effects θ_i = μ + δ_i (δ = between-mission
deviation). Genes come in three kinds:

- kind 1: μ = 0 and every δ_i = 0. Nothing changes in any mission.
- kind 2: μ = 0 but δ_i ≠ 0. Missions differ, and their effects cancel on average.
- kind 3: μ ≠ 0.

- **(A) these missions:** a gene is null when nothing changes in any mission: kind 1 only.
  Fixed effect's question.
- **(B) a new mission:** a gene is null when μ = 0: kinds 1 and 2. Random effects' question.

Kind 2 is where the two disagree: FE calls it correctly under (A) and wrongly under (B). It's
also reported on its own.

What is copied from the real data (``calibrate``), per stratum
--------------------------------------------------------------
- The number of genes G and each gene's per-site SE vector. Rows are resampled whole, so the
  real structure is kept: SpaceX-16's small SEs, which sites are missing for a gene.
- The τ² distribution: V3's fitted prior, a deconvolved estimate, not the raw REML τ̂², which is
  noisy and 0 for 42–47% of genes.
- The multiplicative-heterogeneity prior (d₀, s₀²) from V2, for the multiplicative generator.
- The spread of non-zero effects σ_μ, fitted by marginal likelihood for each assumed fraction of
  μ = 0 genes (``fit_effect_sd``). μ ~ N(0, σ_μ²) for kind 3. A normal is lighter-tailed than
  real effects probably are; that's a stated simplification.

Weakly identified at k = 3–4, so swept rather than estimated: the fraction of μ = 0 genes
(0.80 / 0.90 / 0.95), a τ² scale (×0.5 / 1 / 2), and the share of μ = 0 genes with no
heterogeneity at all (kind 1 vs 2), fixed at 0.5. V3's prior puts almost no mass at exactly
τ² = 0, so without that fixed share there would be no kind 1 genes and estimand (A) would have
no nulls.

Generators (``Setting.generator``)
---------------------------------
- ``additive_eb``: δ_i ~ N(0, τ²), τ² drawn from V3's prior × scale. The main grid.
- ``additive_reml``: τ² drawn from the real REML τ̂² values × scale. Doesn't use V3's own fit,
  so it doesn't favour V3.
- ``multiplicative``: Var(y_i) = φ·v_i, with φ ~ d₀s₀²/χ²(d₀) from V2's prior; the excess
  (φ − 1)₊·v_i is the mission deviation. Favours V2.
- ``se_noise``: as ``additive_eb``, but the reported SE is noisy around the true one,
  v_obs = v·χ²(df)/df with df = n_flight + n_ground − 2 per site. **An approximate stress test**:
  DESeq2 shrinks dispersions toward a trend, so real SEs are less noisy than this.
- ``min3_pattern`` (ISS only): SE rows from the ≥3-site set, so k is 3 or 4 by gene.

Every method refits its own hyperparameters (d₀, s₀², V3's prior) on every simulated dataset,
as it would on real data. Seeds come from ``np.random.SeedSequence([base, stratum, setting,
replicate])`` and are recorded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import optimize

from fedkg_spacebio.evaluation import meta_methods
from fedkg_spacebio.federation import eb_tau2, meta, moderation

GENERATORS = ("additive_eb", "additive_reml", "multiplicative", "se_noise", "min3_pattern")


@dataclass(frozen=True)
class Setting:
    name: str
    generator: str = "additive_eb"
    tau2_scale: float = 1.0
    frac_mu0: float = 0.90
    frac_kind1_of_mu0: float = 0.5


def default_settings(include_min3: bool, free_frac_mu0: float | None = None) -> list[Setting]:
    """The 3 × 3 grid, the alternative generators at the central setting, and (if given) one
    setting at the unconstrained marginal-likelihood fit of the fraction of μ = 0 genes.

    That last one was added after calibration showed the free fit far below the grid (about
    0.37 on ISS): the data look like many small effects rather than a few large ones. It's
    reported as an addition to the pre-specified grid."""
    grid = [Setting(f"eb_c{c:g}_f{f:g}", "additive_eb", c, f)
            for c in (0.5, 1.0, 2.0) for f in (0.80, 0.90, 0.95)]
    extra = [Setting("reml_central", "additive_reml"), Setting("mult_central", "multiplicative"),
             Setting("senoise_central", "se_noise")]
    if free_frac_mu0 is not None:
        extra.append(Setting("eb_freefit", "additive_eb", 1.0, round(free_frac_mu0, 4)))
    if include_min3:
        extra.append(Setting("min3_central", "min3_pattern"))
    return grid + extra


@dataclass
class Calibration:
    """What the simulation copies from one stratum's real data (see the module docstring)."""
    stratum: str
    se: np.ndarray                  # (G, K) per-site SE, NaN where the site didn't enter
    resid_df: np.ndarray            # (K,) n_flight + n_ground − 2
    tau2_grid: np.ndarray
    tau2_pi: np.ndarray
    tau2_reml: np.ndarray           # real REML τ̂² values (finite)
    phi_d0: float
    phi_s02: float
    sigma_mu: dict                  # frac_mu0 → σ_μ
    free_fit: dict                  # unconstrained (frac_mu0, σ_μ) estimate, reported only
    se_alt: np.ndarray | None = None  # ≥3-site SE rows, for ``min3_pattern``

    def summary(self) -> dict:
        cdf = np.cumsum(self.tau2_pi)
        quant = {f"q{int(q * 100)}": float(self.tau2_grid[np.searchsorted(cdf, q)])
                 for q in (0.5, 0.75, 0.9, 0.95, 0.99)}
        return {"stratum": self.stratum, "G": int(self.se.shape[0]), "K": int(self.se.shape[1]),
                "resid_df": self.resid_df.tolist(), "tau2_prior_mass_at_0": float(self.tau2_pi[0]),
                "tau2_prior_quantiles": quant, "phi_d0": self.phi_d0, "phi_s02": self.phi_s02,
                "sigma_mu": {str(k): v for k, v in self.sigma_mu.items()},
                "free_fit": self.free_fit}


# ---------------------------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------------------------


def _marginal_terms(y, v, grid):
    """Per gene and grid value: (Σ log d_i, Σ w y², Σ w, Σ w y, k) with d = v + τ², w = 1/d.

    These give the log density of y under μ ~ N(0, σ²) (σ = 0 for μ = 0) in closed form, using
    the matrix determinant lemma and Sherman–Morrison for D + σ²11ᵀ:
        log det = Σ log d + log(1 + σ² Σw)
        quad    = Σ w y² − σ² (Σ w y)² / (1 + σ² Σw)
    """
    Y, V, present = meta.as_matrix(y, v)
    k = meta.n_sites(present)
    J = len(grid)
    out = [np.empty((len(Y), J)) for _ in range(4)]
    for j, t in enumerate(grid):
        d = np.where(present, V + t, 1.0)
        w = np.where(present, 1.0 / d, 0.0)
        out[0][:, j] = np.where(present, np.log(d), 0.0).sum(axis=1)
        out[1][:, j] = (w * Y**2).sum(axis=1)
        out[2][:, j] = w.sum(axis=1)
        out[3][:, j] = (w * Y).sum(axis=1)
    return (*out, k)


def _log_marginal(terms, pi, frac_mu0, sigma):
    logdet, wy2, sw, swy, k = terms
    s2 = sigma**2
    const = -0.5 * k[:, None] * np.log(2 * np.pi)
    l0 = const - 0.5 * (logdet + wy2)
    l1 = const - 0.5 * (logdet + np.log1p(s2 * sw) + wy2 - s2 * swy**2 / (1 + s2 * sw))
    logpi = np.log(np.maximum(pi, 1e-300))[None, :]
    a0 = np.logaddexp.reduce(l0 + logpi, axis=1)
    a1 = np.logaddexp.reduce(l1 + logpi, axis=1)
    return np.logaddexp(np.log(frac_mu0) + a0, np.log1p(-frac_mu0) + a1)


def fit_effect_sd(y, v, grid, pi, frac_mu0: float | None = None) -> tuple[float, float]:
    """Fit μ ~ frac_mu0·δ₀ + (1 − frac_mu0)·N(0, σ_μ²) by marginal likelihood, τ² prior fixed.

    With ``frac_mu0`` given, only σ_μ is fitted (that's what the grid uses: σ_μ is refitted for
    each assumed fraction, because the two trade off). With None, both are fitted, and the
    result is reported but not used. Returns (frac_mu0, σ_μ). Genes with k < 2 are skipped.
    """
    Y, V, present = meta.as_matrix(y, v)
    keep = meta.n_sites(present) >= 2
    terms = _marginal_terms(np.where(present, Y, np.nan)[keep], np.where(present, V, np.nan)[keep],
                            grid)
    if frac_mu0 is not None:
        res = optimize.minimize_scalar(
            lambda ls: -_log_marginal(terms, pi, frac_mu0, np.exp(ls)).sum(),
            bounds=(np.log(1e-3), np.log(10.0)), method="bounded")
        return float(frac_mu0), float(np.exp(res.x))
    res = optimize.minimize(
        lambda p: -_log_marginal(terms, pi, 1 / (1 + np.exp(-p[0])), np.exp(p[1])).sum(),
        x0=[np.log(0.9 / 0.1), np.log(0.3)], method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000})
    return float(1 / (1 + np.exp(-res.x[0]))), float(np.exp(res.x[1]))


def calibrate(table: pd.DataFrame, stratum: str, *, alt_table: pd.DataFrame | None = None,
              frac_grid=(0.80, 0.90, 0.95)) -> Calibration:
    """Build a stratum's ``Calibration`` from its WS-07 per-gene table (primary variant)."""
    y, v, clients = meta_methods.matrices_from_table(table)
    resid_df = np.array([int(table[f"n_flight__{c}"].iloc[0]) + int(table[f"n_ground__{c}"].iloc[0])
                         - 2 for c in clients], dtype=np.float64)
    v3 = eb_tau2.eb_tau2(y, v)
    v2 = moderation.moderated_fe(y, v)
    tau2_reml, _ = meta.tau2_reml(y, v)
    sigma_mu = {f: fit_effect_sd(y, v, v3["grid"], v3["pi"], f)[1] for f in frac_grid}
    f_hat, s_hat = fit_effect_sd(y, v, v3["grid"], v3["pi"])
    sigma_mu[round(f_hat, 4)] = s_hat          # for the ``eb_freefit`` setting
    se_alt = None
    if alt_table is not None:
        _, v_alt, alt_clients = meta_methods.matrices_from_table(alt_table)
        if alt_clients != clients:
            raise ValueError("alt table has different clients")
        se_alt = np.sqrt(v_alt)
    return Calibration(stratum=stratum, se=np.sqrt(v), resid_df=resid_df, tau2_grid=v3["grid"],
                       tau2_pi=v3["pi"], tau2_reml=tau2_reml[np.isfinite(tau2_reml)],
                       phi_d0=v2["d0"], phi_s02=v2["s02"], sigma_mu=sigma_mu,
                       free_fit={"frac_mu0": f_hat, "sigma_mu": s_hat}, se_alt=se_alt)


# ---------------------------------------------------------------------------------------------
# Generating one dataset
# ---------------------------------------------------------------------------------------------


def generate(rng: np.random.Generator, cal: Calibration, s: Setting) -> tuple[np.ndarray,
                                                                               np.ndarray, dict]:
    """One simulated dataset: (y, v_obs, truth). truth has per-gene ``mu``, ``kind`` (1/2/3),
    ``theta`` (G, K) mission effects, and ``theta_fe`` (the FE-weighted mean of θ with the true
    variances: the quantity FE estimates, i.e. the "these missions" target)."""
    if s.generator not in GENERATORS:
        raise ValueError(f"unknown generator {s.generator!r}")
    pool = cal.se_alt if s.generator == "min3_pattern" else cal.se
    if pool is None:
        raise ValueError("min3_pattern needs a calibration with se_alt")
    G = cal.se.shape[0]
    se = pool[rng.integers(pool.shape[0], size=G)]
    present = np.isfinite(se)
    v = np.where(present, se**2, np.nan)
    K = se.shape[1]

    mu0 = rng.random(G) < s.frac_mu0
    kind1 = mu0 & (rng.random(G) < s.frac_kind1_of_mu0)
    if s.frac_mu0 not in cal.sigma_mu:
        raise ValueError(f"no fitted σ_μ for frac_mu0={s.frac_mu0}; "
                         "add it to calibrate(frac_grid=…)")
    sigma_mu = cal.sigma_mu[s.frac_mu0]
    mu = np.where(mu0, 0.0, rng.normal(0.0, sigma_mu, size=G))

    z = rng.standard_normal((G, K))
    if s.generator == "multiplicative":
        phi = cal.phi_d0 * cal.phi_s02 / rng.chisquare(cal.phi_d0, size=G) \
            if np.isfinite(cal.phi_d0) else np.full(G, cal.phi_s02)
        phi = np.where(kind1, 1.0, phi)
        dev_var = np.maximum(phi - 1.0, 0.0)[:, None] * v          # (φ − 1)₊ v_i
        samp_var = np.minimum(phi, 1.0)[:, None] * v               # so Var(y_i) = φ v_i
    else:
        if s.generator == "additive_reml":
            tau2 = rng.choice(cal.tau2_reml, size=G)
        else:
            tau2 = rng.choice(cal.tau2_grid, size=G, p=cal.tau2_pi)
        tau2 = np.where(kind1, 0.0, s.tau2_scale * tau2)
        dev_var = np.broadcast_to(tau2[:, None], (G, K))
        samp_var = v
    theta = mu[:, None] + np.sqrt(dev_var) * z
    y = theta + np.sqrt(samp_var) * rng.standard_normal((G, K))
    y = np.where(present, y, np.nan)

    if s.generator == "se_noise":
        df = cal.resid_df[None, :]
        v_obs = v * rng.chisquare(np.broadcast_to(df, (G, K))) / df
    else:
        v_obs = v

    no_dev = np.nansum(np.where(present, dev_var, 0.0), axis=1) == 0
    kind = np.where(mu != 0, 3, np.where(no_dev, 1, 2))
    w = np.where(present, 1.0 / np.where(present, v, 1.0), 0.0)
    theta_fe = (w * np.where(present, theta, 0.0)).sum(axis=1) / w.sum(axis=1)
    return y, v_obs, {"mu": mu, "kind": kind, "theta": theta, "theta_fe": theta_fe}


# ---------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------


def score(frames: dict[str, pd.DataFrame], truth: dict) -> pd.DataFrame:
    """One row per method: calls, FDR and power under (A) and (B), the kind-2 call rate,
    wrong-sign calls among kind 3, and 95% interval coverage of μ (B's target) and of the
    FE-weighted mission mean (A's target), overall and by kind.

    FDR is false calls / max(calls, 1), so a replicate with no calls has FDR 0 (the usual
    convention; E[FDR] averages over replicates).
    """
    kind, mu, theta_fe = truth["kind"], truth["mu"], truth["theta_fe"]
    rows = []
    for m, f in frames.items():
        hit = f["hit"].to_numpy(dtype=bool)
        est = f["est"].to_numpy()
        lo, hi = f["ci_low"].to_numpy(), f["ci_high"].to_numpy()
        ok = np.isfinite(lo) & np.isfinite(hi)
        calls = int(hit.sum())
        fa = int((hit & (kind == 1)).sum())
        fb = int((hit & (kind <= 2)).sum())
        cov_mu = (lo <= mu) & (mu <= hi)
        cov_fe = (lo <= theta_fe) & (theta_fe <= hi)
        row = {
            "method": m, "calls": calls,
            "false_A": fa, "fdr_A": fa / max(calls, 1),
            "power_A": float((hit & (kind >= 2)).sum() / max((kind >= 2).sum(), 1)),
            "false_B": fb, "fdr_B": fb / max(calls, 1),
            "power_B": float((hit & (kind == 3)).sum() / max((kind == 3).sum(), 1)),
            "kind2_call_rate": float((hit & (kind == 2)).sum() / max((kind == 2).sum(), 1)),
            "wrong_sign": int((hit & (kind == 3) & (np.sign(est) != np.sign(mu))).sum()),
            "coverage_mu": float(cov_mu[ok].mean()) if ok.any() else np.nan,
            "coverage_theta_fe": float(cov_fe[ok].mean()) if ok.any() else np.nan,
            "ci_width_median": float(np.median((hi - lo)[ok])) if ok.any() else np.nan,
        }
        for kk in (1, 2, 3):
            sel = ok & (kind == kk)
            row[f"coverage_mu_kind{kk}"] = float(cov_mu[sel].mean()) if sel.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run_replicate(cal: Calibration, s: Setting, seed_seq: np.random.SeedSequence,
                  methods=meta_methods.METHODS) -> pd.DataFrame:
    """Generate, run every method (each refits its hyperparameters), score."""
    rng = np.random.default_rng(seed_seq)
    y, v, truth = generate(rng, cal, s)
    frames, info = meta_methods.run_methods(y, v, methods)
    out = score(frames, truth)
    out["n_kind1"] = int((truth["kind"] == 1).sum())
    out["n_kind2"] = int((truth["kind"] == 2).sum())
    out["n_kind3"] = int((truth["kind"] == 3).sum())
    if "v2_prior" in info:
        out["v2_d0"], out["v2_s02"] = info["v2_prior"]["d0"], info["v2_prior"]["s02"]
    if "v1_untrunc_prior" in info:
        out["v1_d0"] = info["v1_untrunc_prior"]["d0"]
    if "v3_prior" in info:
        out["v3_mass_at_0"] = float(info["v3_prior"]["pi"][0])
    for k, val in asdict(s).items():
        out[k] = val
    out["stratum"] = cal.stratum
    out["entropy"] = str(seed_seq.entropy)
    out["spawn_key"] = str(tuple(seed_seq.spawn_key))
    return out


def seed_for(base: int, stratum_idx: int, setting_idx: int, rep: int) -> np.random.SeedSequence:
    """The seed for one replicate: fixed by (base, stratum, setting, replicate) alone, so any
    replicate can be rerun on its own and parallel order doesn't matter."""
    return np.random.SeedSequence([base, stratum_idx, setting_idx, rep])


def summarize(reps: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD over replicates, per stratum × setting × method."""
    keys = ["stratum", "name", "generator", "tau2_scale", "frac_mu0", "method"]
    metrics = ["calls", "fdr_A", "power_A", "fdr_B", "power_B", "kind2_call_rate", "wrong_sign",
               "coverage_mu", "coverage_theta_fe", "ci_width_median", "coverage_mu_kind1",
               "coverage_mu_kind2", "coverage_mu_kind3"]
    g = reps.groupby(keys, sort=False)[metrics]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd"))
    out["n_rep"] = g.size()
    return out.reset_index()


def mc_se(p: float, n: int) -> float:
    """Monte Carlo SE of a proportion-like mean over n replicates (for the write-up)."""
    return float(np.sqrt(p * (1 - p) / max(n, 1)))
