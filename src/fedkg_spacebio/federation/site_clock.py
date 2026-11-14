"""Q-11 site-side clock diagnostic: do flight and ground differ in clock time, and how?

The question
------------
No site recorded when its animals were euthanized. If the flight and ground arms within a site
were taken at different times of day, every gene that cycles in the liver looks like a "flight
effect". If spaceflight shifted the clock, it looks the same. This module measures the size of
any flight–ground difference in clock time and whether it has the shape of a clock-time
difference. **It can't tell a dissection-time mismatch from a real clock shift.**

Why not estimate each animal's phase from z-scored genes (the first plan)
-------------------------------------------------------------------------
Centring a gene within a site subtracts its site mean, which is what carries absolute time.
What's left is each animal's deviation from the site average. For two arms at times t_f and
t_g, a cosine gene's deviations are ±½[cos(ω(t_f − φ)) − cos(ω(t_g − φ))], which is
∝ sin(ω(M − φ)) with M the midpoint (t_f + t_g)/2. The deviation pattern sits 6 h from M,
with opposite signs for the two arms. A template match then puts the arms ~12 h apart **for any
true shift**. The size of the shift lives in the **amplitude** of the deviation, not in its
phase. ``tests/test_site_clock.py`` reproduces this.

The clock plane (what this module does)
---------------------------------------
From the reference (``evaluation/clock_reference.py``), each panel gene has a peak time φ_g and
an amplitude A_g on the RNA-seq log2 scale. With a_g = A_g cos(ωφ_g) and b_g = A_g sin(ωφ_g), a
healthy clock at time t gives x_g(t) − m_g = cos(ωt)·a_g + sin(ωt)·b_g. So any difference in
clock time between two groups lies in the 2-D plane spanned by **a** and **b**.

Inside the site:
1. **Normalize:** log2(median-of-ratios normalized count + 1). Size factors are computed over
   all genes within this site (DESeq2's estimator, implemented here so it doesn't depend on a
   model fit).
   - *Alternatives:* VST. The panel genes are well expressed in liver, where VST ≈ log2. VST
     would add a dispersion fit and a blind/non-blind choice, and the amplitudes must be on the
     same log2(count + 1) scale as the RNA-seq reference.
2. **Flight–ground difference per panel gene:** OLS of log expression on condition plus the
   site's design covariates (``study`` for SpaceX-4, ``age`` for SpaceX-16), the same design as
   site DE. The flight coefficients form the vector **d**.
3. **Fit d ≈ α·a + β·b** (2 parameters, k genes). (α, β) is the **chord**: flight's point on
   the clock circle minus ground's.
   - **Shift size:** both points lie on the unit circle, so L = |chord| = 2·|sin(ωΔ/2)| and
     |Δ| = (24/π)·arcsin(L/2). That assumes both arms have the reference amplitude.
     - **L ≥ 2 is impossible** for two points on the unit circle. It means amplitudes differ
       from the reference, or the noise is large. It's reported as **"not identifiable"**,
       never clipped.
   - **Midpoint:** chord = 2 sin(ωΔ/2)·(−sin ωM, cos ωM). The chord's angle ψ gives
     M = ψ/ω − 6 h if flight is later (Δ > 0), or ψ/ω + 6 h if flight is earlier. Both readings
     are reported. A reading is flagged "light-phase consistent" if both implied arm times fall
     in ZT0–12, with a 1 h margin for the CT/ZT gap. That's an assumption (dissections in the
     light phase), stated wherever it's used.
   - **Coherence: R²_plane** = |projection of d onto the plane|² / |d|². This is the share of
     the flight–ground clock-gene difference that a pure clock-time difference explains. With
     no clock structure it averages about 2/k. A permutation null is reported alongside.
     - It can't separate "shifted" from "amplitude changed": both stay in the plane. It does
       separate both from "disrupted", where genes move independently and leave the plane.
4. **Test:** permute the arm labels within covariate blocks; exact enumeration when there are
   at most 20,000 relabelings, otherwise 9,999 random ones. Statistics: L (any clock-time
   difference) and R²_plane.
5. **Intervals:** percentile bootstrap, stratified by arm × block, 2,000 resamples. Each
   resample is paired with a bootstrap replicate of the reference fit, so the interval carries
   the reference's calibration uncertainty. The sample-only interval and the shift at
   amplitudes ×0.75 and ×1.25 are reported too.
6. **Noise scalar:** the median, over panel genes, of the residual SD from step 2. It feeds the
   power simulation (``evaluation/clock_sim.py``).

The federation boundary
-----------------------
Only ``ClockSummary`` leaves, checked by ``validate_clock_summary`` against exact key
whitelists:
- arm sizes and the panel genes used (with drops);
- the chord and its sample-bootstrap covariance;
- L and |Δ| with CIs, both readings, R²_plane;
- permutation p-values, the noise scalar, settings and versions.

No per-sample value, sample name or expression value leaves. The chord is two numbers per site,
an arm-level aggregate (D-030 list, extended in the WS-09 plan approval).

Output level (CONTRACTS §5.1–§5.2): both ``ClockSummary`` and (B)'s ``AbsoluteSummary`` are
``per_site_summary``. They reveal arm-level aggregates of one site, so a site policy that allows
only masked contributions refuses the clock diagnostic. The field-by-field contract is CONTRACTS
§4.1, and ``tests/test_contracts.py`` checks that it lists every whitelisted field.
"""

from __future__ import annotations

import itertools
import json
import math
from importlib.metadata import version

import numpy as np
import pandas as pd

from fedkg_spacebio.evaluation.clock_reference import OMEGA, PERIOD, ClockBasis
from fedkg_spacebio.harmonize.gene_map import GeneMap
from fedkg_spacebio.sites.loader import SiteData

MIN_MEAN_COUNT = 10.0
MAX_EXACT_PERMUTATIONS = 20_000
N_RANDOM_PERMUTATIONS = 9_999
N_BOOT = 2_000
LIGHT_WINDOW = (-1.0, 13.0)  # ZT0–12 with a 1 h margin for the CT/ZT gap
AMPLITUDE_SENSITIVITY = (0.75, 1.25)

SETTINGS = {
    "normalization": "log2(median-of-ratios normalized count + 1), size factors within site",
    "difference": "OLS per panel gene: log expression ~ design (condition: flight vs ground)",
    "panel_min_mean_normalized_count": MIN_MEAN_COUNT,
    "plane": "d ~ alpha*a + beta*b, a = A cos(w phi), b = A sin(w phi); peaks GSE11923, "
             "amplitudes GSE54651",
    "shift": "|delta| = (24/pi) * arcsin(L/2); not identifiable when L >= 2",
    "permutation": f"labels permuted within covariate blocks; exact if <= "
                   f"{MAX_EXACT_PERMUTATIONS}, else {N_RANDOM_PERMUTATIONS} random",
    "bootstrap": f"{N_BOOT} percentile, stratified by arm x block, paired with reference "
                 "bootstrap replicates",
    "light_window_zt": list(LIGHT_WINDOW),
    "amplitude_sensitivity": list(AMPLITUDE_SENSITIVITY),
}


# --------------------------------------------------------------------------- inside the site

def size_factors(counts: pd.DataFrame) -> np.ndarray:
    """DESeq2 median-of-ratios size factors (genes × samples). Genes with any zero are skipped
    (their geometric mean is 0)."""
    c = counts.to_numpy(dtype=float)
    pos = np.all(c > 0, axis=1)
    if pos.sum() < 1:
        raise ValueError("no gene is non-zero in every sample; size factors undefined")
    logc = np.log(c[pos])
    return np.exp(np.median(logc - logc.mean(axis=1, keepdims=True), axis=0))


def log_normalize(counts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(normalized counts, log2(normalized + 1)), both genes × samples."""
    norm = counts / size_factors(counts)
    return norm, np.log2(norm + 1.0)


def panel_matrix(site: SiteData, gene_map: GeneMap, basis: ClockBasis
                 ) -> tuple[pd.DataFrame, ClockBasis, list[dict]]:
    """The panel genes' log2 expression (samples × NCBI Gene), the basis restricted to the
    genes used, and the drops with their reasons."""
    norm, logx = log_normalize(site.counts)
    tab = gene_map.table
    one = tab[tab["category"] == "one_to_one"]
    by_ncbi = pd.Series(one.index, index=one["ncbi_gene"].astype("int64").to_numpy())
    cols, used, drops = {}, [], []
    for gid, sym in zip(basis.ncbi_gene, basis.symbol, strict=True):
        if gid not in by_ncbi.index:
            drops.append({"ncbi_gene": int(gid), "symbol": sym,
                          "reason": "no 1:1 Ensembl ID in the GeneLab annotation"})
            continue
        ens = by_ncbi.loc[gid]
        if ens not in logx.index:
            drops.append({"ncbi_gene": int(gid), "symbol": sym,
                          "reason": "Ensembl ID not in this site's count matrix"})
            continue
        mean = float(norm.loc[ens].mean())
        if mean < MIN_MEAN_COUNT:
            drops.append({"ncbi_gene": int(gid), "symbol": sym,
                          "reason": f"mean normalized count < {MIN_MEAN_COUNT:g}"})
            continue
        cols[gid] = logx.loc[ens].to_numpy(float)
        used.append(gid)
    X = pd.DataFrame(cols, index=site.counts.columns)
    return X, basis.subset(used), drops


def design_matrix(metadata: pd.DataFrame, design: str) -> tuple[np.ndarray, int, np.ndarray]:
    """(model matrix, column of the flight indicator, block label per sample).

    ``design`` is a formula like ``~ study + condition``; every term is categorical. Covariates
    are one-hot coded with the first sorted level as reference. Blocks are the combinations of
    covariate levels; permutations and the bootstrap stay within them.
    """
    terms = [t.strip() for t in design.replace("~", "").split("+") if t.strip()]
    if "condition" not in terms:
        raise ValueError(f"design {design!r} has no condition term")
    covs = [t for t in terms if t != "condition"]
    cols = [np.ones(len(metadata))]
    for c in covs:
        levels = sorted(metadata[c].astype(str).unique())
        cols += [(metadata[c].astype(str) == lv).to_numpy(float) for lv in levels[1:]]
    cond_col = len(cols)
    cols.append((metadata["condition"] == "flight").to_numpy(float))
    blocks = (metadata[covs].astype(str).agg("|".join, axis=1).to_numpy() if covs
              else np.array([""] * len(metadata)))
    return np.column_stack(cols), cond_col, blocks


def ols_difference(Y: np.ndarray, X: np.ndarray, cond_col: int) -> tuple[np.ndarray, np.ndarray]:
    """Flight coefficient per gene and residual SD per gene. Y is samples × genes."""
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ coef
    dof = X.shape[0] - np.linalg.matrix_rank(X)
    return coef[cond_col], np.sqrt((resid**2).sum(0) / dof)


def plane_fit(d: np.ndarray, basis: ClockBasis) -> dict:
    """Least-squares d ≈ α·a + β·b. Works on a (genes,) vector or a (reps, genes) array."""
    B = np.column_stack([basis.a, basis.b])  # genes × 2
    d2 = np.atleast_2d(d)
    coef, *_ = np.linalg.lstsq(B, d2.T, rcond=None)  # 2 × reps
    proj = (B @ coef).T
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = (proj**2).sum(1) / (d2**2).sum(1)
    alpha, beta = coef
    out = {"alpha": alpha, "beta": beta, "L": np.hypot(alpha, beta), "r2_plane": r2}
    if np.ndim(d) == 2:
        return out
    return {k: float(v[0]) for k, v in out.items()}


def delta_hours(L, scale: float = 1.0):
    """|Δ| in hours from the chord length; NaN where L/scale ≥ 2 (not identifiable)."""
    x = np.asarray(L, float) / scale / 2
    with np.errstate(invalid="ignore"):
        out = np.where(x < 1, (PERIOD / np.pi) * np.arcsin(np.minimum(x, 1)), np.nan)
    return float(out) if np.ndim(out) == 0 else out


def in_window(t: float, window=LIGHT_WINDOW) -> bool:
    lo, hi = window
    t = float(np.mod(t, PERIOD))
    return bool(lo <= t <= hi or lo <= t - PERIOD <= hi)


def readings(alpha: float, beta: float, delta: float | None) -> list[dict]:
    """The two (direction, midpoint) solutions for a chord, with arm times in reference CT.

    If |Δ| isn't identifiable, the midpoints are still reported; arm times aren't.
    """
    psi = math.atan2(beta, alpha) / OMEGA
    out = []
    for direction, mid in (("flight_later", psi - PERIOD / 4), ("flight_earlier",
                                                                psi + PERIOD / 4)):
        mid = float(np.mod(mid, PERIOD))
        rec = {"direction": direction, "mid_ct": round(mid, 3), "flight_ct": None,
               "ground_ct": None, "light_phase_consistent": None}
        if delta is not None and np.isfinite(delta):
            sign = 1 if direction == "flight_later" else -1
            tf = float(np.mod(mid + sign * delta / 2, PERIOD))
            tg = float(np.mod(mid - sign * delta / 2, PERIOD))
            rec.update({"flight_ct": round(tf, 3), "ground_ct": round(tg, 3),
                        "light_phase_consistent": in_window(tf) and in_window(tg)})
        out.append(rec)
    return out


def block_permutations(cond: np.ndarray, blocks: np.ndarray, rng: np.random.Generator,
                       n_random: int = N_RANDOM_PERMUTATIONS) -> tuple[np.ndarray, bool]:
    """Flight indicators (perms × samples), relabelled within blocks. Exact enumeration if the
    count is ≤ ``MAX_EXACT_PERMUTATIONS`` (the observed labelling is one of them); otherwise
    the observed labelling followed by ``n_random`` random ones."""
    groups = [np.flatnonzero(blocks == b) for b in np.unique(blocks)]
    sizes = [(len(g), int(cond[g].sum())) for g in groups]
    total = math.prod(math.comb(n, f) for n, f in sizes)
    if total <= MAX_EXACT_PERMUTATIONS:
        per_block = [list(itertools.combinations(g, f)) for g, (_, f) in zip(groups, sizes,
                                                                              strict=True)]
        perms = np.zeros((total, len(cond)))
        for i, choice in enumerate(itertools.product(*per_block)):
            for idx in choice:
                perms[i, list(idx)] = 1.0
        return perms, True
    perms = np.empty((n_random + 1, len(cond)))
    perms[0] = cond
    for i in range(1, n_random + 1):
        p = np.zeros(len(cond))
        for g, (_, f) in zip(groups, sizes, strict=True):
            p[rng.choice(g, size=f, replace=False)] = 1.0
        perms[i] = p
    return perms, False


def permuted_differences(Y: np.ndarray, X: np.ndarray, cond_col: int, perms: np.ndarray
                         ) -> np.ndarray:
    """Flight coefficients (perms × genes) for every relabelling, by Frisch–Waugh: residualize
    Y and the indicator on the covariates once, then d = c̃ᵀỸ / c̃ᵀc̃."""
    Z = np.delete(X, cond_col, axis=1)
    M = np.eye(len(Z)) - Z @ np.linalg.pinv(Z)
    Yt = M @ Y
    Ct = perms @ M
    return (Ct @ Yt) / (Ct**2).sum(1, keepdims=True)


def bootstrap_indices(cond: np.ndarray, blocks: np.ndarray, n_boot: int,
                      rng: np.random.Generator) -> np.ndarray:
    strata = [np.flatnonzero((blocks == b) & (cond == c))
              for b in np.unique(blocks) for c in (0.0, 1.0)]
    strata = [s for s in strata if len(s)]
    return np.stack([np.concatenate([rng.choice(s, size=len(s)) for s in strata])
                     for _ in range(n_boot)])


def analyse(Y: np.ndarray, metadata: pd.DataFrame, design: str, basis: ClockBasis,
            ref_bases: list[ClockBasis] | None = None, seed: int = 0,
            n_boot: int = N_BOOT, n_random_perm: int = N_RANDOM_PERMUTATIONS) -> dict:
    """Steps 2–6 of the module docstring on a (samples × panel genes) log2 matrix.

    Returns plain numbers only (no per-sample values). Used by ``site_clock_summary`` and by
    the validation and synthetic runs.
    """
    rng = np.random.default_rng(seed)
    X, cond_col, blocks = design_matrix(metadata, design)
    cond = X[:, cond_col]
    d, rsd = ols_difference(Y, X, cond_col)
    fit = plane_fit(d, basis)
    L = fit["L"]

    perms, exact = block_permutations(cond, blocks, rng, n_random_perm)
    Dp = permuted_differences(Y, X, cond_col, perms)
    pf = plane_fit(Dp, basis)
    if exact:
        p_L = float(np.mean(pf["L"] >= L - 1e-12))
        p_r2 = float(np.mean(pf["r2_plane"] >= fit["r2_plane"] - 1e-12))
    else:  # perms[0] is the observed labelling
        p_L = float((1 + np.sum(pf["L"][1:] >= L - 1e-12)) / len(perms))
        p_r2 = float((1 + np.sum(pf["r2_plane"][1:] >= fit["r2_plane"] - 1e-12)) / len(perms))

    idx = bootstrap_indices(cond, blocks, n_boot, rng)
    Db = np.stack([ols_difference(Y[i], X[i], cond_col)[0] for i in idx])
    bs = plane_fit(Db, basis)  # sample uncertainty only
    if ref_bases:
        joint_L = np.empty(n_boot)
        for j in range(n_boot):
            rb = ref_bases[j % len(ref_bases)].subset(basis.ncbi_gene)
            joint_L[j] = plane_fit(Db[j], rb)["L"]
    else:
        joint_L = bs["L"]

    def ci(v):
        return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]

    L_ci, L_ci_s = ci(joint_L), ci(bs["L"])
    delta = delta_hours(L)
    status = ("not_identifiable" if not np.isfinite(delta) else
              "ci_upper_not_identifiable" if L_ci[1] >= 2 else "identifiable")

    def to_h(v):
        h = delta_hours(v)
        return None if not np.isfinite(h) else float(h)

    cov = np.cov(np.vstack([bs["alpha"], bs["beta"]]))
    return {
        "k": int(basis.k), "alpha": float(fit["alpha"]), "beta": float(fit["beta"]),
        "chord_cov_sample": cov.tolist(),
        "L": float(L), "L_ci": L_ci, "L_ci_sample_only": L_ci_s,
        "delta_h": to_h(L), "delta_h_ci": [to_h(L_ci[0]), to_h(L_ci[1])],
        "delta_h_ci_sample_only": [to_h(L_ci_s[0]), to_h(L_ci_s[1])],
        "delta_h_amplitude_x0.75": to_h(L / AMPLITUDE_SENSITIVITY[0]),
        "delta_h_amplitude_x1.25": to_h(L / AMPLITUDE_SENSITIVITY[1]),
        "share_boot_L_ge_2": float(np.mean(joint_L >= 2)),
        "identifiability": status,
        "readings": readings(fit["alpha"], fit["beta"], to_h(L)),
        "r2_plane": float(fit["r2_plane"]), "r2_null_expected": 2.0 / basis.k,
        "r2_plane_perm_median": float(np.median(pf["r2_plane"])),
        "p_perm_L": p_L, "p_perm_r2": p_r2, "n_perm": int(len(perms)), "perm_exact": exact,
        "noise_sd": float(np.median(rsd)),
    }


# --------------------------------------------------------------------------- (B) exploratory

def absolute_phase(Y: np.ndarray, basis: ClockBasis) -> dict[str, np.ndarray]:
    """Per-sample absolute clock time by template matching (exploratory; WS-09 plan §1 (B)).

    Needs the reference baselines ``basis.mesor`` on the same log2(count + 1) scale. For each
    sample, y = x − mesor is fitted as y ≈ k + u·a + v·b by least squares:
    - k absorbs a library-wide offset between this dataset's normalization and the reference's;
    - (u, v) is the sample's point in the clock plane;
    - phase = atan2(v, u)/ω, the time t that best matches cos(ω(t − φ_g)) across genes;
    - scale c = √(u² + v²), the amplitude relative to the reference;
    - template r is the correlation across genes between y and u·a + v·b.

    *Alternative considered:* a grid search for the t maximizing corr(y, cos(ω(t − φ))). It
    gives nearly the same answer, but least squares is exact and doesn't weight genes by a
    grid's resolution.

    *Weakness:* any gene-specific offset between the reference's pipeline (mm9, HTSeq) and the
    site's (GRCm39, RSEM) biases the phase. That can't be checked here, which is why this is
    exploratory. Returns per-sample arrays: **site-internal only, never sent**.
    """
    if basis.mesor is None:
        raise ValueError("absolute phase needs reference mesors")
    y = np.atleast_2d(Y) - basis.mesor
    B = np.column_stack([np.ones(basis.k), basis.a, basis.b])
    coef, *_ = np.linalg.lstsq(B, y.T, rcond=None)
    k, u, v = coef
    fitted = np.outer(u, basis.a) + np.outer(v, basis.b)
    yc = y - y.mean(1, keepdims=True)
    fc = fitted - fitted.mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (yc * fc).sum(1) / np.sqrt((yc**2).sum(1) * (fc**2).sum(1))
    return {"phase": np.mod(np.arctan2(v, u) / OMEGA, PERIOD), "scale": np.hypot(u, v),
            "offset": k, "template_r": r}


def circular_mean(phases: np.ndarray) -> tuple[float, float]:
    """(circular mean in hours, mean resultant length R in [0, 1]). R near 1 means the
    phases are tightly clustered; near 0, spread around the clock."""
    z = np.exp(1j * OMEGA * np.asarray(phases)).mean()
    return float(np.mod(np.angle(z) / OMEGA, PERIOD)), float(abs(z))


# --------------------------------------------------------------------------- the boundary

SUMMARY_FIELDS = ("client", "site", "stratum", "n_flight", "n_ground", "design",
                  "panel_used", "panel_dropped", "result", "settings", "versions")
RESULT_FIELDS = ("k", "alpha", "beta", "chord_cov_sample", "L", "L_ci", "L_ci_sample_only",
                 "delta_h", "delta_h_ci", "delta_h_ci_sample_only", "delta_h_amplitude_x0.75",
                 "delta_h_amplitude_x1.25", "share_boot_L_ge_2", "identifiability", "readings",
                 "r2_plane", "r2_null_expected", "r2_plane_perm_median", "p_perm_L",
                 "p_perm_r2", "n_perm", "perm_exact", "noise_sd")
READING_FIELDS = ("direction", "mid_ct", "flight_ct", "ground_ct", "light_phase_consistent")
ABSOLUTE_SUMMARY_FIELDS = ("client", "stage", "k", "arms", "settings")
ABSOLUTE_ARM_FIELDS = ("arm", "n", "circ_mean_ct", "resultant_length", "median_template_r",
                       "median_scale")
STATUSES = ("identifiable", "ci_upper_not_identifiable", "not_identifiable")


def versions() -> dict[str, str]:
    return {p: version(p) for p in ("numpy", "pandas", "scipy")}


def site_clock_summary(site: SiteData, gene_map: GeneMap, basis: ClockBasis,
                       ref_bases: list[ClockBasis] | None = None, seed: int = 0,
                       n_boot: int = N_BOOT) -> dict:
    """Run the diagnostic on one site's own data and return the validated boundary object."""
    Xp, used, drops = panel_matrix(site, gene_map, basis)
    if used.k < 3:
        raise ValueError(f"{site.client}: only {used.k} panel genes usable; need ≥ 3")
    res = analyse(Xp.to_numpy(), site.metadata, site.manifest["design"], used, ref_bases,
                  seed, n_boot)
    m = site.manifest
    summary = {
        "client": site.client, "site": m["site"], "stratum": m["stratum"],
        "n_flight": int(site.provenance["n_flight"]),
        "n_ground": int(site.provenance["n_ground"]), "design": m["design"],
        "panel_used": [{"ncbi_gene": int(g), "symbol": s}
                       for g, s in zip(used.ncbi_gene, used.symbol, strict=True)],
        "panel_dropped": drops, "result": res, "settings": dict(SETTINGS) | {"seed": seed},
        "versions": versions(),
    }
    validate_clock_summary(summary)
    return summary


def _num(x, name, allow_none=False):
    if x is None and allow_none:
        return
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f"{name} must be a finite number, got {x!r}")


def validate_clock_summary(s: dict) -> None:
    """Raise ValueError unless ``s`` is exactly the site → server clock summary."""
    if set(s) != set(SUMMARY_FIELDS):
        raise ValueError(f"summary fields wrong; extra {sorted(set(s) - set(SUMMARY_FIELDS))}, "
                         f"missing {sorted(set(SUMMARY_FIELDS) - set(s))}")
    for f in ("n_flight", "n_ground"):
        if not isinstance(s[f], int) or isinstance(s[f], bool) or s[f] < 1:
            raise ValueError(f"{f} must be a positive int")
    for p in s["panel_used"]:
        if set(p) != {"ncbi_gene", "symbol"} or not isinstance(p["ncbi_gene"], int):
            raise ValueError("panel_used entries are {ncbi_gene:int, symbol}")
    for p in s["panel_dropped"]:
        if set(p) != {"ncbi_gene", "symbol", "reason"}:
            raise ValueError("panel_dropped entries are {ncbi_gene, symbol, reason}")
    r = s["result"]
    if set(r) != set(RESULT_FIELDS):
        raise ValueError(f"result fields wrong; extra {sorted(set(r) - set(RESULT_FIELDS))}, "
                         f"missing {sorted(set(RESULT_FIELDS) - set(r))}")
    for f in ("alpha", "beta", "L", "r2_plane", "r2_null_expected", "r2_plane_perm_median",
              "p_perm_L", "p_perm_r2", "noise_sd", "share_boot_L_ge_2"):
        _num(r[f], f)
    for f in ("delta_h", "delta_h_amplitude_x0.75", "delta_h_amplitude_x1.25"):
        _num(r[f], f, allow_none=True)
    for f in ("L_ci", "L_ci_sample_only", "delta_h_ci", "delta_h_ci_sample_only"):
        if not isinstance(r[f], list) or len(r[f]) != 2:
            raise ValueError(f"{f} must be a 2-item list")
        for x in r[f]:
            _num(x, f, allow_none=f.startswith("delta"))
    cov = r["chord_cov_sample"]
    if not (isinstance(cov, list) and len(cov) == 2 and all(len(row) == 2 for row in cov)):
        raise ValueError("chord_cov_sample must be 2 × 2")
    if r["identifiability"] not in STATUSES:
        raise ValueError(f"identifiability must be one of {STATUSES}")
    if len(r["readings"]) != 2 or any(set(x) != set(READING_FIELDS) for x in r["readings"]):
        raise ValueError(f"readings: two dicts with keys {READING_FIELDS}")
    if r["k"] != len(s["panel_used"]):
        raise ValueError("k must equal the number of panel genes used")
    json.dumps(s)  # plain JSON only: no arrays smuggled in


# --------------------------------------------------------------------------- (B) boundary

def site_absolute_summary(site: SiteData, gene_map: GeneMap, basis: ClockBasis,
                          stage: str) -> dict:
    """(B), exploratory: per-arm aggregates of each animal's absolute clock time.

    Two stages, so flight-arm values exist only if the ground arms pass the plausibility rule
    (approved WS-09 addition 2):
    - ``stage="ground"`` sends the ground arm only;
    - ``stage="both"`` sends both arms, and is run only after every site's ground arm landed in
      the lights-on window.

    Per arm, the summary carries:
    - n;
    - circular mean time (reference CT);
    - mean resultant length R (1 = all animals at one time; 0 = spread round the clock);
    - median template correlation r (how clock-like each animal's panel is);
    - median amplitude scale c relative to the reference.

    Per-sample phases stay inside this function.
    """
    if stage not in ("ground", "both"):
        raise ValueError("stage is 'ground' or 'both'")
    Xp, used, _ = panel_matrix(site, gene_map, basis)
    est = absolute_phase(Xp.to_numpy(), used)
    arms = []
    for arm in (("ground",) if stage == "ground" else ("ground", "flight")):
        sel = (site.metadata["condition"] == arm).to_numpy()
        mean, R = circular_mean(est["phase"][sel])
        arms.append({"arm": arm, "n": int(sel.sum()), "circ_mean_ct": round(mean, 3),
                     "resultant_length": round(R, 4),
                     "median_template_r": round(float(np.median(est["template_r"][sel])), 4),
                     "median_scale": round(float(np.median(est["scale"][sel])), 4)})
    out = {"client": site.client, "stage": stage, "k": int(used.k), "arms": arms,
           "settings": {"estimator": "least squares x - mesor ~ k + u*a + v*b per animal; "
                                     "phase = atan2(v, u)/w",
                        "mesor_source": "GSE54651 log2(DESeq2-normalized + 1)",
                        "ground_plausible_window_ct": list(LIGHT_WINDOW)}}
    validate_absolute_summary(out)
    return out


def validate_absolute_summary(s: dict) -> None:
    if set(s) != set(ABSOLUTE_SUMMARY_FIELDS):
        raise ValueError(f"absolute summary fields must be exactly {ABSOLUTE_SUMMARY_FIELDS}")
    want = ["ground"] if s["stage"] == "ground" else ["ground", "flight"]
    if [a.get("arm") for a in s["arms"]] != want:
        raise ValueError(f"stage {s['stage']!r} sends arms {want}")
    for a in s["arms"]:
        if set(a) != set(ABSOLUTE_ARM_FIELDS):
            raise ValueError(f"absolute arm entries have keys {ABSOLUTE_ARM_FIELDS}")
        for f in ("circ_mean_ct", "resultant_length", "median_template_r", "median_scale"):
            _num(a[f], f)
    json.dumps(s)
