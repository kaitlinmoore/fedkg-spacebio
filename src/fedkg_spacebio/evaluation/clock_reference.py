"""Q-11 reference clock: peak times and amplitudes of liver clock genes, and the "clock plane".

Why a reference is needed
-------------------------
No site recorded the time of day of euthanasia. To ask whether flight and ground animals were
at different points of the 24 h cycle, we need to know how each clock gene's expression moves
over the day in mouse liver. A public time course with known sampling times supplies that.

Two public series, two jobs
---------------------------
- **GSE11923** (Hughes et al. 2009; Affymetrix Mouse 430 2.0, GPL1261). Liver every hour for
  48 h, CT18–CT65, in constant darkness after LD 12:12 entrainment. Adult male C57BL/6J; each
  array is a pool of 3–5 mice. Values are MAS5 signals (linear scale), so they're log2'd
  before fitting. Hourly sampling over two cycles gives **precise peak times (φ_g)**.
- **GSE54651** (Zhang et al. 2014). RNA-seq of liver every 6 h for 48 h (CT22–CT64), constant
  darkness, C57BL/6, one sample per time point. Values are DESeq2-normalized gene counts. It
  gives **amplitudes on the RNA-seq log scale (A_g)** and gene baselines (mesors).
  - Why not use the array's amplitudes? Microarrays compress fold changes relative to RNA-seq
    (background and saturation at the ends of the intensity range). The shift in hours is read
    off the amplitude scale (``federation/site_clock.py``), so it has to be on the sites' own
    scale.

Both series were sampled in constant darkness, so times are **circadian time (CT)**: CT0 is
subjective dawn, extrapolated from the last light-dark cycle. The sites ran LD 12:12, which is
**zeitgeber time (ZT)**, with ZT0 = lights on. CT ≈ ZT for the first days in darkness. Over
~2 days the free-running clock drifts by less than 1 h (C57BL/6 period ~23.7 h). That gap is
stated as a caveat, not corrected.

The cosinor
-----------
For one gene, with t in hours and ω = 2π/24:

    y(t) = m + a·cos(ωt) + b·sin(ωt) + ε

This is linear in (m, a, b), so ordinary least squares fits it exactly, with no iterations.
Amplitude A = √(a² + b²), peak time φ = atan2(b, a)/ω (mod 24), since a·cos ωt + b·sin ωt =
A·cos(ω(t − φ)). The F test of (a, b) = (0, 0) against a constant has (2, n − 3) degrees of
freedom. R² is the share of variance the 24 h cosine explains.
- *Alternatives considered:* JTK_CYCLE and RAIN (non-parametric rhythm tests) and COSOPT (used
  by Hughes 2009 for Table S1). They test for rhythmicity well but don't give the (a, b) pair
  in log units that the clock plane needs. The cosinor gives it directly, and for strong
  sinusoidal clock genes it's the natural model.

The clock plane
---------------
Write a_g = A_g cos(ωφ_g) and b_g = A_g sin(ωφ_g) for each panel gene g. A healthy clock at
time t then gives, across the panel,

    x_g(t) − m_g = cos(ωt)·a_g + sin(ωt)·b_g

so every clock state is a point (cos ωt, sin ωt) on the unit circle in the plane spanned by the
vectors **a** and **b**. ``ClockBasis`` holds them. The site code projects expression
differences onto this plane (``federation/site_clock.py``).

The panel rule (fixed before any site data was looked at; WS-09 plan §3)
-----------------------------------------------------------------------
Candidates: Bmal1 (Arntl), Npas2, Clock, Nfil3, Cry1, Per1–3, Dbp, Tef, Hlf, Nr1d1/2,
Bhlhe40/41, Ciart. A gene is kept if:
1. on GSE11923, its best probe set has cosinor BH q < 0.001 (BH over all probe sets on the
   array), R² ≥ 0.5 and peak-to-trough ≥ 1.5-fold (2A ≥ log2 1.5);
2. if several of its probe sets pass, they agree on peak time within 2 h; the one with the
   highest R² is used;
3. on GSE54651, it's rhythmic (cosinor p < 0.05) with a peak within 3 h of GSE11923's.

Every drop is recorded with its reason.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from fedkg_spacebio.federation import meta

PERIOD = 24.0
OMEGA = 2 * np.pi / PERIOD

PANEL = ("Bmal1/Arntl", "Npas2", "Clock", "Nfil3", "Cry1", "Per1", "Per2", "Per3", "Dbp", "Tef",
         "Hlf", "Nr1d1", "Nr1d2", "Bhlhe40", "Bhlhe41", "Ciart")

PANEL_RULE = {"array_bh_q_max": 1e-3, "array_r2_min": 0.5,
              "array_peak_to_trough_min_fold": 1.5, "probe_sets_agree_within_h": 2.0,
              "rnaseq_p_max": 0.05, "rnaseq_peak_within_h": 3.0}


# --------------------------------------------------------------------------- reading GEO files

def read_series_matrix(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A GEO series matrix: (probe × sample values, per-sample header fields).

    The header lines start with ``!Sample_``; the table sits between
    ``!series_matrix_table_begin`` and ``!series_matrix_table_end``.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    head: dict[str, list[str]] = {}
    for ln in lines:
        if ln.startswith("!Sample_"):
            key, *vals = ln.split("\t")
            key = key[len("!Sample_"):]
            if key in head:  # repeated keys (e.g. several characteristics lines)
                i = 2
                while f"{key}_{i}" in head:
                    i += 1
                key = f"{key}_{i}"
            head[key] = [v.strip('"') for v in vals]
    start = lines.index("!series_matrix_table_begin") + 1
    end = lines.index("!series_matrix_table_end")
    header = [c.strip('"') for c in lines[start].split("\t")]
    rows = [ln.split("\t") for ln in lines[start + 1:end]]
    table = pd.DataFrame(rows, columns=header)
    ids = table.iloc[:, 0].str.strip('"')
    values = table.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    values.index = pd.Index(ids, name="probe_set")
    samples = pd.DataFrame(head)
    samples.index = pd.Index(samples["geo_accession"], name="gsm")
    return values, samples


_CT = re.compile(r"CT\s*(\d+(?:\.\d+)?)", re.I)


def ct_from_titles(titles) -> np.ndarray:
    """Circadian time in hours from sample titles such as ``Liver_CT18`` or ``Liv_CT22``."""
    out = []
    for t in titles:
        m = _CT.findall(str(t))
        if not m:
            raise ValueError(f"no CT in sample title {t!r}")
        out.append(float(m[-1]))
    return np.asarray(out)


def read_zhang_quant(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """GSE54651 ``*_quant.txt.gz``: Ensembl gene × sample (DESeq2-normalized), and CT hours."""
    df = pd.read_csv(path, sep="\t", index_col=0)
    df.index.name = "ensembl"
    return df, ct_from_titles(df.columns)


# --------------------------------------------------------------------------- cosinor

def cosinor(y: np.ndarray, t: np.ndarray, period: float = PERIOD) -> pd.DataFrame:
    """24 h cosinor fitted by OLS to every row of ``y`` (features × time points).

    Rows with any missing value are fitted on their observed points. Returns mesor, a (cos
    coefficient), b (sin coefficient), amplitude, peak (hours, [0, period)), r2, F, p, n, and
    resid_sd.
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    t = np.asarray(t, dtype=float)
    w = 2 * np.pi / period
    X = np.column_stack([np.ones_like(t), np.cos(w * t), np.sin(w * t)])
    out = np.full((y.shape[0], 8), np.nan)
    complete = np.all(np.isfinite(y), axis=1)
    groups = [(complete, np.ones(len(t), bool))]
    for i in np.flatnonzero(~complete):
        groups.append((np.eye(1, y.shape[0], i, dtype=bool)[0], np.isfinite(y[i])))
    for rows, cols in groups:
        if not rows.any() or cols.sum() < 4:
            continue
        Xs, Y = X[cols], y[rows][:, cols].T  # time × features
        coef, *_ = np.linalg.lstsq(Xs, Y, rcond=None)
        resid = Y - Xs @ coef
        n = cols.sum()
        rss = (resid**2).sum(0)
        tss = ((Y - Y.mean(0))**2).sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r2 = 1 - rss / tss
            f = ((tss - rss) / 2) / (rss / (n - 3))
        p = stats.f.sf(f, 2, n - 3)
        out[rows] = np.column_stack([coef.T, r2, f, p, np.full(rows.sum(), n),
                                     np.sqrt(rss / (n - 3))])
    df = pd.DataFrame(out, columns=["mesor", "a", "b", "r2", "F", "p", "n", "resid_sd"])
    df["amplitude"] = np.hypot(df["a"], df["b"])
    df["peak"] = np.mod(np.arctan2(df["b"], df["a"]) / w, period)
    return df[["mesor", "a", "b", "amplitude", "peak", "r2", "F", "p", "n", "resid_sd"]]


def circ_diff(x, y, period: float = PERIOD):
    """Signed circular difference x − y in (−period/2, period/2]."""
    d = np.mod(np.asarray(x, float) - np.asarray(y, float) + period / 2, period) - period / 2
    return np.where(d == -period / 2, period / 2, d)


# --------------------------------------------------------------------------- panel selection

def select_panel(array_fit: pd.DataFrame, probe_gene: pd.Series, panel_ids: pd.DataFrame,
                 rnaseq_fit: pd.DataFrame | None, rule: dict = PANEL_RULE
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the pre-specified panel rule (module docstring).

    ``array_fit``: ``cosinor`` per probe set (index probe set ID) with a ``q`` column (BH over
    all probe sets). ``probe_gene``: probe set → NCBI Gene (unique mappings only).
    ``panel_ids``: ``symbol``, ``ncbi_gene`` (and ``status``) per candidate gene.
    ``rnaseq_fit``: ``cosinor`` per NCBI Gene on GSE54651 (index ncbi_gene), or None to skip
    rule 3.

    Returns (kept, decisions). ``kept`` has one row per kept gene: symbol, ncbi_gene, probe
    set, peak (array), amplitude (array), r2, q, and the RNA-seq amplitude, peak, mesor and p.
    ``decisions`` has every candidate with ``kept`` and ``reason``.
    """
    min_amp = np.log2(rule["array_peak_to_trough_min_fold"]) / 2
    rows, kept = [], []
    for _, g in panel_ids.iterrows():
        rec = {"symbol": g["symbol"], "ncbi_gene": g["ncbi_gene"], "kept": False, "reason": ""}
        if g.get("status", "ok") != "ok" or pd.isna(g["ncbi_gene"]):
            rec["reason"] = f"ID not resolved ({g.get('status')})"
            rows.append(rec)
            continue
        gid = int(g["ncbi_gene"])
        probes = probe_gene.index[probe_gene.to_numpy() == gid]
        fits = array_fit.reindex(probes).dropna(subset=["r2"])
        rec["n_probe_sets"] = int(len(fits))
        if fits.empty:
            rec["reason"] = "no probe set on GPL1261 with a unique Entrez ID"
            rows.append(rec)
            continue
        ok = fits[(fits["q"] < rule["array_bh_q_max"]) & (fits["r2"] >= rule["array_r2_min"])
                  & (fits["amplitude"] >= min_amp)]
        best_any = fits.sort_values("r2", ascending=False).iloc[0]
        rec.update({"best_probe": best_any.name, "array_r2": float(best_any["r2"]),
                    "array_q": float(best_any["q"]),
                    "array_peak_to_trough_fold": float(2 ** (2 * best_any["amplitude"])),
                    "array_peak": float(best_any["peak"])})
        if ok.empty:
            fails = []
            if best_any["q"] >= rule["array_bh_q_max"]:
                fails.append(f"q {best_any['q']:.2g}")
            if best_any["r2"] < rule["array_r2_min"]:
                fails.append(f"R² {best_any['r2']:.2f}")
            if best_any["amplitude"] < min_amp:
                fails.append(f"peak/trough {2 ** (2 * best_any['amplitude']):.2f}-fold")
            rec["reason"] = "not rhythmic enough on GSE11923: " + ", ".join(fails)
            rows.append(rec)
            continue
        spread = np.abs(circ_diff(ok["peak"].to_numpy()[:, None], ok["peak"].to_numpy()[None]))
        if spread.max() > rule["probe_sets_agree_within_h"]:
            rec["reason"] = (f"passing probe sets disagree on peak by {spread.max():.1f} h")
            rows.append(rec)
            continue
        best = ok.sort_values("r2", ascending=False).iloc[0]
        rec.update({"best_probe": best.name, "array_r2": float(best["r2"]),
                    "array_q": float(best["q"]),
                    "array_peak_to_trough_fold": float(2 ** (2 * best["amplitude"])),
                    "array_peak": float(best["peak"])})
        out = {"symbol": g["symbol"], "ncbi_gene": gid, "probe_set": best.name,
               "peak": float(best["peak"]), "amplitude_array": float(best["amplitude"]),
               "mesor_array": float(best["mesor"]), "r2_array": float(best["r2"]),
               "q_array": float(best["q"])}
        if rnaseq_fit is not None:
            if gid not in rnaseq_fit.index or not np.isfinite(rnaseq_fit.loc[gid, "p"]):
                rec["reason"] = "not measured in GSE54651 liver (after 1:1 Ensembl→NCBI map)"
                rows.append(rec)
                continue
            r = rnaseq_fit.loc[gid]
            dphi = float(abs(circ_diff(r["peak"], best["peak"])))
            rec.update({"rnaseq_p": float(r["p"]), "rnaseq_peak": float(r["peak"]),
                        "peak_diff_h": dphi})
            if r["p"] >= rule["rnaseq_p_max"]:
                rec["reason"] = f"not rhythmic on GSE54651 (p {r['p']:.2g})"
                rows.append(rec)
                continue
            if dphi > rule["rnaseq_peak_within_h"]:
                rec["reason"] = f"GSE54651 peak {dphi:.1f} h from GSE11923's"
                rows.append(rec)
                continue
            out.update({"amplitude_rnaseq": float(r["amplitude"]),
                        "mesor_rnaseq": float(r["mesor"]), "peak_rnaseq": float(r["peak"]),
                        "p_rnaseq": float(r["p"]), "resid_sd_rnaseq": float(r["resid_sd"])})
        rec.update({"kept": True, "reason": "kept"})
        rows.append(rec)
        kept.append(out)
    return pd.DataFrame(kept), pd.DataFrame(rows)


def array_fit_with_q(values_log2: pd.DataFrame, ct: np.ndarray) -> pd.DataFrame:
    """Cosinor on every probe set, with BH q over all probe sets with a finite p."""
    fit = cosinor(values_log2.to_numpy(), ct)
    fit.index = values_log2.index
    q = np.full(len(fit), np.nan)
    ok = np.isfinite(fit["p"].to_numpy())
    q[ok] = meta.bh_qvalues(fit["p"].to_numpy()[ok])
    fit["q"] = q
    return fit


def log2_mas5(values: pd.DataFrame, floor: float = 1.0) -> pd.DataFrame:
    """log2 of MAS5 signal, floored at ``floor`` (MAS5 signals can be < 1 for absent probes)."""
    return np.log2(values.clip(lower=floor))


def log2_counts(values: pd.DataFrame) -> pd.DataFrame:
    """log2(normalized count + 1): the same transform the sites use (``site_clock``)."""
    return np.log2(values + 1.0)


# --------------------------------------------------------------------------- the basis

@dataclass(frozen=True)
class ClockBasis:
    """The clock plane for one panel: per-gene peak (h), amplitude and baseline (log2).

    ``a = amplitude·cos(ω·peak)`` and ``b = amplitude·sin(ω·peak)`` span the plane. ``mesor``
    is used only by the exploratory absolute-phase estimate.
    """

    ncbi_gene: tuple[int, ...]
    symbol: tuple[str, ...]
    peak: np.ndarray
    amplitude: np.ndarray
    mesor: np.ndarray | None = None
    label: str = ""

    @property
    def a(self) -> np.ndarray:
        return self.amplitude * np.cos(OMEGA * self.peak)

    @property
    def b(self) -> np.ndarray:
        return self.amplitude * np.sin(OMEGA * self.peak)

    @property
    def k(self) -> int:
        return len(self.ncbi_gene)

    def subset(self, genes) -> ClockBasis:
        idx = [self.ncbi_gene.index(int(g)) for g in genes]
        return ClockBasis(tuple(self.ncbi_gene[i] for i in idx),
                          tuple(self.symbol[i] for i in idx), self.peak[idx],
                          self.amplitude[idx], None if self.mesor is None else self.mesor[idx],
                          self.label)

    def scaled(self, s: float) -> ClockBasis:
        return ClockBasis(self.ncbi_gene, self.symbol, self.peak, self.amplitude * s,
                          self.mesor, f"{self.label} ×{s:g}")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"ncbi_gene": self.ncbi_gene, "symbol": self.symbol,
                             "peak": self.peak, "amplitude": self.amplitude,
                             **({"mesor": self.mesor} if self.mesor is not None else {})})

    @classmethod
    def from_frame(cls, df: pd.DataFrame, label: str = "") -> ClockBasis:
        return cls(tuple(int(x) for x in df["ncbi_gene"]), tuple(str(s) for s in df["symbol"]),
                   df["peak"].to_numpy(float), df["amplitude"].to_numpy(float),
                   df["mesor"].to_numpy(float) if "mesor" in df else None, label)


def basis_from_panel(kept: pd.DataFrame, amplitude: str = "rnaseq") -> ClockBasis:
    """Peaks from GSE11923; amplitudes (and mesors) from GSE54651 (``rnaseq``) or the array."""
    amp = kept[f"amplitude_{amplitude}"].to_numpy(float)
    mes = kept[f"mesor_{amplitude}"].to_numpy(float)
    return ClockBasis(tuple(int(x) for x in kept["ncbi_gene"]), tuple(kept["symbol"]),
                      kept["peak"].to_numpy(float), amp, mes,
                      f"peaks GSE11923, amplitudes {amplitude}")


def bootstrap_bases(kept: pd.DataFrame, array_log2: pd.DataFrame, ct_array: np.ndarray,
                    rnaseq_log2: pd.DataFrame, ct_rnaseq: np.ndarray, n_boot: int,
                    seed: int) -> list[ClockBasis]:
    """Reference uncertainty: residual bootstrap of both cosinor fits, per gene.

    For each replicate, every panel gene's array series (its chosen probe set) and RNA-seq
    series are rebuilt as fitted value + resampled residual, and refitted. Peaks come from the
    array refit, amplitudes and mesors from the RNA-seq refit. With 8 RNA-seq points this is
    rough (5 residual df), and it is the larger source of uncertainty; it's reported as such.

    ``array_log2`` rows: the panel's probe sets; ``rnaseq_log2`` rows: the panel's NCBI IDs.
    """
    rng = np.random.default_rng(seed)

    def fitted_and_resid(Y, t):
        X = np.column_stack([np.ones_like(t), np.cos(OMEGA * t), np.sin(OMEGA * t)])
        coef, *_ = np.linalg.lstsq(X, Y.T, rcond=None)
        fit = (X @ coef).T
        # Scale residuals by sqrt(n/(n-p)) so their variance isn't biased low (3 parameters).
        n = len(t)
        return fit, (Y - fit) * np.sqrt(n / (n - 3))

    ya = array_log2.loc[list(kept["probe_set"])].to_numpy(float)
    yr = rnaseq_log2.loc[list(kept["ncbi_gene"])].to_numpy(float)
    fa, ra = fitted_and_resid(ya, ct_array)
    fr, rr = fitted_and_resid(yr, ct_rnaseq)
    out = []
    for _ in range(n_boot):
        ia = rng.integers(0, ra.shape[1], ra.shape)
        ir = rng.integers(0, rr.shape[1], rr.shape)
        ca = cosinor(fa + np.take_along_axis(ra, ia, 1), ct_array)
        cr = cosinor(fr + np.take_along_axis(rr, ir, 1), ct_rnaseq)
        out.append(ClockBasis(tuple(int(x) for x in kept["ncbi_gene"]), tuple(kept["symbol"]),
                              ca["peak"].to_numpy(), cr["amplitude"].to_numpy(),
                              cr["mesor"].to_numpy(), "bootstrap"))
    return out
