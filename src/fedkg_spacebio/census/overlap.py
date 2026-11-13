"""Do the same animals, or the same sequencing libraries, appear in more than one study? (WS-06)

AI-drafted (Claude, WS-06).

Why it matters: if one mouse is counted in two studies, a meta-analysis counts its evidence twice
and leave-one-site-out can leak it from a held-out site into training. So we check two different
things, independently, and report both.

1. Animal identity (metadata)
-----------------------------
Two samples come from the same animal when their **source names** match after normalization
(``samples.normalize_source``: upper-case, letters and digits only) **and** strain and sex agree.
The extra condition matters because numbering restarts per study: ``M1`` in one cohort isn't ``M1``
in another, but a C57BL/6J ``RR1_FLT_M21`` in two studies of the same mission almost certainly is.
Name matches that fail the strain/sex condition are reported as collisions, not overlaps.

2. Library identity (data)
--------------------------
Two samples are the **same sequencing library**, reprocessed, when their count vectors are
near-exact copies. Kaitlin's rule (2026-11-12): identical count vectors, **or** Pearson r > 0.9999
**with equal library size**. A high correlation alone is not enough: two different mice from the
same tissue easily reach r ≈ 0.99 on log counts, and we must not call those duplicates.

- r is computed on log1p(counts) over the genes both matrices share. On raw counts, a handful of
  very highly expressed liver genes (albumin, apolipoproteins) dominate r; on the log scale every
  gene contributes, which makes r far more discriminating at the top end.
- Library size is the column total over the shared genes. "Equal" means a relative difference of
  at most ``lib_rtol`` (default 0, exact). The diagnostics table reports the best r and the
  library-size ratio for every sample, so a near miss stays visible.
- Within-study pairs are skipped: a mouse sequenced with and without spike-ins (OSD-168) has two
  different libraries by design; the animal check already catches it.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations

import numpy as np
import pandas as pd

R_THRESHOLD = 0.9999


def animal_overlap(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cross-study animal overlap from sample metadata.

    Returns:
        matrix: studies × studies; the diagonal is each study's number of unique animals, an
            off-diagonal cell the number of animals the two studies share.
        pairs: one row per shared animal (study_a, study_b, source names, strain, sex).
        collisions: same normalized source name in two studies but different strain or sex.
    """
    animals = (samples[["osd_id", "source_name", "source_key", "strain", "sex"]]
               .drop_duplicates(["osd_id", "source_key", "strain", "sex"]))
    studies = sorted(animals["osd_id"].unique(), key=_osd_sort_key)
    matrix = pd.DataFrame(0, index=studies, columns=studies, dtype=int)
    for osd, g in animals.groupby("osd_id"):
        matrix.loc[osd, osd] = g["source_key"].nunique()

    pairs, collisions = [], []
    for a, b in combinations(studies, 2):
        A = animals[animals["osd_id"] == a]
        B = animals[animals["osd_id"] == b]
        joined = A.merge(B, on="source_key", suffixes=("_a", "_b"))
        for _, r in joined.iterrows():
            same = r["strain_a"] == r["strain_b"] and r["sex_a"] == r["sex_b"]
            rec = {"study_a": a, "study_b": b, "source_a": r["source_name_a"],
                   "source_b": r["source_name_b"], "strain_a": r["strain_a"],
                   "strain_b": r["strain_b"], "sex_a": r["sex_a"], "sex_b": r["sex_b"]}
            (pairs if same else collisions).append(rec)
        n = sum(1 for p in pairs if p["study_a"] == a and p["study_b"] == b)
        matrix.loc[a, b] = matrix.loc[b, a] = n
    return matrix, pd.DataFrame(pairs), pd.DataFrame(collisions)


def within_study_duplicates(samples: pd.DataFrame) -> pd.DataFrame:
    """Animals that gave more than one sample in the same study, and which factors differ."""
    rows = []
    for (osd, _key), g in samples.groupby(["osd_id", "source_key"]):
        if len(g) < 2:
            continue
        factor_cols = [c for c in g.columns if c.startswith("factor:")]
        varying = [c.removeprefix("factor:") for c in factor_cols
                   if g[c].astype(str).nunique(dropna=False) > 1]
        rows.append({"osd_id": osd, "source_name": g["source_name"].iloc[0],
                     "n_samples": len(g), "samples": sorted(g["sample_name"]),
                     "factors_that_differ": varying or ["none (technical replicate)"]})
    return pd.DataFrame(rows)


def library_overlap(counts: Mapping[str, pd.DataFrame], r_threshold: float = R_THRESHOLD,
                    lib_rtol: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-study near-exact library matches from count matrices (genes × samples).

    Returns:
        matches: sample pairs that are identical, or have r > ``r_threshold`` and equal library
            size (relative difference ≤ ``lib_rtol``), with r and both library sizes.
        best: for every sample, its highest-r partner in each other study and that pair's
            library-size ratio (diagnostics; not a finding by itself).
    """
    matches, best = [], []
    studies = sorted(counts, key=_osd_sort_key)
    for a, b in combinations(studies, 2):
        genes = counts[a].index.intersection(counts[b].index)
        if len(genes) == 0:
            continue
        A = counts[a].loc[genes].to_numpy(dtype=np.float64)
        B = counts[b].loc[genes].to_numpy(dtype=np.float64)
        lib_a, lib_b = A.sum(axis=0), B.sum(axis=0)
        r = _pearson_columns(np.log1p(A), np.log1p(B))  # shape (n_a, n_b)
        for i, sa in enumerate(counts[a].columns):
            j = int(np.nanargmax(r[i])) if np.isfinite(r[i]).any() else 0
            best.append({"study": a, "sample": sa, "other_study": b,
                         "best_partner": counts[b].columns[j], "r_log1p": float(r[i, j]),
                         "lib_ratio": float(lib_a[i] / lib_b[j]) if lib_b[j] else np.nan})
        for j, sb in enumerate(counts[b].columns):
            i = int(np.nanargmax(r[:, j])) if np.isfinite(r[:, j]).any() else 0
            best.append({"study": b, "sample": sb, "other_study": a,
                         "best_partner": counts[a].columns[i], "r_log1p": float(r[i, j]),
                         "lib_ratio": float(lib_b[j] / lib_a[i]) if lib_a[i] else np.nan})
        # Identical non-constant vectors have r = 1, so every candidate passes r > threshold first.
        for i, j in zip(*np.nonzero(r > r_threshold), strict=True):
            identical = bool(np.array_equal(A[:, i], B[:, j]))
            equal_lib = _rel_diff(lib_a[i], lib_b[j]) <= lib_rtol
            if identical or equal_lib:
                matches.append({"study_a": a, "sample_a": counts[a].columns[i],
                                "study_b": b, "sample_b": counts[b].columns[j],
                                "r_log1p": float(r[i, j]), "identical": identical,
                                "lib_a": float(lib_a[i]), "lib_b": float(lib_b[j]),
                                "n_genes": len(genes)})
        # Identical vectors with zero variance would give r = nan; catch them explicitly.
        for i in range(A.shape[1]):
            for j in range(B.shape[1]):
                if not np.isfinite(r[i, j]) and np.array_equal(A[:, i], B[:, j]):
                    matches.append({"study_a": a, "sample_a": counts[a].columns[i],
                                    "study_b": b, "sample_b": counts[b].columns[j],
                                    "r_log1p": float("nan"), "identical": True,
                                    "lib_a": float(lib_a[i]), "lib_b": float(lib_b[j]),
                                    "n_genes": len(genes)})
    return pd.DataFrame(matches), pd.DataFrame(best)


def _pearson_columns(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pearson r between every column of X and every column of Y (genes are rows).

    Centre each column, scale it to unit length, and r is a dot product: r = x̂ᵀŷ. A constant
    column has no correlation; it comes out as nan rather than dividing by zero.
    """
    Xc, Yc = X - X.mean(axis=0), Y - Y.mean(axis=0)
    nx, ny = np.linalg.norm(Xc, axis=0), np.linalg.norm(Yc, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (Xc / nx).T @ (Yc / ny)


def _rel_diff(x: float, y: float) -> float:
    denom = max(abs(x), abs(y))
    return 0.0 if denom == 0 else abs(x - y) / denom


def _osd_sort_key(osd: str) -> tuple:
    num = osd.rpartition("-")[2]
    return (0, int(num), osd) if num.isdigit() else (1, 0, osd)
