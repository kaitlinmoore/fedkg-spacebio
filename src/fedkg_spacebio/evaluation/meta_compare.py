"""Real-data comparison of phase 3a methods (WS-08): overlaps, ranks, flags, leave-one-out.

Pure functions on ``meta_methods.run_methods`` outputs; ``jobs/phase3a_compare/compare.py``
calls them per stratum and variant. No biological interpretation happens here.

What each comparison is for
---------------------------
- **Hit counts and Jaccard overlap:** how much the methods agree on *which* genes, not just how
  many. Jaccard = |A ∩ B| / |A ∪ B|; 1 = identical lists, 0 = disjoint.
- **Rank agreement:** Spearman correlation of signed −log10 p (the sign of the estimate times
  the strength of evidence), over all genes, plus overlap of the top N by p. Two methods can
  call different numbers of genes and still order genes the same way.
- **Where the heterogeneity flag falls:** of each method's hits, how many have Cochran's Q
  p < 0.10. A "new mission" method should rarely call flagged genes; FE calls many.
- **Leave one mission out (LOMO) sign agreement:** for each mission, rerun the method without
  it, and ask whether the left-out mission's own log2FC has the same sign as the estimate from
  the others, over the full-data hits where that mission has an estimate. 50% is chance. This
  is a weak check at k = 3–4: the left-out estimate is a single noisy mission and the remaining
  pool has only 2–3 missions.
- **Single-site influence:** with each site dropped (and BH rerun on the k − 1 sites), how many
  of the full-data hits are still hits. A method whose hits depend on one site (SpaceX-16
  carries ~40% of FE weight) loses most of them when that site goes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from fedkg_spacebio.evaluation import meta_methods
from fedkg_spacebio.federation import results


def hit_counts(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Hits per method, split by direction of the method's own estimate."""
    rows = []
    for m, f in frames.items():
        h = f["hit"].to_numpy(dtype=bool)
        rows.append({"method": m, "hits": int(h.sum()), "up": int((h & (f["est"] > 0)).sum()),
                     "down": int((h & (f["est"] < 0)).sum())})
    return pd.DataFrame(rows)


def jaccard_matrix(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pairwise Jaccard of hit sets (NaN when both sets are empty)."""
    names = list(frames)
    hits = {m: frames[m]["hit"].to_numpy(dtype=bool) for m in names}
    out = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            union = (hits[a] | hits[b]).sum()
            out.loc[a, b] = (hits[a] & hits[b]).sum() / union if union else np.nan
    return out


def signed_logp(f: pd.DataFrame) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.sign(f["est"].to_numpy()) * -np.log10(np.maximum(f["p"].to_numpy(), 1e-300))


def rank_agreement(frames: dict[str, pd.DataFrame], top_n: int = 500) -> pd.DataFrame:
    """Pairwise Spearman of signed −log10 p, and the share of the top-N genes (by p) in common.

    Methods with identical p-values (``fe`` and ``fe_flag``) correlate 1 by construction.
    """
    names = list(frames)
    sl = {m: signed_logp(frames[m]) for m in names}
    top = {m: set(np.argsort(frames[m]["p"].to_numpy(), kind="mergesort")[:top_n])
           for m in names}
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ok = np.isfinite(sl[a]) & np.isfinite(sl[b])
            rho = stats.spearmanr(sl[a][ok], sl[b][ok]).statistic if ok.sum() > 2 else np.nan
            rows.append({"a": a, "b": b, "spearman_signed_logp": float(rho),
                         f"top{top_n}_overlap": len(top[a] & top[b]) / top_n})
    return pd.DataFrame(rows)


def flag_breakdown(frames: dict[str, pd.DataFrame], q_p: np.ndarray) -> pd.DataFrame:
    """Of each method's hits, how many carry the heterogeneity flag (Q p < HET_Q_P)."""
    flag = np.asarray(q_p) < results.HET_Q_P
    rows = []
    for m, f in frames.items():
        h = f["hit"].to_numpy(dtype=bool)
        rows.append({"method": m, "hits": int(h.sum()), "flagged": int((h & flag).sum()),
                     "unflagged": int((h & ~flag).sum()),
                     "flagged_share": float((h & flag).sum() / h.sum()) if h.any() else np.nan})
    return pd.DataFrame(rows)


def leave_one_out(y, v, clients: list[str], full: dict[str, pd.DataFrame],
                  methods=meta_methods.METHODS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop each site in turn; return (lomo, influence).

    ``lomo``: one row per (method, dropped site): over the full-data hits where that site has an
    estimate, the share where sign(site's log2FC) == sign(method's estimate from the other
    sites). Also the same share over all genes with a nonzero estimate, as a baseline.

    ``influence``: one row per (method, dropped site): hits in the k − 1 analysis, and how many
    of the full-data hits survive.
    """
    Y, V = np.asarray(y, dtype=np.float64), np.asarray(v, dtype=np.float64)
    lomo, infl = [], []
    for i, c in enumerate(clients):
        keep = [j for j in range(Y.shape[1]) if j != i]
        frames, _ = meta_methods.run_methods(Y[:, keep], V[:, keep], methods)
        yi = Y[:, i]
        has = np.isfinite(yi) & (yi != 0)
        for m in methods:
            full_hit = full[m]["hit"].to_numpy(dtype=bool)
            est = frames[m]["est"].to_numpy()
            agree = np.sign(yi) == np.sign(est)
            sel = full_hit & has & np.isfinite(est) & (est != 0)
            base = has & np.isfinite(est) & (est != 0)
            lomo.append({"method": m, "dropped": c, "n_hits_scored": int(sel.sum()),
                         "sign_agreement": float(agree[sel].mean()) if sel.any() else np.nan,
                         "baseline_all_genes": float(agree[base].mean())})
            loo_hit = frames[m]["hit"].to_numpy(dtype=bool)
            infl.append({"method": m, "dropped": c, "full_hits": int(full_hit.sum()),
                         "hits_without_site": int(loo_hit.sum()),
                         "full_hits_surviving": int((full_hit & loo_hit).sum()),
                         "survival_share": float((full_hit & loo_hit).sum() / full_hit.sum())
                         if full_hit.any() else np.nan})
    return pd.DataFrame(lomo), pd.DataFrame(infl)


def per_gene_calls(ncbi_gene, symbol, frames: dict[str, pd.DataFrame], q_p) -> pd.DataFrame:
    """One row per gene: each method's estimate, p, q and hit, plus the heterogeneity flag."""
    out = pd.DataFrame({"ncbi_gene": np.asarray(ncbi_gene), "symbol": np.asarray(symbol),
                        "q_p": np.asarray(q_p),
                        "het_flag": np.asarray(q_p) < results.HET_Q_P})
    for m, f in frames.items():
        for col in ("est", "ci_low", "ci_high", "p", "q", "hit"):
            out[f"{m}__{col}"] = f[col].to_numpy()
    return out


def scaled_f_quantiles(n: int, d: float, d0: float, s02: float) -> np.ndarray:
    """Theoretical quantiles of s₀²·F(d, d₀) at plotting positions (i − 0.5)/n, for the QQ
    check of φ̂ against V2's fitted prior. d₀ = ∞ gives s₀²·χ²(d)/d."""
    pp = (np.arange(1, n + 1) - 0.5) / n
    if np.isinf(d0):
        return s02 * stats.chi2.ppf(pp, d) / d
    return s02 * stats.f.ppf(pp, d, d0)
