"""Every phase 3a pooling method behind one interface, for the WS-08 comparison.

The simulation (``meta_sim``) and the real-data comparison (``meta_compare``) both call
``run_methods``, so a method's definition (its estimate, interval, p-value and hit rule) lives
in one place and the two can't disagree.

Methods and the question each answers
-------------------------------------
| key | method | question |
|---|---|---|
| ``fe`` | inverse-variance fixed effect | these missions; assumes no scatter beyond the SEs |
| ``fe_flag`` | FE hit and Cochran's Q p ≥ 0.10 | these missions, consistent only (D-026) |
| ``hksj_trunc`` | REML + truncated HKSJ, t(k−1) | a new mission (D-025) |
| ``hksj_untrunc`` | REML + HKSJ, t(k−1) | a new mission |
| ``v1_trunc`` / ``v1_untrunc`` | the WS-08 sketch: moderated HKSJ q | a new mission (in intent) |
| ``v2`` | FE + moderated multiplicative φ | these missions, with an honest variance |
| ``v3`` | empirical-Bayes τ² | a new mission |

BH runs within each call (one stratum and variant) for every method. A hit is BH q < 0.05
(``results.FDR_Q``). For ``fe_flag`` the hit is an FE hit without the heterogeneity flag
(``results.HET_Q_P``); its estimate, interval and p are FE's.

Output: one DataFrame per method with columns ``est``, ``se`` (NaN for V3, which reports a
posterior SD in ``se`` instead), ``ci_low``, ``ci_high``, ``p``, ``q``, ``hit``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from fedkg_spacebio.federation import eb_tau2, meta, moderation, results

METHODS = ("fe", "fe_flag", "hksj_trunc", "hksj_untrunc", "v1_trunc", "v1_untrunc", "v2", "v3")

QUESTION = {
    "fe": "these missions (no excess scatter)",
    "fe_flag": "these missions, consistent only",
    "hksj_trunc": "a new mission",
    "hksj_untrunc": "a new mission",
    "v1_trunc": "a new mission (sketch)",
    "v1_untrunc": "a new mission (sketch)",
    "v2": "these missions, honest variance",
    "v3": "a new mission",
}

OUT_COLUMNS = ["est", "se", "ci_low", "ci_high", "p", "q", "hit"]


def matrices_from_table(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(Y, V, clients) from a per-gene table (schema ``phase3a-per-gene/1`` or ``/2``).

    Only the per-site columns are read, so both schema versions give the same matrices.

    Y = log2FC and V = SE², with NaN wherever ``included__<client>`` is False, so exactly the
    cells that entered WS-07's pooling enter here.
    """
    clients = [c[len("log2fc__"):] for c in table.columns if c.startswith("log2fc__")]
    inc = table[[f"included__{c}" for c in clients]].to_numpy(dtype=bool)
    y = np.where(inc, table[[f"log2fc__{c}" for c in clients]].to_numpy(dtype=np.float64), np.nan)
    se = table[[f"se__{c}" for c in clients]].to_numpy(dtype=np.float64)
    return y, np.where(inc, se**2, np.nan), clients


def _frame(est, se, lo, hi, p, hit=None) -> pd.DataFrame:
    q = meta.bh_qvalues(p)
    return pd.DataFrame({"est": est, "se": se, "ci_low": lo, "ci_high": hi, "p": p, "q": q,
                         "hit": (q < results.FDR_Q) if hit is None else hit})


def run_methods(y, v, methods=METHODS, *, level: float = 0.95, min_k: int = 2,
                v3_kwargs: dict | None = None) -> tuple[dict[str, pd.DataFrame], dict]:
    """Run the requested methods on one (stratum, variant): (per-method frames, fit info).

    Genes with k < ``min_k`` get NaN and are left out of BH (as in ``meta.pool``). ``info``
    holds each method's fitted hyperparameters (V1/V2 d₀ and s₀², V3's grid and prior), the
    REML τ², and Cochran's Q p, for the write-up and the simulation's diagnostics.
    """
    Y, V, present = meta.as_matrix(y, v)
    Y = np.where(present, Y, np.nan)
    V = np.where(present, V, np.nan)
    k = meta.n_sites(present)
    drop = k < min_k
    Y[drop], V[drop] = np.nan, np.nan
    out, info = {}, {"k": k}
    z = stats.norm.ppf(1.0 - (1.0 - level) / 2.0)

    fe = meta.fixed_effect(Y, V)
    info["q_p"] = fe["q_p"]
    if {"fe", "fe_flag"} & set(methods):
        lo, hi = fe["est"] - z * fe["se"], fe["est"] + z * fe["se"]
        base = _frame(fe["est"], fe["se"], lo, hi, fe["p"])
        if "fe" in methods:
            out["fe"] = base
        if "fe_flag" in methods:
            flagged = (fe["q_p"] < results.HET_Q_P)
            out["fe_flag"] = base.assign(hit=base["hit"].to_numpy() & ~flagged)

    needs_reml = {"hksj_trunc", "hksj_untrunc", "v1_trunc", "v1_untrunc"} & set(methods)
    if needs_reml:
        tau2, conv = meta.tau2_reml(Y, V)
        info["tau2_reml"], info["reml_converged"] = tau2, conv
        for trunc in (True, False):
            key = "hksj_trunc" if trunc else "hksj_untrunc"
            if key in methods:
                r = meta.random_effects_hksj(Y, V, tau2, truncate=trunc, level=level)
                out[key] = _frame(r["est"], r["se"], r["ci_low"], r["ci_high"], r["p"])
            key = "v1_trunc" if trunc else "v1_untrunc"
            if key in methods:
                r = moderation.moderated_hksj(Y, V, tau2, truncate=trunc, level=level)
                out[key] = _frame(r["est"], r["se"], r["ci_low"], r["ci_high"], r["p"])
                info[f"{key}_prior"] = {"d0": r["d0"], "s02": r["s02"],
                                        "truncation_applied": r["truncation_applied"]}

    if "v2" in methods:
        r = moderation.moderated_fe(Y, V, level=level)
        out["v2"] = _frame(r["est"], r["se"], r["ci_low"], r["ci_high"], r["p"])
        info["v2_prior"] = {"d0": r["d0"], "s02": r["s02"],
                            "min_phi_moderated": float(np.nanmin(r["scale_moderated"]))}

    if "v3" in methods:
        r = eb_tau2.eb_tau2(Y, V, level=level, **(v3_kwargs or {}))
        out["v3"] = _frame(r["est"], r["sd"], r["ci_low"], r["ci_high"], r["p"])
        info["v3_prior"] = {"grid": r["grid"], "pi": r["pi"], "fit": r["fit"]}
        info["v3_tau2_post_mean"] = r["tau2_post_mean"]

    return {m: out[m] for m in methods}, info
