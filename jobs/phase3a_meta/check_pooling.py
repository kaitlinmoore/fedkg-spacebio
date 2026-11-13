"""Check 3, pooling validation on the real data: our pooling vs statsmodels, gene by gene.

Run after job.py for a stratum, from WSL after ``fedkg``:

    uv run python jobs/phase3a_meta/check_pooling.py --stratum iss [--counts standard]

``tests/test_meta.py`` checks ``federation/meta.py`` against published values (metafor's BCG
example) and simulated data. This script runs the same comparison on the real genes × sites
matrices of each variant, with ``statsmodels.stats.meta_analysis.combine_effects`` as an
independent implementation:

1. **Fixed effect and Q** (no τ² involved): FE estimate, FE variance, Cochran's Q and the
   Q-based I² should agree to rounding error.
2. **τ² estimators:** our REML vs statsmodels' Paule–Mandel (PM) and DerSimonian–Laird (DL).
   These are *different* estimators, so they're not expected to be equal; the check reports how
   far apart they are (correlation, |Δ| quantiles, how often each puts τ² at 0). With equal
   variances all three coincide, so large gaps would point at a bug, not a method difference.
3. **HKSJ arithmetic** separately from τ²: give our ``random_effects_hksj`` statsmodels' PM τ²,
   and compare our untruncated HKSJ variance with statsmodels' ``var_hksj_re``.
4. **Does the τ² estimator change the secondary result?** RE-HKSJ (truncated) BH hit counts with
   REML, PM or DL τ². The FE headline doesn't depend on τ².

Two statsmodels conventions differ from ours and from the usual textbook definitions, so they are
normalized before comparing (and the raw counts are reported):
- its DL τ² is **not truncated at 0** (``method_re="dl"`` returns the raw moment estimate, which is
  negative whenever Q < k − 1). Standard DL is max(0, ·), so we truncate. Untruncated, a negative
  τ² makes some RE weights larger than the FE weights, which is meaningless.
- its I² is (Q − df)/Q without the floor at 0. Ours is max(0, ·).

statsmodels is a dev dependency (``uv sync`` installs it). Output:
``$FEDKG_WORKDIR/results/phase3a/<stratum>/<counts>/check3.json``.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats as sps
from statsmodels.stats.meta_analysis import combine_effects

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402
from job import DROP_VARIANTS  # noqa: E402

from fedkg_spacebio.federation import meta  # noqa: E402
from fedkg_spacebio.federation import stratum as st  # noqa: E402
from fedkg_spacebio.federation.meta_flare import load_summaries, variants_for  # noqa: E402
from fedkg_spacebio.federation.results import FDR_Q  # noqa: E402


def absdiff(a, b) -> dict:
    d = np.abs(np.asarray(a) - np.asarray(b))
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {}
    return {"median": float(np.median(d)), "q90": float(np.quantile(d, 0.9)),
            "q99": float(np.quantile(d, 0.99)), "max": float(d.max())}


def reldiff_max(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 0)
    return float(np.max(np.abs(a[ok] - b[ok]) / np.abs(b[ok]), initial=0.0))


def compare_tau2(ours, other) -> dict:
    other = np.asarray(other)
    return {"pearson": float(np.corrcoef(ours, other)[0, 1]),
            "spearman": float(sps.spearmanr(ours, other).statistic),
            "abs_diff": absdiff(ours, other),
            "frac_zero_ours": float(np.mean(ours == 0)),
            "frac_zero_other": float(np.mean(other == 0)),
            "agree_on_zero": float(np.mean((ours == 0) == (other == 0)))}


def statsmodels_per_gene(Y, V, method):
    """Loop over genes (statsmodels is per-meta-analysis). Missing cells are dropped."""
    out = {k: np.full(len(Y), np.nan) for k in
           ("tau2_raw", "fe", "var_fe", "q", "i2_raw", "re", "var_hksj_re")}
    with warnings.catch_warnings():
        # statsmodels takes sqrt of RE variances that are negative when its DL τ² is negative;
        # those values aren't used here (τ² is truncated below and HKSJ is recomputed).
        warnings.simplefilter("ignore", RuntimeWarning)
        for g in range(len(Y)):
            ok = np.isfinite(Y[g]) & np.isfinite(V[g])
            if ok.sum() < 2:
                continue
            r = combine_effects(Y[g, ok], V[g, ok], method_re=method)
            out["tau2_raw"][g], out["fe"][g] = r.tau2, r.mean_effect_fe
            out["var_fe"][g], out["q"][g], out["i2_raw"][g] = r.var_eff_w_fe, r.q, r.i2
            out["re"][g], out["var_hksj_re"][g] = r.mean_effect_re, r.var_hksj_re
    out["tau2"] = np.maximum(0.0, out["tau2_raw"])   # standard truncation (see module docstring)
    out["i2"] = np.maximum(0.0, out["i2_raw"])
    out["n_tau2_negative_raw"] = int(np.sum(out["tau2_raw"] < 0))
    return out


def re_hits(Y, V, tau2) -> int:
    re = meta.random_effects_hksj(Y, V, tau2, truncate=True)
    return int((meta.bh_qvalues(re["p"]) < FDR_Q).sum())


def check_variant(frames, variant) -> dict:
    use = {c: f for c, f in frames.items() if c not in variant.drop_clients}
    genes, Y, V, clients = st.matrices(use, variant.flag)
    need = len(clients) if variant.min_sites is None else variant.min_sites
    keep = np.isfinite(Y).sum(axis=1) >= need
    Y, V = Y[keep], V[keep]
    fe = meta.fixed_effect(Y, V)
    tau2, conv = meta.tau2_reml(Y, V)
    pm = statsmodels_per_gene(Y, V, "pm")
    dl = statsmodels_per_gene(Y, V, "dl")
    hk = meta.random_effects_hksj(Y, V, pm["tau2"], truncate=False)
    return {
        "n_genes": int(len(Y)), "clients": clients,
        "fixed_effect": {
            "est_max_abs_diff": absdiff(fe["est"], pm["fe"]).get("max"),
            "var_max_rel_diff": reldiff_max(fe["se"] ** 2, pm["var_fe"]),
            "q_max_rel_diff": reldiff_max(fe["q"], pm["q"]),
            "i2_max_abs_diff": absdiff(fe["i2"], pm["i2"]).get("max"),
        },
        "tau2_reml_vs_pm": compare_tau2(tau2, pm["tau2"]),
        "tau2_reml_vs_dl": compare_tau2(tau2, dl["tau2"]),
        "tau2_pm_vs_dl": compare_tau2(pm["tau2"], dl["tau2"]),
        "reml_not_converged": int((~conv).sum()),
        "hksj_given_pm_tau2": {
            "re_est_max_abs_diff": absdiff(hk["est"], pm["re"]).get("max"),  # PM τ² ≥ 0 here
            "var_max_rel_diff": reldiff_max(hk["se"] ** 2, pm["var_hksj_re"]),
        },
        "statsmodels_negative_tau2_raw": {"pm": pm["n_tau2_negative_raw"],
                                          "dl": dl["n_tau2_negative_raw"]},
        "re_hksj_truncated_hits": {"reml": re_hits(Y, V, tau2), "pm": re_hits(Y, V, pm["tau2"]),
                                   "dl": re_hits(Y, V, dl["tau2"])},
    }


def run(stratum: str, counts: str, root: Path | None = None) -> dict:
    base = config.results_dir(stratum, counts, "flare", root)
    frames = {c: s.to_frame() for c, s in load_summaries(base / "site_summaries").items()}
    report = {"stratum": stratum, "counts": counts, "variants": {}}
    for v in variants_for(DROP_VARIANTS.get(stratum, [])):
        report["variants"][v.name] = check_variant(frames, v)
        print(v.name, json.dumps(report["variants"][v.name]["fixed_effect"]), flush=True)
    out = base.parent / "check3.json"
    out.write_text(json.dumps(report, indent=1))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stratum", choices=["iss", "earth"], required=True)
    parser.add_argument("--counts", choices=["standard", "rrnarm"], default="standard")
    args = parser.parse_args()
    run(args.stratum, args.counts)


if __name__ == "__main__":
    main()
