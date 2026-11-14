"""Q-11 estimator validation (WS-09), run before any site result is trusted.

    uv run python jobs/phase3a_clock/validate.py

Run after ``reference.py``. Writes
``$FEDKG_WORKDIR/results/phase3a/circadian-phase/validation/``:

1. ``gse11923_absolute_loto.csv``: each array's absolute clock time, estimated with the
   reference refitted without that array. Same platform, gene baselines known: the best case
   for the exploratory absolute estimate (B).
2. ``gse11923_chord.csv``: pseudo-arms of 3 consecutive held-out hours against 3 hours shifted
   by 2, 4 or 6 h (reference refitted without all 6 arrays). This checks the clock-plane
   estimator in hours, on the array's own amplitude scale. Pooled arrays are less noisy than
   single animals, so it's a best case too.
3. ``gse54651_pairs.csv``: cross-platform check. Pairs of RNA-seq samples 6, 12 or 18 h apart,
   with GSE11923 peaks and GSE54651 amplitudes refitted without the pair. 12 h apart is L = 2,
   the edge of identifiability, so those pairs test the "not identifiable" path.
4. ``gse54651_absolute_loo.csv``: (B) on RNA-seq samples, with mesors and amplitudes refitted
   on the other 7 samples and peaks from GSE11923.
5. ``synthetic_grid.csv``: bias, CI coverage, power and R²_plane on site-like synthetic data
   (``evaluation/clock_sim.py``).
6. ``validation.json``: summary numbers.
"""

import json
import os

import numpy as np
import pandas as pd
from common import data_dir, load_array, load_reference, out_dir

from fedkg_spacebio.evaluation import clock_reference as cref
from fedkg_spacebio.evaluation import clock_sim
from fedkg_spacebio.federation import site_clock as sc
from fedkg_spacebio.harmonize.gene_map import load_map, map_summary


def q(v, p):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.quantile(v, p)) if len(v) else None


def gse11923_checks(panel: pd.DataFrame, arr: pd.DataFrame, ct: np.ndarray):
    Y = arr.loc[list(panel["probe_set"])].to_numpy(float)  # genes × arrays
    ids, syms = tuple(panel["ncbi_gene"].astype(int)), tuple(panel["symbol"])

    def basis_without(drop):
        keep = np.setdiff1d(np.arange(len(ct)), drop)
        f = cref.cosinor(Y[:, keep], ct[keep])
        return cref.ClockBasis(ids, syms, f["peak"].to_numpy(), f["amplitude"].to_numpy(),
                               f["mesor"].to_numpy(), "array refit")

    rows = []
    for i in range(len(ct)):
        b = basis_without([i])
        est = sc.absolute_phase(Y[:, i][None], b)
        rows.append({"ct": ct[i], "est_ct": float(est["phase"][0]),
                     "error_h": float(cref.circ_diff(est["phase"][0], ct[i])),
                     "template_r": float(est["template_r"][0]),
                     "scale": float(est["scale"][0])})
    absolute = pd.DataFrame(rows)

    rows = []
    for s in (2, 4, 6):
        for h in range(0, len(ct) - (s + 3) + 1):
            g_idx, f_idx = np.arange(h, h + 3), np.arange(h + s, h + s + 3)
            drop = np.union1d(g_idx, f_idx)
            b = basis_without(drop)
            d = Y[:, f_idx].mean(1) - Y[:, g_idx].mean(1)
            fit = sc.plane_fit(d, b)
            dh = sc.delta_hours(fit["L"])
            rd = sc.readings(fit["alpha"], fit["beta"], dh if np.isfinite(dh) else None)
            later = [x for x in rd if x["direction"] == "flight_later"][0]
            true_mid = np.mod((ct[g_idx].mean() + ct[f_idx].mean()) / 2, 24)
            rows.append({"shift": s, "ground_start_ct": ct[h], "delta_h": dh,
                         "abs_error_h": abs(dh - s) if np.isfinite(dh) else np.nan,
                         "mid_error_h": float(cref.circ_diff(later["mid_ct"], true_mid)),
                         "r2_plane": fit["r2_plane"]})
    chord = pd.DataFrame(rows)
    return absolute, chord


def gse54651_checks(panel: pd.DataFrame):
    rq, ct = cref.read_zhang_quant(data_dir() / "gse54651" / "GSE54651_Liv_quant.txt.gz")
    from phase3a_meta.config import annotation_path
    mapped, _ = map_summary(rq.assign(tested=True), load_map(annotation_path()),
                            tested_col=None)
    Y = cref.log2_counts(mapped.loc[list(panel["ncbi_gene"].astype(int)), rq.columns]
                         ).to_numpy(float)
    ids, syms = tuple(panel["ncbi_gene"].astype(int)), tuple(panel["symbol"])
    peaks = panel["peak"].to_numpy(float)

    def basis_without(drop):
        keep = np.setdiff1d(np.arange(len(ct)), drop)
        f = cref.cosinor(Y[:, keep], ct[keep])
        return cref.ClockBasis(ids, syms, peaks, f["amplitude"].to_numpy(),
                               f["mesor"].to_numpy(), "rnaseq refit")

    rows = []
    for i in range(len(ct)):
        for j in range(i + 1, len(ct)):
            sep = abs(float(cref.circ_diff(ct[j], ct[i])))
            b = basis_without([i, j])
            fit = sc.plane_fit(Y[:, j] - Y[:, i], b)
            dh = sc.delta_hours(fit["L"])
            rows.append({"ct_i": ct[i], "ct_j": ct[j], "separation_h": sep, "L": fit["L"],
                         "delta_h": dh, "identifiable": bool(np.isfinite(dh)),
                         "r2_plane": fit["r2_plane"]})
    pairs = pd.DataFrame(rows)
    rows = []
    for i in range(len(ct)):
        est = sc.absolute_phase(Y[:, i][None], basis_without([i]))
        rows.append({"ct": ct[i], "est_ct": float(est["phase"][0]),
                     "error_h": float(cref.circ_diff(est["phase"][0], ct[i])),
                     "template_r": float(est["template_r"][0])})
    return pairs, pd.DataFrame(rows)


def main() -> int:
    od = out_dir("validation")
    basis, _, _ = load_reference()
    panel = pd.read_csv(out_dir("reference") / "panel_decisions.csv")
    kept = panel[panel["kept"]].rename(columns={"best_probe": "probe_set",
                                                "array_peak": "peak"})
    arr, ct = load_array()

    absolute, chord = gse11923_checks(kept, arr, ct)
    absolute.to_csv(od / "gse11923_absolute_loto.csv", index=False)
    chord.to_csv(od / "gse11923_chord.csv", index=False)
    pairs, abs_r = gse54651_checks(kept)
    pairs.to_csv(od / "gse54651_pairs.csv", index=False)
    abs_r.to_csv(od / "gse54651_absolute_loo.csv", index=False)

    cells = []
    seed = 1
    for n in (6, 10, 20):
        for noise in (0.15, 0.3, 0.6):
            for spread in (0.0, 1.0, 2.0):
                for shift in (0.0, 2.0, 4.0, 6.0):
                    cells.append(clock_sim.Cell(n, n, noise, spread, shift, seed=seed))
                    seed += 1
    for scen in ("damped", "disrupted"):
        for n in (6, 10, 20):
            cells.append(clock_sim.Cell(n, n, 0.3, 1.0, 0.0, scenario=scen, seed=seed))
            seed += 1
    grid = clock_sim.run_grid(basis, cells, workers=min(20, os.cpu_count() or 4))
    grid.to_csv(od / "synthetic_grid.csv", index=False)

    abs_err = absolute["error_h"].abs()
    summary = {
        "gse11923_absolute_loto": {"n": len(absolute),
                                   "median_abs_error_h": float(abs_err.median()),
                                   "p90_abs_error_h": float(abs_err.quantile(0.9)),
                                   "median_template_r": float(absolute["template_r"].median())},
        "gse11923_chord": {
            int(s): {"n": int(len(g)), "median_delta_h": q(g["delta_h"], 0.5),
                     "median_abs_error_h": q(g["abs_error_h"], 0.5),
                     "p90_abs_error_h": q(g["abs_error_h"], 0.9),
                     "median_abs_mid_error_h": q(g["mid_error_h"].abs(), 0.5),
                     "p90_abs_mid_error_h": q(g["mid_error_h"].abs(), 0.9),
                     "median_r2_plane": q(g["r2_plane"], 0.5)}
            for s, g in chord.groupby("shift")},
        "gse54651_pairs": {
            float(s): {"n": int(len(g)), "share_identifiable": float(g["identifiable"].mean()),
                       "median_delta_h": q(g["delta_h"], 0.5),
                       "median_r2_plane": q(g["r2_plane"], 0.5), "median_L": q(g["L"], 0.5)}
            for s, g in pairs.groupby("separation_h")},
        "gse54651_absolute_loo": {"median_abs_error_h": float(abs_r["error_h"].abs().median()),
                                  "max_abs_error_h": float(abs_r["error_h"].abs().max())},
    }
    (od / "validation.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    show = ["n_flight", "noise", "spread", "shift", "scenario", "median_delta_h", "bias_h",
            "coverage_L", "power", "share_not_identifiable", "median_r2_plane",
            "share_r2_p_lt_05"]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(grid[show].round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
