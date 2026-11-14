"""Q-11 server step (WS-09): results table, site comparisons, per-site power.

    uv run python jobs/phase3a_clock/report.py

Reads only the validated site summaries (``summaries/*.json``). Writes
``$FEDKG_WORKDIR/results/phase3a/circadian-phase/report/``:
- ``results_table.csv``: one row per client;
- ``comparisons.json``: chord heterogeneity (ISS, ISS without SpaceX-8, Earth) and SpaceX-8
  vs the other ISS sites;
- ``site_power.csv``: synthetic runs at each site's own n per arm and noise scalar. Gives power
  and CI coverage for 0/2/4/6 h shifts;
- ``n_needed.csv``: power at a grid of n per arm, at each site's noise, and the smallest n per
  arm reaching 80% for 2, 4 and 6 h.

Power runs use spread 0 and the site's noise scalar as independent per-gene noise. The noise
scalar is the within-arm residual SD, which already includes clock variation from animals
taken at different times. Counting that as noise is conservative for the arm difference.
"""

import json
import os

import pandas as pd
from common import load_reference, out_dir

from fedkg_spacebio.evaluation import clock_compare, clock_sim

ISS = ["SpaceX-4", "SpaceX-8", "SpaceX-13-ISST", "SpaceX-16-ISST"]
EARTH = ["SpaceX-13-LAR", "SpaceX-16-LAR", "SpaceX-21"]
N_GRID = (3, 4, 6, 8, 10, 15, 20, 30)


def main() -> int:
    sd = out_dir("summaries")
    od = out_dir("report")
    summ = {c: json.loads((sd / f"{c}.json").read_text()) for c in ISS + EARTH}
    tab = clock_compare.table(list(summ.values()))
    tab.to_csv(od / "results_table.csv", index=False)

    comps = {
        "iss_all": clock_compare.heterogeneity([summ[c] for c in ISS]),
        "iss_without_spacex8": clock_compare.heterogeneity(
            [summ[c] for c in ISS if c != "SpaceX-8"]),
        "spacex8_vs_other_iss": clock_compare.one_vs_rest([summ[c] for c in ISS], "SpaceX-8"),
        "earth_all": clock_compare.heterogeneity([summ[c] for c in EARTH]),
    }
    (od / "comparisons.json").write_text(json.dumps(comps, indent=1))

    basis, _, _ = load_reference()
    cells, seed = [], 1000
    for c in ISS + EARTH:
        s = summ[c]
        noise = round(s["result"]["noise_sd"], 3)
        for shift in (0.0, 2.0, 4.0, 6.0):
            cells.append(clock_sim.Cell(s["n_flight"], s["n_ground"], noise, 0.0, shift,
                                        seed=seed))
            seed += 1
        for n in N_GRID:
            for shift in (2.0, 4.0, 6.0):
                cells.append(clock_sim.Cell(n, n, noise, 0.0, shift, seed=seed, n_boot=200))
                seed += 1
    grid = clock_sim.run_grid(basis, cells, workers=min(20, os.cpu_count() or 4))
    noise_of = {round(summ[c]["result"]["noise_sd"], 3): c for c in summ}
    grid.insert(0, "site", grid["noise"].map(noise_of))
    own = grid[grid["n_boot"] == 500]  # the cells at each site's own n (grid cells use 200)
    own.to_csv(od / "site_power.csv", index=False)
    g = grid[grid["n_flight"].isin(N_GRID) & (grid["n_flight"] == grid["n_ground"])
             & (grid["n_boot"] == 200)]
    g.to_csv(od / "n_grid_power.csv", index=False)
    clock_sim.n_needed(g.drop(columns=["noise"])).to_csv(
        od / "n_needed.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(tab[["client", "n_flight", "n_ground", "delta_h", "delta_lo", "delta_hi",
                   "delta_lo_sample_only", "delta_hi_sample_only", "delta_amp_x0.75",
                   "delta_amp_x1.25", "p_perm_L", "r2_plane", "r2_perm_median", "p_perm_r2",
                   "noise_sd", "identifiability"]].round(3).to_string())
        print(tab[["client", "later_mid_ct", "later_flight_ct", "later_ground_ct",
                   "earlier_mid_ct", "earlier_flight_ct", "earlier_ground_ct",
                   "light_phase_reading"]].round(2).to_string())
        print(json.dumps(comps, indent=1))
        print(own[["site", "n_flight", "n_ground", "noise", "shift", "median_delta_h", "power",
                   "coverage_L", "median_r2_plane"]].round(3).to_string())
        print(pd.read_csv(od / "n_needed.csv").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
