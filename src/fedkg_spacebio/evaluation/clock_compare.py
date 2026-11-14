"""Q-11 server side: tabulate the sites' clock summaries and compare their chords.

The server sees only validated ``ClockSummary`` objects (``federation/site_clock.py``). It never
sees expression or per-sample values.

Comparing sites
---------------
Each site sends its chord c_k = (α, β): flight's point on the clock circle minus ground's, in
the shared reference plane. Chords are comparable across sites without the 12 h ambiguity,
since the ambiguity arises only when a chord is turned into a midpoint time. Each site also
sends V_k, the 2 × 2 covariance of its chord across its own sample bootstrap.

- **Heterogeneity among K sites:** with W_k = V_k⁻¹ and the pooled chord
  c̄ = (Σ W_k)⁻¹ Σ W_k c_k,

      Q = Σ_k (c_k − c̄)ᵀ W_k (c_k − c̄)  ~  χ²(2(K − 1))  if all sites share one chord.

  This is Cochran's Q, with a 2-D estimate in place of a scalar.
- **One site against the rest** (SpaceX-8 against the other ISS sites): z = c_t − c̄_rest,
  with c̄_rest pooled as above and Var = V_t + (Σ_rest W_k)⁻¹. Then zᵀ Var⁻¹ z ~ χ²(2).

Caveats, reported with the results:
- The bootstrap covariances at n = 6–10 per arm are rough, so χ² p-values are approximate.
- The reference fit's uncertainty is left out on purpose. It's shared by every site, since all
  use the same plane, so it moves all chords together and largely cancels in a comparison.
- A chord difference can come from a different shift size **or** a different axis
  (midpoint), so both are shown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from fedkg_spacebio.federation.site_clock import validate_clock_summary


def table(summaries: list[dict]) -> pd.DataFrame:
    rows = []
    for s in summaries:
        validate_clock_summary(s)
        r = s["result"]
        rd = {x["direction"]: x for x in r["readings"]}
        light = [d for d, x in rd.items() if x["light_phase_consistent"]]
        rows.append({
            "client": s["client"], "stratum": s["stratum"], "n_flight": s["n_flight"],
            "n_ground": s["n_ground"], "k": r["k"], "alpha": r["alpha"], "beta": r["beta"],
            "L": r["L"], "L_lo": r["L_ci"][0], "L_hi": r["L_ci"][1],
            "delta_h": r["delta_h"], "delta_lo": r["delta_h_ci"][0],
            "delta_hi": r["delta_h_ci"][1],
            "delta_lo_sample_only": r["delta_h_ci_sample_only"][0],
            "delta_hi_sample_only": r["delta_h_ci_sample_only"][1],
            "delta_amp_x0.75": r["delta_h_amplitude_x0.75"],
            "delta_amp_x1.25": r["delta_h_amplitude_x1.25"],
            "identifiability": r["identifiability"], "p_perm_L": r["p_perm_L"],
            "perm_exact": r["perm_exact"], "n_perm": r["n_perm"],
            "r2_plane": r["r2_plane"], "r2_perm_median": r["r2_plane_perm_median"],
            "p_perm_r2": r["p_perm_r2"], "noise_sd": r["noise_sd"],
            "later_mid_ct": rd["flight_later"]["mid_ct"],
            "later_flight_ct": rd["flight_later"]["flight_ct"],
            "later_ground_ct": rd["flight_later"]["ground_ct"],
            "earlier_mid_ct": rd["flight_earlier"]["mid_ct"],
            "earlier_flight_ct": rd["flight_earlier"]["flight_ct"],
            "earlier_ground_ct": rd["flight_earlier"]["ground_ct"],
            "light_phase_reading": light[0] if len(light) == 1 else
            ("both" if len(light) == 2 else "neither"),
            "panel_dropped": ";".join(d["symbol"] for d in s["panel_dropped"]),
        })
    return pd.DataFrame(rows)


def _pool(chords: np.ndarray, covs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    W = np.linalg.inv(covs)
    S = W.sum(0)
    c = np.linalg.solve(S, np.einsum("kij,kj->i", W, chords))
    return c, np.linalg.inv(S)


def heterogeneity(summaries: list[dict]) -> dict:
    chords = np.array([[s["result"]["alpha"], s["result"]["beta"]] for s in summaries])
    covs = np.array([s["result"]["chord_cov_sample"] for s in summaries])
    c, _ = _pool(chords, covs)
    W = np.linalg.inv(covs)
    dev = chords - c
    Q = float(np.einsum("ki,kij,kj->", dev, W, dev))
    df = 2 * (len(summaries) - 1)
    return {"clients": [s["client"] for s in summaries], "Q": Q, "df": df,
            "p": float(stats.chi2.sf(Q, df)), "pooled_chord": c.tolist()}


def one_vs_rest(summaries: list[dict], target: str) -> dict:
    t = [s for s in summaries if s["client"] == target]
    rest = [s for s in summaries if s["client"] != target]
    if len(t) != 1 or not rest:
        raise ValueError(f"need exactly one {target!r} and at least one other site")
    ct = np.array([t[0]["result"]["alpha"], t[0]["result"]["beta"]])
    Vt = np.array(t[0]["result"]["chord_cov_sample"])
    cr, Vr = _pool(np.array([[s["result"]["alpha"], s["result"]["beta"]] for s in rest]),
                   np.array([s["result"]["chord_cov_sample"] for s in rest]))
    z = ct - cr
    stat = float(z @ np.linalg.solve(Vt + Vr, z))
    return {"target": target, "rest": [s["client"] for s in rest], "chord_target": ct.tolist(),
            "chord_rest_pooled": cr.tolist(), "chi2": stat, "df": 2,
            "p": float(stats.chi2.sf(stat, 2))}
