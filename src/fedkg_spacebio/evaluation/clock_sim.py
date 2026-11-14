"""Synthetic site-like data for the Q-11 clock diagnostic: bias, CI coverage and power.

Each synthetic site has n_f flight and n_g ground animals. Every animal's log2 expression of
each panel gene is

    x_g = m_g + c·A_g·cos(ω(t − φ_g)) + ε_g,   ε_g ~ N(0, σ²)

with the reference basis's m, A, φ.
- **Ground time** t_g is drawn uniformly in ZT0–12 for each replicate, so results average over
  plausible dissection times.
- **Flight time** is t_g − Δ, flight earlier. The estimator reports |Δ|, so the direction
  doesn't matter.
- **Spread:** each animal's own time is its arm's time plus N(0, s²) hours. Dissections are
  spread over hours, sometimes days.

Scenarios:
- ``shift``: as above; c = 1 for both arms.
- ``damped``: flight c = 0.5, no shift. The flight clock is at the right time but weaker.
- ``disrupted``: flight genes get the reference peaks shuffled among genes, no shift. Gene
  relationships are broken.

Noise is one σ for every gene: each site's median residual SD (the noise scalar it sends).
Real genes differ; using one value is a simplification, stated in the write-up.

Scored per cell:
- the median |Δ̂| and its bias against the true |Δ|;
- coverage of the 95% interval for L, against the nominal L = 2 sin(ω|Δ|/2). L is used
  because it's always defined; Δ isn't when L ≥ 2;
- power of the permutation test on L at α = 0.05;
- the share of replicates where the shift is not identifiable;
- median R²_plane and how often its permutation p is < 0.05.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from fedkg_spacebio.evaluation.clock_reference import OMEGA, ClockBasis
from fedkg_spacebio.federation import site_clock as sc


@dataclass(frozen=True)
class Cell:
    n_flight: int
    n_ground: int
    noise: float
    spread: float
    shift: float
    scenario: str = "shift"
    reps: int = 200
    n_boot: int = 500
    n_perm: int = 999
    seed: int = 0


def simulate(basis: ClockBasis, cell: Cell, rng: np.random.Generator
             ) -> tuple[np.ndarray, pd.DataFrame]:
    tg = rng.uniform(0, 12)
    rows, cond = [], []
    for arm, n in (("flight", cell.n_flight), ("ground", cell.n_ground)):
        t0 = tg - cell.shift if arm == "flight" else tg
        c = 0.5 if (arm == "flight" and cell.scenario == "damped") else 1.0
        peaks = (rng.permutation(basis.peak) if (arm == "flight"
                                                 and cell.scenario == "disrupted")
                 else basis.peak)
        for _ in range(n):
            t = t0 + rng.normal(0, cell.spread) if cell.spread else t0
            rows.append(basis.mesor + c * basis.amplitude * np.cos(OMEGA * (t - peaks))
                        + rng.normal(0, cell.noise, basis.k))
            cond.append(arm)
    md = pd.DataFrame({"condition": cond}, index=[f"s{i}" for i in range(len(cond))])
    return np.array(rows), md


def run_cell(args: tuple[ClockBasis, Cell]) -> dict:
    basis, cell = args
    rng = np.random.default_rng(cell.seed)
    true_L = 2 * abs(np.sin(OMEGA * cell.shift / 2))
    out = []
    for r in range(cell.reps):
        Y, md = simulate(basis, cell, rng)
        res = sc.analyse(Y, md, "~ condition", basis, None, seed=cell.seed * 100_003 + r,
                         n_boot=cell.n_boot, n_random_perm=cell.n_perm)
        out.append((res["delta_h"], res["L_ci"][0] <= true_L <= res["L_ci"][1],
                    res["p_perm_L"] < 0.05, res["identifiability"] == "not_identifiable",
                    res["r2_plane"], res["p_perm_r2"] < 0.05))
    d = np.array([np.nan if o[0] is None else o[0] for o in out])
    return asdict(cell) | {
        "true_L": true_L,
        "median_delta_h": float(np.nanmedian(d)) if np.isfinite(d).any() else np.nan,
        "bias_h": float(np.nanmedian(d) - cell.shift) if np.isfinite(d).any() else np.nan,
        "mad_h": float(np.nanmedian(np.abs(d - cell.shift))) if np.isfinite(d).any()
        else np.nan,
        "coverage_L": float(np.mean([o[1] for o in out])),
        "power": float(np.mean([o[2] for o in out])),
        "share_not_identifiable": float(np.mean([o[3] for o in out])),
        "median_r2_plane": float(np.median([o[4] for o in out])),
        "share_r2_p_lt_05": float(np.mean([o[5] for o in out])),
    }


def run_grid(basis: ClockBasis, cells: list[Cell], workers: int = 8) -> pd.DataFrame:
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(run_cell, [(basis, c) for c in cells]))
    return pd.DataFrame(rows)


def n_needed(power: pd.DataFrame, target: float = 0.8) -> pd.DataFrame:
    """Smallest n per arm reaching ``target`` power, per (noise, spread, shift)."""
    keys = [k for k in ("site", "noise", "spread", "shift") if k in power]
    out = []
    for key, g in power.groupby(keys):
        g = g.sort_values("n_flight")
        ok = g[g["power"] >= target]
        out.append(dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True))
                   | {"n_per_arm_for_80pct": int(ok["n_flight"].iloc[0]) if len(ok) else None,
                      "max_n_tried": int(g["n_flight"].max())})
    return pd.DataFrame(out)
