"""WS-08 calibrated simulation: every phase 3a method under known truth.

Not a FLARE job. Every method compared here runs at the server on per-site summaries, so the
simulation runs them in-process on simulated summaries (see
``fedkg_spacebio.evaluation.meta_sim`` for the design).

    uv run python jobs/phase3a_compare/simulate.py                  # 100 replicates, all settings
    uv run python jobs/phase3a_compare/simulate.py --reps 5 --workers 4    # quick look

Reads WS-07's per-gene tables (standard counts; primary, plus ISS min3_sites for the mixed-k
generator) and writes to ``$FEDKG_WORKDIR/results/phase3a/method-comparison/simulation/``:
- ``calibration.json``: what was copied from each stratum (τ² prior quantiles, V2 prior, σ_μ);
- ``replicates.csv.gz``: one row per stratum × setting × replicate × method, with seeds;
- ``summary.csv``: mean and SD over replicates;
- ``run.json``: arguments, base seed, timing, package versions.
"""

import os

# One BLAS thread per worker process; the parallelism is across replicates.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase3a_meta.config import results_dir, workdir  # noqa: E402

from fedkg_spacebio.evaluation import meta_sim  # noqa: E402

STRATA = ("iss", "earth")
BASE_SEED = 20261113

_CALS: dict = {}


def out_dir() -> Path:
    return workdir() / "results" / "phase3a" / "method-comparison" / "simulation"


def table(stratum: str, variant: str) -> pd.DataFrame:
    return pd.read_csv(results_dir(stratum, "standard", "flare") / variant / "per_gene.csv.gz")


def _init(cals):
    _CALS.update(cals)


def _task(args):
    stratum, si, setting, rep = args
    seed = meta_sim.seed_for(BASE_SEED, STRATA.index(stratum), si, rep)
    out = meta_sim.run_replicate(_CALS[stratum], setting, seed)
    out["rep"] = rep
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--strata", nargs="+", default=list(STRATA), choices=STRATA)
    args = ap.parse_args(argv)

    od = out_dir()
    od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cals, tasks, settings_by = {}, [], {}
    for s in args.strata:
        alt = table(s, "min3_sites") if s == "iss" else None
        cal = meta_sim.calibrate(table(s, "primary"), s, alt_table=alt)
        cals[s] = cal
        settings = meta_sim.default_settings(include_min3=(s == "iss"),
                                             free_frac_mu0=cal.free_fit["frac_mu0"])
        settings_by[s] = [st.name for st in settings]
        tasks += [(s, i, st, r) for i, st in enumerate(settings) for r in range(args.reps)]
        print(f"[calibrate] {s}: {json.dumps(cal.summary(), default=float)}", flush=True)
    (od / "calibration.json").write_text(json.dumps(
        {s: c.summary() for s, c in cals.items()}, indent=2, default=float))

    print(f"[simulate] {len(tasks)} replicates on {args.workers} workers", flush=True)
    parts = []
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(cals,)) as ex:
        for i, out in enumerate(ex.map(_task, tasks, chunksize=4), 1):
            parts.append(out)
            if i % 200 == 0 or i == len(tasks):
                print(f"[simulate] {i}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    reps = pd.concat(parts, ignore_index=True)
    reps.to_csv(od / "replicates.csv.gz", index=False, float_format="%.10g")
    summ = meta_sim.summarize(reps)
    summ.to_csv(od / "summary.csv", index=False, float_format="%.6g")
    (od / "run.json").write_text(json.dumps({
        "finished": datetime.now(UTC).isoformat(timespec="seconds"),
        "seconds": round(time.time() - t0, 1), "reps": args.reps, "workers": args.workers,
        "base_seed": BASE_SEED, "seed_rule": "SeedSequence([base, stratum_idx, setting_idx, rep])",
        "strata": list(args.strata), "settings": settings_by,
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                     "pandas": pd.__version__},
    }, indent=2))
    print(f"[done] {od}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
