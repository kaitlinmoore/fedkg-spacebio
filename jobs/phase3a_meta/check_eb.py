"""Check 4: the pipeline's V2/V3 columns equal WS-08's real-data comparison, gene by gene.

Run after report.py has written the schema-2 tables, from WSL after ``fedkg``:

    uv run python jobs/phase3a_meta/check_eb.py

Why: V2 and V3 now run inside the FLARE server (``meta.pool``), not only in WS-08's offline
comparison (``jobs/phase3a_compare/compare.py``). WS-08 ran them on the schema-1 per-gene tables
with ``evaluation.meta_methods.run_methods``. Both paths call the same two functions
(``moderation.moderated_fe``, ``eb_tau2.eb_tau2``) on the same per-site values, so every V2/V3
estimate, interval, p and q should agree, and the hit sets should be identical. A difference
would mean the server pooled different cells, or ran a method with different settings.

For each (stratum, counts, variant) WS-08 wrote (``method-comparison/real/…/calls.csv.gz``), it
reports the largest |Δ| per column, whether the genes line up, and whether the hit sets are
equal. Output: ``$FEDKG_WORKDIR/results/phase3a/check_eb.json``. Exits non-zero on a failure.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402

# V3's credible limits come from an iterative quantile solver (eb_tau2._mixture_quantile) that
# stops when its step is < 1e-10. A last-bit difference in its inputs (e.g. from BLAS summation
# order in the EM fit; seen at ~1e-15 in V3's estimate and p) can move the stopping point by up
# to about that much, so the limits are compared to 1e-9, ten times the solver's own tolerance.
# Everything else uses Check 1's tolerance.
SOLVER_ATOL = 1e-9
SOLVER_COLUMNS = {"v3_ci_low", "v3_ci_high"}
PAIRS = {  # pipeline column → WS-08 column
    "v3": {"v3_est": "v3__est", "v3_ci_low": "v3__ci_low", "v3_ci_high": "v3__ci_high",
           "v3_p": "v3__p", "v3_q": "v3__q"},
    "v2": {"fe_est": "v2__est", "v2_ci_low": "v2__ci_low", "v2_ci_high": "v2__ci_high",
           "v2_p": "v2__p", "v2_q": "v2__q"},
}


def compare(new: pd.DataFrame, old: pd.DataFrame) -> dict:
    rec = {"n_genes": int(len(new)),
           "same_genes": bool(len(new) == len(old)
                              and np.array_equal(new["ncbi_gene"], old["ncbi_gene"]))}
    if not rec["same_genes"]:
        rec["pass"] = False
        return rec
    ok = True
    for method, cols in PAIRS.items():
        m = {}
        for ours, theirs in cols.items():
            a, b = new[ours].to_numpy(dtype=float), old[theirs].to_numpy(dtype=float)
            both = np.isfinite(a) & np.isfinite(b)
            # Same float64 code on identical inputs. Check 1's tolerance leaves room for a
            # last-bit difference from operation order; the solver columns get SOLVER_ATOL.
            atol = SOLVER_ATOL if ours in SOLVER_COLUMNS else config.ATOL
            rtol = 0.0 if ours in SOLVER_COLUMNS else config.RTOL
            close = bool(np.array_equal(np.isfinite(a), np.isfinite(b))
                         and np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))
            m[ours] = {"max_abs_diff": float(np.max(np.abs(a[both] - b[both]), initial=0.0)),
                       "allclose": close}
            ok &= close
        hits_new = set(new.loc[new[f"{method}_hit"].astype(bool), "ncbi_gene"])
        hits_old = set(old.loc[old[f"{method}__hit"].astype(bool), "ncbi_gene"])
        m["hits"] = {"pipeline": len(hits_new), "ws08": len(hits_old),
                     "identical": hits_new == hits_old}
        ok &= hits_new == hits_old
        rec[method] = m
    rec["pass"] = bool(ok)
    return rec


def main(root: Path | None = None) -> dict:
    base = (root or config.workdir()) / "results" / "phase3a"
    ws08 = base / "method-comparison" / "real"
    out = {"rtol": config.RTOL, "atol": config.ATOL, "solver_atol": SOLVER_ATOL,
           "solver_columns": sorted(SOLVER_COLUMNS), "runs": {}}
    for calls in sorted(ws08.glob("*/*/*/calls.csv.gz")):
        variant, counts, stratum = calls.parent.name, calls.parent.parent.name, \
            calls.parent.parent.parent.name
        table = config.results_dir(stratum, counts, "flare", root) / variant / "per_gene.csv.gz"
        key = f"{stratum}/{counts}/{variant}"
        if not table.exists():
            out["runs"][key] = {"pass": False, "detail": f"missing {table}"}
            continue
        out["runs"][key] = compare(pd.read_csv(table), pd.read_csv(calls))
    out["pass"] = bool(out["runs"]) and all(r["pass"] for r in out["runs"].values())
    (base / "check_eb.json").write_text(json.dumps(out, indent=1))
    for key, r in out["runs"].items():
        hits = {m: (r[m]["hits"]["pipeline"], r[m]["hits"]["ws08"]) for m in PAIRS if m in r}
        print(f"{key:40s} pass={r['pass']}  hits (pipeline, WS-08): {hits}")
    print("Check 4 (V2/V3 = WS-08):", "PASS" if out["pass"] else "FAIL")
    return out


if __name__ == "__main__":
    if not main()["pass"]:
        raise SystemExit("Check 4 FAILED; see check_eb.json")
