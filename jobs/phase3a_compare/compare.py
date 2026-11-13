"""WS-08 real-data comparison: every phase 3a method on WS-07's saved per-gene tables.

    uv run python jobs/phase3a_compare/compare.py

No site DE or FLARE rerun: every method runs at the server on per-site (log2FC, SE), and
WS-07's tables hold exactly the cells that entered its pooling. For each stratum, counts matrix
and variant it writes, under ``$FEDKG_WORKDIR/results/phase3a/method-comparison/real/``:

- ``<stratum>/<counts>/<variant>/calls.csv.gz``: per gene, each method's estimate, CI, p, q and
  hit, plus the heterogeneity flag (``meta_compare.per_gene_calls``);
- ``…/hit_counts.csv``, ``jaccard.csv``, ``rank_agreement.csv``, ``flag_breakdown.csv``;
- ``…/hyperparameters.json``: V1 and V2 (d₀, s₀²), V2's smallest moderated φ (whether the
  truncation max(1, ·) could ever bind), V3's prior and EM fit;
- for the primary standard-counts set only: ``lomo.csv`` and ``influence.csv``
  (``meta_compare.leave_one_out``) and ``phi_qq.csv`` (φ̂ against V2's fitted scaled F);
- ``summary.json``: hit counts for every run, and each variant's overlap with the primary.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase3a_meta.config import results_dir, workdir  # noqa: E402

from fedkg_spacebio.evaluation import meta_compare, meta_methods  # noqa: E402

RUNS = {
    "iss": ["primary", "min3_sites", "indep_filter", "finite_p", "primary_without_SpaceX-8"],
    "earth": ["primary", "min3_sites", "indep_filter", "finite_p"],
}
COUNTS = ("standard", "rrnarm")


def out_root() -> Path:
    return workdir() / "results" / "phase3a" / "method-comparison" / "real"


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def run_one(stratum: str, counts: str, variant: str,
            primary_hits: dict | None) -> tuple[dict, dict] | None:
    """Run every method on one saved table; None if WS-07 didn't save that table."""
    src = results_dir(stratum, counts, "flare") / variant / "per_gene.csv.gz"
    if not src.exists():
        return None
    t = pd.read_csv(src)
    y, v, clients = meta_methods.matrices_from_table(t)
    frames, info = meta_methods.run_methods(y, v)
    od = out_root() / stratum / counts / variant
    od.mkdir(parents=True, exist_ok=True)

    calls = meta_compare.per_gene_calls(t["ncbi_gene"], t["symbol"], frames, info["q_p"])
    calls.to_csv(od / "calls.csv.gz", index=False, float_format="%.17g")
    hc = meta_compare.hit_counts(frames)
    hc.to_csv(od / "hit_counts.csv", index=False)
    meta_compare.jaccard_matrix(frames).to_csv(od / "jaccard.csv", float_format="%.4f")
    meta_compare.rank_agreement(frames).to_csv(od / "rank_agreement.csv", index=False,
                                               float_format="%.4f")
    meta_compare.flag_breakdown(frames, info["q_p"]).to_csv(od / "flag_breakdown.csv",
                                                            index=False, float_format="%.4f")
    hyper = {
        "v1_trunc": info["v1_trunc_prior"], "v1_untrunc": info["v1_untrunc_prior"],
        "v2": info["v2_prior"],
        "v3": {"grid": info["v3_prior"]["grid"], "pi": info["v3_prior"]["pi"],
               "fit": info["v3_prior"]["fit"],
               "mass_at_0": float(info["v3_prior"]["pi"][0]),
               "reml_share_tau2_zero": float(np.mean(info["tau2_reml"] == 0))},
        "clients": clients, "n_genes": int(len(t)),
        "k_counts": {int(k): int(n) for k, n in zip(*np.unique(info["k"], return_counts=True),
                                                    strict=True)},
    }
    (od / "hyperparameters.json").write_text(json.dumps(_jsonable(hyper), indent=1))

    summary = {"hits": hc.set_index("method").to_dict(orient="index"),
               "v2_d0": info["v2_prior"]["d0"], "v2_s02": info["v2_prior"]["s02"],
               "v2_min_phi_moderated": info["v2_prior"]["min_phi_moderated"],
               "v1_d0": info["v1_untrunc_prior"]["d0"], "v1_s02": info["v1_untrunc_prior"]["s02"],
               "v3_mass_at_0": hyper["v3"]["mass_at_0"]}
    if primary_hits is not None:
        genes = t["ncbi_gene"].to_numpy()
        summary["shared_with_primary"] = {
            m: int(len(set(genes[frames[m]["hit"].to_numpy(dtype=bool)]) & primary_hits[m]))
            for m in frames}

    if variant == "primary" and counts == "standard":
        lomo, infl = meta_compare.leave_one_out(y, v, clients, frames)
        lomo.to_csv(od / "lomo.csv", index=False, float_format="%.4f")
        infl.to_csv(od / "influence.csv", index=False, float_format="%.4f")
        d = info["k"] - 1
        phi = np.sort(t["q"].to_numpy() / d)       # Cochran's Q / (k − 1), V2's scale factor
        dd = int(np.median(d))
        qq = pd.DataFrame({"phi_hat": phi, "expected": meta_compare.scaled_f_quantiles(
            len(phi), dd, info["v2_prior"]["d0"], info["v2_prior"]["s02"])})
        qq.to_csv(od / "phi_qq.csv", index=False, float_format="%.6g")
    return summary, {m: set(t["ncbi_gene"].to_numpy()[frames[m]["hit"].to_numpy(dtype=bool)])
                     for m in frames}


def main() -> int:
    t0 = time.time()
    summary = {}
    for stratum, variants in RUNS.items():
        for counts in COUNTS:
            primary_hits = None
            for variant in variants:
                res = run_one(stratum, counts, variant, primary_hits)
                if res is None:
                    print(f"[skip] {stratum}/{counts}/{variant}: no table", flush=True)
                    continue
                s, hits = res
                if variant == "primary":
                    primary_hits = hits
                summary[f"{stratum}/{counts}/{variant}"] = s
                print(f"[done] {stratum}/{counts}/{variant} "
                      f"{ {m: h['hits'] for m, h in s['hits'].items()} } ({time.time() - t0:.0f}s)",
                      flush=True)
    (out_root() / "summary.json").write_text(json.dumps(_jsonable(summary), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
