"""Check 1, federated = central: the same pipeline in-process, compared with the FLARE run.

Run after job.py, from WSL after ``fedkg``:

    uv run python jobs/phase3a_meta/central.py --stratum iss [--counts standard]

It runs every client's loader → pydeseq2 → summary in this one process (no FLARE), pools with
the same ``stratum.pool_variants``, and compares at three levels:

1. **transport**: what each FLARE client saved as sent (``flare/sent/``) vs what the server
   received (``flare/site_summaries/``). It must be bit-for-bit identical.
2. **site recomputation**: the in-process summaries vs the ones the server received. This
   shows whether pydeseq2 is deterministic across runs. Any difference is reported, not
   assumed away.
3. **pooled**: every variant's pooled table, central vs FLARE, with
   ``allclose(rtol=config.RTOL, atol=config.ATOL)``. That includes the V2/V3 columns, and the
   priors V2 and V3 fitted across genes (``variant.json`` → ``eb_priors``) must agree too: a
   different prior would change every gene's V2/V3 result at once.

By construction this tests transport and wiring, not a benefit of federation. Fixed-effect
federated pooling of these summaries is mathematically the same as central pooling of them.

Output: ``$FEDKG_WORKDIR/results/phase3a/<stratum>/<counts>/central/`` and ``check1.json`` there.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402
from job import DROP_VARIANTS  # noqa: E402

from fedkg_spacebio.federation import stratum as st  # noqa: E402
from fedkg_spacebio.federation.meta_flare import (  # noqa: E402
    load_summaries,
    variants_for,
    write_pooled,
    write_summaries,
)
from fedkg_spacebio.federation.site_de import run_de, summarize  # noqa: E402
from fedkg_spacebio.harmonize.gene_map import load_map  # noqa: E402
from fedkg_spacebio.sites.loader import load_site  # noqa: E402
from fedkg_spacebio.sites.manifest import clients_in  # noqa: E402


def compare_summaries(a: dict, b: dict) -> dict:
    out = {}
    for client in sorted(set(a) | set(b)):
        if client not in a or client not in b:
            out[client] = {"present_in_both": False}
            continue
        ga, gb = a[client].genes, b[client].genes
        rec = {"present_in_both": True, "meta_equal": a[client].meta == b[client].meta}
        for k in ga:
            same_shape = ga[k].shape == gb[k].shape
            rec[k] = {"bitwise_equal": bool(same_shape and ga[k].dtype == gb[k].dtype
                                            and np.array_equal(ga[k], gb[k]))}
            if same_shape and ga[k].dtype.kind == "f":
                rec[k]["max_abs_diff"] = float(np.max(np.abs(ga[k] - gb[k]), initial=0.0))
        out[client] = rec
    return out


def compare_pooled(central: dict[str, pd.DataFrame], flare_dir: Path) -> dict:
    out = {}
    for name, c in central.items():
        f = pd.read_csv(flare_dir / name / "meta_genes.csv.gz")
        rec = {"n_genes_central": int(len(c)), "n_genes_flare": int(len(f)),
               "same_genes": bool(len(c) == len(f)
                                  and np.array_equal(c["ncbi_gene"], f["ncbi_gene"]))}
        if rec["same_genes"]:
            cols = {}
            for col in c.columns:
                if col == "ncbi_gene" or c[col].dtype == bool:
                    cols[col] = {"equal": bool(np.array_equal(c[col].to_numpy(),
                                                              f[col].to_numpy()))}
                    continue
                x, y = c[col].to_numpy(dtype=float), f[col].to_numpy(dtype=float)
                both = np.isfinite(x) & np.isfinite(y)
                cols[col] = {
                    "nan_pattern_equal": bool(np.array_equal(np.isfinite(x), np.isfinite(y))),
                    "max_abs_diff": float(np.max(np.abs(x[both] - y[both]), initial=0.0)),
                    "allclose": bool(np.allclose(x, y, rtol=config.RTOL, atol=config.ATOL,
                                                 equal_nan=True)),
                }
            rec["columns"] = cols
            rec["pass"] = all(v.get("allclose", v.get("equal", False))
                              and v.get("nan_pattern_equal", True) for v in cols.values())
        else:
            rec["pass"] = False
        rec["eb_priors"] = compare_priors(c.attrs.get("eb_priors"),
                                          json.loads((flare_dir / name / "variant.json")
                                                     .read_text()).get("eb_priors"))
        rec["pass"] = rec["pass"] and rec["eb_priors"]["pass"]
        out[name] = rec
    return out


def compare_priors(central: dict | None, flare: dict | None) -> dict:
    """The V2/V3 priors of one variant, central vs FLARE (read back from JSON on both sides,
    so both have been through the same serialization)."""
    if central is None or flare is None:
        return {"pass": False, "detail": "eb_priors missing"}
    central = json.loads(json.dumps(central))
    out = {"identical": central == flare}
    nums = {}
    for method, keys in (("v2", ("d0", "s02")), ("v3", ("grid", "pi", "em_loglik"))):
        for key in keys:
            a, b = central.get(method, {}).get(key), flare.get(method, {}).get(key)
            if isinstance(a, (int, float, list)) and isinstance(b, type(a)):
                nums[f"{method}.{key}"] = float(np.max(np.abs(np.subtract(a, b)), initial=0.0))
    out["max_abs_diff"] = nums

    def rest(d):  # everything except the float values compared above (fitted, iterations, …)
        return {m: {k: x for k, x in v.items() if f"{m}.{k}" not in nums}
                for m, v in d.items() if isinstance(v, dict)}
    # Same code on the same float64 inputs: expected identical. The tolerance is there only so
    # a last-bit difference would be reported as a number rather than as a bare failure.
    out["pass"] = out["identical"] or (rest(central) == rest(flare)
                                       and all(d <= config.ATOL for d in nums.values()))
    return out


def run(stratum: str, counts: str, root: Path | None = None, n_cpus: int = 8,
        verify_annotation: bool = True, clients: list[str] | None = None,
        drop_variants: list[str] | None = None) -> dict:
    flare_dir = config.results_dir(stratum, counts, "flare", root)
    out = config.results_dir(stratum, counts, "central", root)
    if out.exists():
        shutil.rmtree(out)
    gene_map = load_map(config.annotation_path(root), verify=verify_annotation)
    central = {}
    for client in clients or clients_in(stratum):
        site = load_site(client, config.manifest_dir(counts, root), config.data_root(root))
        res = run_de(site.counts, site.metadata, site.manifest["design"], n_cpus=n_cpus)
        central[client] = summarize(site, res, gene_map)
        print(f"{client}: {len(central[client].genes['ncbi_gene'])} genes", flush=True)
    write_summaries(central, out)
    variants = variants_for(DROP_VARIANTS.get(stratum, []) if drop_variants is None
                            else drop_variants)
    pooled = st.pool_variants({c: s.to_frame() for c, s in central.items()}, variants)
    write_pooled(central, pooled, out, source="central")

    sent = load_summaries(flare_dir / "sent")
    received = load_summaries(flare_dir / "site_summaries")
    transport = compare_summaries(sent, received)
    recompute = compare_summaries(central, received)
    pooled_cmp = compare_pooled(pooled, flare_dir)
    report = {
        "stratum": stratum, "counts": counts, "rtol": config.RTOL, "atol": config.ATOL,
        "transport_bitwise": all(all(v["bitwise_equal"] for k, v in r.items()
                                     if isinstance(v, dict)) and r["meta_equal"]
                                 for r in transport.values()),
        "recompute_bitwise": all(r.get("present_in_both") and all(
            v["bitwise_equal"] for k, v in r.items() if isinstance(v, dict))
            for r in recompute.values()),
        "pooled_pass": all(r["pass"] for r in pooled_cmp.values()),
        "transport": transport, "recompute": recompute, "pooled": pooled_cmp,
    }
    (out / "check1.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("transport_bitwise", "recompute_bitwise",
                                             "pooled_pass")}))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stratum", choices=["iss", "earth"], required=True)
    parser.add_argument("--counts", choices=["standard", "rrnarm"], default="standard")
    parser.add_argument("--n_cpus", type=int, default=8)
    args = parser.parse_args()
    report = run(args.stratum, args.counts, n_cpus=args.n_cpus)
    if not (report["transport_bitwise"] and report["pooled_pass"]):
        raise SystemExit("Check 1 FAILED; see check1.json")


if __name__ == "__main__":
    main()
