"""Exactness check for the simulated secure sum (CONTRACTS §6.3): masked FE (+Q) vs plain.

Run from WSL after ``fedkg``, once the flag-driven phase 3a runs exist:

    uv run python jobs/phase3a_meta/check_secure_sum.py
    uv run python jobs/phase3a_meta/check_secure_sum.py --flare_run <…/jobs/<job slug>> [...]

**Part 1, in process, no DE rerun.** For each stratum × counts variant, it reads the per-site
summaries the plain FLARE server received (``…/flare/site_summaries/``), builds the gene plan from
their flags for two inclusion variants (``primary``: every site; ``min3_sites``: ≥ 3 sites,
which exercises the zeros sent for unassigned genes), and runs the client and server halves of
the secure sum in this process (encode with overflow check, mask, sum, decode, FE and Q). It
compares the result with
- (a) plain pooling of the same summaries (``meta.pool``, classic columns), and
- (b) the ``fe_*``, ``q``, ``q_p``, ``i2`` columns of the existing schema-2 table for that variant
  (``…/flare/<variant>/per_gene.csv.gz``).

**Part 2, ``--flare_run``.** For a secure-sum spec run made by ``job.py --spec``, it compares the
server's table with plain pooling of the same run's site-local summaries (``sites/*/local``: the
same DE results the masked vectors were built from), and with the open-profile ``primary`` table.

Pass criteria, per gene:
- |Δ| ≤ the worst-case rounding bound (``secure_sum.error_bound``, from the plain values) plus
  64 float64 ulps of slack for the plain side's own rounding, for the estimate, SE and Q;
- identical gene sets, ``k``, ``fe_hit`` sets and ``het_flag`` sets.
Also reported: max |Δp|/p, max |Δq|, and how close the nearest gene sits to the q = 0.05
threshold (how far a rounding error would have to move it to flip a hit).

Output: ``$FEDKG_WORKDIR/results/phase3a/secure_sum_check.json`` (and the summary on stdout).
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402

from fedkg_spacebio.federation import meta  # noqa: E402
from fedkg_spacebio.federation import secure_sum as ss  # noqa: E402
from fedkg_spacebio.federation.gene_plan import build_gene_plan  # noqa: E402
from fedkg_spacebio.federation.meta_flare import load_summaries  # noqa: E402
from fedkg_spacebio.federation.results import FDR_Q, HET_Q_P  # noqa: E402
from fedkg_spacebio.sites.inclusion import FLAGS  # noqa: E402

PARAMS = ss.Params(40, 64)
ULP_SLACK = 64 * np.finfo(np.float64).eps
VARIANTS = {"primary": "all", "min3_sites": 3}


def flags_of(summaries) -> dict[str, pd.DataFrame]:
    return {c: pd.DataFrame({k: s.genes[k] for k in ("ncbi_gene", *FLAGS)})
            for c, s in summaries.items()}


def plain_pool(genes_by_client: dict[str, dict], rows: pd.DataFrame) -> pd.DataFrame:
    clients = sorted(rows.columns)
    Y = np.full((len(rows), len(clients)), np.nan)
    V = np.full_like(Y, np.nan)
    for j, c in enumerate(clients):
        g = genes_by_client[c]
        on = rows[c].to_numpy(dtype=bool)
        pos = pd.Index(g["ncbi_gene"]).get_indexer(rows.index[on])
        Y[on, j] = np.asarray(g["log2fc"])[pos]
        V[on, j] = np.asarray(g["se"])[pos] ** 2
    out = meta.pool(Y, V, eb=False)
    out.insert(0, "ncbi_gene", rows.index.to_numpy())
    return out.reset_index(drop=True)


def masked_pool(genes_by_client: dict[str, dict], rows: pd.DataFrame) -> pd.DataFrame:
    """Both halves of the secure sum, in process, with the running totals checked too."""
    clients = sorted(rows.columns)
    seed, job = secrets.token_bytes(32), "job:FAKE-0000"
    payloads, running = {}, {q: np.zeros(len(rows), dtype=object)
                             for q in ss.QUANTITIES_FE_HET}
    limit = (1 << (PARAMS.modulus_bits - 1)) - 1
    for c in clients:
        vals = ss.contribution_values(genes_by_client[c], rows, c, "tested_mean10",
                                      ss.QUANTITIES_FE_HET)
        for q, v in vals.items():  # exact integer running totals (Python ints)
            running[q] = running[q] + np.array([int(x) for x in np.rint(v * PARAMS.scale)],
                                               dtype=object)
            if max(abs(int(t)) for t in running[q]) > limit:
                raise SystemExit(f"running total of {q} out of range after {c}")
        payloads[c] = ss.masked_contribution(vals, c, clients, root_seed=seed, job_id=job,
                                             params=PARAMS)
    return ss.fe_from_totals(ss.aggregate(payloads, rows, PARAMS))


def compare(masked: pd.DataFrame, ref: pd.DataFrame, label: str) -> dict:
    """Per-gene comparison of a masked table with a plain reference (same rows required)."""
    same_genes = np.array_equal(masked["ncbi_gene"].to_numpy(), ref["ncbi_gene"].to_numpy())
    if not same_genes:
        return {"label": label, "pass": False, "same_genes": False,
                "n_masked": int(len(masked)), "n_ref": int(len(ref))}
    est, sw = ref["fe_est"].to_numpy(), 1 / ref["fe_se"].to_numpy() ** 2
    b = ss.error_bound(est, sw, ref["k"].to_numpy(), PARAMS.fixed_point_bits)
    out = {"label": label, "same_genes": True, "n_genes": int(len(ref)),
           "same_k": bool(np.array_equal(masked["k"], ref["k"]))}
    ok = out["same_k"]
    checks = {
        "fe_est": b["est"] + ULP_SLACK * (1 + np.abs(est)),
        "fe_se": b["se"] + ULP_SLACK * ref["fe_se"].to_numpy(),
    }
    if "q" in masked and "q" in ref:
        checks["q"] = b["q"] + ULP_SLACK * (1 + ref["q"].to_numpy())
    for col, bound in checks.items():
        d = np.abs(masked[col].to_numpy() - ref[col].to_numpy())
        use = np.isfinite(d)
        ratio = d[use] / bound[use]
        out[col] = {"max_abs_diff": float(d[use].max()), "max_bound": float(bound[use].max()),
                    "max_diff_over_bound": float(ratio.max()), "within_bound": bool(
                        (ratio <= 1).all())}
        ok &= out[col]["within_bound"]
    p_m, p_r = masked["fe_p"].to_numpy(), ref["fe_p"].to_numpy()
    pos = p_r > 0
    out["fe_p_max_rel_diff"] = float(np.nanmax(np.abs(p_m[pos] - p_r[pos]) / p_r[pos]))
    out["fe_q_max_abs_diff"] = float(np.nanmax(np.abs(masked["fe_q"] - ref["fe_q"])))
    hit_m, hit_r = masked["fe_q"] < FDR_Q, ref["fe_q"] < FDR_Q
    out["fe_hit"] = {"masked": int(hit_m.sum()), "plain": int(hit_r.sum()),
                     "identical": bool(np.array_equal(hit_m, hit_r)),
                     "nearest_q_to_threshold": float(np.nanmin(np.abs(ref["fe_q"] - FDR_Q)))}
    ok &= out["fe_hit"]["identical"]
    if "q_p" in masked and "q_p" in ref:
        het_m, het_r = masked["q_p"] < HET_Q_P, ref["q_p"] < HET_Q_P
        out["het_flag"] = {"masked": int(het_m.sum()), "plain": int(het_r.sum()),
                           "identical": bool(np.array_equal(het_m, het_r)),
                           "q_p_max_abs_diff": float(np.nanmax(np.abs(masked["q_p"]
                                                                       - ref["q_p"]))),
                           "i2_max_abs_diff": float(np.nanmax(np.abs(masked["i2"]
                                                                     - ref["i2"]))),
                           "nearest_q_p_to_threshold": float(np.nanmin(np.abs(ref["q_p"]
                                                                              - HET_Q_P)))}
        ok &= out["het_flag"]["identical"]
    out["n_q_clamped_to_zero"] = int(masked.attrs.get("n_q_clamped", 0))
    out["pass"] = bool(ok)
    return out


def part1(root: Path | None) -> list[dict]:
    reports = []
    for stratum in ("iss", "earth"):
        for counts in ("standard", "rrnarm"):
            flare = config.results_dir(stratum, counts, "flare", root)
            summaries = load_summaries(flare / "site_summaries")
            genes = {c: s.genes for c, s in summaries.items()}
            for variant, min_sites in VARIANTS.items():
                rows = build_gene_plan(flags_of(summaries), "mean10_finite", min_sites)
                masked = masked_pool(genes, rows)
                label = f"{stratum}/{counts}/{variant}"
                reports.append(compare(masked, plain_pool(genes, rows), f"{label} vs plain"))
                table = pd.read_csv(flare / variant / "per_gene.csv.gz")
                reports.append(compare(masked, table, f"{label} vs schema-2 table"))
    return reports


def part2(run_dir: Path, root: Path | None) -> list[dict]:
    run_dir = Path(run_dir)
    spec = json.loads((run_dir / "job_spec.json").read_text())["spec"]
    server = pd.read_csv(run_dir / "server" / "per_gene.csv.gz")
    clients = sorted(spec["clients"])
    local = {}
    for c in clients:
        with np.load(run_dir / "sites" / c / "local" / "summary.npz") as z:
            local[c] = {k: z[k] for k in z.files}
    rows = pd.DataFrame({c: server[f"included__{c}"].to_numpy(dtype=bool) for c in clients},
                        index=pd.Index(server["ncbi_gene"].to_numpy(), name="ncbi_gene"))
    label = f"FLARE {run_dir.name} ({spec['stratum']}/{spec['counts_variant']})"
    reports = [compare(server, plain_pool(local, rows), f"{label} vs its site-local summaries")]
    if spec["inclusion"]["min_sites_per_gene"] == "all":
        primary = pd.read_csv(config.results_dir(spec["stratum"], spec["counts_variant"],
                                                 "flare", root) / "primary" / "per_gene.csv.gz")
        reports.append(compare(server, primary, f"{label} vs open-profile primary table"))
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flare_run", nargs="*", default=[], type=Path)
    parser.add_argument("--skip_part1", action="store_true")
    args = parser.parse_args()
    reports = [] if args.skip_part1 else part1(None)
    for d in args.flare_run:
        reports += part2(d, None)
    for r in reports:
        if not r.get("same_genes", False):
            print(f"FAIL {r['label']}: gene sets differ")
            continue
        line = (f"{'PASS' if r['pass'] else 'FAIL'} {r['label']}: {r['n_genes']} genes; "
                f"max|Δest| {r['fe_est']['max_abs_diff']:.2g} "
                f"({r['fe_est']['max_diff_over_bound']:.2f} of bound); "
                f"fe_hit {r['fe_hit']['masked']}/{r['fe_hit']['plain']} "
                f"identical={r['fe_hit']['identical']}")
        if "q" in r:
            line += (f"; max|ΔQ| {r['q']['max_abs_diff']:.2g} "
                     f"({r['q']['max_diff_over_bound']:.2f} of bound); het_flag identical="
                     f"{r['het_flag']['identical']}")
        print(line)
    out = config.workdir() / "results" / "phase3a" / "secure_sum_check.json"
    existing = json.loads(out.read_text()) if out.exists() and args.skip_part1 else {}
    payload = {**existing, "params": PARAMS.to_dict(), "ulp_slack": ULP_SLACK}
    key = "flare_runs" if args.skip_part1 else "in_process"
    payload[key] = reports
    if not args.skip_part1 and args.flare_run:
        payload["flare_runs"] = [r for r in reports if r["label"].startswith("FLARE")]
        payload["in_process"] = [r for r in reports if not r["label"].startswith("FLARE")]
    out.write_text(json.dumps(payload, indent=1))
    print("Wrote", out)
    if not all(r["pass"] for r in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
