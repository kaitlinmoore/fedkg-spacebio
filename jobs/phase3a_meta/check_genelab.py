"""Check 2, site DE vs GeneLab: our pydeseq2 log2FC and SE vs GeneLab's published DE tables.

Run from WSL after ``fedkg``:

    uv run python jobs/phase3a_meta/check_genelab.py

This is an offline validation over public files, not site code. It reads a study's whole count
matrix for 2a.

For each census row a phase 3a client uses (10 source contrasts), it computes three things:

- **GeneLab**: ``Log2fc_<contrast>``, and SE = ``Log2fc_/Stat_`` (DESeq2's Wald statistic is
  log2FC / lfcSE). It's undefined where Stat = 0 or is missing.
- **2a, replica**: pydeseq2 with GeneLab's design: **every sample in the study's SampleTable**,
  ``~ condition`` over GeneLab's group labels, one fit per study, then the contrast. This
  isolates pydeseq2 vs R DESeq2. One known difference remains: GeneLab's normalized counts vary
  by gene within a sample (normalized ÷ raw has a CV of 7–10% at counts ≥ 100 in OSD-463), i.e.
  gene-specific normalization factors, consistent with ``DESeqDataSetFromTximport`` length
  offsets. A counts-only pydeseq2 fit has none.
- **2b, our design**: the row's own flight and ground samples only, ``~ condition``. For
  single-row clients this is exactly the client's site DE. For covariate clients (SpaceX-4
  study, SpaceX-16 age), it's the per-group run the handoff asks for.

Metrics are computed on genes with our base mean ≥ 10 (the primary inclusion rule) and finite
values on both sides:
- Pearson r and median |Δ| for log2FC and SE;
- the share of genes with |Δlog2FC| > 0.1;
- the 5 largest |Δlog2FC|.

Expect close but not identical.

Outputs: ``$FEDKG_WORKDIR/results/phase3a/check2/`` (``summary.csv``, ``summary.json``, one
per-gene CSV per contrast and arm).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402

from fedkg_spacebio.federation.site_de import MIN_BASE_MEAN, run_de_contrasts  # noqa: E402
from fedkg_spacebio.sites.loader import ERCC  # noqa: E402
from fedkg_spacebio.sites.manifest import CLIENTS, sample_table_name  # noqa: E402


def genelab_columns(de_path: Path, contrast: str) -> pd.DataFrame:
    lf, stc = f"Log2fc_{contrast}", f"Stat_{contrast}"
    t = pd.read_csv(de_path, usecols=["ENSEMBL", "SYMBOL", lf, stc], index_col=0)
    t = t.rename(columns={lf: "gl_log2fc", stc: "gl_stat"})
    with np.errstate(divide="ignore", invalid="ignore"):
        t["gl_se"] = np.where(t["gl_stat"] != 0, t["gl_log2fc"] / t["gl_stat"], np.nan)
    return t


WELL_ESTIMATED_SE = 2.0  # both SEs below this: excludes near-separation genes (one arm ~0)


def metrics_from(j: pd.DataFrame) -> dict:
    """Agreement metrics for one joined per-gene table (ours vs GeneLab).

    Pearson r is reported but is dominated by a handful of **near-separation** genes (one arm
    almost all zeros). There the MLE log2FC runs off to ±20 or more, and the two
    implementations disagree on the SE: pydeseq2 gives hundreds, R DESeq2 gives a few units.
    So the robust numbers (Spearman, quantiles of |Δ|, the median SE ratio) and the
    well-estimated subset are the ones to read."""
    from scipy import stats

    well = (j["se"] < WELL_ESTIMATED_SE) & (j["gl_se"] < WELL_ESTIMATED_SE)
    ad = j["d_log2fc"].abs()
    ratio = j["se"] / j["gl_se"]
    top = j.reindex(ad.sort_values(ascending=False).index).head(5)
    return {
        "n_genes": int(len(j)),
        "r_log2fc": float(np.corrcoef(j["log2fc"], j["gl_log2fc"])[0, 1]),
        "spearman_log2fc": float(stats.spearmanr(j["log2fc"], j["gl_log2fc"])[0]),
        "median_abs_d_log2fc": float(ad.median()),
        "p90_abs_d_log2fc": float(ad.quantile(0.9)),
        "p99_abs_d_log2fc": float(ad.quantile(0.99)),
        "frac_abs_d_log2fc_gt_0.1": float((ad > 0.1).mean()),
        "r_se": float(np.corrcoef(j["se"], j["gl_se"])[0, 1]),
        "spearman_se": float(stats.spearmanr(j["se"], j["gl_se"])[0]),
        "median_se_ratio": float(ratio.median()),
        "p05_se_ratio": float(ratio.quantile(0.05)),
        "p95_se_ratio": float(ratio.quantile(0.95)),
        "median_abs_d_se": float(j["d_se"].abs().median()),
        "n_near_separation": int((~well).sum()),
        "n_well_estimated": int(well.sum()),
        "r_log2fc_well": float(np.corrcoef(j.loc[well, "log2fc"],
                                           j.loc[well, "gl_log2fc"])[0, 1]),
        "r_se_well": float(np.corrcoef(j.loc[well, "se"], j.loc[well, "gl_se"])[0, 1]),
        "top_outliers": [{"ensembl": g, "symbol": r["SYMBOL"], "log2fc": r["log2fc"],
                          "gl_log2fc": r["gl_log2fc"], "se": r["se"], "gl_se": r["gl_se"],
                          "base_mean": r["base_mean"]} for g, r in top.iterrows()],
    }


def compare(ours: pd.DataFrame, gl: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    j = ours[["base_mean", "log2fc", "se"]].join(gl, how="inner")
    ok = (j["base_mean"] >= MIN_BASE_MEAN) & np.isfinite(j[["log2fc", "se", "gl_log2fc",
                                                           "gl_se"]]).all(axis=1)
    j = j[ok].copy()
    j["d_log2fc"] = j["log2fc"] - j["gl_log2fc"]
    j["d_se"] = j["se"] - j["gl_se"]
    return metrics_from(j), j


def summarize_saved(out: Path) -> pd.DataFrame:
    """Recompute the metrics from the saved per-gene tables (no DE rerun)."""
    rows = json.loads((out / "summary.json").read_text())
    for r in rows:
        f = out / f"{r['client']}__{r['osd_id']}__{r['comparison_id']}__{r['arm']}.csv.gz"
        r.update(metrics_from(pd.read_csv(f, index_col=0)))
    (out / "summary.json").write_text(json.dumps(rows, indent=1, default=float))
    flat = pd.DataFrame([{k: v for k, v in r.items() if k not in ("top_outliers", "warnings")}
                         for r in rows])
    flat.to_csv(out / "summary.csv", index=False)
    return flat


def load_counts(root: Path | None, osd: str, counts_file: str, samples: list[str]) -> pd.DataFrame:
    path = config.data_root(root) / osd / counts_file
    id_col = pd.read_csv(path, nrows=0).columns[0]
    t = pd.read_csv(path, usecols=[id_col, *samples], index_col=0)
    t = t.loc[~t.index.to_series().str.match(ERCC).to_numpy(), samples]
    return pd.DataFrame(np.rint(t.to_numpy(dtype=float)).astype(np.int64), index=t.index,
                        columns=samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cpus", type=int, default=16)
    parser.add_argument("--metrics_only", action="store_true",
                        help="recompute metrics from the saved per-gene tables")
    args = parser.parse_args()
    root = None
    out = config.workdir() / "results" / "phase3a" / "check2"
    out.mkdir(parents=True, exist_ok=True)
    if args.metrics_only:
        print(summarize_saved(out).drop(columns=["genelab_contrast"]).to_string())
        return
    mdir = config.manifest_dir("standard")

    # Collect the source contrasts: (client, row spec, census row, samples).
    jobs = []
    for client, spec in CLIENTS.items():
        m = json.loads((mdir / f"{client}.json").read_text())
        for rs, row in zip(spec.rows, m["census_rows"], strict=True):
            src = next(s for s in m["sources"] if s["osd_id"] == rs.osd_id)
            samples = [s for s in src["samples"]
                       if all(s.get(k) == v for k, v in rs.covariates.items())]
            jobs.append({"client": client, "osd": rs.osd_id, "row": row,
                         "counts_file": src["counts_file"].split("/", 1)[1],
                         "samples": samples})

    rows = []
    for osd in sorted({j["osd"] for j in jobs}):
        study_jobs = [j for j in jobs if j["osd"] == osd]
        counts_file = study_jobs[0]["counts_file"]
        st = pd.read_csv(config.data_root(root) / osd / sample_table_name(counts_file),
                         index_col=0)
        ct = pd.read_csv(config.data_root(root) / osd / counts_file.replace(
            "RSEM_Unnormalized_Counts_GLbulkRNAseq", "contrasts_GLbulkRNAseq"), index_col=0)
        levels = sorted(st["condition"].unique())
        code = {lab: f"g{i:02d}" for i, lab in enumerate(levels)}
        # 2a: one all-group fit per study, GeneLab's design.
        all_samples = list(st.index.astype(str))
        counts = load_counts(root, osd, counts_file, all_samples)
        meta = pd.DataFrame({"condition": [code[c] for c in st["condition"]]},
                            index=pd.Index(all_samples, name="sample"))
        contrasts = [("condition", code[ct[j["row"]["genelab_contrast"]].iloc[0]],
                      code[ct[j["row"]["genelab_contrast"]].iloc[1]]) for j in study_jobs]
        print(f"{osd}: replica fit on {len(all_samples)} samples, {len(levels)} groups",
              flush=True)
        replica = run_de_contrasts(counts, meta, "~ condition", contrasts, n_cpus=args.n_cpus)
        de_path = config.data_root(root) / osd / study_jobs[0]["row"]["de_file"]
        for j, rep in zip(study_jobs, replica, strict=True):
            gl = genelab_columns(de_path, j["row"]["genelab_contrast"])
            # 2b: this row's own two arms.
            names = [s["sample"] for s in j["samples"]]
            own = load_counts(root, osd, j["counts_file"], names)
            own_meta = pd.DataFrame({"condition": [s["arm"] for s in j["samples"]]},
                                    index=pd.Index(names, name="sample"))
            ours = run_de_contrasts(own, own_meta, "~ condition",
                                    [("condition", "flight", "ground")], n_cpus=args.n_cpus)[0]
            label = f"{j['client']}__{osd}__{j['row']['comparison_id']}"
            for arm, res in (("2a_replica", rep), ("2b_own_arms", ours)):
                metrics, table = compare(res, gl)
                table.to_csv(out / f"{label}__{arm}.csv.gz", float_format="%.10g")
                rows.append({"client": j["client"], "osd_id": osd,
                             "comparison_id": j["row"]["comparison_id"],
                             "genelab_contrast": j["row"]["genelab_contrast"], "arm": arm,
                             "n_samples_fit": len(all_samples) if arm == "2a_replica"
                             else len(names),
                             "warnings": res.attrs.get("warnings", []), **metrics})
                print(f"  {label} {arm}: r(log2FC)={metrics['r_log2fc']:.4f} "
                      f"med|Δ|={metrics['median_abs_d_log2fc']:.4f} "
                      f"r(SE)={metrics['r_se']:.4f} n={metrics['n_genes']}", flush=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=1, default=float))
    flat = pd.DataFrame([{k: v for k, v in r.items() if k not in ("top_outliers", "warnings")}
                         for r in rows])
    flat.to_csv(out / "summary.csv", index=False)
    print(flat.drop(columns=["genelab_contrast", "comparison_id"]).to_string())


if __name__ == "__main__":
    main()
