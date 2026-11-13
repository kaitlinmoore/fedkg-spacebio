"""Phase 3a results: stable per-gene tables, headline numbers, forest and volcano plots.

Run after job.py (and central.py) for a stratum:

    uv run python jobs/phase3a_meta/report.py --stratum iss [--counts standard]
    uv run python jobs/phase3a_meta/report.py --sensitivity      # cross-run comparison table

It reads the FLARE server's outputs (``…/<stratum>/<counts>/flare/``) and writes, next to them:
- ``<variant>/per_gene.csv.gz`` and ``per_gene.schema.json``: the stable per-gene table
  (``fedkg_spacebio.federation.results``): pooled FE/RE columns, Q, I², τ², hit and
  heterogeneity flags, and each site's log2FC, SE and inclusion. The symbol comes from the
  pinned GeneLab annotation table. It's a display label, never a join key;
- ``headline.json``: hit counts from ``results.hit_counts`` (FE headline, consistent vs
  heterogeneous, RE secondary), τ² and I² distributions, the share of genes where HKSJ
  truncation applied, REML convergence, and the |t| that truncated HKSJ would need to pass BH;
- ``primary/forest_top10.png`` (top 10 by FE p), ``primary/forest_top10_consistent.png`` (top 10
  FE hits without the heterogeneity flag), ``primary/volcano.png`` (FE, the headline) and
  ``primary/volcano_re.png`` (truncated HKSJ, secondary).

Headline: fixed effect, BH q < 0.05, with a per-gene heterogeneity flag (Cochran's Q p < 0.10).
This replaced the pre-specified RE-HKSJ headline after the results were seen (2026-11-13,
Kaitlin); see ``results.py`` and ``docs/phase3a-liver-meta.md``.

Top genes are **results only**. No biological claim is made here (a literature check comes
later).

Plot styling follows the reference data-viz palette: light surface ``#fcfcfb``, text inks
``#0b0b0b`` / ``#52514e``, fixed effect in orange ``#eb6834`` and random effects in blue
``#2a78d6``. Site identity is carried by text labels, never by color. In the FE volcano,
heterogeneity is carried by marker shape as well as color.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402

from fedkg_spacebio.federation import results as rs  # noqa: E402
from fedkg_spacebio.federation.meta_flare import load_summaries  # noqa: E402
from fedkg_spacebio.harmonize.gene_map import load_map  # noqa: E402

Q = rs.FDR_Q
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#a8a7a2", "#e4e3df"
RE_COLOR, FE_COLOR = "#2a78d6", "#eb6834"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def quantiles(x) -> dict:
    x = pd.Series(x).dropna()
    if x.empty:
        return {}
    return {f"q{int(p * 100):02d}": float(x.quantile(p)) for p in (0.05, 0.25, 0.5, 0.75, 0.95)}


def t_needed(m: int, df: int, rank: int) -> float:
    """|t| a gene at BH rank ``rank`` of ``m`` would need (two-sided, t with ``df``) to pass."""
    return float(stats.t.isf(Q * rank / m / 2, df))


def headline(t: pd.DataFrame) -> dict:
    out = rs.hit_counts(t)
    k = int(t["k"].max()) if len(t) else 0
    m = int(t["fe_p"].notna().sum())
    abs_t = (t["re_est"] / t["hksj_se"]).abs()
    out.update({
        "k_values": sorted(int(x) for x in t["k"].unique()),
        "re_p_lt_0.05": int((t["re_p"] < 0.05).sum()),
        "re_p_untruncated_lt_0.05": int((t["re_p_untruncated"] < 0.05).sum()),
        "fe_p_lt_0.05": int((t["fe_p"] < 0.05).sum()),
        "min_fe_q": float(t["fe_q"].min()), "min_re_q": float(t["re_q"].min()),
        "max_abs_t_hksj": float(abs_t.max()),
        "hksj_t_needed": ({"rank_1": t_needed(m, k - 1, 1), "rank_100": t_needed(m, k - 1, 100),
                           "rank_1000": t_needed(m, k - 1, 1000)} if k >= 2 and m else {}),
        "tau2": quantiles(t["tau2"]), "i2": quantiles(t["i2"]),
        "frac_tau2_zero": float((t["tau2"] == 0).mean()),
        "frac_i2_gt_0.5": float((t["i2"] > 0.5).mean()),
        "frac_hksj_truncated": float((t["hksj_multiplier"] < 1).mean()),
        "reml_not_converged": int((~t["reml_converged"].astype(bool)).sum()),
    })
    return out


def top_genes(t: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top n by the headline: FE p (ties, at p ≈ 0, broken by larger |FE estimate|)."""
    return (t.assign(_abs=t["fe_est"].abs()).sort_values(["fe_p", "_abs"], ascending=[True, False])
            .head(n).drop(columns="_abs"))


def forest(top: pd.DataFrame, frames: dict[str, pd.DataFrame], path: Path, title: str):
    clients = sorted(frames)
    n = len(top)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 0.42 * (len(clients) + 3) * nrow
                                                  + 0.8), sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()
    ylab = [*clients, "Fixed effect", "Random effects (HKSJ)"]
    ypos = np.arange(len(ylab))[::-1]
    for ax, (_, g) in zip(axes, top.iterrows(), strict=False):
        style(ax)
        gid = int(g["ncbi_gene"])
        for c, y in zip(clients, ypos[:len(clients)], strict=True):
            f = frames[c]
            if bool(g.get(f"included__{c}", False)):
                e, s = f.loc[gid, "log2fc"], f.loc[gid, "se"]
                ax.plot([e - 1.96 * s, e + 1.96 * s], [y, y], color=INK2, lw=1.2)
                ax.plot(e, y, "s", color=INK2, ms=4)
        y_fe, y_re = ypos[-2], ypos[-1]
        fe_lo, fe_hi = g["fe_est"] - 1.96 * g["fe_se"], g["fe_est"] + 1.96 * g["fe_se"]
        ax.plot([fe_lo, fe_hi], [y_fe, y_fe], color=FE_COLOR, lw=2)
        ax.plot(g["fe_est"], y_fe, "D", color=FE_COLOR, ms=5)
        ax.plot([g["hksj_ci_low"], g["hksj_ci_high"]], [y_re, y_re], color=RE_COLOR, lw=2)
        ax.plot(g["re_est"], y_re, "D", color=RE_COLOR, ms=5)
        ax.axvline(0, color=MUTED, lw=0.8)
        het = "  heterogeneous" if g["het_flag"] else ""
        ax.set_title(f"{g['symbol']}  (NCBIGene:{gid})\nq_FE={g['fe_q']:.2g}  Q p={g['q_p']:.2g}"
                     f"  I²={g['i2']:.0%}{het}", fontsize=8, color=INK)
        ax.set_xlabel("log2 fold change (flight vs ground)", fontsize=7, color=INK2)
    for ax in axes[n:]:
        ax.set_visible(False)
    axes[0].set_yticks(ypos, ylab, fontsize=7, color=INK)
    for row in range(nrow):
        axes[row * ncol].set_yticks(ypos, ylab, fontsize=7, color=INK)
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.005, "Site bars: 95% Wald CI (± 1.96 SE). Pooled: FE 95% CI (headline); "
             "random effects REML τ², truncated HKSJ t(k−1) 95% CI (secondary). Heterogeneous = "
             f"Cochran's Q p < {rs.HET_Q_P}. Results only, not biological claims.",
             fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def volcano(t: pd.DataFrame, path: Path, title: str):
    """Headline volcano: FE estimate vs −log10 FE p. Hits split by the heterogeneity flag."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6), facecolor=SURFACE)
    style(ax)
    y = -np.log10(t["fe_p"].clip(lower=1e-300))
    none, cons, het = ~t["fe_hit"], t["fe_hit_consistent"], t["fe_hit_heterogeneous"]
    ax.scatter(t.loc[none, "fe_est"], y[none], s=6, color=MUTED, alpha=0.6, linewidths=0,
               label=f"q ≥ {Q} ({int(none.sum()):,})")
    ax.scatter(t.loc[cons, "fe_est"], y[cons], s=10, color=FE_COLOR, linewidths=0,
               label=f"q < {Q}, consistent ({int(cons.sum()):,})")
    ax.scatter(t.loc[het, "fe_est"], y[het], s=14, marker="^", facecolors="none",
               edgecolors=INK2, linewidths=0.7,
               label=f"q < {Q}, heterogeneous: Q p < {rs.HET_Q_P} ({int(het.sum()):,})")
    for _, g in top_genes(t, 8).iterrows():
        ax.annotate(str(g["symbol"]), (g["fe_est"], -np.log10(max(g["fe_p"], 1e-300))),
                    fontsize=7, color=INK, xytext=(3, 2), textcoords="offset points")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("pooled log2 fold change, fixed effect (flight vs ground)", fontsize=8,
                  color=INK2)
    ax.set_ylabel("−log10 p (fixed effect)", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def volcano_re(t: pd.DataFrame, path: Path, title: str):
    """Secondary volcano: RE estimate vs −log10 truncated-HKSJ p."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6), facecolor=SURFACE)
    style(ax)
    sig = t["re_hit"]
    y = -np.log10(t["re_p"].clip(lower=1e-300))
    ax.scatter(t.loc[~sig, "re_est"], y[~sig], s=6, color=MUTED, alpha=0.6, linewidths=0,
               label=f"q ≥ {Q} ({int((~sig).sum()):,})")
    ax.scatter(t.loc[sig, "re_est"], y[sig], s=10, color=RE_COLOR, linewidths=0,
               label=f"q < {Q} ({int(sig.sum()):,})")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("pooled log2 fold change, random effects (flight vs ground)", fontsize=8,
                  color=INK2)
    ax.set_ylabel("−log10 p (HKSJ, truncated)", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def report(stratum: str, counts: str, root: Path | None = None) -> dict:
    base = config.results_dir(stratum, counts, "flare", root)
    gmap = load_map(config.annotation_path(root))
    symbols = gmap.table.dropna(subset=["ncbi_gene"]).set_index("ncbi_gene")["symbol"]
    summaries = load_summaries(base / "site_summaries")
    frames = {c: s.to_frame() for c, s in summaries.items()}
    run = json.loads((base / "pooling_run.json").read_text())
    out = {"stratum": stratum, "counts": counts, "clients": sorted(frames),
           "headline": f"FE BH q < {Q}; heterogeneity flag Q p < {rs.HET_Q_P}",
           "thresholds": {"fdr_q": Q, "het_q_p": rs.HET_Q_P}, "variants": {}}
    for vdir in sorted(p for p in base.iterdir() if (p / "meta_genes.csv.gz").exists()):
        pooled = pd.read_csv(vdir / "meta_genes.csv.gz")
        var = json.loads((vdir / "variant.json").read_text())
        t = rs.per_gene_table(pooled, frames, var["flag"], var["clients"], symbols,
                              site_n={c: tuple(run["n"][c]) for c in var["clients"]})
        t.to_csv(vdir / "per_gene.csv.gz", index=False, float_format="%.17g")
        prov = {"stratum": stratum, "counts": counts, "variant": var["variant"],
                "clients": var["clients"], "inclusion_flag": var["flag"],
                "min_sites": var["min_sites"], "pooling_source": run["source"],
                "n_flight_ground": {c: run["n"][c] for c in var["clients"]},
                "site_versions": {c: summaries[c].meta["versions"] for c in var["clients"]},
                "annotation_sha256": gmap.sha256, "written": date.today().isoformat()}
        (vdir / "per_gene.schema.json").write_text(json.dumps(rs.schema(t, prov), indent=1))
        (vdir / "meta_genes_annotated.csv.gz").unlink(missing_ok=True)  # superseded
        out["variants"][vdir.name] = headline(t)
        if vdir.name == "primary":
            label = (f"{'ISS' if stratum == 'iss' else 'Earth'} stratum, {counts} counts, "
                     f"k = {len(frames)} sites, {len(t):,} genes")
            top = top_genes(t)
            out["top10"] = top[["ncbi_gene", "symbol", "fe_est", "fe_se", "fe_q", "q_p", "i2",
                                "het_flag", "re_est", "hksj_ci_low", "hksj_ci_high", "re_q",
                                "tau2"]].to_dict("records")
            forest(top, frames, vdir / "forest_top10.png", f"Top 10 genes by FE p: {label}")
            cons = top_genes(t[t["fe_hit_consistent"]])
            out["top10_consistent"] = cons[["ncbi_gene", "symbol", "fe_est", "fe_se", "fe_q",
                                            "q_p", "i2", "re_est", "re_q", "tau2"]
                                           ].to_dict("records")
            forest(cons, frames, vdir / "forest_top10_consistent.png",
                   f"Top 10 consistent FE hits (Q p ≥ {rs.HET_Q_P}) by FE p: {label}")
            volcano(t, vdir / "volcano.png", f"Fixed effect (headline): {label}")
            volcano_re(t, vdir / "volcano_re.png", f"Random effects, HKSJ (secondary): {label}")
    (base / "headline.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def corr(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna()
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else float("nan")


def sensitivity(root: Path | None = None) -> pd.DataFrame:
    """One row per (stratum, counts, variant): hit counts (``results.hit_counts``), overlap of
    FE hits with the primary analysis (standard counts), and agreement of estimates."""
    rows = []
    for stratum in ("iss", "earth"):
        prim = config.results_dir(stratum, "standard", "flare", root) / "primary"
        if not (prim / "per_gene.csv.gz").exists():
            continue
        p = pd.read_csv(prim / "per_gene.csv.gz").set_index("ncbi_gene")
        p_hits = set(p.index[p["fe_hit"]])
        p_cons = set(p.index[p["fe_hit_consistent"]])
        for counts in ("standard", "rrnarm"):
            base = config.results_dir(stratum, counts, "flare", root)
            for f in sorted(base.glob("*/per_gene.csv.gz")):
                t = pd.read_csv(f).set_index("ncbi_gene")
                common = p.index.intersection(t.index)
                hits, cons = set(t.index[t["fe_hit"]]), set(t.index[t["fe_hit_consistent"]])
                rows.append({
                    "stratum": stratum, "counts": counts, "variant": f.parent.name,
                    **rs.hit_counts(t.reset_index()),
                    "fe_hits_shared_with_primary": len(hits & p_hits),
                    "consistent_shared_with_primary": len(cons & p_cons),
                    "r_fe_est_vs_primary": corr(p.loc[common, "fe_est"], t.loc[common, "fe_est"]),
                    "r_re_est_vs_primary": corr(p.loc[common, "re_est"], t.loc[common, "re_est"]),
                    "median_tau2": float(t["tau2"].median()),
                    "median_i2": float(t["i2"].median()),
                })
    df = pd.DataFrame(rows)
    out = (root or config.workdir()) / "results" / "phase3a" / "sensitivity.csv"
    df.to_csv(out, index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stratum", choices=["iss", "earth"])
    parser.add_argument("--counts", choices=["standard", "rrnarm"], default="standard")
    parser.add_argument("--sensitivity", action="store_true")
    args = parser.parse_args()
    if args.sensitivity:
        print(sensitivity().to_string())
    else:
        out = report(args.stratum, args.counts)
        print(json.dumps(out["variants"], indent=1, default=float))


if __name__ == "__main__":
    main()
