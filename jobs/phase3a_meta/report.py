"""Phase 3a results: stable per-gene tables, headline numbers, forest and volcano plots.

Run after job.py (and central.py) for a stratum:

    uv run python jobs/phase3a_meta/report.py --stratum iss [--counts standard]
    uv run python jobs/phase3a_meta/report.py --sensitivity      # cross-run comparison table

It reads the FLARE server's outputs (``…/<stratum>/<counts>/flare/``) and writes, next to them:
- ``<variant>/per_gene.csv.gz`` and ``per_gene.schema.json``: the stable per-gene table
  (``fedkg_spacebio.federation.results``, schema ``phase3a-per-gene/2``): pooled FE, V3, V2 and
  REML/HKSJ columns, Q, I², τ², hit and heterogeneity flags, and each site's log2FC, SE and
  inclusion. The schema file also carries the V2/V3 priors fitted for that variant. The symbol
  comes from the pinned GeneLab annotation table. It's a display label, never a join key;
- ``headline.json``: which method answers which question, then hit counts from
  ``results.hit_counts`` (FE, V3, V2, the descriptive FE split by heterogeneity, RE-HKSJ), the
  V2/V3 priors, τ² and I² distributions, the share of genes where HKSJ truncation applied,
  REML convergence, and the |t| that truncated HKSJ would need to pass BH;
- ``primary/forest_top10.png`` (top 10 by FE p), ``primary/forest_top10_consistent.png`` (top 10
  FE hits without the heterogeneity flag; descriptive), ``primary/volcano.png`` (FE) and
  ``primary/volcano_v3.png`` (V3).

Which result answers which question (D-029): **FE (BH q < 0.05) for "these missions"**, with
Cochran's Q p and I² as annotations, not filters; **V3 (empirical-Bayes τ²) for "a new
mission"**; V2 as the sensitivity run for misestimated SEs. REML + truncated HKSJ, the
pre-specified headline, stays in the table for comparison; V3 replaces it in the plots. Both
changes of method were made after the results were seen (D-026, then D-029); see
``results.py`` and ``docs/phase3a-liver-meta.md``. Every statement of the result carries
D-035's caveat: consistent with a time-of-day mismatch of a few hours or a clock change; not
separable.

Top genes are **results only**. No biological claim is made here (a literature check comes
later).

Plot styling follows the reference data-viz palette: light surface ``#fcfcfb``, text inks
``#0b0b0b`` / ``#52514e``, fixed effect in orange ``#eb6834`` and V3 (the random-effects,
"new mission" result) in blue ``#2a78d6``. Site identity is carried by text labels, never by
color. In the FE volcano, heterogeneity is carried by marker shape as well as color.
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
QUESTIONS = {
    "these_missions": f"fixed effect (inverse variance), BH q < {Q} (fe_hit)",
    "a_new_mission": f"V3, empirical-Bayes tau2, BH q < {Q} (v3_hit)",
    "sensitivity_misestimated_se": f"V2, moderated multiplicative heterogeneity, BH q < {Q} "
                                   "(v2_hit)",
    "annotations_not_filters": f"Cochran's Q p (het_flag: q_p < {rs.HET_Q_P}) and I^2",
    "kept_for_comparison": "REML + truncated HKSJ (re_hit); FE hits split by het_flag",
    "decisions": "D-029 (amends D-026); both method changes were made after seeing results",
    "caveat": "Consistent with a time-of-day mismatch of a few hours or a clock change; not "
              "separable (D-035).",
}
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#a8a7a2", "#e4e3df"
V3_COLOR, FE_COLOR = "#2a78d6", "#eb6834"


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
        "min_v3_q": float(t["v3_q"].min()),
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
    """Top n by FE p, the "these missions" result.

    Ties (at p ≈ 0) are broken by the larger |FE estimate|.
    """
    return (t.assign(_abs=t["fe_est"].abs()).sort_values(["fe_p", "_abs"], ascending=[True, False])
            .head(n).drop(columns="_abs"))


def forest(top: pd.DataFrame, frames: dict[str, pd.DataFrame], path: Path, title: str):
    clients = sorted(frames)
    n = len(top)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.8 * ncol, 0.42 * (len(clients) + 4) * nrow
                                                  + 1.2), sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()
    ylab = [*clients, "Fixed effect", "V3 (new mission)"]
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
        y_fe, y_v3 = ypos[-2], ypos[-1]
        fe_lo, fe_hi = g["fe_est"] - 1.96 * g["fe_se"], g["fe_est"] + 1.96 * g["fe_se"]
        ax.plot([fe_lo, fe_hi], [y_fe, y_fe], color=FE_COLOR, lw=2)
        ax.plot(g["fe_est"], y_fe, "D", color=FE_COLOR, ms=5)
        ax.plot([g["v3_ci_low"], g["v3_ci_high"]], [y_v3, y_v3], color=V3_COLOR, lw=2)
        ax.plot(g["v3_est"], y_v3, "D", color=V3_COLOR, ms=5)
        ax.axvline(0, color=MUTED, lw=0.8)
        het = "  (Q p < 0.1)" if g["het_flag"] else ""
        # Three short lines, so neighbouring panel titles don't overlap.
        ax.set_title(f"{g['symbol']}  (NCBIGene:{gid})\nq_FE={g['fe_q']:.2g}   q_V3={g['v3_q']:.2g}"
                     f"\nQ p={g['q_p']:.2g}   I²={g['i2']:.0%}{het}", fontsize=7.5, color=INK)
        ax.set_xlabel("log2 fold change (flight vs ground)", fontsize=7, color=INK2)
    for ax in axes[n:]:
        ax.set_visible(False)
    axes[0].set_yticks(ypos, ylab, fontsize=7, color=INK)
    for row in range(nrow):
        axes[row * ncol].set_yticks(ypos, ylab, fontsize=7, color=INK)
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.005, "Site bars: 95% Wald CI (± 1.96 SE). Pooled: FE 95% CI (these missions); "
             "V3 posterior mean and 95% credible interval (a new mission).\nQ p and I² are "
             "annotations, not filters. Consistent with a time-of-day mismatch of a few hours or "
             "a clock change; not separable. Results only, not biological claims.",
             fontsize=7, color=INK2, va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def volcano(t: pd.DataFrame, path: Path, title: str):
    """FE volcano ("these missions"): FE estimate vs −log10 FE p. Hits are split by the
    heterogeneity flag for description only; a flagged gene is still an FE hit."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6), facecolor=SURFACE)
    style(ax)
    y = -np.log10(t["fe_p"].clip(lower=1e-300))
    none, cons, het = ~t["fe_hit"], t["fe_hit_consistent"], t["fe_hit_heterogeneous"]
    ax.scatter(t.loc[none, "fe_est"], y[none], s=6, color=MUTED, alpha=0.6, linewidths=0,
               label=f"q ≥ {Q} ({int(none.sum()):,})")
    ax.scatter(t.loc[cons, "fe_est"], y[cons], s=10, color=FE_COLOR, linewidths=0,
               label=f"q < {Q}, Q p ≥ {rs.HET_Q_P} ({int(cons.sum()):,})")
    ax.scatter(t.loc[het, "fe_est"], y[het], s=14, marker="^", facecolors="none",
               edgecolors=INK2, linewidths=0.7,
               label=f"q < {Q}, Q p < {rs.HET_Q_P} ({int(het.sum()):,})")
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


def volcano_v3(t: pd.DataFrame, path: Path, title: str):
    """V3 volcano ("a new mission"): posterior mean vs −log10 p_EB."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6), facecolor=SURFACE)
    style(ax)
    sig = t["v3_hit"]
    y = -np.log10(t["v3_p"].clip(lower=1e-300))
    ax.scatter(t.loc[~sig, "v3_est"], y[~sig], s=6, color=MUTED, alpha=0.6, linewidths=0,
               label=f"q ≥ {Q} ({int((~sig).sum()):,})")
    ax.scatter(t.loc[sig, "v3_est"], y[sig], s=10, color=V3_COLOR, linewidths=0,
               label=f"q < {Q} ({int(sig.sum()):,})")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("V3 posterior mean log2 fold change (flight vs ground)", fontsize=8,
                  color=INK2)
    ax.set_ylabel("−log10 p (V3, empirical-Bayes τ²)", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def prior_summary(var: dict) -> dict:
    """The fitted V2/V3 priors in a readable form (the full grid and π are in the schema)."""
    pri = var.get("eb_priors") or {}
    out = {"n_genes": pri.get("n_genes")}
    v2, v3 = pri.get("v2", {}), pri.get("v3", {})
    out["v2"] = ({"d0": v2["d0"], "s02": v2["s02"]} if v2.get("fitted")
                 else {"fitted": False, "reason": v2.get("reason")})
    if v3.get("fitted"):
        grid, pi = np.asarray(v3["grid"]), np.asarray(v3["pi"])
        out["v3"] = {"prior_mass_tau2_zero": float(pi[grid == 0].sum()),
                     "prior_mean_tau2": float((grid * pi).sum()),
                     "em_iterations": v3["em_iterations"], "em_converged": v3["em_converged"]}
    else:
        out["v3"] = {"fitted": False, "reason": v3.get("reason")}
    return out


def report(stratum: str, counts: str, root: Path | None = None) -> dict:
    base = config.results_dir(stratum, counts, "flare", root)
    gmap = load_map(config.annotation_path(root))
    symbols = gmap.table.dropna(subset=["ncbi_gene"]).set_index("ncbi_gene")["symbol"]
    summaries = load_summaries(base / "site_summaries")
    frames = {c: s.to_frame() for c, s in summaries.items()}
    run = json.loads((base / "pooling_run.json").read_text())
    out = {"stratum": stratum, "counts": counts, "clients": sorted(frames),
           "questions": QUESTIONS,
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
                "annotation_sha256": gmap.sha256, "eb_priors": var.get("eb_priors"),
                "written": date.today().isoformat()}
        (vdir / "per_gene.schema.json").write_text(json.dumps(rs.schema(t, prov), indent=1))
        (vdir / "meta_genes_annotated.csv.gz").unlink(missing_ok=True)  # superseded
        out["variants"][vdir.name] = headline(t) | {"eb_priors": prior_summary(var)}
        if vdir.name == "primary":
            label = (f"{'ISS' if stratum == 'iss' else 'Earth'} stratum, {counts} counts, "
                     f"k = {len(frames)} sites, {len(t):,} genes")
            top = top_genes(t)
            out["top10"] = top[["ncbi_gene", "symbol", "fe_est", "fe_se", "fe_q", "q_p", "i2",
                                "het_flag", "v3_est", "v3_ci_low", "v3_ci_high", "v3_q",
                                "v2_q", "re_est", "hksj_ci_low", "hksj_ci_high", "re_q",
                                "tau2"]].to_dict("records")
            forest(top, frames, vdir / "forest_top10.png", f"Top 10 genes by FE p: {label}")
            cons = top_genes(t[t["fe_hit_consistent"]])
            out["top10_consistent"] = cons[["ncbi_gene", "symbol", "fe_est", "fe_se", "fe_q",
                                            "q_p", "i2", "v3_est", "v3_q", "re_est",
                                            "re_q", "tau2"]
                                           ].to_dict("records")
            forest(cons, frames, vdir / "forest_top10_consistent.png",
                   f"Top 10 FE hits with Q p ≥ {rs.HET_Q_P} (descriptive) by FE p: {label}")
            volcano(t, vdir / "volcano.png", f"Fixed effect, these missions: {label}")
            volcano_v3(t, vdir / "volcano_v3.png", f"V3, a new mission: {label}")
    (base / "headline.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def corr(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna()
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else float("nan")


def sensitivity(root: Path | None = None) -> pd.DataFrame:
    """One row per (stratum, counts, variant): hit counts (``results.hit_counts``), overlap of
    FE and V3 hits with the primary analysis (standard counts), and agreement of estimates."""
    rows = []
    for stratum in ("iss", "earth"):
        prim = config.results_dir(stratum, "standard", "flare", root) / "primary"
        if not (prim / "per_gene.csv.gz").exists():
            continue
        p = pd.read_csv(prim / "per_gene.csv.gz").set_index("ncbi_gene")
        p_hits = set(p.index[p["fe_hit"]])
        p_cons = set(p.index[p["fe_hit_consistent"]])
        p_v3 = set(p.index[p["v3_hit"]])
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
                    "v3_hits_shared_with_primary": len(set(t.index[t["v3_hit"]]) & p_v3),
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
