"""WS-08 figures and tables for ``docs/phase3a-method-comparison.md``.

    uv run python jobs/phase3a_compare/report.py

Run after ``simulate.py``, ``compare.py`` and ``circadian.py``. Reads their outputs under
``$FEDKG_WORKDIR/results/phase3a/method-comparison/`` and writes ``figures/*.png`` and
``tables.md`` (markdown tables pasted into the write-up) there.

Figures:
- ``sim_fdr.png``: empirical FDR under both estimands, every method, both strata. Large dot =
  central setting; small dots = the other settings. Dashed line = the nominal 0.05.
- ``sim_power.png``: power under estimand B (a new mission) against FDR under B, central
  setting.
- ``phi_qq.png``: V2's scale factor φ̂ = Q/(k − 1) against the fitted scaled-F prior.
- ``tau2_prior.png``: cumulative distribution of V3's fitted τ² prior vs the REML τ̂² values.
- ``core_clock_iss.png``: per-site log2FC of core clock genes, ISS stratum (diverging map).

Styling: light surface ``#fcfcfb``, inks ``#0b0b0b`` / ``#52514e``, categorical slots in fixed
order (blue, orange, aqua, yellow, magenta) from the reference data-viz palette. FE keeps
WS-07's orange and random effects its blue. Every method is also named on the axis, so identity
never rests on colour alone. Diverging map: blue ``#2a78d6`` ↔ gray ``#f0efec`` ↔ red ``#e34948``.
(The palette validator needs Node, which isn't installed; these are the palette's documented,
validated default slots.)
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase3a_meta.config import results_dir, workdir  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLOR = {"fe": "#eb6834", "fe_flag": "#eb6834", "hksj_trunc": "#2a78d6",
         "hksj_untrunc": "#2a78d6", "v1_trunc": "#eda100", "v1_untrunc": "#eda100",
         "v2": "#1baf7a", "v3": "#e87ba4"}
HOLLOW = {"fe_flag", "hksj_untrunc", "v1_untrunc"}       # second member of a family
LABEL = {"fe": "FE", "fe_flag": "FE + flag", "hksj_trunc": "HKSJ (trunc.)",
         "hksj_untrunc": "HKSJ (untrunc.)", "v1_trunc": "V1 sketch (trunc.)",
         "v1_untrunc": "V1 sketch (untrunc.)", "v2": "V2 mod. multiplicative",
         "v3": "V3 EB τ²"}
ORDER = list(LABEL)

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def root() -> Path:
    return workdir() / "results" / "phase3a" / "method-comparison"


def dot(ax, x, y, m, size=60, alpha=1.0, z=3):
    hollow = m in HOLLOW
    ax.scatter(x, y, s=size, facecolor=SURFACE if hollow else COLOR[m], edgecolor=COLOR[m],
               linewidth=1.5 if hollow else 2.0, alpha=alpha, zorder=z)


def fig_sim_fdr(summ: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), sharey=True)
    for i, st in enumerate(("iss", "earth")):
        for j, (col, title) in enumerate((("fdr_A_mean", "Estimand A: these missions"),
                                          ("fdr_B_mean", "Estimand B: a new mission"))):
            ax = axes[i, j]
            ax.axvline(0.05, color=INK2, linestyle="--", linewidth=1, zorder=1)
            for r, m in enumerate(ORDER):
                d = summ[(summ.stratum == st) & (summ.method == m)]
                other = d[d.name != "eb_c1_f0.9"]
                for _, row in other.iterrows():
                    dot(ax, row[col], r, m, size=14, alpha=0.45, z=2)
                c = d[d.name == "eb_c1_f0.9"]
                if len(c):
                    dot(ax, c[col].item(), r, m, size=70)
            ax.set_yticks(range(len(ORDER)), [LABEL[m] for m in ORDER])
            ax.invert_yaxis()
            ax.set_xlim(-0.02, 0.7)
            ax.grid(axis="x", color=GRID, linewidth=0.8)
            ax.set_title(f"{st.upper()} · {title}", loc="left", fontsize=9, color=INK)
            if i == 1:
                ax.set_xlabel("Empirical FDR at BH q < 0.05 (mean of 100 replicates)")
    fig.suptitle("Calibrated simulation: false discovery rate by method and estimand",
                 x=0.01, ha="left", fontsize=11)
    fig.text(0.01, 0.005, "Large dot: central setting (τ² ×1, 90% μ = 0). Small dots: other "
             "settings and generators. Dashed: nominal 0.05. HKSJ makes ~0 calls (FDR 0 by "
             "convention).", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_sim_power(summ: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), sharey=False)
    for ax, st in zip(axes, ("iss", "earth"), strict=True):
        d = summ[(summ.stratum == st) & (summ.name == "eb_c1_f0.9")]
        ax.axvline(0.05, color=INK2, linestyle="--", linewidth=1, zorder=1)
        for _, row in d.iterrows():
            dot(ax, row.fdr_B_mean, row.power_B_mean, row.method, size=60)
            if row.method in ("fe", "v3"):        # selective direct labels; legend has all
                ax.annotate(LABEL[row.method], (row.fdr_B_mean, row.power_B_mean),
                            xytext=(6, 6), textcoords="offset points", fontsize=7.5,
                            color=INK2)
        ax.set_xlabel("FDR under estimand B")
        ax.set_ylabel("Power under estimand B")
        ax.set_title(st.upper(), loc="left", fontsize=9)
        ax.grid(color=GRID, linewidth=0.8)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=7,
                          markerfacecolor=SURFACE if m in HOLLOW else COLOR[m],
                          markeredgecolor=COLOR[m], markeredgewidth=1.8, label=LABEL[m])
               for m in ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.5)
    fig.suptitle("Central setting: power vs FDR for 'a new mission'", x=0.01, ha="left",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_phi_qq(out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, st in zip(axes, ("iss", "earth"), strict=True):
        qq = pd.read_csv(root() / "real" / st / "standard" / "primary" / "phi_qq.csv")
        hp = json.loads((root() / "real" / st / "standard" / "primary" /
                         "hyperparameters.json").read_text())["v2"]
        ax.scatter(np.log10(qq.expected), np.log10(qq.phi_hat), s=6, color="#1baf7a",
                   edgecolor="none", alpha=0.6)
        lim = [min(np.log10(qq.expected).min(), np.log10(qq.phi_hat).min()),
               max(np.log10(qq.expected).max(), np.log10(qq.phi_hat).max())]
        ax.plot(lim, lim, color=INK2, linewidth=1, linestyle="--")
        ax.set_xlabel("log10 expected (s₀²·F(d, d₀))")
        ax.set_ylabel("log10 observed φ̂")
        ax.set_title(f"{st.upper()}: d₀ = {hp['d0']:.1f}, s₀² = {hp['s02']:.2f}", loc="left",
                     fontsize=9)
        ax.grid(color=GRID, linewidth=0.8)
    fig.suptitle("V2 prior check: φ̂ = Q/(k−1) against the fitted scaled F (primary set)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_tau2(out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, st in zip(axes, ("iss", "earth"), strict=True):
        hp = json.loads((root() / "real" / st / "standard" / "primary" /
                         "hyperparameters.json").read_text())["v3"]
        g, pi = np.asarray(hp["grid"]), np.asarray(hp["pi"])
        t = pd.read_csv(results_dir(st, "standard", "flare") / "primary" / "per_gene.csv.gz")
        floor = g[1]
        xr = np.sort(np.maximum(t["tau2"].to_numpy(), floor))
        ax.step(np.log10(np.maximum(g, floor)), np.cumsum(pi), where="post", color="#e87ba4",
                linewidth=2, label="V3 fitted prior (all genes)")
        ax.step(np.log10(xr), np.arange(1, len(xr) + 1) / len(xr), where="post",
                color="#2a78d6", linewidth=2, label="REML τ̂², gene by gene")
        ax.annotate(f"REML: {np.mean(t['tau2'] == 0):.0%} exactly 0\n(drawn at grid floor)",
                    (np.log10(floor) + 0.1, 0.50), color=INK2, fontsize=8)
        ax.legend(loc="lower right", fontsize=7.5, frameon=False)
        ax.set_xlim(np.log10(floor) - 0.2, 0.5)
        ax.set_xlabel("log10 τ² (log2FC² units)")
        ax.set_ylabel("Cumulative share of genes")
        ax.set_title(st.upper(), loc="left", fontsize=9)
        ax.grid(color=GRID, linewidth=0.8)
    fig.suptitle("Between-mission variance: V3's deconvolved prior vs per-gene REML",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_core_clock(out: Path):
    c = pd.read_csv(root() / "circadian" / "core_clock.csv")
    c = c[c.stratum == "iss"]
    sites = sorted(x for x in c.columns if x.startswith("log2fc__")
                   and c[x].notna().any())
    mat = c[sites].to_numpy(dtype=float)
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    fig, ax = plt.subplots(figsize=(5.6, 6.4))
    lim = np.nanmax(np.abs(mat))
    im = ax.imshow(mat, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for (i, j), val in np.ndenumerate(mat):
        ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(sites)), [s.split("__")[1] for s in sites], rotation=20)
    ax.set_yticks(range(len(c)), c["symbol"])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label("site log2FC, flight vs ground")
    ax.set_title("ISS: core clock genes, per-site log2FC", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) and not np.isnan(v)
                 else ("–" if isinstance(v, float) else str(v)) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def tables(summ: pd.DataFrame) -> str:
    out = []
    for st in ("iss", "earth"):
        for name in summ[summ.stratum == st].name.unique():
            d = summ[(summ.stratum == st) & (summ.name == name)].copy()
            d["method"] = d["method"].map(LABEL)
            keep = ["method", "calls_mean", "fdr_A_mean", "power_A_mean", "fdr_B_mean",
                    "power_B_mean", "kind2_call_rate_mean", "coverage_mu_mean",
                    "coverage_theta_fe_mean"]
            out.append(f"### Simulation · {st} · {name}\n\n" + md_table(d[keep]))
    real = json.loads((root() / "real" / "summary.json").read_text())
    rows = []
    for run, s in real.items():
        rows.append({"run": run, **{LABEL[m]: s["hits"][m]["hits"] for m in ORDER}})
    out.append("### Real data: hits per method\n\n" + md_table(pd.DataFrame(rows), "{:.0f}"))
    return "\n\n".join(out) + "\n"


def main() -> int:
    figs = root() / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    summ = pd.read_csv(root() / "simulation" / "summary.csv")
    fig_sim_fdr(summ, figs / "sim_fdr.png")
    fig_sim_power(summ, figs / "sim_power.png")
    fig_phi_qq(figs / "phi_qq.png")
    fig_tau2(figs / "tau2_prior.png")
    fig_core_clock(figs / "core_clock_iss.png")
    (root() / "tables.md").write_text(tables(summ))
    print(f"[done] {figs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
