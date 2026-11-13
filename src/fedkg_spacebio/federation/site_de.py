"""Site-side differential expression and the summary that leaves the site (phase 3a).

What a site does
----------------
1. ``run_de``: pydeseq2 on the site's own rounded counts (``sites/loader.py``). Each client
   fits **only its own flight and ground samples**, plus design covariates. There are no
   all-group fits (Kaitlin, WS-07).
   - Wald test, **unshrunk** (MLE) log2 fold change and its ``lfcSE``. LFC shrinkage would
     change what the SE means (it becomes a posterior SD under a prior fitted to this site), and
     a meta-analysis needs each site's sampling SE. GeneLab's published DE is unshrunk too.
   - Contrast: flight vs ground, so a positive log2FC means higher in flight. That matches the
     KG's group-1-relative-to-group-2 convention.
   - Everything else is a pydeseq2 default, recorded in ``DE_SETTINGS``: ratio size factors,
     parametric dispersion trend, Cook's outlier refit at ≥ 7 replicates and Cook's filtering
     below that, and independent filtering at alpha = 0.05. R's ``results()`` defaults to
     alpha = 0.1; it only affects ``padj``, which feeds sensitivity run (a) and nothing else.
2. ``tested`` flags, one per inclusion rule (Kaitlin, WS-07):
   - **primary** ``tested_mean10``: base mean (the mean of DESeq2-normalized counts over the
     site's samples) ≥ 10. It's a fixed rule, pre-specified, applied identically at every site.
     Wald SEs for very low counts are unreliable, and an inverse-variance meta-analysis trusts
     the SE completely. It also needs a finite log2FC, SE and p-value. The p-value requirement
     excludes genes pydeseq2 flags as Cook's outliers (p = NaN, an estimate driven by one
     sample). That last part is a WS-07 choice, reported in the return note.
   - sensitivity (a) ``tested_indep_filter``: padj is not NaN (pydeseq2's own independent filter).
   - sensitivity (b) ``tested_finite_p``: any finite p-value (plus a finite log2FC and SE).
3. ``summarize``: map the per-gene table to NCBI Gene (1:1 only, ``harmonize/gene_map.py``)
   and package it as the **site → server summary**.

The federation boundary (CONTRACTS §4)
--------------------------------------
Per gene, the summary carries ``ncbi_gene``, ``log2fc``, ``se``, ``base_mean`` and the three
``tested`` flags. Per site, it carries ``n_flight`` and ``n_ground`` and a metadata dict (design,
DE settings, rounding, mapping drop log, versions). **No counts, no per-sample values, no sample
names.** ``validate_summary`` enforces this with exact key whitelists. The site calls it before
sending, and the server calls it again on receipt.

Governance note (for phase 4): REML random effects needs each site's per-gene estimate and SE at
the server, so these per-site summaries do leave the site. A fixed-effect-only analysis would
need just Σw·y and Σw per gene (w = 1/SE²), which secure aggregation could deliver without
revealing any single site's estimate.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from importlib.metadata import version

import numpy as np
import pandas as pd

from fedkg_spacebio.harmonize.gene_map import GeneMap, map_summary
from fedkg_spacebio.sites.loader import SiteData

MIN_BASE_MEAN = 10.0
ALPHA = 0.05
DE_SETTINGS = {
    "tool": "pydeseq2", "test": "Wald", "lfc": "MLE (unshrunk)", "size_factors": "ratio",
    "fit_type": "parametric", "refit_cooks": True, "min_replicates": 7, "cooks_filter": True,
    "independent_filter": True, "alpha": ALPHA,
    "contrast": "condition: flight vs ground (positive = higher in flight)",
    "primary_inclusion": f"base_mean >= {MIN_BASE_MEAN:g} and finite log2fc, se, pvalue",
}

GENE_FIELDS = ("ncbi_gene", "log2fc", "se", "base_mean", "tested_mean10", "tested_indep_filter",
               "tested_finite_p")
META_FIELDS = ("client", "site", "stratum", "n_flight", "n_ground", "design", "counts_variant",
               "de_settings", "rounding", "n_genes_de", "n_ercc_dropped", "mapping", "versions",
               "census_rows")
TESTED_FLAGS = ("tested_mean10", "tested_indep_filter", "tested_finite_p")


def run_de(counts: pd.DataFrame, metadata: pd.DataFrame, design: str,
           contrast: tuple[str, str, str] = ("condition", "flight", "ground"),
           n_cpus: int | None = None, quiet: bool = True) -> pd.DataFrame:
    """pydeseq2 Wald test. ``counts`` is genes × samples (int). Returns an Ensembl-indexed
    table: base_mean, log2fc, se, stat, pvalue, padj.

    pydeseq2's warnings are captured into ``result.attrs["warnings"]`` and end up in the site's
    metadata. Some of them change the method, for example "dispersion trend curve fitting did
    not converge. Switching to a mean-based dispersion trend", so they must not be lost."""
    return run_de_contrasts(counts, metadata, design, [contrast], n_cpus, quiet)[0]


_DEPRECATIONS = (DeprecationWarning, PendingDeprecationWarning, FutureWarning)


def _method_warnings(caught) -> list[str]:
    # Keep method-relevant warnings; drop library deprecation notices (pandas 3 → 4 etc.).
    return sorted({f"{w.category.__name__}: {w.message}" for w in caught
                   if "pydeseq2" in (w.filename or "") and not issubclass(w.category,
                                                                          _DEPRECATIONS)})


def run_de_contrasts(counts: pd.DataFrame, metadata: pd.DataFrame, design: str,
                     contrasts: list[tuple[str, str, str]], n_cpus: int | None = None,
                     quiet: bool = True) -> list[pd.DataFrame]:
    """One pydeseq2 fit, several Wald contrasts (used by Check 2a's GeneLab replica, which
    takes several contrasts from one all-group fit, as GeneLab does)."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats

    if list(metadata.index) != list(counts.columns):
        raise ValueError("metadata rows must match count columns, in order")
    inference = DefaultInference(n_cpus=n_cpus)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dds = DeseqDataSet(counts=counts.T, metadata=metadata, design=design,
                           refit_cooks=True, inference=inference, quiet=quiet)
        dds.deseq2()
    fit_warnings = _method_warnings(caught)
    out = []
    for contrast in contrasts:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stat = DeseqStats(dds, contrast=list(contrast), alpha=ALPHA, cooks_filter=True,
                              independent_filter=True, inference=inference, quiet=quiet)
            stat.summary()
        res = stat.results_df.rename(columns={"baseMean": "base_mean",
                                              "log2FoldChange": "log2fc", "lfcSE": "se"})
        res.index.name = "ensembl"
        res = res[["base_mean", "log2fc", "se", "stat", "pvalue", "padj"]].copy()
        res.attrs["warnings"] = sorted(set(fit_warnings) | set(_method_warnings(caught)))
        out.append(res)
    return out


def inclusion_flags(res: pd.DataFrame, min_base_mean: float = MIN_BASE_MEAN) -> pd.DataFrame:
    est_ok = np.isfinite(res["log2fc"]) & np.isfinite(res["se"]) & (res["se"] > 0)
    p_ok = np.isfinite(res["pvalue"])
    out = res.copy()
    out["tested_mean10"] = est_ok & p_ok & (res["base_mean"] >= min_base_mean)
    out["tested_indep_filter"] = est_ok & np.isfinite(res["padj"])
    out["tested_finite_p"] = est_ok & p_ok
    return out


@dataclass
class SiteSummary:
    """What leaves a site: per-gene arrays plus site-level metadata. See ``validate_summary``."""

    genes: dict[str, np.ndarray]
    meta: dict

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.genes).set_index("ncbi_gene")


def versions() -> dict[str, str]:
    return {p: version(p) for p in ("pydeseq2", "numpy", "pandas", "scipy", "nvflare")}


def summarize(site: SiteData, res: pd.DataFrame, gene_map: GeneMap) -> SiteSummary:
    """Map a site's Ensembl DE table to NCBI Gene and package the boundary object."""
    flagged = inclusion_flags(res)
    mapped, drop_log = map_summary(flagged, gene_map, tested_col="tested_mean10")
    # Send genes with a usable estimate; the server applies the inclusion rule via the flags.
    keep = np.isfinite(mapped["log2fc"]) & np.isfinite(mapped["se"]) & (mapped["se"] > 0)
    mapped = mapped[keep]
    genes = {
        "ncbi_gene": mapped.index.to_numpy(dtype=np.int64),
        "log2fc": mapped["log2fc"].to_numpy(dtype=np.float64),
        "se": mapped["se"].to_numpy(dtype=np.float64),
        "base_mean": mapped["base_mean"].to_numpy(dtype=np.float64),
        **{f: mapped[f].to_numpy(dtype=bool) for f in TESTED_FLAGS},
    }
    m = site.manifest
    meta = {
        "client": site.client, "site": m["site"], "stratum": m["stratum"],
        "n_flight": int(site.provenance["n_flight"]), "n_ground": int(site.provenance["n_ground"]),
        "design": m["design"], "counts_variant": m["counts_variant"],
        "de_settings": dict(DE_SETTINGS) | {"warnings": list(res.attrs.get("warnings", []))},
        "rounding": site.provenance["rounding"],
        "n_genes_de": int(len(res)), "n_ercc_dropped": int(site.provenance["n_ercc_dropped"]),
        "mapping": drop_log | {"n_sent": int(len(mapped)),
                               "n_tested_mean10_sent": int(mapped["tested_mean10"].sum())},
        "versions": versions(),
        "census_rows": [{k: r[k] for k in ("osd_id", "comparison_id", "genelab_contrast")}
                        for r in m["census_rows"]],
    }
    summary = SiteSummary(genes, meta)
    validate_summary(summary.genes, summary.meta)
    return summary


def validate_summary(genes: dict, meta: dict) -> None:
    """Raise ValueError unless ``genes``/``meta`` match the site→server contract exactly."""
    if set(genes) != set(GENE_FIELDS):
        raise ValueError(f"per-gene fields must be exactly {GENE_FIELDS}; got {sorted(genes)}")
    if set(meta) != set(META_FIELDS):
        extra, missing = set(meta) - set(META_FIELDS), set(META_FIELDS) - set(meta)
        raise ValueError(f"site metadata fields wrong; extra {sorted(extra)}, "
                         f"missing {sorted(missing)}")
    n = len(genes["ncbi_gene"])
    for key, arr in genes.items():
        a = np.asarray(arr)
        if a.ndim != 1 or len(a) != n:
            raise ValueError(f"{key}: expected a 1-D per-gene array of length {n}")
    ids = np.asarray(genes["ncbi_gene"])
    if ids.dtype.kind not in "iu" or np.any(ids <= 0) or len(np.unique(ids)) != n:
        raise ValueError("ncbi_gene must be unique positive integers")
    se = np.asarray(genes["se"], dtype=float)
    if not np.all(np.isfinite(se) & (se > 0)) or not np.all(np.isfinite(genes["log2fc"])):
        raise ValueError("log2fc must be finite and se finite and > 0 for every gene sent")
    for f in TESTED_FLAGS:
        if np.asarray(genes[f]).dtype != bool:
            raise ValueError(f"{f} must be boolean")
    for f in ("n_flight", "n_ground"):
        if not isinstance(meta[f], int) or meta[f] < 1:
            raise ValueError(f"{f} must be a positive int")
    json.dumps(meta)  # metadata must be plain JSON (no arrays smuggled in)
