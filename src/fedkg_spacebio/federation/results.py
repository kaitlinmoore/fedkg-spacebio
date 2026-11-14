"""Phase 3a result definitions: hit calls, the heterogeneity flag, and the stable per-gene table.

Pure functions with no FLARE or plotting import. ``jobs/phase3a_meta/report.py`` calls them for
every (stratum, counts, variant), so the headline, the sensitivity table and the write-up all
take their counts from one function (``hit_counts``) and can't disagree.

Which result answers which question (D-029, 2026-11-13; amends D-026)
---------------------------------------------------------------------
Phase 3a asks two different questions, and each has its own method:

- **"These missions" (estimand A): the FE hit.** Fixed-effect (inverse-variance) BH q < 0.05
  within the stratum and variant (``fe_hit``). FE estimates the precision-weighted average of
  the missions in this analysis, so it answers "is there a change across *these* missions?".
  It is not an estimate of what a new mission would show. Cochran's Q p and I² are reported
  next to it as **annotations, not filters**: a gene with a high Q is still an FE hit.
- **"A new mission" (estimand B): the V3 hit.** Empirical-Bayes τ² (``v3_hit``, D-028). It
  replaced REML + truncated HKSJ in this role, because in WS-08's calibrated simulation it was
  the only method that held FDR near 0.05 for B in every setting. At k = 3 (Earth) it has no
  power, and that is reported as a result, not a failure.
- **Sensitivity for A: the V2 hit** (``v2_hit``): FE with a moderated multiplicative
  heterogeneity factor. It protects A when the site SEs are misestimated, at a power cost.

Descriptive columns, kept for comparison with earlier write-ups:
- ``het_flag`` (Q p < 0.10) and the split of FE hits by it, ``fe_hit_consistent`` /
  ``fe_hit_heterogeneous``. The split was D-026's headline. Under D-029 it describes FE hits;
  it doesn't select them. WS-08's simulation showed that "unflagged" doesn't make an FE hit
  generalize to a new mission.
- ``re_hit``: REML + truncated HKSJ (the pre-specified D-025 headline). With 3–4 sites it can't
  reach BH q < 0.05 (a t with 2–3 df needs |t| ≥ 8–16), and it found no gene.

Both method changes (D-026, then D-029) were made after seeing results. Wherever the phase 3a
result is reported, it says so, along with D-029's fragility notes and D-035's caveat:
"consistent with a time-of-day mismatch of a few hours or a clock change; not separable."

The per-gene table (schema ``SCHEMA_VERSION``)
-----------------------------------------------
One row per gene that entered the variant, keyed by ``ncbi_gene``. It holds every pooled column
(``meta.POOL_COLUMNS``), the hit and flag columns, and each site's inputs as
``log2fc__<client>``, ``se__<client>``, ``base_mean__<client>``, ``n_flight__<client>``,
``n_ground__<client>`` and ``included__<client>``. n is per site (the same on every row); it is
repeated so the table is self-contained. ``included`` is False when
the site didn't send the gene or didn't flag it under this variant's inclusion rule; its
log2fc/se are then still shown if the site sent them, but they didn't enter the pooling.
Written as gzip CSV with ``%.17g`` floats, so values round-trip exactly, plus a JSON schema
file with column definitions. Schema 2 added the V2 and V3 columns (``meta.EB_COLUMNS``) and
their hit columns; the fitted priors are in the schema file's ``provenance.eb_priors``. Every
schema-1 column is unchanged in meaning and value.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from fedkg_spacebio.federation import meta

FDR_Q = 0.05        # BH threshold for every hit call (FE, V3, V2, RE)
HET_Q_P = 0.10      # Cochran's Q p-value below which a gene is flagged as heterogeneous
SCHEMA_VERSION = "phase3a-per-gene/2"

FLAG_COLUMNS = ["fe_hit", "v3_hit", "v2_hit", "het_flag", "fe_hit_consistent",
                "fe_hit_heterogeneous", "re_hit"]

COLUMN_DOC = {
    "ncbi_gene": "NCBI Gene ID (int). The join key; never join on symbol.",
    "symbol": "Gene symbol from the pinned GeneLab annotation table. Display label only.",
    "k": "Number of sites that entered pooling for this gene.",
    "fe_est": "Fixed-effect pooled log2 fold change (flight vs ground).",
    "fe_se": "Fixed-effect SE, 1/sqrt(sum w), w = 1/se_i^2.",
    "fe_p": "Fixed-effect two-sided normal p-value.",
    "fe_q": "BH q-value of fe_p within this stratum and variant.",
    "re_est": "Random-effects pooled log2 fold change (REML tau2 weights).",
    "hksj_se": "Truncated HKSJ SE: sqrt(max(1, multiplier) / sum w*).",
    "hksj_ci_low": "Lower 95% CI, t(k-1) quantile times hksj_se.",
    "hksj_ci_high": "Upper 95% CI.",
    "re_p": "Random-effects p-value, truncated HKSJ, t with k-1 df.",
    "re_q": "BH q-value of re_p within this stratum and variant.",
    "hksj_multiplier": "HKSJ q = sum w*(y - re_est)^2 / (k-1), before truncation.",
    "re_p_untruncated": "Random-effects p-value with untruncated HKSJ.",
    "tau2": "REML between-site variance (log2 scale squared); >= 0.",
    "reml_converged": "Whether the REML iteration converged.",
    "i2": "Q-based I^2 = max(0, (Q - df)/Q).",
    "q": "Cochran's Q (fixed-effect weights).",
    "q_p": "Cochran's Q chi-square p-value, df = k-1.",
    "v2_se": "V2 SE: sqrt(phi_moderated / sum w), w = 1/se_i^2. V2's estimate is fe_est.",
    "v2_ci_low": "V2 lower 95% CI, t quantile on v2_df.",
    "v2_ci_high": "V2 upper 95% CI.",
    "v2_df": "V2 t degrees of freedom: (k-1) + d0 (inf when d0 is inf: normal).",
    "v2_p": "V2 two-sided t p-value.",
    "v2_q": "BH q-value of v2_p within this stratum and variant.",
    "v2_phi": "Raw multiplicative heterogeneity phi = Q/(k-1), FE weights.",
    "v2_phi_moderated": "phi moderated toward the prior s0^2 with d0 prior df (Smyth 2004).",
    "v3_est": "V3 posterior mean of the cross-mission mean effect (EB tau2 grid prior).",
    "v3_sd": "V3 posterior SD of that mean.",
    "v3_ci_low": "V3 lower 95% equal-tailed credible limit.",
    "v3_ci_high": "V3 upper 95% equal-tailed credible limit.",
    "v3_p": "V3 p_EB = 2 min(P(mu > 0 | y), P(mu < 0 | y)).",
    "v3_q": "BH q-value of v3_p within this stratum and variant.",
    "v3_tau2_post_mean": "V3 posterior mean of this gene's between-mission variance tau2.",
    "v3_p_tau2_zero": "V3 posterior probability that tau2 = 0 (grid point mass at 0).",
    "fe_hit": f"fe_q < {FDR_Q}. Answers 'these missions' (D-029). Q p and I^2 annotate it; "
              "they don't filter it.",
    "v3_hit": f"v3_q < {FDR_Q}. Answers 'a new mission' (D-029).",
    "v2_hit": f"v2_q < {FDR_Q}. Sensitivity for 'these missions' (misestimated SEs; D-029).",
    "het_flag": f"q_p < {HET_Q_P}: heterogeneity across sites flagged. Annotation only.",
    "fe_hit_consistent": "fe_hit and not het_flag. Descriptive (D-026's split); not a filter.",
    "fe_hit_heterogeneous": "fe_hit and het_flag. Descriptive (D-026's split).",
    "re_hit": f"re_q < {FDR_Q} (REML + truncated HKSJ; kept for comparison, superseded by "
              "v3_hit for 'a new mission').",
    "log2fc__<client>": "That site's pydeseq2 log2 fold change (unshrunk MLE).",
    "se__<client>": "That site's pydeseq2 lfcSE.",
    "base_mean__<client>": "That site's mean of DESeq2-normalized counts over its samples.",
    "n_flight__<client>": "Flight animals in that site's DE fit (same on every row).",
    "n_ground__<client>": "Ground-control animals in that site's DE fit (same on every row).",
    "included__<client>": "Whether that site's estimate entered pooling for this gene.",
}


def classify(pooled: pd.DataFrame) -> pd.DataFrame:
    """Add the hit and flag columns to a pooled table (one call site for every definition).

    NaN q-values or Q p-values (genes with k < 2) give False, never a hit or a flag.
    """
    t = pooled.copy()
    fe_hit = (t["fe_q"] < FDR_Q).to_numpy()
    het = (t["q_p"] < HET_Q_P).to_numpy()
    t["fe_hit"] = fe_hit
    t["v3_hit"] = (t["v3_q"] < FDR_Q).to_numpy()
    t["v2_hit"] = (t["v2_q"] < FDR_Q).to_numpy()
    t["het_flag"] = het
    t["fe_hit_consistent"] = fe_hit & ~het
    t["fe_hit_heterogeneous"] = fe_hit & het
    t["re_hit"] = (t["re_q"] < FDR_Q).to_numpy()
    return t


def hit_counts(t: pd.DataFrame) -> dict:
    """The one source of hit counts for a classified table (``classify`` output)."""
    up, down = t["fe_est"] > 0, t["fe_est"] < 0
    return {
        "n_genes": int(len(t)),
        "fe_hit": int(t["fe_hit"].sum()),
        "fe_hit_up": int((t["fe_hit"] & up).sum()),
        "fe_hit_down": int((t["fe_hit"] & down).sum()),
        "v3_hit": int(t["v3_hit"].sum()),
        "v3_hit_up": int((t["v3_hit"] & (t["v3_est"] > 0)).sum()),
        "v3_hit_down": int((t["v3_hit"] & (t["v3_est"] < 0)).sum()),
        "v2_hit": int(t["v2_hit"].sum()),
        "fe_hit_consistent": int(t["fe_hit_consistent"].sum()),
        "fe_hit_heterogeneous": int(t["fe_hit_heterogeneous"].sum()),
        "het_flag": int(t["het_flag"].sum()),
        "re_hit": int(t["re_hit"].sum()),
    }


def per_gene_table(pooled: pd.DataFrame, frames: Mapping[str, pd.DataFrame], flag: str,
                   clients: list[str], symbols: pd.Series | None = None,
                   site_n: Mapping[str, tuple[int, int]] | None = None) -> pd.DataFrame:
    """Build the stable per-gene table for one variant.

    ``pooled`` is a ``stratum.pool_variant`` output (``ncbi_gene`` plus ``POOL_COLUMNS``).
    ``frames`` are the site summary frames (index ``ncbi_gene``) and ``clients`` the clients
    that entered this variant, in the order used for pooling. ``flag`` is the variant's
    inclusion flag; a site counts as included for a gene when it sent the gene, flagged it and
    the gene is in the pooled table. ``site_n`` maps client → (n_flight, n_ground); if given,
    the n columns are added.
    """
    t = classify(pooled)
    genes = t["ncbi_gene"].to_numpy()
    cols = {"ncbi_gene": genes}
    cols["symbol"] = (pd.Series(genes).map(symbols).to_numpy() if symbols is not None
                      else np.full(len(genes), None))
    out = pd.DataFrame(cols)
    for c in [*meta.POOL_COLUMNS, *FLAG_COLUMNS]:
        out[c] = t[c].to_numpy()
    for c in clients:
        f = frames[c].reindex(genes)
        out[f"log2fc__{c}"] = f["log2fc"].to_numpy(dtype=np.float64)
        out[f"se__{c}"] = f["se"].to_numpy(dtype=np.float64)
        out[f"base_mean__{c}"] = f["base_mean"].to_numpy(dtype=np.float64)
        if site_n is not None:
            out[f"n_flight__{c}"] = int(site_n[c][0])
            out[f"n_ground__{c}"] = int(site_n[c][1])
        out[f"included__{c}"] = f[flag].eq(True).to_numpy()  # not sent (NaN) → False
    k_check = out[[f"included__{c}" for c in clients]].sum(axis=1).to_numpy()
    if not np.array_equal(k_check, out["k"].to_numpy()):
        raise ValueError("per-site inclusion doesn't reproduce k; table and summaries disagree")
    return out


def schema(table: pd.DataFrame, provenance: dict) -> dict:
    """JSON schema for a per-gene table: version, columns with dtypes and meanings, provenance."""
    def doc(col):
        if "__" in col:
            return COLUMN_DOC[col.split("__")[0] + "__<client>"]
        return COLUMN_DOC[col]
    return {
        "schema_version": SCHEMA_VERSION,
        "format": "gzip CSV, UTF-8, header row, no index; floats %.17g (exact round trip); "
                  "missing = empty field; booleans True/False",
        "key": "ncbi_gene",
        "thresholds": {"fdr_q": FDR_Q, "het_q_p": HET_Q_P},
        "columns": [{"name": c, "dtype": str(table[c].dtype), "description": doc(c)}
                    for c in table.columns],
        "provenance": provenance,
    }
