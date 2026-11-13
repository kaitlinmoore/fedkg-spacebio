"""Q-11 diagnostics: are phase 3a hits enriched for genes that cycle in mouse liver?

Why this exists
---------------
No site records the time of day of dissection. Mouse liver expression swings strongly over 24 h
(thousands of transcripts cycle), so if flight and ground animals within a site were dissected at
different times of day, cycling genes would show up as "flight effects". This module checks
whether the hits over-represent cycling genes, and whether the hit lists hold up without them.
It's a **diagnostic, not a correction**: spaceflight may shift the clock itself, so a clock-gene
signal isn't automatically an artifact, and removing cycling genes can remove real biology.

The rhythmic list
-----------------
Hughes et al. 2009, PLoS Genet 5:e1000442, Table S1 ("24 h cycling genes"): mouse liver sampled
every hour for 48 h (GEO GSE11923, Affymetrix Mouse Genome 430 2.0); a probe set is listed if
COSOPT and Fisher's G-test both give FDR < 0.05 and the COSOPT period is 20–30 h. CC-BY.

Mapping to NCBI Gene uses **probe set IDs**, not symbols: NCBI GEO's platform annotation for
GPL1261 gives each probe set's Entrez Gene ID (``probe_to_entrez``). Every drop is counted:
probe sets with no Entrez ID, probe sets annotated to several genes (dropped as ambiguous), and
mapped IDs that aren't in the pinned GeneLab annotation (e.g. IDs retired since the 2016
annotation).

Enrichment
----------
Per method: of the genes tested in the stratum **and measurable on the array** (≥ 1 GPL1261
probe set with a unique Entrez ID; otherwise a gene could never be "rhythmic" here and would
dilute the background), how many are rhythmic among hits vs among non-hits. One-sided Fisher
exact test for over-representation, with the odds ratio. Then the hits are recounted with
rhythmic genes removed, two ways: dropping them from the hit list, and rerunning BH on the
non-rhythmic genes only (BH depends on which genes are tested).
"""

from __future__ import annotations

import gzip
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from fedkg_spacebio.federation import meta, results

# Core clock and first-order clock-output genes, by symbol for display. IDs are never typed in
# by hand: ``resolve_symbols`` looks each one up in the pinned GeneLab annotation and checks it
# against the GPL1261 annotation, and the resolved IDs are written to the outputs. An entry
# "A/B" means GeneLab's current symbol A and the 2016 array annotation's older symbol B
# (Arntl was renamed Bmal1).
CORE_CLOCK = ("Bmal1/Arntl", "Clock", "Npas2", "Per1", "Per2", "Per3", "Cry1", "Cry2", "Nr1d1",
              "Nr1d2", "Rora", "Rorc", "Dbp", "Tef", "Hlf", "Nfil3", "Bhlhe40", "Bhlhe41",
              "Ciart")


def read_geo_annot(path: Path) -> pd.DataFrame:
    """The platform table of a GEO ``.annot.gz`` file (between the begin/end markers)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    start = lines.index("!platform_table_begin") + 1
    end = lines.index("!platform_table_end") if "!platform_table_end" in lines else len(lines)
    header = lines[start].split("\t")
    rows = [ln.split("\t") for ln in lines[start + 1:end]]
    width = len(header)
    rows = [r + [""] * (width - len(r)) if len(r) < width else r[:width] for r in rows]
    return pd.DataFrame(rows, columns=header)


def probe_to_entrez(annot: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Probe set ID → NCBI Gene ID (int), unique mappings only; plus the drop counts.

    GEO writes several genes as ``"12345///67890"``; such probe sets are dropped as ambiguous.
    """
    gid = annot["Gene ID"].fillna("").astype(str).str.strip()
    none = gid == ""
    multi = gid.str.contains("///", regex=False)
    keep = ~none & ~multi
    m = pd.Series(gid[keep].astype(np.int64).to_numpy(), index=annot.loc[keep, "ID"].to_numpy())
    log = {"probe_sets": int(len(annot)), "unique_entrez": int(keep.sum()),
           "no_entrez": int(none.sum()), "multiple_entrez": int(multi.sum())}
    return m, log


def rhythmic_genes(table_s1: pd.DataFrame, pmap: pd.Series, platform_ids: Iterable[str],
                   known_ids: Iterable[int] | None = None) -> tuple[set[int], dict]:
    """NCBI Gene IDs of the rhythmic probe sets, with every drop counted.

    ``pmap`` is ``probe_to_entrez``'s unique mapping; ``platform_ids`` is every probe set ID in
    the platform annotation, so a probe set missing from it can be told apart from one that is
    there but has no unique Entrez ID. ``known_ids``: IDs present in the pinned GeneLab
    annotation; mapped IDs outside it are counted (``not_in_genelab_annotation``) and dropped.
    """
    probes = table_s1["Probe Set ID"].astype(str).str.strip()
    mapped = probes.map(pmap)
    ok = mapped.notna()
    in_annot = probes.isin(set(platform_ids))
    genes = set(mapped[ok].astype(np.int64))
    log = {"probe_sets": int(len(probes)),
           "probe_not_in_platform_annotation": int((~in_annot).sum()),
           "probe_without_unique_entrez": int((in_annot & ~ok).sum()),
           "probe_mapped": int(ok.sum()),
           "distinct_genes": len(genes)}
    if known_ids is not None:
        known = set(int(i) for i in known_ids)
        missing = genes - known
        genes = genes & known
        log["not_in_genelab_annotation"] = len(missing)
        log["distinct_genes_kept"] = len(genes)
    log["drop_rate_probes"] = round(1 - log["probe_mapped"] / max(log["probe_sets"], 1), 4)
    return genes, log


def resolve_symbols(symbols: Iterable[str], genelab: pd.DataFrame, annot: pd.DataFrame
                    ) -> pd.DataFrame:
    """Look up display symbols in the pinned GeneLab annotation (``SYMBOL`` → ``ENTREZID``) and
    check each against GPL1261's Entrez ID for the same symbol. A symbol that is missing or
    ambiguous in either source, or where the two disagree, is reported, never guessed."""
    gl = genelab[["SYMBOL", "ENTREZID"]].dropna().copy()
    gl["ENTREZID"] = gl["ENTREZID"].astype(str)
    geo = annot[["Gene symbol", "Gene ID"]].copy()
    rows = []
    for entry in symbols:
        s, s_geo = (entry.split("/", 1) + [entry])[:2] if "/" in entry else (entry, entry)
        a = sorted(set(gl.loc[gl["SYMBOL"] == s, "ENTREZID"]) - {""})
        b = sorted(set(geo.loc[geo["Gene symbol"] == s_geo, "Gene ID"]) - {""})
        a_ids = {x for v in a for x in v.split("|")}
        status = ("ok" if len(a_ids) == 1 and set(b) == a_ids else
                  "missing_in_genelab" if not a_ids else
                  "ambiguous_in_genelab" if len(a_ids) > 1 else
                  "missing_on_array" if not b else "disagree")
        rows.append({"symbol": s, "array_symbol": s_geo,
                     "ncbi_gene": int(next(iter(a_ids))) if len(a_ids) == 1 else None,
                     "genelab_ids": "|".join(sorted(a_ids)),
                     "gpl1261_ids": "|".join(b), "status": status})
    return pd.DataFrame(rows)


def enrichment(hit: np.ndarray, rhythmic: np.ndarray, measurable: np.ndarray) -> dict:
    """One-sided Fisher exact test: are rhythmic genes over-represented among hits?

    All three are boolean arrays over the tested genes. Only measurable genes count.
    """
    h, r = hit[measurable], rhythmic[measurable]
    a = int((h & r).sum())
    b = int((h & ~r).sum())
    c = int((~h & r).sum())
    d = int((~h & ~r).sum())
    if a + b == 0:
        return {"hits_measurable": 0, "rhythmic_hits": 0, "background": int(measurable.sum()),
                "rhythmic_background": int(r.sum()), "share_hits": np.nan,
                "share_background": float(r.mean()) if r.size else np.nan,
                "odds_ratio": np.nan, "p_one_sided": np.nan}
    res = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
    return {"hits_measurable": a + b, "rhythmic_hits": a, "background": int(measurable.sum()),
            "rhythmic_background": int(r.sum()), "share_hits": a / (a + b),
            "share_background": float(r.mean()), "odds_ratio": float(res.statistic),
            "p_one_sided": float(res.pvalue)}


def enrichment_stratified(hit: np.ndarray, rhythmic: np.ndarray, measurable: np.ndarray,
                          strata: np.ndarray) -> dict:
    """Mantel–Haenszel version of ``enrichment``, within strata (e.g. base-mean deciles).

    Why: cycling liver genes tend to be well expressed, and well-expressed genes have smaller
    SEs, so they're easier to call. An unstratified odds ratio mixes "rhythmic" with "easy to
    detect". Within each expression stratum, compare rhythmic share among hits vs non-hits, then
    pool the 2×2 tables:

        OR_MH = Σ_s (a_s d_s / n_s) / Σ_s (b_s c_s / n_s)

    and the Cochran–Mantel–Haenszel test (no continuity correction), one-sided toward
    over-representation: z = (Σ a_s − Σ E[a_s]) / sqrt(Σ Var[a_s]) with the hypergeometric mean
    and variance of each table's a_s.
    """
    m = measurable & np.isfinite(strata)
    num = den = a_sum = e_sum = v_sum = 0.0
    for s in np.unique(strata[m]):
        sel = m & (strata == s)
        h, r = hit[sel], rhythmic[sel]
        a, b = float((h & r).sum()), float((h & ~r).sum())
        c, d = float((~h & r).sum()), float((~h & ~r).sum())
        n = a + b + c + d
        if n < 2:
            continue
        num += a * d / n
        den += b * c / n
        r1, c1 = a + b, a + c
        a_sum += a
        e_sum += r1 * c1 / n
        v_sum += r1 * (n - r1) * c1 * (n - c1) / (n**2 * (n - 1))
    if v_sum == 0:
        return {"or_mh": np.nan, "p_cmh_one_sided": np.nan}
    z = (a_sum - e_sum) / np.sqrt(v_sum)
    return {"or_mh": num / den if den > 0 else np.inf,
            "p_cmh_one_sided": float(stats.norm.sf(z))}


def without_rhythmic(calls: pd.DataFrame, method: str, rhythmic: np.ndarray) -> dict:
    """Hits once rhythmic genes are removed: (a) dropped from the list; (b) BH rerun on the
    non-rhythmic genes. For ``fe_flag``, (b) applies the heterogeneity flag after the new BH.

    V2's and V3's priors are not refitted without the rhythmic genes; only BH changes.
    """
    hit = calls[f"{method}__hit"].to_numpy(dtype=bool)
    keep = ~rhythmic
    p = calls[f"{method}__p"].to_numpy()
    q = meta.bh_qvalues(p[keep])
    rerun = q < results.FDR_Q
    if method == "fe_flag":
        rerun &= ~calls["het_flag"].to_numpy(dtype=bool)[keep]
    return {"hits": int(hit.sum()), "hits_dropping_rhythmic": int((hit & keep).sum()),
            "hits_bh_rerun_nonrhythmic": int(rerun.sum())}
