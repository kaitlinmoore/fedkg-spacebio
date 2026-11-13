"""Which processed files a study publishes, and what their headers say (WS-06).

GeneLab's bulk RNA-seq pipeline (GL-DPPD-7101; github.com/nasa/GeneLab_Data_Processing) publishes,
per study, under the file category ``GeneLab Processed RNA-Seq Files``:

- ``*_RSEM_Unnormalized_Counts[_rRNArm]_GLbulkRNAseq.csv``: per-sample gene counts from RSEM
  (genes × samples). ``_rRNArm`` is the same matrix with ribosomal-RNA genes removed. These are
  what a site would feed to (py)DESeq2.
- ``*_STAR_Unnormalized_Counts_GLbulkRNAseq.csv``: STAR's own gene counts (GeneLab's DE doesn't
  use them).
- ``*_Normalized_Counts[_rRNArm]_GLbulkRNAseq.csv``: DESeq2 median-of-ratios normalized counts.
- ``*_differential_expression[_rRNArm]_GLbulkRNAseq.csv``: the DE table. Annotation columns
  (``ENSEMBL, SYMBOL, ..., ENTREZID``), one normalized-count column per sample, then per contrast
  ``Log2fc_``, ``Stat_`` (7101-E and later: the DESeq2 Wald statistic, so SE = log2FC / Stat),
  ``P.value_`` and ``Adj.p.value_``.
- ``*_SampleTable_GLbulkRNAseq.csv`` (sample → condition used in DE) and
  ``*_contrasts_GLbulkRNAseq.csv`` (contrast name → the two conditions, in order).

Older studies use names without ``_GLbulkRNAseq``; the patterns below accept both.

The DE tables are large (up to ~525 MB), so the census reads only their first 64 KB with an HTTP
Range request: enough for the header and a few hundred gene rows.
"""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

PROCESSED_CATEGORY = "GeneLab Processed RNA-Seq Files"

# role → regex on the file name. ``(?P<rrna>_rRNArm)?`` is captured so both variants map cleanly.
FILE_ROLES = {
    "counts_rsem": r"_RSEM_Unnormalized_Counts(?:_GLbulkRNAseq)?\.csv$",
    "counts_rsem_rrnarm": r"_RSEM_Unnormalized_Counts_rRNArm(?:_GLbulkRNAseq)?\.csv$",
    "counts_star": r"_STAR_Unnormalized_Counts(?:_GLbulkRNAseq)?\.csv$",
    "normalized": r"_Normalized_Counts(?:_GLbulkRNAseq)?\.csv$",
    "normalized_rrnarm": r"_Normalized_Counts_rRNArm(?:_GLbulkRNAseq)?\.csv$",
    "de": r"_differential_expression(?:_GLbulkRNAseq)?\.csv$",
    "de_rrnarm": r"_differential_expression_rRNArm(?:_GLbulkRNAseq)?\.csv$",
    "sample_table": r"_SampleTable(?:_GLbulkRNAseq)?\.csv$",
    "contrasts": r"_contrasts(?:_GLbulkRNAseq)?\.csv$",
    "runsheet": r"_runsheet\.csv$",
}


def select_rnaseq_files(listing: Iterable[dict]) -> dict[str, dict]:
    """Pick one file per role from a study's file listing.

    Returns ``role → {file_name, file_size, date_created}``. A role with several candidates (e.g.
    two RNA-seq assays in one study) raises, so nothing is silently picked.
    """
    found: dict[str, list[dict]] = {}
    for f in listing:
        if f.get("category") != PROCESSED_CATEGORY:
            continue
        name = f["file_name"]
        if "rna_seq" not in name and "rna-seq" not in name.lower():
            continue
        for role, pattern in FILE_ROLES.items():
            if re.search(pattern, name):
                found.setdefault(role, []).append(
                    {"file_name": name, "file_size": f.get("file_size"),
                     "date_created": f.get("date_created")})
    out = {}
    for role, files in found.items():
        if len(files) > 1:
            names = ", ".join(sorted(x["file_name"] for x in files))
            raise ValueError(f"several files for role {role!r}: {names}")
        out[role] = files[0]
    return out


def parse_partial_csv(head: bytes) -> pd.DataFrame:
    """Parse the first bytes of a CSV, dropping a trailing line cut off by the byte range."""
    text = head.decode("utf-8", errors="replace")
    if "\n" not in text:
        raise ValueError("byte range too short: no complete header line")
    complete = text[: text.rfind("\n") + 1]
    return pd.read_csv(io.StringIO(complete), dtype=str, keep_default_na=False)


def header_columns(head: bytes) -> list[str]:
    """Column names from the first line of a CSV."""
    text = head.decode("utf-8", errors="replace")
    if "\n" not in text:
        raise ValueError("byte range too short: no complete header line")
    return next(csv.reader([text.split("\n", 1)[0].rstrip("\r")]))


def de_header_info(columns: list[str]) -> dict:
    """What a DE table header offers: its contrasts, and which have a ``Stat_`` column."""
    contrasts = [c.removeprefix("Log2fc_") for c in columns if c.startswith("Log2fc_")]
    stat = {c.removeprefix("Stat_") for c in columns if c.startswith("Stat_")}
    return {
        "id_columns": [c for c in columns if c in {"ENSEMBL", "SYMBOL", "ENTREZID", "TAIR"}],
        "contrasts": contrasts,
        "stat_contrasts": [c for c in contrasts if c in stat],
        "has_any_stat": bool(stat),
    }


def contrast_for(flight_samples: Iterable[str], control_samples: Iterable[str],
                 sample_table: pd.DataFrame, contrasts: pd.DataFrame) -> dict:
    """Find GeneLab's contrast for one flight → control comparison.

    ``sample_table`` is the SampleTable CSV (first column sample, ``condition``); ``contrasts`` is
    the contrasts CSV (columns are contrast names; row 0 is group 1, row 1 is group 2). GeneLab's
    contrast ``(A)v(B)`` gives log2(A/B), so we want group 1 = flight condition.

    Returns ``{contrast, n_flight_in_de, n_control_in_de, flight_condition, control_condition}``;
    ``contrast`` is None if the samples don't map to exactly one condition each or no contrast fits.
    """
    st = sample_table.rename(columns={sample_table.columns[0]: "sample"})
    cond = dict(zip(st["sample"], st["condition"], strict=True))
    fc = {cond[s] for s in flight_samples if s in cond}
    cc = {cond[s] for s in control_samples if s in cond}
    result = {
        "contrast": None,
        "n_flight_in_de": sum(s in cond for s in flight_samples),
        "n_control_in_de": sum(s in cond for s in control_samples),
        "flight_condition": "|".join(sorted(fc)) or None,
        "control_condition": "|".join(sorted(cc)) or None,
    }
    if len(fc) != 1 or len(cc) != 1:
        return result
    f, c = fc.pop(), cc.pop()
    body = contrasts.drop(columns=[contrasts.columns[0]])
    for name in body.columns:
        if (body[name].iloc[0], body[name].iloc[1]) == (f, c):
            result["contrast"] = name
            break
    return result


ENSEMBL_MOUSE = re.compile(r"^ENSMUSG\d{11}(\.\d+)?$")
ENSEMBL_ANY = re.compile(r"^ENS[A-Z]*G\d{11}(\.\d+)?$")
ERCC = re.compile(r"^ERCC-\d{5}$")


def detect_namespace(ids: Iterable[str]) -> dict:
    """Classify the gene IDs of a matrix's first column.

    Returns counts per pattern and a verdict: ``ENSEMBL`` (≥ 99% of non-spike-in IDs are Ensembl
    gene IDs), ``NCBIGene`` (all digits), ``SYMBOL`` or ``mixed``. ERCC spike-in rows are counted
    separately. ``versioned`` says whether Ensembl IDs carry a ``.N`` version suffix.
    """
    ids = [i for i in ids if i]
    n_ercc = sum(bool(ERCC.match(i)) for i in ids)
    genes = [i for i in ids if not ERCC.match(i)]
    n_ens = sum(bool(ENSEMBL_ANY.match(i)) for i in genes)
    n_ens_mouse = sum(bool(ENSEMBL_MOUSE.match(i)) for i in genes)
    n_digit = sum(i.isdigit() for i in genes)
    n = len(genes) or 1
    if n_ens / n >= 0.99:
        verdict = "ENSEMBL"
    elif n_digit / n >= 0.99:
        verdict = "NCBIGene"
    elif n_ens == 0 and n_digit == 0:
        verdict = "SYMBOL"
    else:
        verdict = "mixed"
    return {"namespace": verdict, "n_ids": len(ids), "n_ensembl": n_ens,
            "n_ensembl_mouse": n_ens_mouse, "n_ercc": n_ercc,
            "n_other": len(genes) - n_ens - n_digit,
            "versioned": any("." in i for i in genes if ENSEMBL_ANY.match(i))}


def processing_info(protocol_descriptions: Iterable[str]) -> dict:
    """Pipeline and annotation versions, read from the study's RNA-seq processing protocol text.

    Takes the description(s) that cite ``GL-DPPD-7101-<letter>`` (the RNA-seq pipeline; methylation
    uses 7113). Returns ``pipeline_version``, ``nf_rcp_version``, ``ensembl_release``,
    ``genome_build`` and ``gtf``, each None if absent, and ``conflict`` = True if the text names
    more than one value for any of them (then the joined values are reported).
    """
    texts = [d for d in protocol_descriptions if d and re.search(r"GL-DPPD-7101-[A-Z]", d)]
    blob = " ".join(texts)
    found = {
        "pipeline_version": re.findall(r"GL-DPPD-7101-[A-Z]", blob),
        "nf_rcp_version": re.findall(r"NF_RCP(?: version |-)(\d+\.\d+\.\d+)", blob),
        "ensembl_release": re.findall(r"Ensembl release (\d+)", blob),
        "genome_build": re.findall(r"\b(GRCm\d+|mm\d+)\b", blob),
        "gtf": re.findall(r"(Mus_musculus\.[\w.]+\.gtf)", blob),
    }
    out: dict = {"conflict": False, "n_protocols": len(texts)}
    for key, values in found.items():
        uniq = sorted(set(values))
        out[key] = "|".join(uniq) if uniq else None
        out["conflict"] |= len(uniq) > 1
    return out


def compare_partial_tables(a: pd.DataFrame, b: pd.DataFrame, id_col: str,
                           value_cols: Iterable[str]) -> dict:
    """How closely two partially read tables agree on shared rows and columns.

    Used to check which normalized-count file a DE table's per-sample columns came from: the DE
    table carries the normalized counts it was computed on. Found on 2026-11-12: GeneLab's DE
    table stores **normalized count + 1** in every per-sample cell (a pseudocount; it's also why
    the KG's group means bottom out at exactly 1.000). So the comparison reports the median
    difference as ``offset`` and how far any cell strays from it (``max_dev_from_offset``). A DE
    table "pairs" with a normalized file when that deviation is ~0.
    """
    cols = [c for c in value_cols if c in a.columns and c in b.columns]
    shared = a[[id_col] + cols].merge(b[[id_col] + cols], on=id_col, suffixes=("_a", "_b"))
    if shared.empty or not cols:
        return {"n_rows": 0, "n_cols": len(cols), "offset": None, "max_dev_from_offset": None}
    da = shared[[c + "_a" for c in cols]].apply(pd.to_numeric, errors="coerce").to_numpy()
    db = shared[[c + "_b" for c in cols]].apply(pd.to_numeric, errors="coerce").to_numpy()
    diff = da - db
    offset = float(np.nanmedian(diff))
    return {"n_rows": len(shared), "n_cols": len(cols), "offset": offset,
            "max_dev_from_offset": float(np.nanmax(np.abs(diff - offset)))}
