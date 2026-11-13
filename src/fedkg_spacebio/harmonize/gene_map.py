"""Ensembl → NCBI Gene mapping for per-gene summaries (D-012).

Source
------
GeneLab's reference annotation table for mouse, GL-DPPD-7110-A, Ensembl 112
(``Mus_musculus.GRCm39.112-GL-annotations.tsv``, figshare file 48354460). That's the table the
OSDR liver DE files were annotated with, built from ``org.Mm.eg.db`` 3.19.1. It has one row per
Ensembl gene, and ``ENTREZID`` may hold several IDs joined by ``|``. The SHA-256 is checked on
load (``ANNOTATION_SHA256``), so a silently changed file fails instead of shifting the map.

When the mapping happens (the reading of D-012 approved in WS-07)
-----------------------------------------------------------------
Each site runs DE on its **native Ensembl IDs** and maps its **per-gene summaries** to NCBI Gene
before sending them. Counts are never summed across IDs, and only ``NCBIGene`` rows leave a
site.

Only one-to-one pairs are kept
------------------------------
The map is built from the **whole table**, not from any one site's genes, so every site drops
exactly the same IDs and the result can't depend on which genes a site happened to test.
Each Ensembl ID falls into exactly one category:

- ``no_entrez``: ENTREZID is empty. Dropped.
- ``one_to_many``: one Ensembl ID lists several Entrez IDs. Dropped, because picking one would
  be a guess.
- ``many_to_one``: an Entrez ID claimed by several Ensembl IDs (counting only those that list
  exactly one Entrez ID). **All** of them are dropped. Keeping one would mean choosing which
  locus's estimate represents the gene, and summing them would merge two estimates.
- ``one_to_one``: kept.

An Ensembl ID absent from the table is counted as ``not_in_table`` by ``map_summary``. It
shouldn't happen here, since both the counts and the table are Ensembl 112. Every category is
counted, at the table level (``GeneMap.table_counts``) and per site (``map_summary`` →
``drop_log``). Silent loss is a bug.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fedkg_spacebio.census.osdr_client import default_workdir

ANNOTATION_FILE = "Mus_musculus.GRCm39.112-GL-annotations.tsv"
ANNOTATION_SHA256 = "123a08990f1982b266115852d720e8d0482af6e873f8eca45c8449d3f54044e2"
ANNOTATION_SOURCE = "GL-DPPD-7110-A, Mus musculus, Ensembl 112 (figshare 48354460)"
CATEGORIES = ("one_to_one", "no_entrez", "one_to_many", "many_to_one")


@dataclass
class GeneMap:
    table: pd.DataFrame  # index ENSEMBL; columns: category, ncbi_gene (Int64), symbol
    source: str
    sha256: str | None

    @property
    def table_counts(self) -> dict[str, int]:
        counts = self.table["category"].value_counts()
        return {c: int(counts.get(c, 0)) for c in CATEGORIES} | {"total": int(len(self.table))}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_map(annotations: pd.DataFrame, source: str = "", sha256: str | None = None) -> GeneMap:
    """Classify every Ensembl ID in an annotation frame with ENSEMBL / SYMBOL / ENTREZID."""
    a = annotations[["ENSEMBL", "SYMBOL", "ENTREZID"]].copy()
    if a["ENSEMBL"].duplicated().any():
        raise ValueError("annotation table has duplicate ENSEMBL rows")
    entrez = a["ENTREZID"].fillna("").astype(str).str.strip()
    parts = entrez.str.split("|").apply(lambda xs: sorted({x.strip() for x in xs if x.strip()}))
    n_ids = parts.apply(len)
    single = parts.where(n_ids == 1).apply(lambda xs: xs[0] if isinstance(xs, list) else None)
    claims = single.dropna().value_counts()
    shared = single.map(lambda e: e is not None and claims.get(e, 0) > 1)
    category = pd.Series("one_to_one", index=a.index)
    category[n_ids == 0] = "no_entrez"
    category[n_ids > 1] = "one_to_many"
    category[(n_ids == 1) & shared] = "many_to_one"
    ncbi = pd.array([int(e) if c == "one_to_one" else pd.NA
                     for e, c in zip(single, category, strict=True)], dtype="Int64")
    table = pd.DataFrame({"category": category.to_numpy(), "ncbi_gene": ncbi,
                          "symbol": a["SYMBOL"].to_numpy()},
                         index=pd.Index(a["ENSEMBL"].to_numpy(), name="ensembl"))
    return GeneMap(table, source, sha256)


def load_map(path: Path | None = None, verify: bool = True) -> GeneMap:
    """Load and classify the pinned GeneLab mouse annotation table."""
    path = path or default_workdir() / "data" / "annotations" / ANNOTATION_FILE
    digest = sha256_of(path)
    if verify and digest != ANNOTATION_SHA256:
        raise ValueError(f"{path.name}: SHA-256 {digest} does not match the pinned "
                         f"{ANNOTATION_SHA256}")
    ann = pd.read_csv(path, sep="\t", dtype=str, usecols=["ENSEMBL", "SYMBOL", "ENTREZID"])
    return build_map(ann, ANNOTATION_SOURCE, digest)


def map_summary(per_gene: pd.DataFrame, gene_map: GeneMap,
                tested_col: str | None = "tested") -> tuple[pd.DataFrame, dict]:
    """Map an Ensembl-indexed per-gene table to NCBI Gene, keeping 1:1 IDs only.

    Returns the mapped table (index ``ncbi_gene``, plus a ``symbol`` display column, never a
    join key) and a drop log counted over all input genes and, if ``tested_col`` is given,
    over the tested genes only.
    """
    cat = gene_map.table["category"].reindex(per_gene.index).fillna("not_in_table")

    def tally(mask) -> dict[str, int]:
        vc = cat[mask].value_counts()
        return {c: int(vc.get(c, 0)) for c in (*CATEGORIES, "not_in_table")}

    log = {"all_genes": tally(pd.Series(True, index=per_gene.index)),
           "source": gene_map.source, "sha256": gene_map.sha256}
    if tested_col is not None:
        log["tested_genes"] = tally(per_gene[tested_col].astype(bool))
    keep = cat == "one_to_one"
    out = per_gene[keep.to_numpy()].copy()
    out.insert(0, "ensembl", out.index)
    out["ncbi_gene"] = gene_map.table.loc[out.index, "ncbi_gene"].astype("int64").to_numpy()
    out["symbol"] = gene_map.table.loc[out.index, "symbol"].to_numpy()
    out = out.set_index("ncbi_gene")
    if not out.index.is_unique:  # impossible after the 1:1 filter; checked anyway
        raise AssertionError("duplicate NCBI Gene IDs after 1:1 mapping")
    return out, log
