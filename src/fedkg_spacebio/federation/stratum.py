"""Server-side assembly: site summaries → genes × sites matrices → pooled tables.

Pure functions with no FLARE import. The FLARE controller
(``meta_flare.py``) and the in-process central check (``jobs/phase3a_meta/central.py``) both call
``pool_variants``, so Check 1 compares two transports feeding identical code.

Inclusion variants (Kaitlin, WS-07)
-----------------------------------
Each site sends every gene with a usable estimate, plus three ``tested_*`` flags. The server
picks genes per variant:

| variant | gene must be flagged by | at how many sites |
|---|---|---|
| ``primary`` | ``tested_mean10`` (base mean ≥ 10, finite p) | every site in the stratum |
| ``min3_sites`` | ``tested_mean10`` | at least 3 sites |
| ``indep_filter`` | ``tested_indep_filter`` (pydeseq2 padj not NaN) | every site |
| ``finite_p`` | ``tested_finite_p`` | every site |

A gene that isn't flagged at a site counts as **missing** there (NaN), even though the site
sent an estimate. BH runs within each variant, over exactly that variant's genes.
Leave-one-client-out variants (e.g. the ISS stratum without SpaceX-8) come from the same
summaries by dropping that client's column.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fedkg_spacebio.federation import meta


@dataclass(frozen=True)
class Variant:
    name: str
    flag: str
    min_sites: int | None  # None = every client in the variant
    drop_clients: tuple[str, ...] = ()


VARIANTS = (
    Variant("primary", "tested_mean10", None),
    Variant("min3_sites", "tested_mean10", 3),
    Variant("indep_filter", "tested_indep_filter", None),
    Variant("finite_p", "tested_finite_p", None),
)


def matrices(frames: Mapping[str, pd.DataFrame], flag: str) -> tuple[pd.Index, np.ndarray,
                                                                      np.ndarray, list[str]]:
    """Align per-site frames (index ncbi_gene) on the union of genes.

    Returns (genes, Y, V, clients). Y = log2FC and V = SE², with NaN where the site didn't send
    the gene or didn't flag it as tested under ``flag``. Clients come in sorted order, so the
    column order doesn't depend on arrival order.
    """
    clients = sorted(frames)
    genes = pd.Index(sorted(set().union(*(f.index for f in frames.values()))), name="ncbi_gene")
    Y = np.full((len(genes), len(clients)), np.nan)
    V = np.full_like(Y, np.nan)
    for j, c in enumerate(clients):
        f = frames[c]
        ok = f[flag].astype(bool)
        pos = genes.get_indexer(f.index[ok])
        Y[pos, j] = f.loc[ok, "log2fc"].to_numpy(dtype=np.float64)
        V[pos, j] = f.loc[ok, "se"].to_numpy(dtype=np.float64) ** 2
    return genes, Y, V, clients


def pool_variant(frames: Mapping[str, pd.DataFrame], variant: Variant) -> pd.DataFrame:
    use = {c: f for c, f in frames.items() if c not in variant.drop_clients}
    missing = set(variant.drop_clients) - set(frames)
    if missing:
        raise ValueError(f"{variant.name}: cannot drop clients that didn't report: {missing}")
    genes, Y, V, clients = matrices(use, variant.flag)
    k = np.isfinite(Y).sum(axis=1)
    need = len(clients) if variant.min_sites is None else variant.min_sites
    keep = k >= need
    pooled = meta.pool(Y[keep], V[keep])
    pooled.insert(0, "ncbi_gene", genes[keep].to_numpy())
    pooled.attrs.update({"variant": variant.name, "clients": clients, "min_sites": need,
                         "flag": variant.flag, "n_genes_union": int(len(genes))})
    return pooled.reset_index(drop=True)


def pool_variants(frames: Mapping[str, pd.DataFrame],
                  variants: tuple[Variant, ...] = VARIANTS) -> dict[str, pd.DataFrame]:
    return {v.name: pool_variant(frames, v) for v in variants}
