"""A site's per-gene inclusion flags and their SHA-256 (CONTRACTS §4, §7 ``sites.list``).

What the hash is for
--------------------
Under the governance contracts (D-033), each site's inclusion flags (``tested_mean10``,
``tested_indep_filter``, ``tested_finite_p`` per gene) are released as ``site_metadata``
**before** a job. The coordinator needs them to build the gene plan (which sites contribute to
each gene) and to run the per-gene minimum-k and differencing checks before submission. The
site registry (``sites.list``) reports a hash of those flags with each client, so:
- a job spec, a decision record or a trace can say exactly which flags it was checked against;
- if a site's flags change (a new DE run, a different counts variant), the hash changes and an
  old decision visibly no longer matches.

The canonical form (what gets hashed)
-------------------------------------
A hash is only useful if every party computes it the same way from the same flags. So the
flags are written to bytes in one fixed way before hashing:
- UTF-8 CSV, header ``ncbi_gene,tested_mean10,tested_indep_filter,tested_finite_p``;
- one row per gene the site reports, **sorted by ncbi_gene ascending** (so the order genes were
  stored in doesn't matter);
- ``ncbi_gene`` as a plain decimal integer; each flag as ``1`` or ``0``;
- ``\\n`` line endings, including after the last row; no quoting, no spaces.

The hash is the lowercase hex SHA-256 of those bytes.

Alternatives considered: hashing the ``.npz`` the site sends (depends on numpy's file format and
array order, and covers the estimates too, which aren't site metadata); hashing a JSON dump
(key order and float formatting are easy to get subtly different). A sorted, minimal CSV is
easy to reproduce by hand in any language.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np
import pandas as pd

FLAGS = ("tested_mean10", "tested_indep_filter", "tested_finite_p")
HEADER = ",".join(("ncbi_gene", *FLAGS))


def _columns(genes) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(ncbi_gene, {flag: bool array}) from a summary's gene dict or a frame.

    A frame may carry ``ncbi_gene`` as a column or as its index (``SiteSummary.to_frame()``).
    """
    if isinstance(genes, pd.DataFrame) and "ncbi_gene" not in genes.columns:
        ids = genes.index.to_numpy()
    else:
        ids = np.asarray(genes["ncbi_gene"])
    if ids.dtype.kind not in "iu":
        raise ValueError("ncbi_gene must be integers")
    flags = {}
    for f in FLAGS:
        a = np.asarray(genes[f])
        if a.dtype != bool or a.shape != ids.shape:
            raise ValueError(f"{f} must be a boolean array with one entry per gene")
        flags[f] = a
    if len(np.unique(ids)) != len(ids):
        raise ValueError("ncbi_gene must be unique")
    return ids, flags


def canonical_flags_csv(genes: Mapping | pd.DataFrame) -> bytes:
    """The canonical bytes described in the module docstring."""
    ids, flags = _columns(genes)
    order = np.argsort(ids, kind="mergesort")
    lines = [HEADER]
    cols = [flags[f][order] for f in FLAGS]
    for i, g in enumerate(ids[order]):
        lines.append(",".join([str(int(g)), *("1" if c[i] else "0" for c in cols)]))
    return ("\n".join(lines) + "\n").encode("utf-8")


def inclusion_flags_sha256(genes: Mapping | pd.DataFrame) -> str:
    """Lowercase hex SHA-256 of ``canonical_flags_csv(genes)``."""
    return hashlib.sha256(canonical_flags_csv(genes)).hexdigest()


__all__ = ["FLAGS", "HEADER", "canonical_flags_csv", "inclusion_flags_sha256"]
