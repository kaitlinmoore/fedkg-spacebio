"""The gene plan: which clients contribute to which gene in a job (CONTRACTS §5.3, §6.1).

What it is for
--------------
In phase 3a the set of sites pooled for a gene varies by gene (a site doesn't test a gene that
fails its expression threshold; ≥ 3-site inclusion keeps genes some sites lack). So the policy
checks that count sites (minimum k, differencing) have to work gene by gene, and they have to
know the per-gene site sets **before** the job runs. The gene plan is that table. The
coordinator builds it from the inclusion flags each site released as ``site_metadata`` (§4,
§5.1) and the spec's inclusion rule. The spec names it by path and SHA-256, so the decision is
bound to one exact plan; later (WS-13) each client sends values only for its genes in it.

Building it
-----------
For rule r with flag column f = ``INCLUSION_FLAG[r]``:
- a client contributes to gene g when its released flags have f true for g;
- a gene a client didn't report at all counts as not included (it can't send a value for it);
- the plan keeps genes with at least ``min_sites_per_gene`` contributing clients, where ``all``
  means every client in the job (complete case).

Canonical form (so the hash can be reproduced by hand)
-------------------------------------------------------
UTF-8, ``\\n`` after every row including the last, no quoting, no spaces:
- **gene plan:** header ``ncbi_gene,<client>,…`` (clients sorted by name); one row per gene,
  ascending by ``ncbi_gene`` (decimal integer); each client cell ``1`` or ``0``; every row has at
  least one ``1``;
- **gene list** (``genes_exclude``, a decision's failing genes): header ``ncbi_gene``, one gene
  per row, ascending, no duplicates.

Same idea as the inclusion-flag hash (``sites.inclusion``): a sorted, minimal CSV is easy to
rebuild in any language, where a pickled frame or a ``.npz`` depends on library internals.
Readers check the bytes are canonical (re-serialise and compare), so two files with the same
content always have the same hash.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

INCLUSION_FLAG = {
    "mean10_finite": "tested_mean10",
    "indep_filter": "tested_indep_filter",
    "finite_p": "tested_finite_p",
}


class GenePlanError(ValueError):
    """A plan or gene list that can't be built, isn't canonical, or doesn't match its hash."""


# ------------------------------------------------------------------ building


def _flag_frame(client: str, flags) -> pd.DataFrame:
    df = pd.DataFrame(flags) if not isinstance(flags, pd.DataFrame) else flags
    if "ncbi_gene" not in df.columns:
        if df.index.name != "ncbi_gene":
            raise GenePlanError(f"{client}: flags need an ncbi_gene column or index")
        df = df.reset_index()
    if df["ncbi_gene"].duplicated().any():
        dup = sorted(df.loc[df["ncbi_gene"].duplicated(), "ncbi_gene"].unique())[:5]
        raise GenePlanError(f"{client}: genes reported more than once, e.g. {dup}")
    return df


def build_gene_plan(flags_by_client: Mapping[str, pd.DataFrame | Mapping], rule: str,
                    min_sites_per_gene: int | str) -> pd.DataFrame:
    """The plan for one job: index ``ncbi_gene``, one bool column per client (sorted by name).

    ``flags_by_client`` maps each job client to its released flags (a frame with an
    ``ncbi_gene`` column or index and the ``tested_*`` columns).
    """
    if rule not in INCLUSION_FLAG:
        raise GenePlanError(f"unknown inclusion rule {rule!r}; expected one of "
                            f"{sorted(INCLUSION_FLAG)}")
    if not flags_by_client:
        raise GenePlanError("no clients")
    clients = sorted(flags_by_client)
    if min_sites_per_gene == "all":
        threshold = len(clients)
    elif isinstance(min_sites_per_gene, int | np.integer) and not isinstance(
            min_sites_per_gene, bool) and min_sites_per_gene >= 1:
        threshold = int(min_sites_per_gene)
    else:
        raise GenePlanError(f"min_sites_per_gene must be 'all' or an int >= 1, got "
                            f"{min_sites_per_gene!r}")

    flag = INCLUSION_FLAG[rule]
    included: dict[str, set[int]] = {}
    for client in clients:
        df = _flag_frame(client, flags_by_client[client])
        if flag not in df.columns:
            raise GenePlanError(f"{client}: flags lack column {flag!r}")
        values = df[flag]
        if not values.isin([0, 1, True, False]).all():
            raise GenePlanError(f"{client}: {flag} must be 0/1 or bool")
        included[client] = {int(g) for g in df.loc[values.astype(bool), "ncbi_gene"]}

    genes = sorted(set().union(*included.values()))
    plan = pd.DataFrame({c: np.array([g in included[c] for g in genes], dtype=bool)
                         for c in clients},
                        index=pd.Index(np.array(genes, dtype=np.int64), name="ncbi_gene"))
    return plan[plan.sum(axis=1) >= threshold]


def exclude_genes(plan: pd.DataFrame, genes: Iterable[int]) -> pd.DataFrame:
    """The plan minus ``genes`` (genes not in the plan are ignored)."""
    return plan[~plan.index.isin(list(genes))]


# ------------------------------------------------------------------ canonical gene plan


def _gene_ids(values) -> np.ndarray:
    arr = np.asarray(list(values))
    if arr.size and (arr.dtype.kind not in "iu" or (arr < 0).any()):
        raise GenePlanError("ncbi_gene values must be non-negative integers")
    return arr.astype(np.int64)


def canonical_plan_csv(plan: pd.DataFrame) -> bytes:
    """The plan's canonical bytes (see the module docstring)."""
    clients = sorted(str(c) for c in plan.columns)
    for c in clients:
        if not c or any(ch in c for ch in ",\n\r\" "):
            raise GenePlanError(f"client name {c!r} can't be a CSV header cell")
    genes = _gene_ids(plan.index)
    if len(set(genes.tolist())) != len(genes):
        raise GenePlanError("a gene appears more than once in the plan")
    cells = plan[clients].to_numpy()
    if cells.size and not np.isin(cells, [0, 1, True, False]).all():
        raise GenePlanError("plan cells must be 0/1 or bool")
    cells = cells.astype(bool)
    if len(genes) and not cells.any(axis=1).all():
        bad = genes[~cells.any(axis=1)][:5].tolist()
        raise GenePlanError(f"genes with no contributing client: {bad}")
    order = np.argsort(genes, kind="stable")
    lines = ["ncbi_gene," + ",".join(clients)]
    lines += [f"{g}," + ",".join("1" if v else "0" for v in row)
              for g, row in zip(genes[order].tolist(), cells[order].tolist(), strict=True)]
    return ("\n".join(lines) + "\n").encode("utf-8")


def gene_plan_sha256(plan: pd.DataFrame) -> str:
    return hashlib.sha256(canonical_plan_csv(plan)).hexdigest()


def write_gene_plan(plan: pd.DataFrame, path: str | Path) -> str:
    """Write the canonical plan; return its SHA-256."""
    data = canonical_plan_csv(plan)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _read_checked(path: str | Path, sha256: str | None) -> bytes:
    path = Path(path)
    if not path.is_file():
        raise GenePlanError(f"{path}: file not found")
    data = path.read_bytes()
    if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
        raise GenePlanError(f"{path}: SHA-256 doesn't match {sha256}")
    return data


def _rows(data: bytes, path) -> list[list[str]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise GenePlanError(f"{path}: not UTF-8") from None
    if not text.endswith("\n"):
        raise GenePlanError(f"{path}: not canonical (no newline after the last row)")
    return [line.split(",") for line in text[:-1].split("\n")]


def read_gene_plan(path: str | Path, sha256: str | None = None) -> pd.DataFrame:
    """Read a plan, checking its hash (if given) and that the bytes are canonical."""
    data = _read_checked(path, sha256)
    rows = _rows(data, path)
    header = rows[0]
    if header[0] != "ncbi_gene" or len(header) < 2:
        raise GenePlanError(f"{path}: header must be ncbi_gene,<client>,...")
    clients = header[1:]
    try:
        genes = [int(r[0]) for r in rows[1:]]
        if any(len(r) != len(header) or any(v not in ("0", "1") for v in r[1:])
               for r in rows[1:]):
            raise ValueError
    except ValueError:
        raise GenePlanError(f"{path}: rows must be <gene>,<0|1>,... matching the header") from None
    cells = np.array([[v == "1" for v in r[1:]] for r in rows[1:]], dtype=bool).reshape(
        len(genes), len(clients))
    plan = pd.DataFrame(cells, columns=clients,
                        index=pd.Index(np.array(genes, dtype=np.int64), name="ncbi_gene"))
    if canonical_plan_csv(plan) != data:
        raise GenePlanError(f"{path}: not in canonical form (CONTRACTS §6.1)")
    return plan


# ------------------------------------------------------------------ canonical gene list


def canonical_gene_list_csv(genes: Iterable[int]) -> bytes:
    arr = _gene_ids(genes)
    if len(set(arr.tolist())) != len(arr):
        raise GenePlanError("a gene appears more than once in the list")
    return ("ncbi_gene\n" + "".join(f"{g}\n" for g in sorted(arr.tolist()))).encode("utf-8")


def gene_list_sha256(genes: Iterable[int]) -> str:
    return hashlib.sha256(canonical_gene_list_csv(genes)).hexdigest()


def write_gene_list(genes: Iterable[int], path: str | Path) -> str:
    """Write the canonical gene list; return its SHA-256."""
    data = canonical_gene_list_csv(genes)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def read_gene_list(path: str | Path, sha256: str | None = None) -> list[int]:
    """Read a gene list (ascending ints), checking its hash (if given) and canonical form."""
    data = _read_checked(path, sha256)
    rows = _rows(data, path)
    if rows[0] != ["ncbi_gene"]:
        raise GenePlanError(f"{path}: header must be ncbi_gene")
    try:
        genes = [int(r[0]) for r in rows[1:] if len(r) == 1]
    except ValueError:
        raise GenePlanError(f"{path}: rows must be integers") from None
    if len(genes) != len(rows) - 1 or canonical_gene_list_csv(genes) != data:
        raise GenePlanError(f"{path}: not in canonical form (CONTRACTS §6.1)")
    return genes
