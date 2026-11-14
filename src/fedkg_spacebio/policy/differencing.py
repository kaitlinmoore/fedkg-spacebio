"""The exact differencing test (CONTRACTS §5.3): can a set of releases isolate a site?

The question
------------
Take one gene. Every release that covered it is a 0/1 row over the stratum's clients (1 = that
client's value went into the released sum). A per-site summary is its own row with a single 1:
the receiver reads that client's value directly. Stack the rows into a matrix R (releases ×
clients). Anyone holding the released numbers can form any linear combination of the rows,
cᵀR, and so learn the matching combination of the clients' values.

Site i (min_sites = m) is **exposed** on this gene if some combination cᵀR is non-zero at i and
zero at every client outside a set T, where T contains i and covers fewer than m sites. Then the
releases give i's own value (T = {i}), or the sum over a group smaller than m that includes i.

Example B (columns A, B, C, D): R1 = 1111 and J4 = 1110 give 1111 − 1110 = 0001, D alone.

The criterion we compute
------------------------
    some cᵀR is non-zero at i and zero outside T
    ⇔  column i of R is NOT in the span of the columns outside T.

Why: write v = Rᵀc (v_j is the coefficient client j gets). "v zero outside T" says c is
orthogonal to every column outside T. Among such c, v_i = c · (column i) can be non-zero exactly
when column i has a component orthogonal to those columns, i.e. isn't in their span. (If it is
in their span, every c orthogonal to them is orthogonal to it too.)

So each test is "is one 0/1 vector a linear combination of a few others", decided exactly
with rational arithmetic (``fractions.Fraction``). With at most 7 clients per stratum and a few
dozen releases, that is a tiny system. Floating point would also work at this size, but a
policy decision shouldn't depend on a tolerance.

Two simplifications, both exact
-------------------------------
- **Monotone in T.** Making T bigger removes columns from "outside T", which can only shrink
  their span. So if T exposes i, every larger T does too. For a yes/no answer it is enough to
  test the *largest* allowed sets: all columns of m − 1 sites including i's (or every column, if
  the stratum has fewer than m sites). Adding more columns of a site already in T doesn't change
  the site count, so a maximal T always takes whole sites. |T| counts **sites**, not clients
  (§5.3): SpaceX-16-ISST and SpaceX-16-LAR together are one site.
- **Grouping genes.** The answer for a gene depends only on its *set* of distinct non-zero rows
  (repeats and order don't change a span). Real plans have thousands of genes but few distinct
  histories (most genes are covered by the same few releases), so each distinct history is
  solved once and the answer copied back.

Alternatives considered: a per-job "no two releases differ by one site" rule (misses
three-way combinations, such as J5's 1011 − 1000 = 0011); a float rank test with a tolerance
(right answer, wrong kind of argument for a policy); an LP or SAT formulation (the linear-algebra
form is exact and simpler).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from fractions import Fraction

import numpy as np


def _bits(row: int, n_cols: int) -> list[int]:
    return [(row >> j) & 1 for j in range(n_cols)]


def _in_span(vectors: list[list[int]], target: list[int]) -> bool:
    """Is ``target`` a rational linear combination of ``vectors``? (Gaussian elimination.)"""
    if not any(target):
        return True
    basis: list[tuple[int, list[Fraction]]] = []  # (pivot index, row with 1 at the pivot)

    def reduce(v: list[Fraction]) -> list[Fraction]:
        for p, b in basis:
            if v[p]:
                f = v[p]
                v = [x - f * y for x, y in zip(v, b, strict=True)]
        return v

    for vec in vectors:
        v = reduce([Fraction(x) for x in vec])
        pivot = next((k for k, x in enumerate(v) if x), None)
        if pivot is not None:
            v = [x / v[pivot] for x in v]
            # keep the basis fully reduced so later reductions only need one pass
            basis = [(p, [x - b[pivot] * y for x, y in zip(b, v, strict=True)] if b[pivot]
                      else b) for p, b in basis]
            basis.append((pivot, v))
    return not any(reduce([Fraction(x) for x in target]))


def isolates(rows: Iterable[int], col: int, group: Iterable[int], n_cols: int) -> bool:
    """Do the releases ``rows`` (bitmasks) expose column ``col`` within ``group``?

    True when some combination of the rows is non-zero at ``col`` and zero outside ``group``,
    i.e. column ``col`` isn't in the span of the columns outside ``group``.
    """
    group = set(group)
    if col not in group:
        raise ValueError("the group must contain the protected column")
    rows = sorted({int(r) for r in rows if r})
    columns = list(zip(*(_bits(r, n_cols) for r in rows), strict=True)) if rows else []
    if not columns:
        return False
    outside = [list(columns[j]) for j in range(n_cols) if j not in group]
    return not _in_span(outside, list(columns[col]))


def _site_groups(col: int, col_sites: Sequence[str], min_sites: int) -> list[frozenset[int]]:
    """The largest allowed groups: whole sites, i's site plus (m − 2) others (or all sites)."""
    sites = list(dict.fromkeys(col_sites))
    own = col_sites[col]
    others = [s for s in sites if s != own]
    k = min(min_sites - 1, len(sites)) - 1  # how many other sites fit beside i's own
    groups = []
    for chosen in itertools.combinations(others, max(k, 0)):
        keep = {own, *chosen}
        groups.append(frozenset(j for j, s in enumerate(col_sites) if s in keep))
    return groups


def exposed(rows: Iterable[int], col: int, col_sites: Sequence[str], min_sites: int) -> bool:
    """Yes/no: does any group of fewer than ``min_sites`` sites (with ``col``) get exposed?"""
    rows = {int(r) for r in rows if r}
    if not any((r >> col) & 1 for r in rows):
        return False  # the client never contributed: nothing of it was released
    n = len(col_sites)
    return any(isolates(rows, col, g, n) for g in _site_groups(col, col_sites, min_sites))


def isolating_set(rows: Iterable[int], col: int, col_sites: Sequence[str],
                  min_sites: int) -> frozenset[int] | None:
    """The smallest exposed group (column indices), or None if ``col`` is safe.

    Smallest first, then in index order, so the witness is the most specific one: {D} rather
    than {C, D} when D alone is exposed. Used to explain a failure; ``failing_genes`` answers
    yes/no for many genes at once.
    """
    rows = {int(r) for r in rows if r}
    if not exposed(rows, col, col_sites, min_sites):
        return None
    n = len(col_sites)
    others = [j for j in range(n) if j != col]
    for size in range(n):
        for extra in itertools.combinations(others, size):
            group = frozenset({col, *extra})
            if len({col_sites[j] for j in group}) < min_sites and isolates(rows, col, group, n):
                return group
    return None  # unreachable: exposed() found a group


def failing_genes(masks: np.ndarray, col: int, col_sites: Sequence[str],
                  min_sites: int) -> np.ndarray:
    """Per gene (row of ``masks``): is column ``col`` exposed? ``masks`` is genes × releases,
    each cell a bitmask over the columns; 0 means that release didn't cover the gene."""
    masks = np.asarray(masks)
    if masks.ndim != 2:
        raise ValueError("masks must be 2-D: genes × release rows")
    if masks.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    # canonical key per gene: its distinct non-zero rows, sorted (zeros pushed to the front)
    keys = np.sort(masks.astype(np.uint64), axis=1)
    keys[:, 1:][keys[:, 1:] == keys[:, :-1]] = 0
    keys = np.sort(keys, axis=1)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    answers = np.array([exposed(row.tolist(), col, col_sites, min_sites) for row in unique],
                       dtype=bool)
    return answers[inverse.reshape(-1)]
