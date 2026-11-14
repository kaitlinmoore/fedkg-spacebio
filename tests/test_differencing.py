"""The exact differencing test (CONTRACTS §5.3).

Notation used throughout: rows are releases written as bitmasks over the stratum's clients, with
bit j set when client j contributed (column j). Example B's columns are A, B, C, D = bits 0..3,
so {A,B,C,D} = 0b1111 = 15, {A,C,D} = 0b1101 = 13, {A,B,C} = 0b0111 = 7, {A} = 0b0001 = 1.

The criterion being tested: some combination of the rows is non-zero at client i and zero outside
a group T (i ∈ T) **iff** column i is not in the span of the columns outside T.
"""

import itertools
import time

import numpy as np
import pytest
from fedkg_spacebio.policy.differencing import failing_genes, isolates, isolating_set

ABCD = ["SpaceX-4", "SpaceX-8", "SpaceX-13", "SpaceX-16"]  # one site per column
A, B, C, D = 0, 1, 2, 3
FULL, NO_B, NO_D, ONLY_A = 0b1111, 0b1101, 0b0111, 0b0001


# ------------------------------------------------------------------ example B by hand


def test_r1_alone_isolates_nobody():
    for col in range(4):
        assert isolating_set([FULL], col, ABCD, 3) is None


def test_j3_isolates_b_but_not_d():
    rows = [FULL, NO_B]  # 1111 − 1011 = 0100
    assert isolating_set(rows, B, ABCD, 3) == frozenset({B})
    assert isolating_set(rows, D, ABCD, 3) is None


def test_j4_isolates_d():
    rows = [FULL, NO_B, NO_D]  # 1111 − 1110 = 0001
    assert isolating_set(rows, D, ABCD, 3) == frozenset({D})


def test_j5_isolates_the_pair_c_d():
    rows = [FULL, NO_B, ONLY_A]  # 1011 − 1000 = 0011: C + D, a group of 2
    assert isolating_set(rows, D, ABCD, 3) == frozenset({C, D})
    assert isolating_set(rows, D, ABCD, 2) is None  # a pair is allowed when min_sites is 2


def test_gene_d_never_contributed_to_passes():
    assert isolating_set([NO_D, NO_D, ONLY_A], D, ABCD, 3) is None


def test_a_small_release_is_itself_a_small_group():
    """Example C: an Earth leave-one-out that keeps SpaceX-16-LAR, with an empty log."""
    earth = ["SpaceX-13", "SpaceX-16", "SpaceX-21"]
    assert isolating_set([0b110], 1, earth, 3) == frozenset({1, 2})


def test_example_c_drop_16_needs_the_full_release():
    earth = ["SpaceX-13", "SpaceX-16", "SpaceX-21"]
    assert isolating_set([0b101], 1, earth, 3) is None             # empty log: passes
    assert isolating_set([0b111, 0b101], 1, earth, 3) == frozenset({1})  # 111 − 101 = 010


def test_fewer_sites_than_min_sites_in_total():
    assert isolating_set([0b11], 0, ["X", "Y"], 3) == frozenset({0, 1})


def test_group_size_counts_sites_not_clients():
    """Two columns of one site: their sum is one site's total."""
    rows = [0b0011]
    assert isolating_set(rows, 0, ["X", "X", "Y", "Z"], 2) == frozenset({0, 1})
    assert isolating_set(rows, 0, ["X", "W", "Y", "Z"], 2) is None


def test_mixed_outputs_leak():
    """Plain per-site summaries from A, B, C plus a masked FE over all four isolate D."""
    rows = [0b0001, 0b0010, 0b0100, FULL]
    assert isolating_set(rows, D, ABCD, 3) == frozenset({D})


def test_isolates_directly():
    assert isolates([FULL, NO_D], D, {D}, 4)
    assert not isolates([FULL, NO_B], D, {D}, 4)
    assert isolates([FULL, NO_B, ONLY_A], D, {C, D}, 4)
    with pytest.raises(ValueError):
        isolates([FULL], D, {A}, 4)  # the group must contain the protected column


# ------------------------------------------------------------------ properties


def random_rows(rng, n_cols, n_rows):
    return [int(x) for x in rng.integers(1, 2 ** n_cols, size=n_rows)]


def float_isolates(rows, col, group, n_cols):
    """Independent check: is column ``col`` outside the least-squares span of the others?"""
    R = np.array([[(r >> j) & 1 for j in range(n_cols)] for r in rows], dtype=float)
    others = [j for j in range(n_cols) if j not in group]
    target = R[:, col]
    if not others:
        return bool(np.any(target))
    S = R[:, others]
    x, *_ = np.linalg.lstsq(S, target, rcond=None)
    return bool(np.linalg.norm(S @ x - target) > 1e-9)


def test_exact_matches_least_squares_on_random_matrices():
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(2, 8))
        rows = random_rows(rng, n, int(rng.integers(1, 9)))
        col = int(rng.integers(0, n))
        others = [j for j in range(n) if j != col]
        k = int(rng.integers(0, n))
        group = {col, *rng.choice(others, size=min(k, len(others)), replace=False).tolist()}
        assert isolates(rows, col, group, n) == float_isolates(rows, col, group, n)


def test_failure_is_monotone_in_the_group():
    """If T isolates i, every larger T does too, so only the largest allowed T matters."""
    rng = np.random.default_rng(1)
    for _ in range(100):
        n = int(rng.integers(3, 7))
        rows = random_rows(rng, n, int(rng.integers(1, 6)))
        col = int(rng.integers(0, n))
        others = [j for j in range(n) if j != col]
        for size in range(len(others) + 1):
            for extra in itertools.combinations(others, size):
                T = {col, *extra}
                if isolates(rows, col, T, n):
                    for x in others:
                        assert isolates(rows, col, T | {x}, n)


def test_duplicate_rows_and_order_dont_matter():
    a = isolating_set([FULL, NO_B, ONLY_A], D, ABCD, 3)
    assert isolating_set([ONLY_A, FULL, ONLY_A, NO_B, FULL], D, ABCD, 3) == a


# ------------------------------------------------------------------ many genes at once


def per_gene_loop(masks, col, sites, m):
    return np.array([isolating_set([int(x) for x in row if x], col, sites, m) is not None
                     for row in masks])


def test_grouped_equals_per_gene_loop():
    rng = np.random.default_rng(2)
    pool = np.array([0, FULL, NO_B, NO_D, ONLY_A, 0b1100, 0b1010, 0b0110], dtype=np.uint16)
    masks = rng.choice(pool, size=(500, 5))
    for col in range(4):
        for m in (2, 3, 4):
            np.testing.assert_array_equal(failing_genes(masks, col, ABCD, m),
                                          per_gene_loop(masks, col, ABCD, m))


def test_zero_rows_mean_no_release():
    masks = np.array([[0, 0, 0], [FULL, 0, 0], [FULL, NO_D, 0]], dtype=np.uint16)
    np.testing.assert_array_equal(failing_genes(masks, D, ABCD, 3), [False, False, True])


def test_twenty_thousand_genes_is_quick():
    """7 clients, ~8 logged releases, 20k genes: a few hundred distinct histories."""
    rng = np.random.default_rng(3)
    sites = [f"S{i}" for i in range(7)]
    common = np.array([0b1111111, 0b0111111, 0b1011111, 0b1111110, 0b1110111, 0], np.uint16)
    masks = rng.choice(common, size=(20_000, 8), p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
    t0 = time.perf_counter()
    for col in range(7):
        failing_genes(masks, col, sites, 3)
    elapsed = time.perf_counter() - t0
    print(f"\n20k genes × 7 protected clients: {elapsed:.3f} s")
    assert elapsed < 5.0
