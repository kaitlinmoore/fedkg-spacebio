"""The inclusion-flag hash (``fedkg_spacebio.sites.inclusion``; CONTRACTS §4). Fake gene IDs."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from fedkg_spacebio.sites.inclusion import canonical_flags_csv, inclusion_flags_sha256


def genes():
    return {"ncbi_gene": np.array([9_000_000_003, 9_000_000_001, 9_000_000_002], dtype=np.int64),
            "log2fc": np.array([0.1, 0.2, 0.3]),  # not part of the hash
            "tested_mean10": np.array([True, False, True]),
            "tested_indep_filter": np.array([False, False, True]),
            "tested_finite_p": np.array([True, True, True])}


def test_canonical_bytes_by_hand():
    # Written out by hand from the rule: header, rows sorted by ncbi_gene, 1/0, LF after each.
    expected = (b"ncbi_gene,tested_mean10,tested_indep_filter,tested_finite_p\n"
                b"9000000001,0,0,1\n"
                b"9000000002,1,1,1\n"
                b"9000000003,1,0,1\n")
    assert canonical_flags_csv(genes()) == expected
    assert inclusion_flags_sha256(genes()) == hashlib.sha256(expected).hexdigest()


def test_hash_ignores_row_order_and_estimates():
    g = genes()
    perm = np.array([2, 0, 1])
    shuffled = {k: v[perm] for k, v in g.items()} | {"log2fc": np.zeros(3)}
    assert inclusion_flags_sha256(shuffled) == inclusion_flags_sha256(g)


def test_hash_changes_when_one_flag_flips():
    g = genes()
    flipped = g | {"tested_finite_p": np.array([True, False, True])}
    assert inclusion_flags_sha256(flipped) != inclusion_flags_sha256(g)


def test_frame_with_index_gives_the_same_hash():
    g = genes()
    frame = pd.DataFrame(g).set_index("ncbi_gene")  # SiteSummary.to_frame() layout
    assert inclusion_flags_sha256(frame) == inclusion_flags_sha256(g)


@pytest.mark.parametrize("bad", [
    {"tested_mean10": np.array([1, 0, 1])},                        # not boolean
    {"tested_mean10": np.array([True, False])},                    # wrong length
    {"ncbi_gene": np.array([1, 1, 2], dtype=np.int64)},            # duplicate gene
    {"ncbi_gene": np.array(["1", "2", "3"])},                      # not integers
])
def test_malformed_flags_are_refused(bad):
    with pytest.raises(ValueError):
        inclusion_flags_sha256(genes() | bad)
