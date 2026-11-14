"""Server-side assembly (``federation.stratum``) and result validation (``meta_flare``).

Fake NCBI-style IDs (≥ 9,000,000,001). The pooling itself is tested in ``test_meta.py``. Tests
here that call ``pool`` need REML/HKSJ, so they fail until those exist.
"""

import numpy as np
import pandas as pd
import pytest
from nvflare.app_common.abstract.fl_model import FLModel

from fedkg_spacebio.federation import stratum
from fedkg_spacebio.federation.meta_flare import SUMMARY_META_KEY, summary_from_result
from fedkg_spacebio.sites.inclusion import inclusion_flags_sha256


def frame(ids, lfc, se, mean10, indep=None, finite_p=None):
    n = len(ids)
    return pd.DataFrame({
        "log2fc": lfc, "se": se, "base_mean": [50.0] * n, "tested_mean10": mean10,
        "tested_indep_filter": indep if indep is not None else [True] * n,
        "tested_finite_p": finite_p if finite_p is not None else [True] * n,
    }, index=pd.Index(ids, name="ncbi_gene"))


G1, G2, G3 = 9_000_000_001, 9_000_000_002, 9_000_000_003


@pytest.fixture
def frames():
    return {
        "B": frame([G1, G2, G3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [True, True, False]),
        "A": frame([G1, G2], [1.0, 2.0], [1.0, 2.0], [True, True]),
        "C": frame([G1, G3], [-1.0, -3.0], [0.5, 0.5], [True, True]),
    }


def test_matrices_align_on_gene_and_sort_clients(frames):
    genes, Y, V, clients = stratum.matrices(frames, "tested_mean10")
    assert clients == ["A", "B", "C"]
    assert genes.tolist() == [G1, G2, G3]
    np.testing.assert_array_equal(Y[0], [1.0, 0.1, -1.0])
    assert np.isnan(Y[1, 2])  # C didn't send G2
    assert np.isnan(Y[2, 0]) and np.isnan(Y[2, 1])  # A didn't send G3; B sent it, not tested
    np.testing.assert_allclose(V[0], [1.0, 0.01, 0.25])


def test_flag_changes_which_cells_count(frames):
    _, Y, _, _ = stratum.matrices(frames, "tested_finite_p")
    assert Y[2, 1] == pytest.approx(0.3)  # B's G3 counts under the looser flag


def test_primary_is_complete_case_and_drop_variant(frames):
    pooled = stratum.pool_variant(frames, stratum.Variant("primary", "tested_mean10", None))
    assert pooled["ncbi_gene"].tolist() == [G1]  # the only gene tested at A, B and C
    assert pooled["k"].tolist() == [3]
    without_c = stratum.pool_variant(frames, stratum.Variant("x", "tested_mean10", None, ("C",)))
    assert without_c["ncbi_gene"].tolist() == [G1, G2]
    with pytest.raises(ValueError, match="didn't report"):
        stratum.pool_variant(frames, stratum.Variant("x", "tested_mean10", None, ("Z",)))


def _good_result(client="SITE-A"):
    genes = {"ncbi_gene": np.array([G1, G2], dtype=np.int64), "log2fc": np.array([0.1, -0.2]),
             "se": np.array([0.3, 0.4]), "base_mean": np.array([20.0, 30.0]),
             "tested_mean10": np.array([True, True]),
             "tested_indep_filter": np.array([True, True]),
             "tested_finite_p": np.array([True, True])}
    meta = {k: None for k in ("site", "stratum", "design", "counts_variant", "de_settings",
                              "rounding", "n_genes_de", "n_ercc_dropped", "mapping", "versions",
                              "census_rows")}
    meta.update(client=client, n_flight=3, n_ground=3,
                inclusion_flags_sha256=inclusion_flags_sha256(genes))
    return FLModel(params=genes, meta={SUMMARY_META_KEY: meta, "client_name": "SITE-A",
                                       "props": {}})


def test_server_accepts_a_valid_result():
    s = summary_from_result(_good_result())
    assert s.meta["client"] == "SITE-A"


def test_server_rejects_impersonation_and_extra_fields():
    with pytest.raises(ValueError, match="came from"):
        summary_from_result(_good_result(client="SITE-B"))
    r = _good_result()
    r.params["counts"] = np.zeros(2)
    with pytest.raises(ValueError, match="per-gene fields"):
        summary_from_result(r)
    r = _good_result()
    del r.meta[SUMMARY_META_KEY]
    with pytest.raises(ValueError, match="has no"):
        summary_from_result(r)


def test_server_rejects_flags_that_dont_match_their_hash():
    # A summary whose flags were changed after hashing (or a hash copied from another run) is
    # refused on receipt: the registry's hash must describe the flags the server pools with.
    r = _good_result()
    r.params["tested_mean10"] = np.array([True, False])
    with pytest.raises(ValueError, match="inclusion_flags_sha256"):
        summary_from_result(r)
