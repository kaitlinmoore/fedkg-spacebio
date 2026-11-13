"""Ensembl → NCBI Gene 1:1 mapping and its drop log (``fedkg_spacebio.harmonize.gene_map``).

Fixture ``tests/fixtures/phase3a/annotations.tsv`` (fake IDs; the NCBI-style numbers start at
9,000,000,001, far above any real NCBI Gene ID) has one gene of each kind:
FAKEENSG0001/0004/0007 one-to-one, 0002 no Entrez ID, 0003 one-to-many, and 0005 + 0006 sharing
one Entrez ID (many-to-one).
"""

from pathlib import Path

import pandas as pd
import pytest

from fedkg_spacebio.harmonize.gene_map import build_map, load_map, map_summary

ANN = Path(__file__).parent / "fixtures" / "phase3a" / "annotations.tsv"


@pytest.fixture(scope="module")
def gmap():
    return load_map(ANN, verify=False)  # the checksum pin is for the real table


def test_categories(gmap):
    cat = gmap.table["category"]
    assert cat["FAKEENSG0001"] == "one_to_one"
    assert cat["FAKEENSG0002"] == "no_entrez"
    assert cat["FAKEENSG0003"] == "one_to_many"
    assert cat["FAKEENSG0005"] == "many_to_one" and cat["FAKEENSG0006"] == "many_to_one"
    assert gmap.table_counts == {"one_to_one": 3, "no_entrez": 1, "one_to_many": 1,
                                 "many_to_one": 2, "total": 7}


def test_many_to_one_drops_every_claimant_not_one(gmap):
    per_gene = pd.DataFrame({"log2fc": [0.1, 0.2], "tested": [True, True]},
                            index=["FAKEENSG0005", "FAKEENSG0006"])
    mapped, log = map_summary(per_gene, gmap)
    assert mapped.empty
    assert log["tested_genes"]["many_to_one"] == 2


def test_map_summary_drop_log_and_output(gmap):
    per_gene = pd.DataFrame(
        {"log2fc": [1.0, 2.0, 3.0, 4.0, 5.0],
         "tested": [True, True, False, True, True]},
        index=["FAKEENSG0001", "FAKEENSG0002", "FAKEENSG0003", "FAKEENSG0004", "NOT_IN_TABLE"])
    mapped, log = map_summary(per_gene, gmap)
    assert mapped.index.tolist() == [9000000001, 9000000004]
    assert mapped["symbol"].tolist() == ["Fake1", "Fake4"]
    assert mapped["ensembl"].tolist() == ["FAKEENSG0001", "FAKEENSG0004"]
    assert mapped["log2fc"].tolist() == [1.0, 4.0]
    assert log["all_genes"] == {"one_to_one": 2, "no_entrez": 1, "one_to_many": 1,
                                "many_to_one": 0, "not_in_table": 1}
    assert log["tested_genes"] == {"one_to_one": 2, "no_entrez": 1, "one_to_many": 0,
                                   "many_to_one": 0, "not_in_table": 1}
    # Nothing is lost silently: every input gene is counted exactly once.
    assert sum(log["all_genes"].values()) == len(per_gene)


def test_map_is_independent_of_which_genes_a_site_tested(gmap):
    """The 1:1 rule is decided on the whole table, so a site that never saw FAKEENSG0006 still
    drops FAKEENSG0005."""
    mapped, _ = map_summary(pd.DataFrame({"tested": [True]}, index=["FAKEENSG0005"]), gmap)
    assert mapped.empty


def test_duplicate_ensembl_rows_are_rejected():
    ann = pd.DataFrame({"ENSEMBL": ["X", "X"], "SYMBOL": ["a", "b"], "ENTREZID": ["1", "2"]})
    with pytest.raises(ValueError, match="duplicate"):
        build_map(ann)


def test_checksum_pin_is_enforced():
    with pytest.raises(ValueError, match="SHA-256"):
        load_map(ANN, verify=True)
