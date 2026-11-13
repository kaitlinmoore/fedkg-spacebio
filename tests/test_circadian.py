"""Tests for the Q-11 rhythmic-gene diagnostics (``fedkg_spacebio.evaluation.circadian``).
Toy tables with obviously fake probe sets and IDs; no real annotation is read."""

import gzip

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fedkg_spacebio.evaluation import circadian


def test_read_geo_annot_and_probe_mapping(tmp_path):
    text = "\n".join([
        "^Annotation", "!Annotation_date = Jan 01 2000", "!platform_table_begin",
        "ID\tGene title\tGene symbol\tGene ID",
        "FAKE_1_at\tfake a\tFakea\t900001",
        "FAKE_2_at\tfake b\tFakeb\t900002///900003",
        "FAKE_3_at\t\t\t",
        "FAKE_4_at\tfake a\tFakea\t900001",
        "!platform_table_end", ""])
    p = tmp_path / "toy.annot.gz"
    with gzip.open(p, "wt") as fh:
        fh.write(text)
    annot = circadian.read_geo_annot(p)
    assert list(annot["ID"]) == ["FAKE_1_at", "FAKE_2_at", "FAKE_3_at", "FAKE_4_at"]
    pmap, log = circadian.probe_to_entrez(annot)
    assert dict(pmap) == {"FAKE_1_at": 900001, "FAKE_4_at": 900001}
    assert log == {"probe_sets": 4, "unique_entrez": 2, "no_entrez": 1, "multiple_entrez": 1}


def test_rhythmic_genes_counts_every_drop():
    pmap = pd.Series({"P1": 900001, "P2": 900002, "P4": 900001})   # P3: on array, ambiguous
    s1 = pd.DataFrame({"Probe Set ID": ["P1", "P2", "P3", "P4", "P9"]})
    platform = ["P1", "P2", "P3", "P4"]                            # P9 not on the platform table
    genes, log = circadian.rhythmic_genes(s1, pmap, platform, known_ids=[900001])
    assert genes == {900001}
    assert log["probe_sets"] == 5 and log["probe_mapped"] == 3
    assert log["probe_not_in_platform_annotation"] == 1       # P9
    assert log["probe_without_unique_entrez"] == 1            # P3
    assert log["distinct_genes"] == 2 and log["not_in_genelab_annotation"] == 1
    assert log["drop_rate_probes"] == pytest.approx(0.4)


def test_resolve_symbols_reports_rather_than_guesses():
    genelab = pd.DataFrame({"SYMBOL": ["Fakea", "Fakeb", "Fakeb", "Fakec"],
                            "ENTREZID": ["900001", "900002", "900009", "900003"]})
    annot = pd.DataFrame({"Gene symbol": ["Fakea", "Fakeb", "Fakec"],
                          "Gene ID": ["900001", "900002", "900004"]})
    out = circadian.resolve_symbols(["Fakea", "Fakeb", "Fakec", "Faked"], genelab, annot)
    st = dict(zip(out["symbol"], out["status"], strict=True))
    assert st == {"Fakea": "ok", "Fakeb": "ambiguous_in_genelab", "Fakec": "disagree",
                  "Faked": "missing_in_genelab"}
    assert out.loc[out.symbol == "Fakea", "ncbi_gene"].item() == 900001


def test_enrichment_matches_fisher_and_uses_measurable_background():
    hit = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 0], dtype=bool)
    rhy = np.array([1, 1, 0, 1, 0, 0, 0, 0, 1, 1], dtype=bool)
    meas = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=bool)   # last two not on the array
    e = circadian.enrichment(hit, rhy, meas)
    # Measurable: hits = genes 0,1,2 (2 rhythmic); non-hits = 3..7 (1 rhythmic).
    assert (e["hits_measurable"], e["rhythmic_hits"], e["background"]) == (3, 2, 8)
    want = stats.fisher_exact([[2, 1], [1, 4]], alternative="greater")
    assert e["p_one_sided"] == pytest.approx(want.pvalue)
    assert e["odds_ratio"] == pytest.approx(want.statistic)
    empty = circadian.enrichment(np.zeros(10, bool), rhy, meas)
    assert empty["hits_measurable"] == 0 and np.isnan(empty["p_one_sided"])


def test_without_rhythmic_reruns_bh_on_the_rest():
    p = np.array([1e-6, 2e-6, 0.01, 0.02, 0.5, 0.9])
    calls = pd.DataFrame({"fe__p": p, "fe__hit": [True, True, True, True, False, False],
                          "fe_flag__p": p,
                          "fe_flag__hit": [True, False, True, False, False, False],
                          "het_flag": [False, True, False, True, False, False]})
    rhy = np.array([True, False, False, False, False, False])
    out = circadian.without_rhythmic(calls, "fe", rhy)
    assert out["hits"] == 4 and out["hits_dropping_rhythmic"] == 3
    # BH on the 5 remaining p: 2e-6·5/1, 0.01·5/2 = 0.025, 0.02·5/3 = 0.033 → 3 hits.
    assert out["hits_bh_rerun_nonrhythmic"] == 3
    flag = circadian.without_rhythmic(calls, "fe_flag", rhy)
    assert flag["hits_bh_rerun_nonrhythmic"] == 1             # flagged genes removed again


def test_enrichment_stratified_removes_a_pure_expression_effect():
    """Hits and rhythmic genes both concentrate in the high-expression stratum, but within each
    stratum they're independent: the crude OR is > 1, the Mantel–Haenszel OR is 1."""
    strata = np.repeat([0.0, 1.0], 200)
    rhy = np.r_[np.arange(200) < 20, np.arange(200) < 100]        # 10% low, 50% high
    hit = np.r_[np.arange(200) % 10 == 0, np.arange(200) % 2 == 0]  # 10% low, 50% high
    meas = np.ones(400, bool)
    crude = circadian.enrichment(hit, rhy, meas)
    mh = circadian.enrichment_stratified(hit, rhy, meas, strata)
    assert crude["odds_ratio"] > 2
    assert mh["or_mh"] == pytest.approx(1.0, abs=1e-12)
    assert mh["p_cmh_one_sided"] == pytest.approx(0.5, abs=1e-9)


def test_resolve_symbols_alias_pair():
    genelab = pd.DataFrame({"SYMBOL": ["Newname"], "ENTREZID": ["900001"]})
    annot = pd.DataFrame({"Gene symbol": ["Oldname"], "Gene ID": ["900001"]})
    out = circadian.resolve_symbols(["Newname/Oldname"], genelab, annot)
    assert out["status"].item() == "ok" and out["ncbi_gene"].item() == 900001
    assert out["symbol"].item() == "Newname" and out["array_symbol"].item() == "Oldname"
