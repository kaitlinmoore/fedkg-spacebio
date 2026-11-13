"""File selection, DE header parsing, contrast mapping, namespace and pipeline detection."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fedkg_spacebio.census import files as fl

FIX = Path(__file__).parent / "fixtures" / "census"


def test_select_rnaseq_files_by_role_only_from_processed_category():
    listing = json.loads((FIX / "file_listing.json").read_text())["studies"]["OSD-FAKE-1"]
    files = fl.select_rnaseq_files(listing["study_files"])
    assert files["counts_rsem"]["file_size"] == 120  # not the raw-data-category duplicate
    assert files["counts_rsem_rrnarm"]["file_name"].endswith("_rRNArm_GLbulkRNAseq.csv")
    assert set(files) == {"counts_rsem", "counts_rsem_rrnarm", "de", "de_rrnarm",
                          "sample_table", "contrasts"}  # no FASTQ role


def test_two_candidates_for_one_role_raise():
    f = {"category": fl.PROCESSED_CATEGORY, "file_name": "X_rna_seq_contrasts.csv"}
    g = {"category": fl.PROCESSED_CATEGORY, "file_name": "Y_rna_seq_contrasts_GLbulkRNAseq.csv"}
    with pytest.raises(ValueError, match="contrasts"):
        fl.select_rnaseq_files([f, g])


def test_de_header_stat_detection_per_contrast():
    head = (FIX / "de_head.csv").read_bytes()
    info = fl.de_header_info(fl.header_columns(head))
    assert info["contrasts"] == ["(Space Flight & Carcass)v(Ground Control & Carcass)",
                                 "(Space Flight & Carcass)v(Ground Control & Upon euthanasia)"]
    assert info["stat_contrasts"] == ["(Space Flight & Carcass)v(Ground Control & Carcass)"]
    assert info["id_columns"] == ["ENSEMBL", "SYMBOL", "ENTREZID"]


def test_partial_csv_drops_the_cut_off_last_line():
    # A Range read ends mid-line with no newline; the fixture file has one, so strip it.
    df = fl.parse_partial_csv((FIX / "de_head.csv").read_bytes().rstrip(b"\n"))
    assert df["ENSEMBL"].tolist() == ["ENSMUSG00000000001", "ENSMUSG00000000002"]


def test_contrast_for_finds_flight_first_contrast_and_n_in_de():
    st = pd.read_csv(FIX / "sample_table.csv")
    ct = pd.read_csv(FIX / "contrasts.csv")
    res = fl.contrast_for(["F1_FLT_C1", "F1_FLT_C2", "F1_FLT_C3_techrep1", "F1_FLT_C3_techrep2"],
                          ["F1_GC_C1", "F1_GC_C2", "F1_GC_C3"], st, ct)
    assert res["contrast"] == "(Space Flight & Carcass)v(Ground Control & Carcass)"
    assert res["n_flight_in_de"] == 3  # techrep2 isn't in the SampleTable
    mixed = fl.contrast_for(["F1_FLT_C1"], ["F1_GC_C1", "F1_GC_U1"], st, ct)
    assert mixed["contrast"] is None  # control samples span two conditions


def test_detect_namespace():
    ens = fl.detect_namespace(["ENSMUSG00000000001", "ENSMUSG00000000003", "ERCC-00002"])
    assert ens["namespace"] == "ENSEMBL" and ens["n_ercc"] == 1 and not ens["versioned"]
    assert fl.detect_namespace(["1001", "1002"])["namespace"] == "NCBIGene"
    assert fl.detect_namespace(["FakeA", "FakeB"])["namespace"] == "SYMBOL"
    assert fl.detect_namespace(["ENSMUSG00000000001.4"])["versioned"]


def test_processing_info_reads_versions_and_flags_conflicts():
    text = ("Data were processed as described in GL-DPPD-7101-G, using NF_RCP version 2.0.2. "
            "Ensembl release 112, genome assembly GRCm39 (Mus_musculus.GRCm39.112.gtf).")
    info = fl.processing_info(["unrelated methylation text GL-DPPD-7113", text])
    assert info["pipeline_version"] == "GL-DPPD-7101-G"
    assert info["nf_rcp_version"] == "2.0.2"
    assert info["ensembl_release"] == "112" and info["genome_build"] == "GRCm39"
    assert not info["conflict"]
    two = fl.processing_info([text, "GL-DPPD-7101-E with Ensembl release 107, GRCm39"])
    assert two["conflict"] and two["pipeline_version"] == "GL-DPPD-7101-E|GL-DPPD-7101-G"


def test_compare_partial_tables_reports_constant_offset():
    a = pd.DataFrame({"ENSEMBL": ["g1", "g2"], "s1": ["11", "2"], "s2": ["5", "1"]})
    b = pd.DataFrame({"ENSEMBL": ["g1", "g2"], "s1": ["10", "1"], "s2": ["4", "0"]})
    res = fl.compare_partial_tables(a, b, "ENSEMBL", ["s1", "s2"])
    assert res["offset"] == 1.0 and res["max_dev_from_offset"] == 0.0
