"""Manifest builder (``fedkg_spacebio.sites.manifest``): row selection, techrep handling,
disjointness. Census-like frames are built here with fake IDs."""

import json

import pandas as pd
import pytest

from fedkg_spacebio.sites.manifest import (
    CLIENTS,
    ClientSpec,
    RowSpec,
    build_manifest,
    check_disjoint,
    clients_in,
    find_row,
)

COUNTS = "GLDS-FAKE-1_rna_seq_RSEM_Unnormalized_Counts_GLbulkRNAseq.csv"


def census_row(flight_factors, flight_animals, control_animals, location_aware=True):
    return {"osd_id": "OSD-FAKE-1", "site": "FAKE-MISSION-1", "control_type": "Ground Control",
            "location_aware_match": location_aware, "strict_match": location_aware,
            "factors_flight": json.dumps(["Space Flight", *flight_factors]),
            "factors_control": json.dumps(["Ground Control", *flight_factors]),
            "flight_animals": json.dumps(flight_animals),
            "control_animals": json.dumps(control_animals),
            "comparison_id": "OSD-FAKE-1-00000000000000000000000000000000",
            "genelab_contrast": "(Space Flight)v(Ground Control)",
            "euthanasia_location_flight": "ISS", "euthanasia_location_flight_source": "fake",
            "de_file": "fake_de.csv", "counts_file": COUNTS, "counts_file_rrnarm": COUNTS}


@pytest.fixture
def setup(tmp_path):
    census = pd.DataFrame([
        census_row(["Carcass"], ["F1", "F2"], ["G1", "G2"]),
        census_row(["Upon euthanasia"], ["F3", "F4"], ["G3", "G4"]),
        census_row(["Other"], ["F5"], ["G5"], location_aware=False),
    ])
    samples = pd.DataFrame([
        {"osd_id": "OSD-FAKE-1", "sample_name": n, "source_key": k, "spaceflight_label": lab}
        for n, k, lab in [
            ("S_F1", "F1", "Space Flight"), ("S_F2_techrep1", "F2", "Space Flight"),
            ("S_F2_techrep2", "F2", "Space Flight"), ("S_G1", "G1", "Ground Control"),
            ("S_G2", "G2", "Ground Control"), ("S_F3", "F3", "Space Flight"),
            ("S_F4", "F4", "Space Flight"), ("S_G3", "G3", "Ground Control"),
            ("S_G4", "G4", "Ground Control"),
        ]])
    (tmp_path / "OSD-FAKE-1").mkdir()
    in_de = ["S_F1", "S_F2_techrep1", "S_G1", "S_G2", "S_F3", "S_F4", "S_G3", "S_G4"]
    pd.DataFrame({"condition": ["x"] * 8}, index=in_de).to_csv(
        tmp_path / "OSD-FAKE-1" / "GLDS-FAKE-1_rna_seq_SampleTable_GLbulkRNAseq.csv")
    return census, samples, tmp_path


def spec(name, factors, stratum="iss"):
    return ClientSpec(name, "FAKE-MISSION-1", stratum, "~ condition",
                      (RowSpec("OSD-FAKE-1", tuple(factors)),))


def test_find_row_needs_exactly_one_location_aware_ground_row(setup):
    census, _, _ = setup
    assert find_row(census, "FAKE-MISSION-1", RowSpec("OSD-FAKE-1", ("Carcass",))) is not None
    with pytest.raises(ValueError, match="found 0"):
        find_row(census, "FAKE-MISSION-1", RowSpec("OSD-FAKE-1", ("Other",)))  # not loc-aware
    with pytest.raises(ValueError, match="found 2"):
        find_row(pd.concat([census, census.iloc[[0]]]), "FAKE-MISSION-1",
                 RowSpec("OSD-FAKE-1", ("Carcass",)))


def test_technical_replicate_resolved_by_sample_table(setup):
    census, samples, root = setup
    m = build_manifest(spec("A", ["Carcass"]), census, samples, root)
    names = [s["sample"] for s in m["sources"][0]["samples"]]
    assert names == ["S_F1", "S_F2_techrep1", "S_G1", "S_G2"]
    assert (m["n_flight"], m["n_ground"]) == (2, 2)


def test_disjointness_check(setup):
    census, samples, root = setup
    a = build_manifest(spec("A", ["Carcass"]), census, samples, root)
    b = build_manifest(spec("B", ["Upon euthanasia"], "earth"), census, samples, root)
    check_disjoint([a, b])  # same file, different animals: fine
    b2 = build_manifest(spec("B", ["Carcass"], "earth"), census, samples, root)
    with pytest.raises(ValueError, match="both A and B"):
        check_disjoint([a, b2])


def test_real_client_table_matches_d020():
    assert clients_in("iss") == ["SpaceX-4", "SpaceX-8", "SpaceX-13-ISST", "SpaceX-16-ISST"]
    assert clients_in("earth") == ["SpaceX-13-LAR", "SpaceX-16-LAR", "SpaceX-21"]
    assert CLIENTS["SpaceX-4"].design == "~ study + condition"
    assert CLIENTS["SpaceX-16-ISST"].design == CLIENTS["SpaceX-16-LAR"].design == \
        "~ age + condition"
    with pytest.raises(ValueError):
        clients_in("mars")
