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
    check_policy_id,
    clients_in,
    find_row,
    write_manifests,
)

COUNTS = "GLDS-FAKE-1_rna_seq_RSEM_Unnormalized_Counts_GLbulkRNAseq.csv"


def census_row(flight_factors, flight_animals, control_animals, location_aware=True,
               tissue="UBERON:0002107"):
    return {"osd_id": "OSD-FAKE-1", "site": "FAKE-MISSION-1", "control_type": "Ground Control",
            "location_aware_match": location_aware, "strict_match": location_aware,
            "factors_flight": json.dumps(["Space Flight", *flight_factors]),
            "factors_control": json.dumps(["Ground Control", *flight_factors]),
            "flight_animals": json.dumps(flight_animals),
            "control_animals": json.dumps(control_animals),
            "comparison_id": "OSD-FAKE-1-00000000000000000000000000000000",
            "genelab_contrast": "(Space Flight)v(Ground Control)",
            "euthanasia_location_flight": "ISS", "euthanasia_location_flight_source": "fake",
            "de_file": "fake_de.csv", "counts_file": COUNTS, "counts_file_rrnarm": COUNTS,
            "tissue_uberon": tissue}


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


# ---------------------------------------------------------------- policy_id (CONTRACTS §3)


def test_manifest_points_at_its_site_policy_version_1_by_default(setup):
    census, samples, root = setup
    m = build_manifest(spec("A", ["Carcass"]), census, samples, root)
    assert m["policy_id"] == "pol:FAKE-MISSION-1:1"
    m2 = build_manifest(spec("A", ["Carcass"]), census, samples, root,
                        policy_id="pol:FAKE-MISSION-1:2")
    assert m2["policy_id"] == "pol:FAKE-MISSION-1:2"


@pytest.mark.parametrize("bad", ["pol:FAKE-MISSION-2:1",   # another site's policy
                                 "pol:FAKE-MISSION-1",     # no version
                                 "pol:FAKE-MISSION-1:0",   # versions start at 1
                                 "pol:project:1",          # the project policy isn't a site's
                                 "FAKE-MISSION-1:1", 3])
def test_policy_id_must_name_this_sites_policy(setup, bad):
    census, samples, root = setup
    with pytest.raises(ValueError, match="policy_id"):
        build_manifest(spec("A", ["Carcass"]), census, samples, root, policy_id=bad)
    with pytest.raises(ValueError):
        check_policy_id(bad, "FAKE-MISSION-1")


def test_write_manifests_applies_policy_ids_per_site(setup, tmp_path):
    census, samples, root = setup
    work = tmp_path / "work"
    (work / "census").mkdir(parents=True)
    census.to_csv(work / "census" / "liver_census.csv", index=False)
    samples.to_csv(work / "census" / "liver_samples.csv", index=False)
    (work / "data" / "osdr").mkdir(parents=True)
    (root / "OSD-FAKE-1").rename(work / "data" / "osdr" / "OSD-FAKE-1")
    clients = {"A": spec("A", ["Carcass"]), "B": spec("B", ["Upon euthanasia"], "earth")}
    paths = write_manifests(out_dir=tmp_path / "m", workdir=work, clients=clients,
                            policy_ids={"FAKE-MISSION-1": "pol:FAKE-MISSION-1:2"})
    # One site, two clients: both point at the same site policy.
    assert {json.loads(p.read_text())["policy_id"] for p in paths.values()} == \
        {"pol:FAKE-MISSION-1:2"}
    with pytest.raises(ValueError, match="no client"):
        write_manifests(out_dir=tmp_path / "m", workdir=work, clients=clients,
                        policy_ids={"FAKE-MISSION-9": "pol:FAKE-MISSION-9:1"})


# ---------------------------------------------------------------- tissue_uberon (CONTRACTS §3)


def test_manifest_carries_the_census_tissue_term(setup):
    census, samples, root = setup
    m = build_manifest(spec("A", ["Carcass"]), census, samples, root)
    assert m["tissue_uberon"] == "UBERON:0002107"
    census.loc[0, "tissue_uberon"] = "UBERON:0006727"   # left lateral lobe of liver
    assert build_manifest(spec("A", ["Carcass"]), census, samples, root)["tissue_uberon"] == \
        "UBERON:0006727"


@pytest.mark.parametrize("bad", [None, float("nan"), "liver", "UBERON:12"])
def test_missing_or_malformed_tissue_refused(setup, bad):
    census, samples, root = setup
    census["tissue_uberon"] = census["tissue_uberon"].astype(object)
    census.loc[0, "tissue_uberon"] = bad
    with pytest.raises(ValueError, match="tissue_uberon"):
        build_manifest(spec("A", ["Carcass"]), census, samples, root)


def test_client_with_two_tissue_terms_refused(setup):
    census, samples, root = setup
    census.loc[1, "tissue_uberon"] = "UBERON:0006727"
    two_rows = ClientSpec("A", "FAKE-MISSION-1", "iss", "~ condition",
                          (RowSpec("OSD-FAKE-1", ("Carcass",)),
                           RowSpec("OSD-FAKE-1", ("Upon euthanasia",))))
    with pytest.raises(ValueError, match="one tissue"):
        build_manifest(two_rows, census, samples, root)
