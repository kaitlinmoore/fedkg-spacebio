"""Samples → groups → flight-vs-control comparisons → KG join, on synthetic fixtures.

Fixture ``census/samples_query.csv`` (fake IDs):
- OSD-FAKE-1: flight/Carcass (3 animals, one with 2 technical replicates), ground/Carcass (3),
  ground/Upon euthanasia (1), plus a proteomics row that must be ignored.
- OSD-FAKE-2: flight ``~30 {day}, On ISS`` vs ground ``32 {day}, On Earth`` (not strict).
- OSD-FAKE-3: flight only (a site with one arm: no comparison possible).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fedkg_spacebio.census import comparisons as cmp
from fedkg_spacebio.census.build import location_aware_match
from fedkg_spacebio.census.samples import normalize_label, normalize_source, parse_samples

FIX = Path(__file__).parent / "fixtures" / "census"


@pytest.fixture(scope="module")
def samples() -> pd.DataFrame:
    return parse_samples((FIX / "samples_query.csv").read_text())


@pytest.fixture(scope="module")
def comps(samples) -> pd.DataFrame:
    return cmp.build_comparisons(cmp.build_groups(samples))


def _one(df: pd.DataFrame, **eq) -> pd.Series:
    sel = df
    for k, v in eq.items():
        sel = sel[sel[k].map(lambda x, v=v: x == v)]
    assert len(sel) == 1, f"expected one row for {eq}, got {len(sel)}"
    return sel.iloc[0]


def test_only_rnaseq_assays_are_kept(samples):
    assert samples["assay_name"].str.contains("rna-sequencing").all()
    assert len(samples) == 14


def test_arms_and_normalization(samples):
    assert set(samples["arm"]) == {"flight", "ground"}
    assert normalize_source("RR-1_FLT_M1") == normalize_source("rr1 flt m1") == "RR1FLTM1"
    assert normalize_label("~60 {day}") == "~60 day"


def test_n_counts_animals_not_technical_replicates(comps):
    row = _one(comps, osd_id="OSD-FAKE-1", factors_control=["Ground Control", "Carcass"])
    assert row["n_flight_samples"] == 4  # M3 sequenced twice
    assert row["n_flight"] == 3
    assert row["n_control"] == 3
    assert row["strict_match"]


def test_confounded_pair_is_kept_but_not_strict(comps):
    row = _one(comps, osd_id="OSD-FAKE-1", factors_control=["Ground Control", "Upon euthanasia"])
    assert not row["strict_match"]
    assert row["differing_factors"] == ["dissection condition"]


def test_site_with_one_arm_has_no_comparison(comps):
    assert "OSD-FAKE-3" not in set(comps["osd_id"])


def test_duration_and_location_differences_fail_strict(comps):
    row = _one(comps, osd_id="OSD-FAKE-2")
    assert not row["strict_match"]
    assert row["differing_factors"] == ["duration", "euthanasia location"]


def test_location_aware_rule_accepts_only_iss_terminal_differences(comps):
    row = _one(comps, osd_id="OSD-FAKE-2").copy()
    assert location_aware_match(row)  # On ISS vs On Earth, 30 vs 32 days
    row["control_factors"] = {**row["control_factors"], "duration": "45 day"}
    assert not location_aware_match(row)  # 15-day gap is a different design
    confounded = _one(comps, osd_id="OSD-FAKE-1",
                      factors_control=["Ground Control", "Upon euthanasia"])
    assert not location_aware_match(confounded)


def test_label_matching_handles_kg_truncation():
    assert cmp.label_matches("SpaceX-4 (RR1", "SpaceX-4 (RR1)")
    assert cmp.label_matches("SpaceX-4", "SpaceX-4 (RR1)")
    assert not cmp.label_matches("1 day", "0 day")
    assert cmp.match_side(["Space Flight"], ["Space Flight", "Carcass"]) == "partial"
    assert cmp.match_side(["Space Flight", "Carcass"], ["Space Flight", "Carcass"]) == "exact"
    assert cmp.match_side(["Space Flight", "Carcass"], ["Space Flight"]) is None


def test_kg_join_grades(comps):
    kg = json.loads((FIX / "kg_comparisons.json").read_text())
    out, unmatched = cmp.match_kg(comps, kg)
    carcass = _one(out, osd_id="OSD-FAKE-1", factors_control=["Ground Control", "Carcass"])
    # Assay name differs by a stray space in the KG: matched after normalization.
    assert carcass["comparison_id"] == "OSD-FAKE-1-00000000000000000000000000000001"
    assert carcass["kg_match"] == "exact" and carcass["kg_assay_match"] == "normalized"
    # The KG's bare ["Space Flight"] vs ["Ground Control"] fits two OSDR pairs: no ID assigned.
    upon = _one(out, osd_id="OSD-FAKE-1", factors_control=["Ground Control", "Upon euthanasia"])
    assert upon["comparison_id"] is None and upon["kg_match"] == "ambiguous"
    fake2 = _one(out, osd_id="OSD-FAKE-2")
    assert fake2["comparison_id"] == "OSD-FAKE-2-00000000000000000000000000000003"
    reasons = {u["comparison_id"]: u["reason"] for u in unmatched}
    assert reasons["OSD-FAKE-2-00000000000000000000000000000004"] == "no OSDR pair fits the labels"
    assert reasons["OSD-FAKE-1-00000000000000000000000000000002"].startswith("ambiguous")
