"""Site assignment, independent-animal counting and the D-016 gate (synthetic census rows)."""

from __future__ import annotations

import pandas as pd

from fedkg_spacebio.census import gate as gt
from fedkg_spacebio.census.samples import normalize_source


def _row(osd, n_f, n_c, strict=True, control="Ground Control", prefix=None, loc="ISS", cid=None):
    # Census rows hold *normalized* source keys, as build_comparisons produces them.
    p = prefix or osd
    return {"osd_id": osd, "comparison_id": cid or f"{osd}-{'0' * 32}", "strict_match": strict,
            "control_type": control, "euthanasia_location_flight": loc,
            "flight_animals": [normalize_source(f"{p}F{i}") for i in range(n_f)],
            "control_animals": [normalize_source(f"{p}C{i}") for i in range(n_c)]}


SITES = {"OSD-FAKE-1": "FAKE-MISSION-A", "OSD-FAKE-2": "FAKE-MISSION-B",
         "OSD-FAKE-3": "FAKE-MISSION-C", "OSD-FAKE-4": None}


def test_assign_sites_leaves_multi_mission_studies_unassigned():
    s = gt.assign_sites({"OSD-FAKE-1": ["M-A"], "OSD-FAKE-4": ["M-A", "M-B"], "OSD-FAKE-5": []})
    assert s == {"OSD-FAKE-1": "M-A", "OSD-FAKE-4": None, "OSD-FAKE-5": None}


def test_shared_animals_reduce_n_only_across_assigned_sites():
    census = pd.DataFrame([_row("OSD-FAKE-1", 4, 4), _row("OSD-FAKE-2", 4, 4),
                           _row("OSD-FAKE-4", 4, 4)])
    # OSD-FAKE-2's F0 is OSD-FAKE-1's F0 (different sites); OSD-FAKE-4 (unassigned) shares F1.
    pairs = pd.DataFrame([
        {"study_a": "OSD-FAKE-1", "study_b": "OSD-FAKE-2", "source_a": "OSD-FAKE-1F0",
         "source_b": "OSD-FAKE-2F0"},
        {"study_a": "OSD-FAKE-1", "study_b": "OSD-FAKE-4", "source_a": "OSD-FAKE-1F1",
         "source_b": "OSD-FAKE-4F1"},
    ])
    out = gt.independent_counts(census, pairs, SITES).set_index("osd_id")
    assert out.loc["OSD-FAKE-1", "n_flight_indep"] == 3  # F0 shared with another site
    assert out.loc["OSD-FAKE-2", "n_flight_indep"] == 3
    assert out.loc["OSD-FAKE-1", "n_control_indep"] == 4
    assert out.loc["OSD-FAKE-4", "n_flight_indep"] == 4  # unassigned: its own count unchanged


def _gate_census():
    rows = [_row("OSD-FAKE-1", 3, 3), _row("OSD-FAKE-2", 5, 2),
            _row("OSD-FAKE-3", 6, 6, strict=False),
            _row("OSD-FAKE-3", 6, 6, control="Vivarium Control")]
    census = pd.DataFrame(rows)
    return gt.independent_counts(census, pd.DataFrame(), SITES)


def test_gate_pass_and_fail_reasons():
    census = _gate_census()
    counts = {k: True for k in SITES}
    table, verdict = gt.assess_gate(census, SITES, counts, min_animals=3, min_sites=3)
    t = table.set_index("site")
    assert t.loc["FAKE-MISSION-A", "passes"]
    assert not t.loc["FAKE-MISSION-B", "passes"]  # 5 vs 2: the smaller arm decides
    assert "2 independent animals" in t.loc["FAKE-MISSION-B", "reason"]
    # FAKE-3 has only a non-strict ground comparison and a strict *vivarium* one: neither counts.
    assert t.loc["FAKE-MISSION-C", "reason"] == "no strict flight-vs-ground comparison"
    assert verdict["n_sites_passing"] == 1 and not verdict["go"]


def test_gate_requires_published_counts():
    census = _gate_census()
    counts = {k: True for k in SITES} | {"OSD-FAKE-1": False}
    table, _ = gt.assess_gate(census, SITES, counts)
    reason = table.set_index("site").loc["FAKE-MISSION-A", "reason"]
    assert reason == "no published unnormalized counts"


def test_go_when_enough_sites_pass():
    census = gt.independent_counts(pd.DataFrame([_row(o, 3, 3) for o in
                                                 ("OSD-FAKE-1", "OSD-FAKE-2", "OSD-FAKE-3")]),
                                   pd.DataFrame(), SITES)
    _, verdict = gt.assess_gate(census, SITES, {k: True for k in SITES})
    assert verdict["go"] and verdict["n_sites_passing"] == 3
    sens = gt.sensitivity(census, SITES, {k: True for k in SITES}, thresholds=(3, 4))
    assert sens["go"].tolist() == [True, False]


def test_strata_split_sites_by_euthanasia_location():
    rows = [_row("OSD-FAKE-1", 3, 3, loc="ISS"), _row("OSD-FAKE-2", 3, 3, loc="Earth"),
            _row("OSD-FAKE-3", 3, 3, loc="Earth")]
    census = gt.independent_counts(pd.DataFrame(rows), pd.DataFrame(), SITES)
    st = gt.strata(census, SITES, {k: True for k in SITES}, min_sites=2)
    passing = st[st["passes"]].groupby("stratum")["site"].apply(sorted).to_dict()
    assert passing == {"Earth": ["FAKE-MISSION-B", "FAKE-MISSION-C"], "ISS": ["FAKE-MISSION-A"]}
    go = st.groupby("stratum")["stratum_go"].first().to_dict()
    assert go == {"Earth": True, "ISS": False}
