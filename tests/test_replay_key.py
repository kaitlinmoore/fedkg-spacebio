"""The replay key (``agent/replay.py``): exactly ``REPLAY_DROPS`` are ignored, nothing else."""

from __future__ import annotations

import copy

import pytest
import yaml

from fedkg_spacebio.agent.replay import REPLAY_DROPS, replay_key, replay_payload
from fedkg_spacebio.federation.jobspec import job_spec_sha256, parse_job_spec
from tests_paths import SPECS

BASE = yaml.safe_load((SPECS / "fe_masked_iss.yaml").read_text())


def spec(**edits):
    d = copy.deepcopy(BASE)
    d["genes_exclude"] = {"decision_id": "dec:FAKE-0001-1", "ref": "plans/a/excl.csv",
                          "sha256": "FAKE-sha256-0020"}
    for path, value in edits.items():
        node = d
        *parents, leaf = path.split("__")
        for p in parents:
            node = node[p]
        node[leaf] = value
    return parse_job_spec(d)


def test_drop_list_is_exactly_this():
    assert REPLAY_DROPS == ("job_id", "policy_decision", "gene_plan.ref", "genes_exclude.ref",
                            "genes_exclude.decision_id")


@pytest.mark.parametrize("edit", [
    {"job_id": "job:20261120T150000Z-abcdef"},
    {"policy_decision": "trace:FAKE-0002/dec/4"},
    {"gene_plan__ref": "plans/another-run/gene_plan.csv"},
    {"genes_exclude__ref": "plans/another-run/excl.csv"},
    {"genes_exclude__decision_id": "trace:FAKE-0002/dec/3"},
])
def test_differences_only_in_dropped_fields_match(edit):
    a, b = spec(), spec(**edit)
    if "policy_decision" not in edit:  # the spec hash already ignores policy_decision
        assert job_spec_sha256(a) != job_spec_sha256(b)
    assert replay_key(a) == replay_key(b)


def test_all_dropped_fields_at_once_match():
    b = spec(job_id="job:20261120T150000Z-abcdef", policy_decision="trace:FAKE-0002/dec/4",
             gene_plan__ref="plans/x/gene_plan.csv", genes_exclude__ref="plans/x/e.csv",
             genes_exclude__decision_id="trace:FAKE-0002/dec/3")
    assert replay_key(spec()) == replay_key(b)


@pytest.mark.parametrize("edit", [
    {"question": "Another wording"},
    {"stratum": "earth"},
    {"clients": ["SpaceX-4", "SpaceX-8", "SpaceX-13-ISST"]},
    {"methods": ["fe", "fe_het"]},
    {"counts_variant": "rrnarm"},
    {"inclusion__rule": "finite_p"},
    {"inclusion__min_sites_per_gene": 3},
    {"gene_plan__sha256": "FAKE-sha256-0099"},
    {"genes_exclude__sha256": "FAKE-sha256-0098"},
    {"aggregation__fixed_point_bits": 39},
    {"aggregation__modulus_bits": 63},
    {"provenance__kg_version": "SPOKE-GeneLab v0.3.2"},
    {"provenance__census_access_date": "2026-12-01"},
    {"provenance__annotation_sha256": "FAKE-sha256-0097"},
    {"provenance__manifest_sha256__SpaceX-4": "FAKE-sha256-0096"},
])
def test_any_other_difference_does_not_match(edit):
    assert replay_key(spec()) != replay_key(spec(**edit))


def test_outputs_and_weighting_are_in_the_key():
    a = spec()
    d = replay_payload(a)
    assert "outputs" in d and "weighting" in d and "provenance" in d
    assert "job_id" not in d and "ref" not in d["gene_plan"]
    assert set(d["genes_exclude"]) == {"sha256"}
