"""Job-spec parsing and hashing (CONTRACTS §6.1, §1 run-time IDs).

The parser checks *shape*: required fields, types, ID syntax, no unknown fields at any level.
Whether a well-formed spec makes sense against the contract tables and the site registry
(unknown client, method of another task, outputs that don't cover a method) is the policy check's
``spec#…`` stage. Those cases are in ``test_policy_edges.py``.
"""

import copy
import json
from datetime import UTC, datetime

import pytest
import yaml

from fedkg_spacebio import ids
from fedkg_spacebio.federation.jobspec import (
    METHODS,
    OUTPUT_KINDS,
    TASKS,
    JobSpecError,
    canonical_spec_json,
    job_spec_sha256,
    load_job_spec,
    parse_job_spec,
    spec_to_dict,
    with_job_id,
)
from policy_helpers import SPECS


def raw(name="fe_masked_iss.yaml") -> dict:
    return yaml.safe_load((SPECS / name).read_text())


# ------------------------------------------------------------------ the §6.4 examples parse


@pytest.mark.parametrize("name", ["fe_masked_iss.yaml", "fedavg.yaml"])
def test_contract_examples_parse(name):
    spec = load_job_spec(SPECS / name)
    d = raw(name)
    assert spec.task == d["task"]
    assert spec.clients == d["clients"]
    assert spec.outputs == d["outputs"]
    assert spec.job_id == d["job_id"]


def test_fields_are_read_as_documented():
    spec = load_job_spec(SPECS / "fe_masked_iss.yaml")
    assert spec.stratum == "iss"
    assert spec.methods == ["fe"]
    assert spec.inclusion.rule == "mean10_finite"
    assert spec.inclusion.min_sites_per_gene == "all"
    assert spec.gene_plan.ref == "plans/job-FAKE-0002/gene_plan.csv"
    assert spec.aggregation.strategy == "secure_sum"
    assert spec.aggregation.fixed_point_bits == 40  # §6.4 example (WS-13 choice)
    assert spec.weighting.scheme == "inverse_variance"
    assert spec.provenance.manifest_sha256["SpaceX-8"] == "FAKE-sha256-0006"
    assert spec.provenance.inclusion_flags_sha256["SpaceX-16-ISST"] == "FAKE-sha256-0015"
    assert spec.policy_decision == "trace:FAKE-0001/dec/3"


def test_contract_tables_match_section_5_2():
    assert set(TASKS) == {"de_meta", "clock_plane", "fedavg_classifier"}
    assert set(OUTPUT_KINDS) == {"site_metadata", "masked_contribution", "per_site_summary",
                                 "model_update", "per_sample"}
    assert set(METHODS) == {"fe", "fe_het", "reml_hksj", "v2", "v3", "clock_plane", "fedavg"}
    assert METHODS["fe"].task == "de_meta"
    assert {"masked_contribution", "per_site_summary"} <= set(METHODS["fe"].outputs)
    # fe_het takes masked contributions since WS-13 (D-039): Q and I² need only the totals
    # Σw·y², Σw·y and Σw, which the secure sum delivers without any site's estimate.
    assert {"masked_contribution", "per_site_summary"} <= set(METHODS["fe_het"].outputs)
    for m in ("reml_hksj", "v2", "v3", "clock_plane"):
        assert "masked_contribution" not in METHODS[m].outputs
        assert "per_site_summary" in METHODS[m].outputs
    assert "model_update" in METHODS["fedavg"].outputs
    # §5.2: per_sample covers every method's need (the project policy refuses it instead)
    assert all("per_sample" in m.outputs for m in METHODS.values())


# ------------------------------------------------------------------ malformed specs raise


def _broken(edit):
    d = raw()
    edit(d)
    return d


BROKEN = {
    "unknown top-level field": lambda d: d.update(priority="high"),
    "unknown nested field": lambda d: d["aggregation"].update(noise=0.1),
    "unknown provenance field": lambda d: d["provenance"].update(operator="x"),
    "wrong schema": lambda d: d.update(schema="job-spec/2"),
    "missing task": lambda d: d.pop("task"),
    "unknown task": lambda d: d.update(task="gwas"),
    "bad stratum": lambda d: d.update(stratum="mars"),
    "bad job id": lambda d: d.update(job_id="job-123"),
    "bad decision id": lambda d: d.update(policy_decision="decision 3"),
    "client with a space": lambda d: d.update(clients=["SpaceX 4", "SpaceX-8"]),
    "duplicate client": lambda d: d.update(clients=["SpaceX-4", "SpaceX-4"]),
    "empty clients": lambda d: d.update(clients=[]),
    "empty methods": lambda d: d.update(methods=[]),
    "methods not a list": lambda d: d.update(methods="fe"),
    "bad counts variant": lambda d: d.update(counts_variant="raw"),
    "bad inclusion rule": lambda d: d["inclusion"].update(rule="mean5"),
    "min sites 0": lambda d: d["inclusion"].update(min_sites_per_gene=0),
    "min sites word": lambda d: d["inclusion"].update(min_sites_per_gene="most"),
    "bad plan sha": lambda d: d["gene_plan"].update(sha256="abc"),
    "plan without ref": lambda d: d["gene_plan"].pop("ref"),
    "unknown strategy": lambda d: d["aggregation"].update(strategy="trusted_party"),
    "secure_sum without bits": lambda d: d["aggregation"].pop("modulus_bits"),
    "modulus above 64 bits": lambda d: d["aggregation"].update(modulus_bits=65),
    "plain with bits": lambda d: d.update(aggregation={"strategy": "plain",
                                                       "fixed_point_bits": 32}),
    "unknown weighting scheme": lambda d: d["weighting"].update(scheme="by_vibes"),
    "de_meta without weighting": lambda d: d.pop("weighting"),
    "de_meta rounds 2": lambda d: d.update(rounds=2),
    "rounds 0": lambda d: d.update(rounds=0),
    "split on de_meta": lambda d: d.update(split={"scheme": "loso", "folds_ref": "x",
                                                  "sha256": "FAKE-sha256-0001"}),
    "genes_exclude bad decision": lambda d: d.update(genes_exclude={
        "decision_id": "nope", "ref": "x.csv", "sha256": "FAKE-sha256-0001"}),
    "outputs not a map": lambda d: d.update(outputs=["masked_contribution"]),
    "question not a string": lambda d: d.update(question=3),
}


@pytest.mark.parametrize("case", sorted(BROKEN))
def test_malformed_spec_raises(case):
    with pytest.raises(JobSpecError):
        parse_job_spec(_broken(BROKEN[case]))


def test_error_lists_every_problem():
    d = raw()
    d.update(stratum="mars", counts_variant="raw", priority="high")
    with pytest.raises(JobSpecError) as exc:
        parse_job_spec(d)
    text = str(exc.value)
    assert "stratum" in text and "counts_variant" in text and "priority" in text
    assert len(exc.value.problems) >= 3


def test_custom_weighting_needs_weights():
    d = raw("fedavg.yaml")
    d["weighting"] = {"scheme": "custom", "reason": "site 3 is noisier"}
    with pytest.raises(JobSpecError):
        parse_job_spec(d)


def test_fedavg_without_weighting_parses():
    """§6.1: absent for fedavg_classifier only, so pol:project:1#fedavg-weighting can deny it."""
    d = raw("fedavg.yaml")
    d.pop("weighting")
    assert parse_job_spec(d).weighting is None


def test_fedavg_needs_a_split():
    d = raw("fedavg.yaml")
    d["split"] = None
    with pytest.raises(JobSpecError):
        parse_job_spec(d)


def test_unknown_method_and_output_names_parse():
    """Unknown method keys and output kinds are well-formed strings. The policy check denies
    them by name (spec#methods-known, spec#outputs-cover) rather than the parser raising."""
    d = raw()
    d["methods"] = ["fe", "fe_magic"]
    d["outputs"]["SpaceX-4"] = "raw_counts"
    spec = parse_job_spec(d)
    assert spec.methods == ["fe", "fe_magic"]


def test_null_job_id_and_mock_ids_parse():
    d = raw()
    d["job_id"] = None
    assert parse_job_spec(d).job_id is None
    d["job_id"] = "job:20261116T142301Z-a1b2c3"
    assert parse_job_spec(d).job_id == "job:20261116T142301Z-a1b2c3"
    d["policy_decision"] = "dec:20261116T142301Z-a1b2c3-2"
    assert parse_job_spec(d).policy_decision == "dec:20261116T142301Z-a1b2c3-2"


def test_contact_windows_shape():
    d = raw()
    d["contact_windows"] = {"deadline": "2026-11-20T00:00:00Z", "clients": {
        "SpaceX-4": [{"start": "2026-11-19T10:00:00Z", "end": "2026-11-19T10:20:00Z"}]}}
    assert parse_job_spec(d).contact_windows is not None
    d["contact_windows"] = {"SpaceX-4": []}
    with pytest.raises(JobSpecError):
        parse_job_spec(d)


# ------------------------------------------------------------------ the spec hash


def test_hash_ignores_policy_decision():
    d = raw()
    h = job_spec_sha256(parse_job_spec(d))
    d["policy_decision"] = None
    assert job_spec_sha256(parse_job_spec(d)) == h
    d["policy_decision"] = "dec:FAKE-0002-7"
    assert job_spec_sha256(parse_job_spec(d)) == h


def test_hash_covers_job_id_and_content():
    d = raw()
    h = job_spec_sha256(parse_job_spec(d))
    for edit in (lambda x: x.update(job_id="job:FAKE-0009"),
                 lambda x: x.update(question="something else"),
                 lambda x: x["outputs"].update({"SpaceX-4": "per_site_summary"})):
        e = copy.deepcopy(d)
        edit(e)
        assert job_spec_sha256(parse_job_spec(e)) != h


def test_hash_is_independent_of_key_order_and_format(tmp_path):
    d = raw()
    h = job_spec_sha256(parse_job_spec(d))
    reordered = dict(reversed(list(d.items())))
    assert job_spec_sha256(parse_job_spec(reordered)) == h
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec_to_dict(parse_job_spec(d)), default=str))
    assert job_spec_sha256(load_job_spec(p)) == h


def test_canonical_json_by_hand():
    """Sorted keys, compact separators, dates as ISO strings, no policy_decision."""
    spec = load_job_spec(SPECS / "fe_masked_iss.yaml")
    text = canonical_spec_json(spec).decode("utf-8")
    assert text.startswith('{"aggregation":{')
    assert '"census_access_date":"2026-11-12"' in text
    assert "policy_decision" not in text
    assert ", " not in text and ": " not in text.replace("Which liver genes change with "
                                                         "spaceflight across ISS missions?", "")
    obj = json.loads(text)
    assert list(obj) == sorted(obj)


def test_spec_to_dict_round_trips():
    spec = load_job_spec(SPECS / "fedavg.yaml")
    again = parse_job_spec(json.loads(json.dumps(spec_to_dict(spec), default=str)))
    assert job_spec_sha256(again) == job_spec_sha256(spec)
    assert spec_to_dict(spec)["schema"] == "job-spec/1"


def test_with_job_id_changes_the_hash_and_nothing_else():
    d = raw()
    d["job_id"] = None
    spec = parse_job_spec(d)
    minted = with_job_id(spec, "job:FAKE-0042")
    assert minted.job_id == "job:FAKE-0042" and spec.job_id is None
    assert job_spec_sha256(minted) != job_spec_sha256(spec)
    a, b = spec_to_dict(minted), spec_to_dict(spec)
    a.pop("job_id"), b.pop("job_id")
    assert a == b


# ------------------------------------------------------------------ run-time IDs (§1)


def test_mint_job_id_form():
    jid = ids.mint_job_id(now=datetime(2026, 11, 16, 14, 23, 1, tzinfo=UTC), token="a1b2c3")
    assert jid == "job:20261116T142301Z-a1b2c3"
    assert ids.JOB_ID_RE.fullmatch(ids.mint_job_id())
    assert ids.job_suffix(jid) == "20261116T142301Z-a1b2c3"
    assert ids.job_suffix("job:FAKE-0002") == "FAKE-0002"


def test_decision_and_release_ids():
    assert ids.decision_id_for("job:FAKE-0002", 1) == "dec:FAKE-0002-1"
    assert ids.release_id_for("job:20261116T142301Z-a1b2c3", 2) == \
        "rel:20261116T142301Z-a1b2c3-2"
    for good in ("dec:FAKE-0002-1", "dec:20261116T142301Z-a1b2c3-12", "trace:FAKE-0001/dec/2",
                 "trace:20261120T150000Z-0f3c9a12/dec/3"):
        assert ids.DECISION_ID_RE.fullmatch(good), good
    for bad in ("dec:FAKE-0002", "decision-1", "trace:FAKE-0001/call/2"):
        assert not ids.DECISION_ID_RE.fullmatch(bad), bad


def test_sha256_forms():
    assert ids.is_sha256("0" * 64)
    assert ids.is_sha256("FAKE-sha256-0003")
    assert not ids.is_sha256("FAKE-sha256-0003", allow_fake=False)
    assert not ids.is_sha256("A" * 64)  # lowercase only
    assert not ids.is_sha256("abc")


def test_modulus_bits_above_64_is_a_parse_error():
    """§6.3: the secure sum runs modulo 2^64, so modulus_bits > 64 can't be run (WS-13)."""
    d = raw()
    d["aggregation"] = {"strategy": "secure_sum", "fixed_point_bits": 40, "modulus_bits": 65}
    with pytest.raises(JobSpecError, match="modulus_bits must be at most 64"):
        parse_job_spec(d)
    d["aggregation"]["modulus_bits"] = 64
    assert parse_job_spec(d).aggregation.modulus_bits == 64
