"""Policy files: the site-policy/1 loader and the demo profiles (CONTRACTS §5.1, §5.5).

The demo's own policy files (``policies/<profile>/``) are what these tests load, so a rule
changed there shows up here.
"""

import hashlib

import pytest
from fedkg_spacebio.policy.registry import (
    PolicyMismatchError,
    check_against_profile,
    load_registry,
)
from fedkg_spacebio.policy.schema import PolicyError, load_policy, load_profile

from policy_helpers import FIX, POLICIES, make_env

SITES = ["SpaceX-4", "SpaceX-8", "SpaceX-13", "SpaceX-16", "SpaceX-21"]
TESTBED_LINE = "INVENTED FOR THE TESTBED"


def all_policy_files():
    return sorted([*POLICIES.glob("*/*.yaml"), *(FIX / "synthetic").glob("*.yaml")])


@pytest.mark.parametrize("path", all_policy_files(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_every_policy_file_says_it_is_invented(path):
    assert TESTBED_LINE in path.read_text(encoding="utf-8").splitlines()[0]


def test_readme_says_invented():
    assert "invented for the testbed" in (POLICIES / "README.md").read_text().lower()


@pytest.mark.parametrize("profile", ["open", "restricted"])
def test_profiles_load(profile):
    prof = load_profile(POLICIES / profile)
    assert prof.name == profile
    assert prof.project.policy.policy_id == "pol:project:1"
    assert prof.project.policy.site is None
    assert sorted(prof.sites) == sorted(SITES)
    for site, lp in prof.sites.items():
        assert lp.policy.site == site
        assert lp.policy.default == "deny"
        assert lp.sha256 == hashlib.sha256(lp.path.read_bytes()).hexdigest()


def test_profile_policy_ids():
    assert load_profile(POLICIES / "open").site_policy_ids() == {s: f"pol:{s}:1" for s in SITES}
    restricted = load_profile(POLICIES / "restricted").site_policy_ids()
    assert restricted == {**{s: f"pol:{s}:1" for s in SITES}, "SpaceX-16": "pol:SpaceX-16:2"}


def test_restricted_differs_from_open_only_at_spacex16():
    for f in (POLICIES / "open").glob("*.yaml"):
        same = f.read_bytes() == (POLICIES / "restricted" / f.name).read_bytes()
        assert same == (f.name != "SpaceX-16.yaml"), f.name


def test_synthetic_project_policy_is_the_demo_one():
    assert (FIX / "synthetic" / "project.yaml").read_bytes() == \
        (POLICIES / "open" / "project.yaml").read_bytes()


def test_rules_match_the_contract_examples():
    """CONTRACTS §5.5, field by field, for the files the demo uses."""
    project = load_profile(POLICIES / "open").project.policy
    assert [(r.key, r.kind) for r in project.rules] == [("no-per-sample", "release"),
                                                      ("fedavg-weighting", "weighting")]
    assert project.rules[0].outputs == ["site_metadata", "masked_contribution",
                                        "per_site_summary", "model_update"]
    assert project.rule_id("no-per-sample") == "pol:project:1#no-per-sample"

    p16 = load_profile(POLICIES / "restricted").sites["SpaceX-16"].policy
    assert [(r.key, r.kind) for r in p16.rules] == [
        ("analyses", "analyses"), ("release", "release"), ("min-k", "min_aggregation"),
        ("diff", "differencing"), ("dp", "dp")]
    rules = {r.key: r for r in p16.rules}
    assert rules["analyses"].tasks == ["de_meta"] and rules["analyses"].methods == "*"
    assert rules["release"].outputs == ["site_metadata", "masked_contribution"]
    assert (rules["min-k"].min_sites, rules["min-k"].min_n_per_arm) == (3, 5)
    assert rules["diff"].min_sites == 3
    assert rules["dp"].required is False
    assert p16.rule_id("diff") == "pol:SpaceX-16:2#diff"

    p4 = load_profile(POLICIES / "open").sites["SpaceX-4"].policy
    assert [r.key for r in p4.rules] == ["analyses", "release", "min-k", "dp"]
    assert {r.key: r for r in p4.rules}["analyses"].tasks == ["de_meta", "clock_plane"]


BAD = sorted(p.name for p in (FIX / "bad").glob("*.yaml"))


@pytest.mark.parametrize("name", BAD)
def test_malformed_policy_raises(name):
    with pytest.raises(PolicyError):
        load_policy(FIX / "bad" / name)


def test_bad_fixtures_present():
    assert {"wrong_schema.yaml", "default_allow.yaml", "duplicate_key.yaml", "unknown_kind.yaml",
            "unknown_output.yaml", "extra_field.yaml", "site_mismatch.yaml",
            "weighting_in_site.yaml", "diff_min_sites_1.yaml", "no_reason.yaml"} <= set(BAD)


def test_profile_file_name_must_match_site(tmp_path):
    d = tmp_path / "p"
    d.mkdir()
    for f in (POLICIES / "open").glob("*.yaml"):
        (d / f.name).write_bytes(f.read_bytes())
    (d / "SpaceX-4.yaml").rename(d / "SpaceX-99.yaml")
    with pytest.raises(PolicyError):
        load_profile(d)


def test_profile_needs_exactly_one_project_policy(tmp_path):
    d = tmp_path / "p"
    d.mkdir()
    for f in (POLICIES / "open").glob("SpaceX-*.yaml"):
        (d / f.name).write_bytes(f.read_bytes())
    with pytest.raises(PolicyError):
        load_profile(d)


# ------------------------------------------------------------------ manifests point at policies


def test_registry_reads_manifest_fields(tmp_path):
    env = make_env(tmp_path, "restricted")
    rec = env.registry["SpaceX-16-ISST"]
    assert (rec.site, rec.stratum, rec.policy_id) == ("SpaceX-16", "iss", "pol:SpaceX-16:2")
    assert (rec.n_flight, rec.n_ground, rec.n_k) == (20, 20, None)
    path = env.manifest_dir / "SpaceX-16-ISST.json"
    assert rec.manifest_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert list(env.registry) == ["SpaceX-4", "SpaceX-8", "SpaceX-13-ISST", "SpaceX-16-ISST",
                                  "SpaceX-13-LAR", "SpaceX-16-LAR", "SpaceX-21"]


def test_registry_client_must_equal_file_name(tmp_path):
    env = make_env(tmp_path, "open")
    (env.manifest_dir / "SpaceX-8.json").rename(env.manifest_dir / "SpaceX-9.json")
    with pytest.raises(ValueError):
        load_registry(env.manifest_dir)


def test_manifest_policy_matching_profile_passes(tmp_path):
    env = make_env(tmp_path, "restricted")
    check_against_profile(env.registry, env.profile)  # no error


def test_manifest_policy_mismatch_is_an_error(tmp_path):
    """An open-profile manifest checked under the restricted profile: error, not a silent pick."""
    env = make_env(tmp_path, "open")
    with pytest.raises(PolicyMismatchError, match="SpaceX-16"):
        check_against_profile(env.registry, load_profile(POLICIES / "restricted"))


def test_manifest_naming_a_site_without_policy_is_an_error(tmp_path):
    env = make_env(tmp_path, "open", clients_file="clients_fedavg.json",
                   policies_dir=FIX / "synthetic")
    with pytest.raises(PolicyMismatchError):
        check_against_profile(env.registry, load_profile(POLICIES / "open"))
