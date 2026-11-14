"""Policy-profile plumbing for the phase 3a job (D-036; CONTRACTS §3).

``jobs/phase3a_meta/job.py --profile {open,restricted}`` builds manifests whose ``policy_id``
matches the profile's policy files. The open profile keeps today's paths. Any other profile adds
a ``profiles/<profile>/`` segment, so manifests and results of two profiles never overwrite each
other.
"""

import importlib.util
from pathlib import Path

import pytest

JOBS = Path(__file__).resolve().parents[1] / "jobs" / "phase3a_meta"


@pytest.fixture(scope="module")
def config():
    spec = importlib.util.spec_from_file_location("phase3a_config_profile", JOBS / "config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_open_profile_keeps_todays_paths(config, tmp_path):
    assert config.manifest_dir("standard", tmp_path) == tmp_path / "phase3a/manifests/standard"
    assert config.manifest_dir("standard", tmp_path, profile="open") == \
        tmp_path / "phase3a/manifests/standard"
    assert config.results_dir("iss", "standard", "flare", tmp_path, profile="open") == \
        tmp_path / "results/phase3a/iss/standard/flare"


def test_other_profiles_get_their_own_paths(config, tmp_path):
    assert config.manifest_dir("rrnarm", tmp_path, profile="restricted") == \
        tmp_path / "phase3a/manifests/profiles/restricted/rrnarm"
    assert config.results_dir("iss", "standard", "flare", tmp_path, profile="restricted") == \
        tmp_path / "results/phase3a/profiles/restricted/iss/standard/flare"


def test_unknown_profile_is_refused(config, tmp_path):
    with pytest.raises(ValueError):
        config.manifest_dir("standard", tmp_path, profile="../open")
    with pytest.raises(ValueError):
        config.policy_ids("nonexistent")


def test_policy_ids_come_from_the_policy_files(config):
    assert config.PROFILES == ("open", "restricted")
    open_ids = config.policy_ids("open")
    assert open_ids["SpaceX-16"] == "pol:SpaceX-16:1"
    assert config.policy_ids("restricted")["SpaceX-16"] == "pol:SpaceX-16:2"
    assert {k: v for k, v in config.policy_ids("restricted").items() if k != "SpaceX-16"} == \
        {k: v for k, v in open_ids.items() if k != "SpaceX-16"}
