"""The inspection CLIs: check a spec against a profile; show, verify or reset a release log.

``python -m fedkg_spacebio.policy.check`` exits 0 on allow, 1 on deny, 2 when the spec can't be
evaluated (malformed, or a manifest/profile mismatch). No job runs.
"""

import json

import yaml
from fedkg_spacebio.federation.jobspec import spec_to_dict

from fedkg_spacebio.policy import check as check_cli
from fedkg_spacebio.policy import log as log_cli
from policy_helpers import FULL_SET, G1, ISS, POLICIES, check, make_env, make_spec


def write_spec(tmp_path, spec, name="spec.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(json.loads(json.dumps(spec_to_dict(spec), default=str)),
                                sort_keys=False))
    return p


def args(env, spec_path, profile="restricted", *extra):
    return [str(spec_path), "--profile", profile, "--policies", str(POLICIES / profile),
            "--manifests", str(env.manifest_dir), "--root", str(env.root), *extra]


def test_check_prints_the_decision_and_exits_by_outcome(tmp_path, capsys):
    env = make_env(tmp_path, "restricted")
    deny = write_spec(tmp_path, make_spec(env, clients=ISS, plan={G1: ISS}, methods=FULL_SET,
                                          outputs="per_site_summary"), "deny.yaml")
    assert check_cli.main(args(env, deny)) == 1
    record = json.loads(capsys.readouterr().out)
    assert record["outcome"] == "deny"
    assert record["deciding_rule"] == "pol:SpaceX-16:2#release"

    allow = write_spec(tmp_path, make_spec(env, clients=ISS, plan={G1: ISS}), "allow.yaml")
    assert check_cli.main(args(env, allow)) == 0
    assert json.loads(capsys.readouterr().out)["outcome"] == "allow"


def test_check_writes_the_record_and_the_minted_spec(tmp_path, capsys):
    env = make_env(tmp_path, "restricted")
    p = write_spec(tmp_path, make_spec(env, clients=ISS, plan={G1: ISS}))
    check_cli.main(args(env, p))
    record = json.loads(capsys.readouterr().out)
    stem = record["decision_id"].replace(":", "_").replace("/", "_")
    assert (env.root / "decisions" / f"{stem}.json").exists()
    suffix = record["job_id"].split(":", 1)[1]
    minted = yaml.safe_load((env.root / "specs" / f"{suffix}.yaml").read_text())
    assert minted["job_id"] == record["job_id"]


def test_dry_run_writes_nothing(tmp_path, capsys):
    env = make_env(tmp_path, "restricted")
    p = write_spec(tmp_path, make_spec(env, clients=ISS, plan={G1: ISS}))
    before = sorted(x.relative_to(env.root) for x in env.root.rglob("*"))
    assert check_cli.main(args(env, p, "restricted", "--dry-run")) == 0
    assert json.loads(capsys.readouterr().out)["outcome"] == "allow"
    assert sorted(x.relative_to(env.root) for x in env.root.rglob("*")) == before


def test_malformed_spec_exits_2(tmp_path, capsys):
    env = make_env(tmp_path, "restricted")
    p = tmp_path / "bad.yaml"
    p.write_text("schema: job-spec/1\ntask: gwas\n")
    assert check_cli.main(args(env, p)) == 2
    assert "task" in capsys.readouterr().err


def test_profile_mismatch_exits_2(tmp_path, capsys):
    env = make_env(tmp_path, "open")  # open-profile manifests
    p = write_spec(tmp_path, make_spec(env, clients=ISS, plan={G1: ISS}))
    assert check_cli.main(args(env, p, "restricted")) == 2
    assert "SpaceX-16" in capsys.readouterr().err


def test_log_cli_show_verify_reset(tmp_path, capsys):
    env = make_env(tmp_path, "restricted")
    p = write_spec(tmp_path, make_spec(env, clients=ISS, plan={G1: ISS}))
    check_cli.main(args(env, p))
    capsys.readouterr()
    base = ["--profile", "restricted", "--root", str(env.root)]
    assert log_cli.main(["verify", *base]) == 0
    assert log_cli.main(["show", *base]) == 0
    assert "restricted-1" in capsys.readouterr().out
    assert log_cli.main(["reset", *base]) == 2  # a reset needs --reason
    assert log_cli.main(["reset", *base, "--reason", "rehearsal"]) == 0
    assert "restricted-2" in capsys.readouterr().out


def test_log_cli_verify_fails_on_tampering(tmp_path, capsys):
    env = make_env(tmp_path, "restricted")
    spec, dec = check(env, make_spec(env, clients=ISS, plan={G1: ISS}))
    env.log.append(spec, dec)
    env.log.path.write_text(env.log.path.read_text().replace('"iss"', '"earth"'))
    assert log_cli.main(["verify", "--profile", "restricted", "--root", str(env.root)]) == 1
