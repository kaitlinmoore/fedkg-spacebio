"""``sites.list`` (CONTRACTS §7): manifests of the profile, filters, released-flag hashes."""

from __future__ import annotations

import json

import numpy as np
import pytest

from agent_helpers import assert_ids_cover_data, start
from fedkg_spacebio.agent.tool_sites import LIVER_TERMS, TOOL
from fedkg_spacebio.agent.tools import Registry, ToolContext
from fedkg_spacebio.sites.inclusion import FLAGS, inclusion_flags_sha256

CLIENTS = [  # (client, site, stratum, tissue, n, studies)
    ("SpaceX-4", "SpaceX-4", "iss", "UBERON:0002107", 8, ["OSD-47", "OSD-48"]),
    ("SpaceX-16-ISST", "SpaceX-16", "iss", "UBERON:0006727", 20, ["OSD-379"]),
    ("SpaceX-13-LAR", "SpaceX-13", "earth", "UBERON:0002107", 10, ["OSD-245"]),
    ("FAKE-SITE-1", "FAKE-SITE-1", "iss", "UBERON:0000955", 5, ["OSD-FAKE-1"]),  # not liver
]


def _flags(seed: int):
    rng = np.random.default_rng(seed)
    genes = np.arange(9000000001, 9000000011, dtype=np.int64)
    return {"ncbi_gene": genes, **{f: rng.random(10) > 0.3 for f in FLAGS}}


@pytest.fixture
def world(tmp_path):
    mdir, sent = tmp_path / "manifests", tmp_path / "sent"
    mdir.mkdir()
    for i, (client, site, stratum, tissue, n, studies) in enumerate(CLIENTS):
        m = {"client": client, "site": site, "stratum": stratum, "policy_id": f"pol:{site}:1",
             "tissue_uberon": tissue, "design": "~ condition", "counts_variant": "standard",
             "studies": studies, "n_flight": n, "n_ground": n, "sources": []}
        (mdir / f"{client}.json").write_text(json.dumps(m))
        d = sent / stratum
        d.mkdir(parents=True, exist_ok=True)
        f = _flags(i)
        np.savez(d / f"{client}.npz", **f, log2fc=np.zeros(10))
        (d / f"{client}.meta.json").write_text(
            json.dumps({"inclusion_flags_sha256": inclusion_flags_sha256(f)}))
    ctx = ToolContext(profile="open", workdir=tmp_path, manifest_dir=mdir,
                      flags_dir=lambda stratum: sent / stratum)
    return ctx, mdir, sent


def call(ctx, inp):
    return Registry([TOOL]).call("sites.list", inp, ctx)


def test_lists_every_client_with_its_facts(world):
    ctx, _, _ = world
    r = call(ctx, {})
    assert r["ok"], r["error"]
    rows = {c["client"]: c for c in r["data"]["clients"]}
    assert set(rows) == {c[0] for c in CLIENTS}
    x = rows["SpaceX-16-ISST"]
    assert (x["site"], x["stratum"], x["tissue_uberon"], x["n_flight"], x["policy_id"]) == \
        ("SpaceX-16", "iss", "UBERON:0006727", 20, "pol:SpaceX-16:1")
    assert x["flags_released"] and len(x["inclusion_flags_sha256"]) == 64
    assert len(x["manifest_sha256"]) == 64
    for want in ("SpaceX-16-ISST", "SpaceX-16", "OSD-379", "UBERON:0006727", "pol:SpaceX-16:1",
                 "OSD-47", "OSD-48"):
        assert want in r["ids"]
    assert r["cache"] is None
    assert_ids_cover_data(r)


def test_liver_filter_accepts_every_liver_term(world):
    ctx, _, _ = world
    r = call(ctx, {"tissue": "liver"})
    assert {c["client"] for c in r["data"]["clients"]} == {"SpaceX-4", "SpaceX-16-ISST",
                                                           "SpaceX-13-LAR"}
    assert r["data"]["tissue_terms"] == list(LIVER_TERMS)
    assert set(LIVER_TERMS) <= set(r["ids"])
    assert_ids_cover_data(r)


def test_exact_tissue_and_stratum_filters(world):
    ctx, _, _ = world
    r = call(ctx, {"tissue": "UBERON:0006727"})
    assert [c["client"] for c in r["data"]["clients"]] == ["SpaceX-16-ISST"]
    r = call(ctx, {"stratum": "earth", "tissue": "liver"})
    assert [c["client"] for c in r["data"]["clients"]] == ["SpaceX-13-LAR"]


@pytest.mark.parametrize("inp", [{"tissue": "brain"}, {"stratum": "mars"}, {"profile": "open"}])
def test_bad_input_refused(world, inp):
    ctx, _, _ = world
    r = call(ctx, inp)
    assert not r["ok"] and r["error"]["code"] == "invalid_input" and r["ids"] == []


def test_flag_hash_is_recomputed_not_trusted(world):
    ctx, _, sent = world
    (sent / "iss" / "SpaceX-4.meta.json").write_text(
        json.dumps({"inclusion_flags_sha256": "0" * 64}))
    r = call(ctx, {})
    assert not r["ok"] and r["error"]["code"] == "flags_mismatch"
    assert "SpaceX-4" in r["error"]["message"]


def test_unreleased_flags_are_null(world):
    ctx, _, sent = world
    (sent / "iss" / "SpaceX-4.npz").unlink()
    r = call(ctx, {})
    row = next(c for c in r["data"]["clients"] if c["client"] == "SpaceX-4")
    assert row["inclusion_flags_sha256"] is None and row["flags_released"] is False


def test_manifest_without_tissue_is_refused(world):
    ctx, mdir, _ = world
    p = mdir / "SpaceX-4.json"
    m = json.loads(p.read_text())
    del m["tissue_uberon"]
    p.write_text(json.dumps(m))
    r = call(ctx, {})
    assert not r["ok"] and r["error"]["code"] == "manifest_incomplete"


def test_no_manifests(tmp_path):
    ctx = ToolContext(profile="restricted", workdir=tmp_path)
    r = call(ctx, {})
    assert r["error"]["code"] == "no_manifests"
    assert ctx.manifest_dir == tmp_path / "phase3a/manifests/profiles/restricted/standard"


def test_trace_records_call_and_result(world, tmp_path):
    ctx, _, _ = world
    ctx.trace = start(tmp_path / "t")
    r = call(ctx, {"stratum": "iss"})
    from fedkg_spacebio.agent.trace import read_trace
    recs = read_trace(ctx.trace.path)
    assert [x["type"] for x in recs] == ["run_start", "tool_call", "tool_result"]
    assert recs[1]["input"] == {"stratum": "iss"}
    assert recs[2]["call_id"] == r["call_id"] == "trace:FAKE-0001/call/1"
    assert recs[2]["ids"] == r["ids"]
