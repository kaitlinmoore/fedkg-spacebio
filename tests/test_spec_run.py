"""Job-spec runs without FLARE: spec checks (``spec_run``), server pooling and tables
(``spec_server``), and the client's spec mode (``jobs/phase3a_meta/client.py``).

Synthetic summaries, fake gene IDs (≥ 9,000,000,001), fake clients ``SITE-*``.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fedkg_spacebio.federation import secure_sum as ss
from fedkg_spacebio.federation import spec_server, stratum
from fedkg_spacebio.federation.gene_plan import build_gene_plan, write_gene_list, write_gene_plan
from fedkg_spacebio.federation.jobspec import parse_job_spec
from fedkg_spacebio.federation.site_de import SiteSummary
from fedkg_spacebio.federation.spec_run import SpecRunError, plan_rows, prepare
from fedkg_spacebio.sites.inclusion import inclusion_flags_sha256

CLIENTS = ["SITE-A", "SITE-B", "SITE-C"]
JOBS = Path(__file__).resolve().parents[1] / "jobs" / "phase3a_meta"


def summaries(n=200, seed=1):
    rng = np.random.default_rng(seed)
    ids = np.arange(9_000_000_001, 9_000_000_001 + n, dtype=np.int64)
    out = {}
    for j, c in enumerate(CLIENTS):
        flag = rng.random(n) > 0.15 * j
        genes = {"ncbi_gene": ids, "log2fc": rng.normal(0, 1, n),
                 "se": rng.uniform(0.05, 1.0, n), "base_mean": rng.uniform(5, 500, n),
                 "tested_mean10": flag, "tested_indep_filter": flag.copy(),
                 "tested_finite_p": np.ones(n, dtype=bool)}
        meta = {"client": c, "site": c, "stratum": "iss", "n_flight": 4 + j, "n_ground": 4 + j,
                "design": "~ condition", "counts_variant": "standard", "de_settings": {},
                "rounding": {}, "n_genes_de": n, "n_ercc_dropped": 0,
                "mapping": {"n_sent": n, "n_tested_mean10_sent": int(flag.sum())},
                "versions": {}, "census_rows": [],
                "inclusion_flags_sha256": inclusion_flags_sha256(genes)}
        out[c] = SiteSummary(genes, meta)
    return out


def env(tmp_path, s, *, outputs="masked_contribution", methods=("fe", "fe_het"),
        min_sites="all", aggregation=None, exclude=None, **over):
    root = tmp_path / "policy"
    mdir = tmp_path / "manifests"
    mdir.mkdir(exist_ok=True)
    msha = {}
    for c in CLIENTS:
        data = json.dumps({"client": c, "site": c, "stratum": "iss", "counts_variant": "standard",
                           "policy_id": f"pol:{c}:1"}).encode()
        (mdir / f"{c}.json").write_bytes(data)
        msha[c] = hashlib.sha256(data).hexdigest()
    plan = build_gene_plan({c: pd.DataFrame(s[c].genes) for c in CLIENTS}, "mean10_finite",
                           min_sites)
    d = {"schema": "job-spec/1", "job_id": "job:FAKE-0007", "task": "de_meta", "question": "q",
         "stratum": "iss", "clients": CLIENTS, "methods": list(methods),
         "counts_variant": "standard",
         "inclusion": {"rule": "mean10_finite", "min_sites_per_gene": min_sites},
         "gene_plan": {"ref": "plans/p/gene_plan.csv",
                       "sha256": write_gene_plan(plan, root / "plans/p/gene_plan.csv")},
         "outputs": {c: outputs for c in CLIENTS},
         "aggregation": aggregation or ({"strategy": "secure_sum", "fixed_point_bits": 40,
                                         "modulus_bits": 64}
                                        if outputs == "masked_contribution"
                                        else {"strategy": "plain"}),
         "weighting": {"scheme": "inverse_variance"}, "rounds": 1,
         "provenance": {"kg_version": "FAKE", "census_access_date": "2026-11-12",
                        "annotation_sha256": "FAKE-sha256-0001", "manifest_sha256": msha,
                        "inclusion_flags_sha256": {c: s[c].meta["inclusion_flags_sha256"]
                                                   for c in CLIENTS}}}
    if exclude is not None:
        d["genes_exclude"] = {"decision_id": "dec:FAKE-0007-1", "ref": "plans/p/excl.csv",
                              "sha256": write_gene_list(exclude, root / "plans/p/excl.csv")}
    d.update(over)
    return parse_job_spec(d), mdir, root, plan


# ----------------------------------------------------------------- spec checks


def test_prepare_secure_sum(tmp_path):
    s = summaries()
    spec, mdir, root, plan = env(tmp_path, s)
    sr = prepare(spec, manifest_dir=mdir, policy_root=root)
    assert sr.mode == "secure_sum" and sr.params == ss.Params(40, 64)
    assert sr.quantities == ss.QUANTITIES_FE_HET
    assert sr.rows.equals(plan) and sr.slug == "job-FAKE-0007"
    assert sr.aggregation()["strategy"] == "secure_sum"
    spec2, *_ = env(tmp_path, s, methods=("fe",))
    assert prepare(spec2, manifest_dir=mdir, policy_root=root).quantities == ss.QUANTITIES_FE


def test_prepare_applies_exclusions(tmp_path):
    s = summaries()
    spec, mdir, root, plan = env(tmp_path, s, exclude=[])
    gone = plan.index[[0, 5]].tolist()
    spec, mdir, root, plan = env(tmp_path, s, exclude=gone)
    sr = prepare(spec, manifest_dir=mdir, policy_root=root)
    assert len(sr.rows) == len(plan) - 2 and not set(gone) & set(sr.rows.index)
    rows, plan_sha, excl_sha = plan_rows(spec, root)
    assert rows.equals(sr.rows) and excl_sha == spec.genes_exclude.sha256


@pytest.mark.parametrize("case, match", [
    (dict(outputs="per_site_summary", aggregation={"strategy": "secure_sum",
                                                   "fixed_point_bits": 40, "modulus_bits": 64}),
     "aggregation: plain"),
    (dict(aggregation={"strategy": "plain"}), "needs aggregation: secure_sum"),
    (dict(methods=("fe", "v3")), "can't serve"),
    (dict(job_id=None), "job_id is null"),
])
def test_prepare_refuses(tmp_path, case, match):
    spec, mdir, root, _ = env(tmp_path, summaries(), **case)
    with pytest.raises(SpecRunError, match=match):
        prepare(spec, manifest_dir=mdir, policy_root=root)


def test_prepare_refuses_mixed_outputs_and_hash_mismatches(tmp_path):
    s = summaries()
    spec, mdir, root, _ = env(tmp_path, s)
    d = spec.model_dump(mode="json", by_alias=True)
    mixed = parse_job_spec({**d, "outputs": {**d["outputs"], "SITE-A": "per_site_summary"}})
    with pytest.raises(SpecRunError, match="mixed outputs"):
        prepare(mixed, manifest_dir=mdir, policy_root=root)
    bad_plan = parse_job_spec({**d, "gene_plan": {**d["gene_plan"], "sha256": "0" * 64}})
    with pytest.raises(SpecRunError, match="SHA-256"):
        prepare(bad_plan, manifest_dir=mdir, policy_root=root)
    (mdir / "SITE-B.json").write_text("{}")
    with pytest.raises(SpecRunError, match="SITE-B: manifest_sha256"):
        prepare(spec, manifest_dir=mdir, policy_root=root)


# ----------------------------------------------------------------- server pooling and tables


def test_plain_spec_pooling_equals_the_flag_driven_primary(tmp_path):
    s = summaries()
    _, _, _, plan = env(tmp_path, s, outputs="per_site_summary")
    restricted = {c: _client().restricted_summary(s[c], plan, c) for c in CLIENTS}
    pooled, frames = spec_server.pool_plain_spec(restricted, plan, "tested_mean10",
                                                 ["fe", "fe_het", "reml_hksj"])
    flag_driven = stratum.pool_variant({c: s[c].to_frame() for c in CLIENTS},
                                       stratum.VARIANTS[0])
    flag_driven = flag_driven.set_index("ncbi_gene").loc[pooled["ncbi_gene"]].reset_index()
    for col in ("k", "fe_est", "fe_se", "fe_p", "fe_q", "q", "q_p", "re_est", "re_q"):
        assert np.array_equal(pooled[col].to_numpy(), flag_driven[col].to_numpy(),
                              equal_nan=True), col


def test_plain_spec_refuses_a_summary_with_extra_genes(tmp_path):
    s = summaries()
    _, _, _, plan = env(tmp_path, s, outputs="per_site_summary")
    sent = {c: _client().restricted_summary(s[c], plan, c) for c in CLIENTS}
    sent["SITE-A"] = s["SITE-A"]  # the full summary: more than the plan assigns
    with pytest.raises(spec_server.SpecPoolError, match="exactly its plan genes"):
        spec_server.pool_plain_spec(sent, plan, "tested_mean10", ["fe"])


def _site_n(s):
    return {c: (s[c].meta["n_flight"], s[c].meta["n_ground"]) for c in CLIENTS}


def test_secure_sum_table_has_only_fe_and_het_columns(tmp_path):
    s = summaries()
    spec, mdir, root, plan = env(tmp_path, s, min_sites=2)
    seed = bytes(32)
    contributions = {}
    for c in CLIENTS:
        arrays, meta, key = _client().spec_payload(s[c], spec, c, str(root), seed)
        assert key == "fedkg_masked_contribution"
        contributions[c] = (arrays, meta)
    expected = {"job_id": spec.job_id, "gene_plan_sha256": spec.gene_plan.sha256,
                "n_rows": len(plan), "fixed_point_bits": 40, "modulus_bits": 64,
                "quantities": list(ss.QUANTITIES_FE_HET)}
    pooled, info = spec_server.pool_secure_sum(contributions, plan, ss.Params(40, 64), expected)
    t = spec_server.write_outputs(tmp_path / "server", pooled, methods=["fe", "fe_het"],
                                  rows=plan, site_n=_site_n(s),
                                  aggregation={"strategy": "secure_sum"}, provenance={},
                                  run_info=info)
    want = ["ncbi_gene", "symbol", "k", "fe_est", "fe_se", "fe_p", "fe_q", "i2", "q", "q_p",
            "fe_hit", "het_flag", "fe_hit_consistent", "fe_hit_heterogeneous"]
    want += [f"{p}__{c}" for c in CLIENTS for p in ("n_flight", "n_ground", "included")]
    assert list(t.columns) == want
    assert not any(c.startswith(("log2fc__", "se__", "base_mean__")) for c in t.columns)
    sch = json.loads((tmp_path / "server" / "per_gene.schema.json").read_text())
    assert sch["schema_version"] == "phase3a-per-gene/2"
    assert sch["methods_run"] == ["fe", "fe_het"] and sch["aggregation"]["strategy"] == "secure_sum"
    # The masked FE equals plain FE within the bound (exactness in miniature).
    ref = spec_server.pool_plain_spec({c: _client().restricted_summary(s[c], plan, c)
                                       for c in CLIENTS}, plan, "tested_mean10", ["fe"])[0]
    b = ss.error_bound(ref["fe_est"], 1 / ref["fe_se"] ** 2, ref["k"], 40)
    assert np.all(np.abs(t["fe_est"] - ref["fe_est"]) <= b["est"] + 1e-14)


def test_secure_sum_server_refuses_a_payload_from_another_plan(tmp_path):
    s = summaries()
    spec, mdir, root, plan = env(tmp_path, s)
    contributions = {c: _client().spec_payload(s[c], spec, c, str(root), bytes(32))[:2]
                     for c in CLIENTS}
    expected = {"gene_plan_sha256": "f" * 64}
    with pytest.raises(ss.SecureSumError, match="gene_plan_sha256"):
        spec_server.pool_secure_sum(contributions, plan, ss.Params(40, 64), expected)
    a, m = contributions["SITE-A"]
    contributions["SITE-A"] = (a, {**m, "client": "SITE-B"})
    with pytest.raises(spec_server.SpecPoolError, match="claims client"):
        spec_server.pool_secure_sum(contributions, plan, ss.Params(40, 64), {})


def test_plain_spec_table_drops_methods_not_run(tmp_path):
    s = summaries()
    _, _, _, plan = env(tmp_path, s, outputs="per_site_summary")
    sent = {c: _client().restricted_summary(s[c], plan, c) for c in CLIENTS}
    pooled, frames = spec_server.pool_plain_spec(sent, plan, "tested_mean10", ["fe"])
    t = spec_server.write_outputs(tmp_path / "srv", pooled, methods=["fe"], rows=plan,
                                  site_n=_site_n(s), aggregation={"strategy": "plain"},
                                  provenance={}, run_info={}, frames=frames)
    assert "fe_est" in t and "log2fc__SITE-A" in t
    assert not {"q", "i2", "re_est", "v3_est", "v2_se", "het_flag"} & set(t.columns)


# ----------------------------------------------------------------- client spec mode

_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        sys.path.insert(0, str(JOBS))
        spec = importlib.util.spec_from_file_location("phase3a_client", JOBS / "client.py")
        _CLIENT = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_CLIENT)
    return _CLIENT


def test_client_refuses_flags_other_than_the_plans(tmp_path):
    s = summaries()
    spec, mdir, root, plan = env(tmp_path, s)
    other = SiteSummary(s["SITE-A"].genes, {**s["SITE-A"].meta,
                                             "inclusion_flags_sha256": "0" * 64})
    with pytest.raises(ValueError, match="aren't the ones"):
        _client().spec_payload(other, spec, "SITE-A", str(root), bytes(32))


def test_client_sends_masked_vectors_or_its_plan_genes(tmp_path):
    s = summaries()
    spec, mdir, root, plan = env(tmp_path, s, min_sites=2)
    arrays, meta, _ = _client().spec_payload(s["SITE-C"], spec, "SITE-C", str(root), bytes(32))
    assert set(arrays) == {"masked_w", "masked_wy", "masked_wy2"}
    assert all(a.dtype == np.uint64 and len(a) == len(plan) for a in arrays.values())
    assert meta["n_rows"] == len(plan) and meta["gene_plan_sha256"] == spec.gene_plan.sha256
    with pytest.raises(ValueError, match="mask_seed"):
        _client().spec_payload(s["SITE-C"], spec, "SITE-C", str(root), None)
    pspec, _, proot, pplan = env(tmp_path, s, outputs="per_site_summary", min_sites=2)
    genes, pmeta, key = _client().spec_payload(s["SITE-C"], pspec, "SITE-C", str(proot), None)
    assert key == "fedkg_site_summary"
    assert set(genes["ncbi_gene"]) == set(pplan.index[pplan["SITE-C"]])
    assert pmeta["mapping"]["n_sent"] == len(genes["ncbi_gene"])
