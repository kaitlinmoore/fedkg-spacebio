"""``results.read`` (CONTRACTS §7): releases only, pooled rows only, the notes that apply."""

from __future__ import annotations

import gzip
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from agent_helpers import assert_ids_cover_data
from fedkg_spacebio.agent.notes import NOTES
from fedkg_spacebio.agent.tool_results import TOOL, write_release_record
from fedkg_spacebio.agent.tools import Registry, ToolContext
from policy_helpers import G1, ISS, A, B, C, check, make_env, make_spec

REG = Registry([TOOL])
GENES = [9000000001, 9000000002, 9000000003, 9000000004, 9000000005]


def _table(clients, plain: bool) -> pd.DataFrame:
    t = pd.DataFrame({
        "ncbi_gene": GENES,
        "symbol": ["Fake1", "Nkx2-1", None, "Fake4", "Fake5"],
        "k": [4, 4, 3, 4, 4],
        "fe_est": [-1.2, 0.8, 0.1, np.nan, 2.5],
        "fe_se": [0.1, 0.2, 0.3, np.nan, 0.5],
        "fe_p": [1e-9, 1e-4, 0.5, np.nan, 1e-6],
        "fe_q": [5e-9, 2e-4, 0.6, np.nan, 3e-6],
        "fe_hit": [True, True, False, False, True],
        "i2": [0.1, 0.9, 0.0, np.nan, 0.5],
        "q": [3.0, 30.0, 1.0, np.nan, 6.0],
        "q_p": [0.4, 1e-5, 0.8, np.nan, 0.1],
        "het_flag": [False, True, False, False, False],
    })
    for c in clients:
        if plain:
            t[f"log2fc__{c}"] = 0.5
            t[f"se__{c}"] = 0.1
        t[f"n_flight__{c}"] = 5
        t[f"included__{c}"] = True
    return t


def released(tmp_path, profile="restricted", clients=ISS, plan=None, methods_refused=None,
             aggregation="secure_sum"):
    """A real release in the env's log, plus its table, schema and release record."""
    env = make_env(tmp_path, profile)
    spec = make_spec(env, clients=clients, plan=plan or {G1: clients},
                     outputs="masked_contribution" if aggregation == "secure_sum"
                     else "per_site_summary",
                     methods=["fe", "fe_het"])
    spec, dec = check(env, spec)
    assert dec.outcome == "allow", dec.reason
    entry = env.log.append(spec, dec)
    work = tmp_path / "work"
    out = work / "results" / "job"
    out.mkdir(parents=True)
    table_path = out / "per_gene.csv.gz"
    with gzip.open(table_path, "wt") as f:
        _table(clients, aggregation == "plain").to_csv(f, index=False)
    schema = {"schema_version": "phase3a-per-gene/2", "methods_run": ["fe", "fe_het"],
              "aggregation": ({"strategy": "secure_sum", "fixed_point_bits": 40,
                               "modulus_bits": 64} if aggregation == "secure_sum"
                              else {"strategy": "plain"})}
    schema_path = out / "per_gene.schema.json"
    schema_path.write_text(json.dumps(schema))
    ctx = ToolContext(profile=profile, workdir=work, policy_root=env.root)

    def ref(p):
        return {"path": p.relative_to(work).as_posix(),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    write_release_record(ctx, {
        "release_id": entry["release_id"], "job_id": spec.job_id,
        "job_spec_sha256": entry["job_spec_sha256"], "profile": profile,
        "log_id": env.log.log_id, "stratum": "iss", "counts_variant": "standard",
        "inclusion": {"rule": "mean10_finite", "min_sites_per_gene": "all"},
        "clients": list(clients), "table": ref(table_path), "schema": ref(schema_path),
        "methods_refused": methods_refused or {}})
    return ctx, entry["release_id"], table_path


def read(ctx, **inp):
    return REG.call("results.read", inp, ctx)


def test_top_rows_pooled_only(tmp_path):
    ctx, rel, _ = released(tmp_path, aggregation="plain", profile="open")
    r = read(ctx, release_id=rel, select={"top": {"sort_by": "q", "n": 3}})
    assert r["ok"], r["error"]
    rows = r["data"]["rows"]
    assert [x["ncbi_gene"] for x in rows] == ["NCBIGene:9000000001", "NCBIGene:9000000005",
                                              "NCBIGene:9000000002"]
    assert set(rows[0]) == {"ncbi_gene", "symbol", "k", "estimate", "se", "p", "q", "hit"}
    assert rows[0]["estimate"] == -1.2 and rows[0]["hit"] is True
    assert not any("__" in k for x in rows for k in x)  # never per-site columns
    assert r["data"]["n_tested"] == 4 and r["data"]["n_hits"] == 3
    assert r["data"]["k_by_gene"]["NCBIGene:9000000001"] == 4
    assert rel in r["ids"] and "NCBIGene:9000000001" in r["ids"]
    assert "NCBIGene:9000000003" not in r["ids"]  # not returned, not citable
    assert_ids_cover_data(r)


def test_sort_by_abs_estimate_and_missing_numbers(tmp_path):
    ctx, rel, _ = released(tmp_path)
    r = read(ctx, release_id=rel, select={"top": {"sort_by": "abs_estimate", "n": 5}})
    assert [x["ncbi_gene"] for x in r["data"]["rows"]][:2] == ["NCBIGene:9000000005",
                                                               "NCBIGene:9000000001"]
    assert len(r["data"]["rows"]) == 4  # the gene with no estimate isn't ranked


def test_named_genes_and_absent_ones(tmp_path):
    ctx, rel, _ = released(tmp_path)
    r = read(ctx, release_id=rel, select={"genes": ["NCBIGene:9000000004",
                                                    "NCBIGene:9000000002",
                                                    "NCBIGene:9000000099"]})
    rows = r["data"]["rows"]
    assert [x["ncbi_gene"] for x in rows] == ["NCBIGene:9000000004", "NCBIGene:9000000002"]
    assert rows[0]["estimate"] is None and rows[1]["symbol"] == "Nkx2-1"
    assert r["data"]["not_in_release"] == ["NCBIGene:9000000099"]
    assert_ids_cover_data(r)


def test_heterogeneity_rows(tmp_path):
    ctx, rel, _ = released(tmp_path)
    r = read(ctx, release_id=rel, method="fe_het", select={"top": {"sort_by": "p", "n": 2}})
    rows = r["data"]["rows"]
    assert rows[0]["ncbi_gene"] == "NCBIGene:9000000002"
    assert set(rows[0]) == {"ncbi_gene", "symbol", "k", "cochran_q", "q_p", "i2", "het_flag"}
    assert rows[0]["het_flag"] is True and r["data"]["n_hits"] == 1
    assert "note:het-annotation" in r["ids"]


def test_method_not_run_names_the_refusing_decision(tmp_path):
    ctx, rel, _ = released(tmp_path, methods_refused={"v3": "trace:FAKE-0001/dec/2"})
    r = read(ctx, release_id=rel, method="v3")
    assert r["error"]["code"] == "method_not_run"
    assert "trace:FAKE-0001/dec/2" in r["error"]["message"]
    ok = read(ctx, release_id=rel)
    assert ok["data"]["methods_refused"] == {"v3": "trace:FAKE-0001/dec/2"}
    assert "trace:FAKE-0001/dec/2" in ok["ids"]


def note_ids(r):
    return {n["note_id"] for n in r["data"]["notes"]}


def test_notes_for_the_restricted_masked_main_result(tmp_path):
    ctx, rel, _ = released(tmp_path)
    r = read(ctx, release_id=rel)
    assert note_ids(r) == {"note:fe-these-missions", "note:het-annotation", "note:time-of-day",
                           "note:rhythmic-share-fe", "note:methods-chosen-after-results",
                           "note:policies-invented", "note:secure-sum-simulation"}
    tod = next(n for n in r["data"]["notes"] if n["note_id"] == "note:time-of-day")
    assert tod["text"] == ("These results are consistent with a time-of-day mismatch of a few "
                           "hours or a clock change; not separable.")
    assert_ids_cover_data(r)


def test_notes_for_the_open_plain_main_result(tmp_path):
    ctx, rel, _ = released(tmp_path, profile="open", aggregation="plain")
    r = read(ctx, release_id=rel)
    ids = note_ids(r)
    assert {"note:spacex16-dominance", "note:spacex8-stays-in",
            "note:open-counterfactual"} <= ids
    assert "note:secure-sum-simulation" not in ids
    assert {"SpaceX-16", "SpaceX-8"} <= set(r["ids"])  # named in the notes, so citable
    assert_ids_cover_data(r)


def test_finding_notes_only_for_the_analysis_they_describe(tmp_path):
    ctx, rel, _ = released(tmp_path, profile="open", aggregation="plain", clients=[A, B, C])
    ids = note_ids(read(ctx, release_id=rel))
    assert not ids & {"note:spacex16-dominance", "note:rhythmic-share-fe",
                      "note:spacex8-stays-in"}
    assert "note:time-of-day" in ids


def test_note_catalog_has_no_decision_log_numbers():
    import re
    for n in NOTES:
        assert not re.search(r"\b[DQ]-\d", n.text), n.key
        assert n.source.startswith("docs/")


def test_unknown_release(tmp_path):
    ctx, rel, _ = released(tmp_path)
    r = read(ctx, release_id="rel:FAKE-0009-1")
    assert r["error"]["code"] == "unknown_release" and r["ids"] == []


def test_record_without_a_log_entry_is_refused(tmp_path):
    ctx, rel, _ = released(tmp_path)
    write_release_record(ctx, {**json.loads(next(ctx.releases_dir.glob("*.json")).read_text()),
                               "release_id": "rel:FAKE-0009-1"})
    assert read(ctx, release_id="rel:FAKE-0009-1")["error"]["code"] == "release_not_in_log"


def test_changed_table_is_refused(tmp_path):
    ctx, rel, table = released(tmp_path)
    table.write_bytes(table.read_bytes() + b"\n")
    assert read(ctx, release_id=rel)["error"]["code"] == "table_changed"


def test_record_is_written_once(tmp_path):
    ctx, rel, _ = released(tmp_path)
    rec = json.loads(next(ctx.releases_dir.glob("*.json")).read_text())
    with pytest.raises(FileExistsError):
        write_release_record(ctx, rec)


@pytest.mark.parametrize("inp", [
    {"release_id": "rel:FAKE-0001"},
    {"release_id": "rel:FAKE-0001-1", "select": {"top": {"n": 51}}},
    {"release_id": "rel:FAKE-0001-1", "select": {"top": {}, "genes": ["NCBIGene:1"]}},
    {"release_id": "rel:FAKE-0001-1", "select": {"genes": ["Fake1"]}},
    {"release_id": "rel:FAKE-0001-1", "select": {"genes": ["FAKE:0001"]}},
    {"release_id": "rel:FAKE-0001-1", "method": "per_site"},
])
def test_bad_input(tmp_path, inp):
    ctx = ToolContext(profile="open", workdir=tmp_path)
    assert REG.call("results.read", inp, ctx)["error"]["code"] == "invalid_input"
