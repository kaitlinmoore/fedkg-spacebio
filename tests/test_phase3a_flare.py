"""End to end in the FLARE simulator: 3 synthetic sites → site DE → FLARE → pooled = central.

``@pytest.mark.slow`` (D-018): run with ``uv run pytest -m slow``. It needs REML/HKSJ.

The synthetic workdir is built under ``tmp_path``: fake studies (``OSD-FAKE-N``), fake Ensembl
IDs, fake NCBI-style IDs (≥ 9,000,000,001), and seeded negative-binomial counts with a shared
flight effect plus site-specific shifts.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

JOBS = Path(__file__).resolve().parents[1] / "jobs" / "phase3a_meta"
CLIENTS = ["SITE-A", "SITE-B", "SITE-C"]
N_GENES = 400


def _load(name):
    sys.path.insert(0, str(JOBS))
    spec = importlib.util.spec_from_file_location(f"phase3a_{name}", JOBS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_workdir(root: Path) -> None:
    rng = np.random.default_rng(42)
    genes = [f"FAKEENSG{i:05d}" for i in range(N_GENES)]
    ann = root / "data" / "annotations"
    ann.mkdir(parents=True)
    pd.DataFrame({"ENSEMBL": genes, "SYMBOL": genes, "GENENAME": genes,
                  "ENTREZID": [str(9_000_000_001 + i) for i in range(N_GENES)]}).to_csv(
        ann / "Mus_musculus.GRCm39.112-GL-annotations.tsv", sep="\t", index=False)
    mu = rng.lognormal(4, 1.2, N_GENES)
    effect = np.zeros(N_GENES)
    effect[:40] = 1.0  # shared log2FC +1 in flight
    mdir = root / "phase3a" / "manifests" / "standard"
    mdir.mkdir(parents=True)
    for j, client in enumerate(CLIENTS):
        osd = f"OSD-FAKE-{j + 1}"
        site_effect = effect + rng.normal(0, 0.3, N_GENES) * (effect != 0)
        names, cols, samples = [], {}, []
        for arm, n in (("flight", 4 + j), ("ground", 4 + j)):
            for i in range(n):
                name = f"{osd}_{arm}_{i}"
                m = mu * (2 ** site_effect if arm == "flight" else 1.0)
                disp = 0.03 + 1 / mu
                cols[name] = rng.negative_binomial(1 / disp, 1 / (1 + m * disp)) + 0.3
                samples.append({"sample": name, "animal": f"{osd}:{arm}{i}", "arm": arm})
                names.append(name)
        d = root / "data" / "osdr" / osd
        d.mkdir(parents=True)
        fname = f"GLDS-FAKE-{j + 1}_rna_seq_RSEM_Unnormalized_Counts_GLbulkRNAseq.csv"
        pd.DataFrame(cols, index=pd.Index(genes, name="")).to_csv(d / fname)
        manifest = {"client": client, "site": f"FAKE-MISSION-{j + 1}", "stratum": "iss",
                    "policy_id": f"pol:FAKE-MISSION-{j + 1}:1", "design": "~ condition",
                    "contrast": ["condition", "flight", "ground"],
                    "counts_variant": "standard", "studies": [osd],
                    "sources": [{"osd_id": osd, "counts_file": f"{osd}/{fname}",
                                 "samples": samples}],
                    "census_rows": [{"osd_id": osd,
                                     "comparison_id": f"{osd}-00000000000000000000000000000000",
                                     "genelab_contrast": "(Space Flight)v(Ground Control)"}],
                    "n_flight": 4 + j, "n_ground": 4 + j}
        (mdir / f"{client}.json").write_text(json.dumps(manifest))


@pytest.mark.slow
def test_flare_pooling_equals_central(tmp_path, monkeypatch):
    monkeypatch.setenv("NVFLARE_SIMULATOR_WORKSPACE_ROOT", str(tmp_path / "flare-sim"))
    build_workdir(tmp_path)
    job, central = _load("job"), _load("central")
    job.run("iss", "standard", root=tmp_path, clients=CLIENTS, drop_variants=["SITE-C"],
            n_cpus=1, verify_annotation=False, build_manifests=False)
    report = central.run("iss", "standard", root=tmp_path, n_cpus=1, verify_annotation=False,
                         clients=CLIENTS, drop_variants=["SITE-C"])
    assert report["transport_bitwise"]
    assert report["pooled_pass"]
    primary = report["pooled"]["primary"]
    assert primary["n_genes_flare"] > 100
    assert "primary_without_SITE-C" in report["pooled"]
    # The FLARE server emits the V2/V3 columns (schema 2), and its fitted priors equal the
    # central run's.
    for col in ("v3_est", "v3_p", "v3_q", "v2_se", "v2_p", "v2_q"):
        assert primary["columns"][col]["allclose"], col
    assert primary["eb_priors"]["pass"]


@pytest.mark.slow
def test_flare_spec_runs_secure_sum_and_plain(tmp_path, monkeypatch):
    """Job-spec runs (WS-13) on the same synthetic sites:

    - **secure_sum** (fe + fe_het): the server's FE and Q equal the flag-driven run's within the
      fixed-point bound, with the same hits; the server received only uint64 masked vectors and
      whitelisted metadata, bit-identical to what each client recorded as sent;
    - **plain** (all five methods): reproduces the flag-driven ``primary`` pooled table.
    """
    from fedkg_spacebio.federation import secure_sum as ss

    monkeypatch.setenv("NVFLARE_SIMULATOR_WORKSPACE_ROOT", str(tmp_path / "flare-sim"))
    build_workdir(tmp_path)
    job, draft = _load("job"), _load("draft_spec")
    flare = job.run("iss", "standard", root=tmp_path, clients=CLIENTS, drop_variants=[],
                    n_cpus=1, verify_annotation=False, build_manifests=False)
    primary = pd.read_csv(flare / "primary" / "meta_genes.csv.gz")
    policy = tmp_path / "policy"
    common = dict(stratum="iss", clients=CLIENTS, root=tmp_path, policy_root=policy,
                  access_date="2026-11-12", annotation_sha256="FAKE-sha256-0001",
                  build_manifests=False)

    spec = draft.draft(aggregation="secure_sum", methods=["fe", "fe_het"],
                       job_id="job:FAKE-0101", **common)
    out = job.run_spec(spec, root=tmp_path, policy_root=policy, n_cpus=1,
                       verify_annotation=False, mask_seed=bytes(range(32)))
    table = pd.read_csv(out / "server" / "per_gene.csv.gz")
    assert np.array_equal(table["ncbi_gene"], primary["ncbi_gene"])
    b = ss.error_bound(primary["fe_est"], 1 / primary["fe_se"] ** 2, primary["k"], 40)
    assert np.all(np.abs(table["fe_est"] - primary["fe_est"]) <= b["est"] + 1e-14)
    assert np.all(np.abs(table["q"] - primary["q"]) <= b["q"] + 1e-13)
    assert np.array_equal(table["fe_hit"], primary["fe_q"] < 0.05)
    assert not any(c.startswith(("log2fc__", "se__")) for c in table.columns)
    schema = json.loads((out / "server" / "per_gene.schema.json").read_text())
    assert schema["methods_run"] == ["fe", "fe_het"]
    assert schema["aggregation"]["strategy"] == "secure_sum"
    for c in CLIENTS:
        sent = np.load(out / "sites" / c / "sent" / f"{c}.npz")
        got = np.load(out / "server" / "received" / f"{c}.npz")
        assert sorted(got.files) == ["masked_w", "masked_wy", "masked_wy2"]
        for k in got.files:
            assert got[k].dtype == np.uint64 and np.array_equal(got[k], sent[k])
        m = json.loads((out / "server" / "received" / f"{c}.meta.json").read_text())
        assert set(m) == set(ss.CONTRIBUTION_META_FIELDS)
    assert not list((out / "server").rglob("summary*"))  # no plain summary on the server side

    spec = draft.draft(aggregation="plain", methods=["fe", "fe_het", "reml_hksj", "v2", "v3"],
                       job_id="job:FAKE-0102", **common)
    out = job.run_spec(spec, root=tmp_path, policy_root=policy, n_cpus=1,
                       verify_annotation=False)
    table = pd.read_csv(out / "server" / "per_gene.csv.gz")
    for col in ("k", "fe_est", "fe_q", "q_p", "re_est", "re_q", "v2_q", "v3_est", "v3_q"):
        assert np.allclose(table[col], primary[col], rtol=1e-12, atol=1e-14,
                           equal_nan=True), col
