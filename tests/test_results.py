"""Phase 3a result definitions (``fedkg_spacebio.federation.results``): hit calls, the
heterogeneity flag, the per-gene table and its schema. Fake gene IDs (≥ 9,000,000,001) and fake
client names."""

import numpy as np
import pandas as pd
import pytest

from fedkg_spacebio.federation import results as rs
from fedkg_spacebio.federation import stratum as st
from fedkg_spacebio.federation.meta import POOL_COLUMNS

GENES = [9_000_000_001 + i for i in range(6)]


def site_frame(rng, flagged):
    return pd.DataFrame({
        "log2fc": rng.normal(0, 1, len(GENES)), "se": rng.uniform(0.1, 0.5, len(GENES)),
        "base_mean": 100.0, "tested_mean10": flagged, "tested_indep_filter": flagged,
        "tested_finite_p": True,
    }, index=pd.Index(GENES, name="ncbi_gene"))


@pytest.fixture
def frames():
    rng = np.random.default_rng(1)
    a = site_frame(rng, [True, True, True, True, True, False])
    b = site_frame(rng, [True, True, True, True, False, False])
    c = site_frame(rng, [True, True, True, False, False, True]).drop(index=GENES[2])
    return {"SITE-A": a, "SITE-B": b, "SITE-C": c}


def test_classify_definitions_and_nan_handling():
    t = pd.DataFrame({"fe_q": [0.01, 0.01, 0.2, np.nan], "q_p": [0.5, 0.05, 0.01, np.nan],
                      "re_q": [0.04, 0.9, np.nan, np.nan], "v3_q": [0.01, 0.2, np.nan, 0.01],
                      "v2_q": [0.2, 0.01, np.nan, np.nan]})
    c = rs.classify(t)
    # V3 and V2 hits don't depend on the FE call or the heterogeneity flag.
    assert c["v3_hit"].tolist() == [True, False, False, True]
    assert c["v2_hit"].tolist() == [False, True, False, False]
    assert c["fe_hit"].tolist() == [True, True, False, False]
    assert c["het_flag"].tolist() == [False, True, True, False]
    assert c["fe_hit_consistent"].tolist() == [True, False, False, False]
    assert c["fe_hit_heterogeneous"].tolist() == [False, True, False, False]
    assert c["re_hit"].tolist() == [True, False, False, False]


def test_thresholds_are_strict_inequalities():
    t = pd.DataFrame({"fe_q": [rs.FDR_Q], "q_p": [rs.HET_Q_P], "re_q": [rs.FDR_Q],
                      "v3_q": [rs.FDR_Q], "v2_q": [rs.FDR_Q]})
    c = rs.classify(t)
    assert not c["fe_hit"][0] and not c["het_flag"][0] and not c["re_hit"][0]
    assert not c["v3_hit"][0] and not c["v2_hit"][0]


def test_hit_counts_match_columns_and_partition():
    rng = np.random.default_rng(2)
    n = 500
    t = rs.classify(pd.DataFrame({"fe_q": rng.uniform(0, 0.2, n), "q_p": rng.uniform(0, 1, n),
                                  "re_q": rng.uniform(0, 1, n), "fe_est": rng.normal(0, 1, n),
                                  "v3_q": rng.uniform(0, 0.5, n), "v3_est": rng.normal(0, 1, n),
                                  "v2_q": rng.uniform(0, 0.5, n)}))
    h = rs.hit_counts(t)
    assert h["v3_hit"] == int(t["v3_hit"].sum()) > 0
    assert h["v3_hit_up"] + h["v3_hit_down"] == h["v3_hit"]
    assert h["v2_hit"] == int(t["v2_hit"].sum())
    assert h["fe_hit"] == int(t["fe_hit"].sum())
    assert h["fe_hit_consistent"] + h["fe_hit_heterogeneous"] == h["fe_hit"]
    assert h["fe_hit_up"] + h["fe_hit_down"] == h["fe_hit"]


def test_per_gene_table_inclusion_reproduces_k(frames):
    pooled = st.pool_variant(frames, st.Variant("min2", "tested_mean10", 2))
    clients = sorted(frames)
    t = rs.per_gene_table(pooled, frames, "tested_mean10", clients)
    assert list(t.columns[:2]) == ["ncbi_gene", "symbol"]
    assert set(POOL_COLUMNS + rs.FLAG_COLUMNS) <= set(t.columns)
    inc = t[[f"included__{c}" for c in clients]].sum(axis=1)
    assert (inc == t["k"]).all()
    # SITE-C didn't send GENES[2]: not included, log2fc missing.
    row = t.set_index("ncbi_gene").loc[GENES[2]]
    assert not row["included__SITE-C"] and np.isnan(row["log2fc__SITE-C"])
    # Per-site inputs are the values the site sent.
    a = frames["SITE-A"].loc[t["ncbi_gene"]]
    np.testing.assert_array_equal(t["log2fc__SITE-A"], a["log2fc"])
    np.testing.assert_array_equal(t["se__SITE-A"], a["se"])


def test_per_gene_table_rejects_inconsistent_inputs(frames):
    pooled = st.pool_variant(frames, st.Variant("min2", "tested_mean10", 2))
    wrong = {c: f.assign(tested_mean10=True) for c, f in frames.items()}
    with pytest.raises(ValueError, match="reproduce k"):
        rs.per_gene_table(pooled, wrong, "tested_mean10", sorted(frames))


def test_per_gene_csv_round_trips_exactly(frames, tmp_path):
    pooled = st.pool_variant(frames, st.Variant("min2", "tested_mean10", 2))
    t = rs.per_gene_table(pooled, frames, "tested_mean10", sorted(frames),
                          pd.Series({g: f"FAKE{i}" for i, g in enumerate(GENES)}))
    path = tmp_path / "per_gene.csv.gz"
    t.to_csv(path, index=False, float_format="%.17g")
    back = pd.read_csv(path)
    for c in t.columns:
        if t[c].dtype.kind == "f":
            np.testing.assert_array_equal(back[c].to_numpy(), t[c].to_numpy(), err_msg=c)
        else:
            assert back[c].tolist() == t[c].tolist(), c


def test_schema_documents_every_column(frames):
    pooled = st.pool_variant(frames, st.Variant("min2", "tested_mean10", 2))
    t = rs.per_gene_table(pooled, frames, "tested_mean10", sorted(frames),
                          site_n={"SITE-A": (4, 4), "SITE-B": (5, 4), "SITE-C": (3, 3)})
    assert (t["n_ground__SITE-B"] == 4).all() and (t["n_flight__SITE-C"] == 3).all()
    s = rs.schema(t, {"variant": "min2"})
    assert s["schema_version"] == rs.SCHEMA_VERSION
    assert s["schema_version"] == "phase3a-per-gene/2"
    assert [c["name"] for c in s["columns"]] == list(t.columns)
    assert all(c["description"] for c in s["columns"])
    names = {c["name"] for c in s["columns"]}
    assert {"v3_est", "v3_q", "v3_hit", "v2_se", "v2_q", "v2_hit"} <= names
