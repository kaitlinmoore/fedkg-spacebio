"""Site DE wrapper and the site→server summary contract (``fedkg_spacebio.federation.site_de``).

Uses a small seeded negative-binomial simulation (fake gene IDs). The point is our seams
(direction, flags, mapping, the boundary object), not pydeseq2's internals.
"""

import json

import numpy as np
import pandas as pd
import pytest

from fedkg_spacebio.federation.site_de import (
    GENE_FIELDS,
    inclusion_flags,
    run_de,
    summarize,
    validate_summary,
)
from fedkg_spacebio.harmonize.gene_map import build_map
from fedkg_spacebio.sites.loader import SiteData

N_GENES, N_UP, N_PER_ARM = 300, 30, 4


@pytest.fixture(scope="module")
def site():
    rng = np.random.default_rng(0)
    mu = rng.lognormal(mean=4, sigma=1.5, size=N_GENES)
    mu[:5] = 2.0  # low-count genes: base mean well under 10
    fold = np.ones(N_GENES)
    fold[5:5 + N_UP] = 4.0  # higher in flight (log2FC = +2)
    samples = [f"S_FLT_{i}" for i in range(N_PER_ARM)] + [f"S_GC_{i}" for i in range(N_PER_ARM)]
    disp = 0.02 + 1.0 / mu  # dispersion falls with mean, as DESeq2's parametric trend assumes
    cols = {}
    for s in samples:
        m = mu * (fold if "FLT" in s else 1.0)
        cols[s] = rng.negative_binomial(1 / disp, 1 / (1 + m * disp))
    genes = [f"FAKEENSG{i:04d}" for i in range(N_GENES)]
    counts = pd.DataFrame(cols, index=pd.Index(genes, name="ensembl"))
    metadata = pd.DataFrame({"condition": ["flight"] * N_PER_ARM + ["ground"] * N_PER_ARM},
                            index=pd.Index(samples, name="sample"))
    manifest = {"site": "FAKE-MISSION-1", "stratum": "iss", "design": "~ condition",
                "counts_variant": "standard",
                "census_rows": [{"osd_id": "OSD-FAKE-1",
                                 "comparison_id": "OSD-FAKE-1-00000000000000000000000000000000",
                                 "genelab_contrast": "(Space Flight)v(Ground Control)"}]}
    prov = {"n_flight": N_PER_ARM, "n_ground": N_PER_ARM, "n_ercc_dropped": 0,
            "rounding": {"rule": "half to even", "fraction_cells_changed": 0.0,
                         "max_abs_change": 0.0}}
    return SiteData("SITE-A", counts, metadata, manifest, prov)


@pytest.fixture(scope="module")
def res(site):
    return run_de(site.counts, site.metadata, "~ condition", n_cpus=1)


@pytest.fixture(scope="module")
def gmap():
    ids = [f"FAKEENSG{i:04d}" for i in range(N_GENES)]
    entrez = [str(9_000_000_000 + i) for i in range(N_GENES)]
    entrez[100] = ""  # one gene with no Entrez ID
    return build_map(pd.DataFrame({"ENSEMBL": ids, "SYMBOL": ids, "ENTREZID": entrez}))


def test_direction_flight_up_is_positive(res):
    up = res.iloc[5:5 + N_UP]
    assert (up["log2fc"] > 0).all()
    assert up["log2fc"].median() == pytest.approx(2.0, abs=0.4)
    assert res.iloc[5 + N_UP:]["log2fc"].abs().median() < 0.3


def test_parametric_trend_used_and_warnings_recorded(res):
    assert res.attrs["warnings"] == []  # no fallback to the mean-based trend on this data


def test_columns_and_unshrunk_se(res):
    assert list(res.columns) == ["base_mean", "log2fc", "se", "stat", "pvalue", "padj"]
    # Wald statistic = MLE log2FC / lfcSE: holds for the unshrunk estimate only.
    ok = res["stat"].notna()
    np.testing.assert_allclose(res.loc[ok, "stat"], res.loc[ok, "log2fc"] / res.loc[ok, "se"],
                               rtol=1e-8)


def test_primary_filter_is_base_mean_10(res):
    flagged = inclusion_flags(res)
    assert not flagged.iloc[:5]["tested_mean10"].any()
    assert (flagged["tested_mean10"] <= flagged["tested_finite_p"]).all()
    assert (flagged.loc[flagged["tested_mean10"], "base_mean"] >= 10).all()


def test_summary_is_mapped_and_carries_no_sample_level_data(site, res, gmap):
    summ = summarize(site, res, gmap)
    assert set(summ.genes) == set(GENE_FIELDS)
    assert 9_000_000_100 not in set(summ.genes["ncbi_gene"])  # unmapped gene dropped
    assert summ.meta["mapping"]["all_genes"]["no_entrez"] == 1
    assert summ.meta["n_flight"] == N_PER_ARM
    # No sample name, and no array as long as the sample count, anywhere in what leaves.
    blob = json.dumps(summ.meta) + json.dumps({k: v.tolist() for k, v in summ.genes.items()})
    assert not any(s in blob for s in site.counts.columns)
    assert all(len(v) == len(summ.genes["ncbi_gene"]) for v in summ.genes.values())


def test_validate_summary_rejects_extra_fields(site, res, gmap):
    summ = summarize(site, res, gmap)
    with pytest.raises(ValueError, match="per-gene fields"):
        validate_summary(summ.genes | {"counts": np.zeros(3)}, summ.meta)
    with pytest.raises(ValueError, match="metadata fields"):
        validate_summary(summ.genes, summ.meta | {"samples": ["S_FLT_0"]})
    bad = dict(summ.genes)
    bad["se"] = bad["se"].copy()
    bad["se"][0] = 0.0
    with pytest.raises(ValueError, match="se finite"):
        validate_summary(bad, summ.meta)
