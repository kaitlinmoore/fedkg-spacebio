"""Q-11 reference clock (``evaluation/clock_reference.py``): cosinor, GEO parsing, panel rule."""

import gzip

import numpy as np
import pandas as pd
import pytest

from fedkg_spacebio.evaluation import clock_reference as cref

W = 2 * np.pi / 24


def test_cosinor_recovers_known_parameters_without_noise():
    t = np.arange(18, 66, 1.0)
    truth = [(8.0, 1.5, 3.0), (5.0, 0.4, 21.5), (12.0, 2.0, 0.25)]  # mesor, amplitude, peak
    y = np.array([m + a * np.cos(W * (t - p)) for m, a, p in truth])
    fit = cref.cosinor(y, t)
    for (m, a, p), (_, row) in zip(truth, fit.iterrows(), strict=True):
        assert row["mesor"] == pytest.approx(m)
        assert row["amplitude"] == pytest.approx(a)
        assert abs(cref.circ_diff(row["peak"], p)) < 1e-9
        assert row["r2"] == pytest.approx(1.0)


def test_cosinor_flat_series_is_not_rhythmic_and_missing_values_are_skipped():
    t = np.arange(0, 48, 6.0)
    rng = np.random.default_rng(1)
    flat = 3 + rng.normal(0, 0.1, len(t))
    gappy = 2 + np.cos(W * (t - 6))
    gappy[3] = np.nan
    fit = cref.cosinor(np.vstack([flat, gappy]), t)
    assert fit.loc[0, "p"] > 0.01
    assert fit.loc[1, "n"] == len(t) - 1
    assert fit.loc[1, "amplitude"] == pytest.approx(1.0)


def test_circ_diff_wraps():
    assert cref.circ_diff(23.0, 1.0) == pytest.approx(-2.0)
    assert cref.circ_diff(1.0, 23.0) == pytest.approx(2.0)
    assert cref.circ_diff(12.0, 0.0) == pytest.approx(12.0)


def test_series_matrix_and_ct_titles(tmp_path):
    text = "\n".join([
        '!Series_title\t"x"',
        '!Sample_title\t"Liver_CT18"\t"Liver_CT19"',
        '!Sample_geo_accession\t"GSM1"\t"GSM2"',
        '!Sample_characteristics_ch1\t"male"\t"male"',
        '!Sample_characteristics_ch1\t"liver"\t"liver"',
        "!series_matrix_table_begin",
        '"ID_REF"\t"GSM1"\t"GSM2"',
        '"FAKE_at"\t10.5\t20',
        '"FAKE2_at"\tnull\t3',
        "!series_matrix_table_end", ""])
    p = tmp_path / "fake_series_matrix.txt.gz"
    with gzip.open(p, "wt") as fh:
        fh.write(text)
    values, samples = cref.read_series_matrix(p)
    assert list(values.index) == ["FAKE_at", "FAKE2_at"]
    assert np.isnan(values.loc["FAKE2_at", "GSM1"])
    assert "characteristics_ch1_2" in samples.columns
    assert cref.ct_from_titles(samples["title"]).tolist() == [18.0, 19.0]
    with pytest.raises(ValueError):
        cref.ct_from_titles(["no time here"])


def _array_fit(rows):
    return pd.DataFrame(rows, columns=["probe", "peak", "amplitude", "r2", "q", "mesor"]
                        ).set_index("probe")


def test_panel_rule_keeps_and_drops_with_reasons():
    afit = _array_fit([
        ("p_ok", 10.0, 1.0, 0.9, 1e-10, 8.0),
        ("p_low_r2", 3.0, 1.0, 0.3, 1e-10, 8.0),
        ("p_dis_1", 2.0, 1.0, 0.9, 1e-10, 8.0), ("p_dis_2", 5.0, 1.0, 0.8, 1e-10, 8.0),
        ("p_two_1", 20.0, 1.0, 0.7, 1e-10, 8.0), ("p_two_2", 21.0, 1.2, 0.9, 1e-10, 8.0),
        ("p_flat", 1.0, 0.1, 0.9, 1e-10, 8.0),
        ("p_far", 6.0, 1.0, 0.9, 1e-10, 8.0),
    ])
    pmap = pd.Series({"p_ok": 1, "p_low_r2": 2, "p_dis_1": 3, "p_dis_2": 3, "p_two_1": 4,
                      "p_two_2": 4, "p_flat": 5, "p_far": 6})
    ids = pd.DataFrame({"symbol": ["Ok", "LowR2", "Disagree", "Two", "Flat", "Far", "Gone"],
                        "ncbi_gene": [1, 2, 3, 4, 5, 6, None],
                        "status": ["ok"] * 6 + ["missing_in_genelab"]})
    rfit = pd.DataFrame({"peak": [10.5, 3, 2, 21.5, 1, 11.0], "amplitude": [2.0] * 6,
                         "mesor": [9.0] * 6, "p": [0.01] * 6, "resid_sd": [0.1] * 6},
                        index=[1, 2, 3, 4, 5, 6])
    kept, dec = cref.select_panel(afit, pmap, ids, rfit)
    reasons = dict(zip(dec["symbol"], dec["reason"], strict=True))
    assert list(kept["symbol"]) == ["Ok", "Two"]
    assert kept.set_index("symbol").loc["Two", "probe_set"] == "p_two_2"  # higher R²
    assert "R² 0.30" in reasons["LowR2"]
    assert "disagree" in reasons["Disagree"]
    assert "peak/trough" in reasons["Flat"]
    assert "from GSE11923" in reasons["Far"]
    assert "not resolved" in reasons["Gone"]
    assert kept.set_index("symbol").loc["Ok", "amplitude_rnaseq"] == 2.0


def test_basis_vectors_subset_and_scale():
    b = cref.ClockBasis((1, 2), ("A", "B"), np.array([0.0, 6.0]), np.array([2.0, 1.0]))
    assert np.allclose(b.a, [2.0, 0.0]) and np.allclose(b.b, [0.0, 1.0])
    assert b.subset([2]).symbol == ("B",)
    assert np.allclose(b.scaled(0.5).amplitude, [1.0, 0.5])


def test_reference_bootstrap_is_exact_without_noise():
    ta, tr = np.arange(18, 66, 1.0), np.arange(22, 70, 6.0)
    kept = pd.DataFrame({"ncbi_gene": [1, 2], "symbol": ["A", "B"], "probe_set": ["pa", "pb"]})
    arr = pd.DataFrame([5 + np.cos(W * (ta - 3)), 6 + 2 * np.cos(W * (ta - 15))],
                       index=["pa", "pb"])
    rna = pd.DataFrame([9 + 1.5 * np.cos(W * (tr - 3)), 7 + 3 * np.cos(W * (tr - 15))],
                       index=[1, 2])
    boots = cref.bootstrap_bases(kept, arr, ta, rna, tr, n_boot=3, seed=0)
    for b in boots:
        assert np.allclose(b.peak, [3, 15]) and np.allclose(b.amplitude, [1.5, 3])
