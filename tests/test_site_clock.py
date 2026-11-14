"""Q-11 site-side clock diagnostic (``federation/site_clock.py``).

Fixtures: ``tests/fixtures/clock/`` (fake IDs only). One count file holds two sites' samples,
as OSD-245 and OSD-379 do:
- SITE-X: 4 flight at t ≈ 4 h, 4 ground at t ≈ 2 h;
- SITE-Y: 3 + 3, both at t ≈ 9 h.

Six fake clock genes peak every 4 h with amplitude 1 (log2), plus two flat housekeeping genes.
``basis.csv`` is their clock plane.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fedkg_spacebio.evaluation.clock_reference import ClockBasis, circ_diff
from fedkg_spacebio.federation import site_clock as sc
from fedkg_spacebio.harmonize.gene_map import build_map
from fedkg_spacebio.sites.loader import load_site

FIX = Path(__file__).parent / "fixtures" / "clock"
W = 2 * np.pi / 24

_OPENED: list[str] | None = None


def _audit(event, args):
    if _OPENED is not None and event == "open" and isinstance(args[0], (str, Path)):
        _OPENED.append(str(Path(args[0]).resolve()))


sys.addaudithook(_audit)


@pytest.fixture(scope="module")
def gmap():
    return build_map(pd.read_csv(FIX / "annotations.tsv", sep="\t", dtype=str), "fixture")


@pytest.fixture(scope="module")
def basis():
    return ClockBasis.from_frame(pd.read_csv(FIX / "basis.csv"), "fixture")


def simulate(basis, t_flight, t_ground, n=6, noise=0.0, spread=0.0, seed=0,
             flight_peaks=None, design_cov=False):
    """Samples × genes log2 matrix and metadata for two arms at given clock times."""
    rng = np.random.default_rng(seed)
    rows, meta = [], []
    for arm, t0 in (("flight", t_flight), ("ground", t_ground)):
        peaks = flight_peaks if (arm == "flight" and flight_peaks is not None) else basis.peak
        for j in range(n):
            t = t0 + rng.normal(0, spread) if spread else t0
            rows.append(9 + basis.amplitude * np.cos(W * (t - peaks))
                        + rng.normal(0, noise, basis.k))
            meta.append({"condition": arm, "study": f"S{j % 2}"})
    md = pd.DataFrame(meta, index=[f"s{i}" for i in range(len(meta))])
    return np.array(rows), md


# ----------------------------------------------------------------- why the plane, not z-scores

def test_zscored_template_matching_puts_arms_12h_apart_whatever_the_shift(basis):
    grid = np.arange(0, 24, 0.1)
    T = np.cos(W * (grid[:, None] - basis.peak[None]))
    for shift in (1.0, 6.0):
        Y, md = simulate(basis, 3 + shift, 3, n=4, noise=0.02, seed=1)
        Z = (Y - Y.mean(0)) / Y.std(0)
        ph = grid[np.argmax(Z @ T.T, axis=1)]
        mean = lambda p: np.mod(np.angle(np.exp(1j * W * p).mean()) / W, 24)  # noqa: E731
        gap = abs(circ_diff(mean(ph[:4]), mean(ph[4:])))
        assert gap == pytest.approx(12, abs=1.0), shift


# ----------------------------------------------------------------- the estimator

@pytest.mark.parametrize("tf,tg", [(4, 2), (10, 3), (1, 23), (6, 6.5)])
def test_noiseless_chord_recovers_shift_and_one_reading_is_the_truth(basis, tf, tg):
    Y, md = simulate(basis, tf, tg)
    r = sc.analyse(Y, md, "~ condition", basis, n_boot=20)
    true = abs(circ_diff(tf, tg))
    assert r["delta_h"] == pytest.approx(true, abs=1e-6)
    assert r["r2_plane"] == pytest.approx(1.0)
    assert r["identifiability"] == "identifiable"
    hit = [x for x in r["readings"]
           if abs(circ_diff(x["flight_ct"], tf)) < 1e-4 and abs(circ_diff(x["ground_ct"], tg))
           < 1e-4]
    assert len(hit) == 1
    other = [x for x in r["readings"] if x is not hit[0]][0]
    assert abs(circ_diff(other["mid_ct"], hit[0]["mid_ct"])) == pytest.approx(12)


def test_light_phase_flag(basis):
    Y, md = simulate(basis, 5, 2)
    r = sc.analyse(Y, md, "~ condition", basis, n_boot=20)
    flags = {x["direction"]: x["light_phase_consistent"] for x in r["readings"]}
    assert flags == {"flight_later": True, "flight_earlier": False}


def test_chord_beyond_the_circle_is_not_identifiable_not_clipped(basis):
    Y, md = simulate(basis, 12, 0)  # L = 2 exactly ...
    Y[:6] = 9 + 1.5 * (Y[:6] - 9)  # ... and flight amplitude 1.5× pushes it past 2
    r = sc.analyse(Y, md, "~ condition", basis, n_boot=20)
    assert r["L"] > 2
    assert r["delta_h"] is None and r["identifiability"] == "not_identifiable"
    assert all(x["flight_ct"] is None for x in r["readings"])
    sc.validate_clock_summary(_wrap(r))


def test_amplitude_sensitivity_brackets_the_estimate(basis):
    Y, md = simulate(basis, 5, 2, noise=0.05, seed=3)
    r = sc.analyse(Y, md, "~ condition", basis, n_boot=50)
    assert r["delta_h_amplitude_x1.25"] < r["delta_h"] < r["delta_h_amplitude_x0.75"]
    assert r["delta_h"] == pytest.approx(sc.delta_hours(r["L"]))


def test_scrambled_flight_clock_leaves_the_plane(basis):
    rng = np.random.default_rng(4)
    scrambled = rng.permutation(basis.peak) + rng.normal(0, 3, basis.k)
    Y, md = simulate(basis, 4, 4, flight_peaks=scrambled)
    r = sc.analyse(Y, md, "~ condition", basis, n_boot=20)
    Y2, md2 = simulate(basis, 7, 4)
    r2 = sc.analyse(Y2, md2, "~ condition", basis, n_boot=20)
    assert r["r2_plane"] < 0.8 < r2["r2_plane"]


def test_covariate_adjustment_removes_a_study_offset(basis):
    Y, md = simulate(basis, 5, 3, n=6)
    Y_off = Y + np.where(md["study"].to_numpy() == "S1", 0.7, 0.0)[:, None] * np.linspace(
        -1, 1, basis.k)
    r = sc.analyse(Y_off, md, "~ study + condition", basis, n_boot=20)
    assert r["delta_h"] == pytest.approx(2.0, abs=1e-6)


# ----------------------------------------------------------------- permutations and bootstrap

def test_block_permutations_are_exact_and_respect_blocks():
    cond = np.array([1, 1, 0, 0, 1, 0, 0, 1.0])
    blocks = np.array(["a"] * 4 + ["b"] * 4)
    perms, exact = sc.block_permutations(cond, blocks, np.random.default_rng(0))
    assert exact and len(perms) == math.comb(4, 2) ** 2
    assert np.all(perms[:, :4].sum(1) == 2) and np.all(perms[:, 4:].sum(1) == 2)
    assert any(np.array_equal(p, cond) for p in perms)
    assert len({tuple(p) for p in perms}) == len(perms)


def test_permuted_differences_match_direct_least_squares():
    rng = np.random.default_rng(5)
    md = pd.DataFrame({"condition": ["flight", "ground"] * 5,
                       "study": ["A"] * 4 + ["B"] * 6})
    Y = rng.normal(size=(10, 4))
    X, c, blocks = sc.design_matrix(md, "~ study + condition")
    perms, _ = sc.block_permutations(X[:, c], blocks, rng)
    D = sc.permuted_differences(Y, X, c, perms[:5])
    for p, d in zip(perms[:5], D, strict=True):
        Xp = X.copy()
        Xp[:, c] = p
        assert np.allclose(sc.ols_difference(Y, Xp, c)[0], d)


def test_bootstrap_keeps_arm_and_block_sizes():
    cond = np.array([1, 1, 1, 0, 0, 0, 1, 0.0])
    blocks = np.array(["a"] * 6 + ["b"] * 2)
    idx = sc.bootstrap_indices(cond, blocks, 50, np.random.default_rng(0))
    for i in idx:
        assert cond[i].sum() == cond.sum()
        assert (blocks[i] == "b").sum() == 2


def test_permutation_p_is_not_anticonservative_under_the_null(basis):
    ps = []
    for seed in range(60):
        Y, md = simulate(basis, 6, 6, n=5, noise=0.3, seed=seed)
        ps.append(sc.analyse(Y, md, "~ condition", basis, n_boot=10, seed=seed)["p_perm_L"])
    assert np.mean(np.array(ps) <= 0.05) <= 0.12  # 252 exact relabellings each


def test_size_factors_follow_library_size():
    counts = pd.DataFrame({"s1": [10, 20, 30, 40], "s2": [20, 40, 60, 80], "s3": [5, 0, 15, 20]})
    sf = sc.size_factors(counts)
    assert sf[1] / sf[0] == pytest.approx(2.0)


# ----------------------------------------------------------------- the federation boundary

def _wrap(result):
    return {"client": "SITE-X", "site": "FAKE", "stratum": "iss", "n_flight": 6, "n_ground": 6,
            "design": "~ condition",
            "panel_used": [{"ncbi_gene": i, "symbol": f"G{i}"} for i in range(result["k"])],
            "panel_dropped": [], "result": result, "settings": {}, "versions": {}}


def test_site_x_runs_on_its_own_data_and_only_aggregates_leave(gmap, basis):
    global _OPENED
    _OPENED = []
    try:
        site = load_site("SITE-X", FIX / "manifests", FIX / "data")
        s = sc.site_clock_summary(site, gmap, basis, n_boot=50)
        opened = list(_OPENED)
    finally:
        _OPENED = None
    ours = {p for p in opened if p.startswith(str(FIX.resolve()))}
    assert ours == {str((FIX / "manifests" / "SITE-X.json").resolve()),
                    str((FIX / "data" / "OSD-FAKE-9" / "GLDS-FAKE-9_rna_seq_RSEM_Unnormalized_"
                         "Counts_GLbulkRNAseq.csv").resolve())}
    text = json.dumps(s)
    assert not any(name in text for name in site.counts.columns)
    assert not any(name in text for name in ("Y_FLT", "Y_GC", "SITE-Y"))
    assert s["n_flight"] == 4 and s["n_ground"] == 4 and s["result"]["k"] == 6
    # 4 h vs 2 h with ±0.3 h jitter: the estimate is near 2 h.
    assert s["result"]["delta_h"] == pytest.approx(2.0, abs=0.6)


def test_validator_rejects_extra_fields_and_per_sample_payloads(basis):
    Y, md = simulate(basis, 5, 2)
    good = _wrap(sc.analyse(Y, md, "~ condition", basis, n_boot=20))
    sc.validate_clock_summary(good)
    bad = json.loads(json.dumps(good))
    bad["phases"] = [1.0, 2.0]
    with pytest.raises(ValueError, match="extra"):
        sc.validate_clock_summary(bad)
    bad = json.loads(json.dumps(good))
    bad["result"]["per_sample_phase"] = [1.0] * 12
    with pytest.raises(ValueError, match="extra"):
        sc.validate_clock_summary(bad)
    bad = json.loads(json.dumps(good))
    bad["result"]["readings"][0]["samples"] = ["s1"]
    with pytest.raises(ValueError, match="readings"):
        sc.validate_clock_summary(bad)


def test_absolute_phase_recovers_known_times_with_a_library_offset(basis):
    Y, md = simulate(basis, 7.5, 3.0, n=3)
    Y = Y - 9 + basis.mesor + 0.8  # a global offset, as between two normalizations
    est = sc.absolute_phase(Y, basis)
    assert np.allclose(est["phase"][:3], 7.5) and np.allclose(est["phase"][3:], 3.0)
    assert np.allclose(est["scale"], 1.0) and np.allclose(est["offset"], 0.8)
    assert np.allclose(est["template_r"], 1.0)


def test_absolute_stages_send_only_arm_aggregates(gmap, basis):
    site = load_site("SITE-X", FIX / "manifests", FIX / "data")
    g = sc.site_absolute_summary(site, gmap, basis, "ground")
    assert [a["arm"] for a in g["arms"]] == ["ground"]
    assert abs(circ_diff(g["arms"][0]["circ_mean_ct"], 2.0)) < 0.5
    both = sc.site_absolute_summary(site, gmap, basis, "both")
    assert abs(circ_diff(both["arms"][1]["circ_mean_ct"], 4.0)) < 0.5
    assert not any(n in json.dumps(both) for n in site.counts.columns)
    bad = json.loads(json.dumps(g))
    bad["arms"].append(dict(bad["arms"][0], arm="flight"))
    with pytest.raises(ValueError, match="stage"):
        sc.validate_absolute_summary(bad)
    bad = json.loads(json.dumps(g))
    bad["arms"][0]["phases"] = [1.0, 2.0]
    with pytest.raises(ValueError, match="keys"):
        sc.validate_absolute_summary(bad)
