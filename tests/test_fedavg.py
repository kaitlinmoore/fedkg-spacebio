"""Tests for the hand-written FedAvg (owned core #3) and the seams around it.

We test our own code: fedavg(), the synthetic sites, and the in-process protocol. FLARE's
internals aren't tested here. The optional slow test runs the real FLARE job end to end and
checks that it matches our reference.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fedkg_spacebio.federation.fedavg import fedavg
from fedkg_spacebio.federation.logreg import centralized_gd, run_fedavg_inprocess
from fedkg_spacebio.federation.synthetic import SITE_NAMES, make_all_sites, make_site_data, pool

# ---------------------------------------------------------------- fedavg(): arithmetic


def test_equal_weights_is_plain_mean():
    a, b, c = np.array([1.0, 2.0]), np.array([3.0, 6.0]), np.array([5.0, 10.0])
    np.testing.assert_allclose(fedavg([a, b, c], [7, 7, 7]), np.mean([a, b, c], axis=0))


def test_weighted_case_worked_by_hand():
    # Client A: theta = [0, 10], 3 samples. Client B: theta = [4, 2], 1 sample.
    # By hand: (3·[0, 10] + 1·[4, 2]) / (3 + 1) = [4, 32] / 4 = [1, 8].
    out = fedavg([np.array([0.0, 10.0]), np.array([4.0, 2.0])], [3, 1])
    np.testing.assert_allclose(out, [1.0, 8.0])


def test_weights_only_matter_relatively():
    # Multiplying every weight by the same constant changes nothing: n_k / Σ n is scale-free.
    arrays = [np.array([1.0, -1.0]), np.array([2.0, 5.0])]
    np.testing.assert_allclose(fedavg(arrays, [40, 10]), fedavg(arrays, [4, 1]))


def test_one_client_is_identity():
    theta = np.array([[1.5, -2.0], [0.25, 3.0]])
    out = fedavg([theta], [123])
    np.testing.assert_array_equal(out, theta)
    assert out is not theta  # a new array, not an alias of the input


def test_zero_weight_client_contributes_nothing():
    out = fedavg([np.array([1.0]), np.array([1000.0])], [5, 0])
    np.testing.assert_allclose(out, [1.0])


def test_output_is_float64_even_for_float32_or_int_input():
    out = fedavg([np.array([1, 2], dtype=np.int64), np.array([3, 4], dtype=np.float32)], [1, 1])
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, [2.0, 3.0])


def test_dict_inputs_average_each_key_with_same_weights():
    c1 = {"coef": np.array([0.0, 10.0]), "intercept": np.array([1.0])}
    c2 = {"coef": np.array([4.0, 2.0]), "intercept": np.array([5.0])}
    out = fedavg([c1, c2], [3, 1])
    assert set(out) == {"coef", "intercept"}
    np.testing.assert_allclose(out["coef"], [1.0, 8.0])
    np.testing.assert_allclose(out["intercept"], [2.0])  # (3·1 + 1·5) / 4


# ---------------------------------------------------------------- fedavg(): validation


def test_empty_input_raises():
    with pytest.raises(ValueError, match="at least one"):
        fedavg([], [])


def test_wrong_number_of_weights_raises():
    with pytest.raises(ValueError, match="one per client"):
        fedavg([np.zeros(2), np.zeros(2)], [1])


def test_shape_mismatch_raises_instead_of_broadcasting():
    # (2,) and (1, 2) would broadcast silently in numpy; fedavg must refuse.
    with pytest.raises(ValueError, match="shape"):
        fedavg([np.zeros(2), np.zeros((1, 2))], [1, 1])


@pytest.mark.parametrize("weights", [[0, 0], [0.0, 0.0]])
def test_zero_total_weight_raises(weights):
    with pytest.raises(ValueError, match="total weight must be positive"):
        fedavg([np.zeros(2), np.ones(2)], weights)


def test_negative_weight_raises():
    with pytest.raises(ValueError, match="non-negative"):
        fedavg([np.zeros(2), np.ones(2)], [3, -1])


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_weight_raises(bad):
    with pytest.raises(ValueError, match="finite"):
        fedavg([np.zeros(2), np.ones(2)], [1, bad])


def test_dict_key_mismatch_raises():
    with pytest.raises(ValueError, match="keys"):
        fedavg([{"a": np.zeros(1)}, {"b": np.zeros(1)}], [1, 1])


def test_dict_shape_mismatch_names_the_key():
    with pytest.raises(ValueError, match="coef"):
        fedavg([{"coef": np.zeros(2)}, {"coef": np.zeros(3)}], [1, 1])


def test_mixing_dict_and_array_raises():
    with pytest.raises(TypeError):
        fedavg([{"a": np.zeros(1)}, np.zeros(1)], [1, 1])
    with pytest.raises(TypeError):
        fedavg([np.zeros(1), {"a": np.zeros(1)}], [1, 1])


# ---------------------------------------------------------------- synthetic sites


def test_synthetic_sites_are_deterministic_and_independent():
    a = make_site_data("site-2", seed=0)
    b = make_site_data("site-2", seed=0)
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.y, b.y)
    # A site's data depends only on (seed, site): generating it alone or with the others gives
    # the same samples. That's what lets each FLARE client build only its own data.
    np.testing.assert_array_equal(make_all_sites(seed=0)[1].X, a.X)
    assert not np.array_equal(make_site_data("site-2", seed=1).X, a.X)


def test_synthetic_sites_are_heterogeneous_and_two_class():
    sites = make_all_sites(seed=0)
    assert [s.name for s in sites] == list(SITE_NAMES)
    sizes = [s.n_samples for s in sites]
    assert len(set(sizes)) == len(sizes), "site sizes must differ, or weighting is untestable"
    for s in sites:
        assert set(np.unique(s.y)) == {0.0, 1.0}, f"{s.name} must contain both classes"
    rates = [s.y.mean() for s in sites]
    assert max(rates) - min(rates) > 0.2, "expected label shift between sites"
    means = np.array([s.X.mean(axis=0) for s in sites])
    assert np.ptp(means, axis=0).max() > 0.5, "expected feature shift between sites"


def test_unknown_site_raises():
    with pytest.raises(KeyError):
        make_site_data("site-99")


# ---------------------------------------------------------------- protocol properties


def test_one_local_step_fedavg_equals_centralized_gd():
    # With one full-batch step per round and n_k weights:
    #   Σ_k (n_k/N)(θ − lr·g_k) = θ − lr·Σ_k (n_k/N) g_k = θ − lr·g_pooled.
    sites = make_all_sites(seed=0)
    X, y = pool(sites)
    theta0 = np.zeros(X.shape[1] + 1)
    fed = run_fedavg_inprocess(sites, theta0, num_rounds=15, local_steps=1, lr=0.5)
    central = centralized_gd(X, y, theta0, steps=15, lr=0.5)
    np.testing.assert_allclose(fed, central, rtol=1e-12, atol=1e-14)


def test_weighting_changes_the_result_on_heterogeneous_sites():
    # If this ever fails, the synthetic data is too homogeneous for the FLARE check to catch a
    # weighting bug.
    sites = make_all_sites(seed=0)
    theta0 = np.zeros(sites[0].X.shape[1] + 1)
    kw = dict(num_rounds=20, local_steps=5, lr=0.5)
    by_n = run_fedavg_inprocess(sites, theta0, weighting="samples", **kw)
    equal = run_fedavg_inprocess(sites, theta0, weighting="equal", **kw)
    assert np.max(np.abs(by_n - equal)) > 1e-2


# ---------------------------------------------------------------- FLARE end to end (slow)

REPO = Path(__file__).resolve().parents[1]
JOB_DIR = REPO / "jobs" / "fedavg_check"


@pytest.mark.slow
def test_flare_fedavg_matches_reference():
    """Run the FLARE simulator job, then reference.py, which exits 0 only if they match.

    Runs in about a minute. Skipped by default; run with ``uv run pytest -m slow``. FLARE writes
    to $NVFLARE_SIMULATOR_WORKSPACE_ROOT (or $FEDKG_WORKDIR/flare-sim), never into the repo.
    """
    env = os.environ.copy()
    workdir = Path(env.get("FEDKG_WORKDIR", Path.home() / "fedkg-work"))
    env.setdefault("NVFLARE_SIMULATOR_WORKSPACE_ROOT", str(workdir / "flare-sim"))
    for script, extra in (("job.py", []), ("reference.py", [])):
        proc = subprocess.run(
            [sys.executable, str(JOB_DIR / script), *extra],
            env=env, capture_output=True, text=True, timeout=600,
        )
        tail = f"{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
        assert proc.returncode == 0, f"{script} failed:\n{tail}"
    assert "FLARE (n_k) vs matching reference" in proc.stdout
