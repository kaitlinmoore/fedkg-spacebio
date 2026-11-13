"""FedAvg aggregation in plain numpy, with a line-by-line walkthrough in the comments.

It is checked against NVIDIA FLARE's built-in FedAvg in ``jobs/fedavg_check/``.

The formula
-----------
K clients each send back a parameter array θ_k after local training, along with the number of
training samples n_k they used. The new global model is the sample-weighted mean:

    θ_global = Σ_k n_k · θ_k / Σ_k n_k          (k = 1..K)

Equivalently, θ_global = Σ_k p_k · θ_k, with p_k = n_k / N and N = Σ_k n_k. The p_k sum to 1, so
the result is a weighted average and stays on the "same scale" as each θ_k.

Why weight by n_k: a client with 400 samples has a gradient estimate 400 samples deep, and one
with 40 has one 40 samples deep. Equal weights would let the small site pull the model as hard
as the big one. With n_k weights, one round of FedAvg with a single full-batch gradient step per
client is exactly one gradient step on the pooled data (see docs/methodology.md §5).

Validation choices (and how FLARE differs)
------------------------------------------
- Empty input, a weights/updates length mismatch, mismatched shapes or mismatched dict keys raise
  ``ValueError``. A silently broadcast or truncated average is the bug we're trying to catch.
- Weights must be finite and non-negative, and must sum to more than 0. A single client weight of
  0 is allowed: that client contributes nothing.
- FLARE does **not** raise on bad weights. In 2.9.1, a missing, non-numeric, non-finite or ≤ 0
  ``NUM_STEPS_CURRENT_ROUND`` silently becomes 1.0 (``base_fedavg._get_num_steps_weight``). This
  module raises instead, so a missing sample count can't quietly turn into equal weighting.

The math runs in float64 whatever the input dtype, so the reference itself adds no rounding error
beyond float64's.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

# A client's parameters: one array, or a dict of named arrays, e.g. {"coef": ..., "bias": ...}.
Params = np.ndarray | Mapping[str, np.ndarray]


def _check_weights(weights: Sequence[float], n_clients: int) -> np.ndarray:
    """Return the weights as a float64 vector, or raise if they can't define a weighted mean."""
    w = np.asarray(weights, dtype=np.float64)  # e.g. [400, 150, 50] → array([400., 150., 50.])
    if w.ndim != 1 or w.shape[0] != n_clients:
        # One weight per client, no more and no less.
        raise ValueError(f"expected {n_clients} weights (one per client), got shape {w.shape}")
    if not np.all(np.isfinite(w)):
        raise ValueError(f"weights must be finite, got {w.tolist()}")
    if np.any(w < 0):
        raise ValueError(f"weights must be non-negative, got {w.tolist()}")
    if w.sum() <= 0:
        # Σ n_k = 0 would mean dividing by zero: nobody contributed anything.
        raise ValueError(f"total weight must be positive, got {w.sum()}")
    return w


def _weighted_mean(arrays: Sequence[np.ndarray], w: np.ndarray, name: str = "params") -> np.ndarray:
    """Σ_k w_k · arrays[k] / Σ_k w_k for same-shaped arrays (w already validated)."""
    # float64 copies, so int or float32 inputs don't change the arithmetic.
    stacked = [np.asarray(a, dtype=np.float64) for a in arrays]
    shape = stacked[0].shape
    for k, a in enumerate(stacked):
        if a.shape != shape:
            # numpy would happily broadcast (3,) against (1, 3). We refuse.
            raise ValueError(f"{name}: client {k} has shape {a.shape}, client 0 has {shape}")
    # Numerator: Σ_k w_k · θ_k, built one client at a time (like FLARE's running total).
    total = np.zeros(shape, dtype=np.float64)
    for w_k, a in zip(w, stacked, strict=True):
        total += w_k * a
    # Denominator: Σ_k w_k. Dividing turns the weighted sum into a weighted mean.
    return total / w.sum()


def fedavg(client_params: Sequence[Params], weights: Sequence[float]) -> Params:
    """Sample-weighted federated average of client parameters.

    Args:
        client_params: one entry per client. Either every entry is a numpy array (all the same
            shape), or every entry is a dict mapping parameter names to arrays (same keys, and
            each key has the same shape across clients).
        weights: one non-negative number per client, normally its training sample count n_k.

    Returns:
        Σ_k n_k · θ_k / Σ_k n_k, as a float64 array or as a dict with the same keys as the inputs.

    Raises:
        ValueError: empty input, wrong number of weights, non-finite / negative / zero-total
            weights, shape mismatch, or dict-key mismatch.
        TypeError: arrays and dicts mixed in one call.

    Example:
        Two clients with 3 and 1 samples:
        ``fedavg([np.array([0.0]), np.array([4.0])], [3, 1])`` gives ``(3·0 + 1·4) / 4 = 1.0``.
    """
    if len(client_params) == 0:
        raise ValueError("fedavg needs at least one client update")
    w = _check_weights(weights, len(client_params))

    first = client_params[0]
    if isinstance(first, Mapping):
        # Dict case: average each named parameter separately, with the same weights.
        if not all(isinstance(p, Mapping) for p in client_params):
            raise TypeError("cannot mix dict and array client parameters")
        keys = set(first)
        for k, p in enumerate(client_params):
            if set(p) != keys:
                # A client missing a layer (or carrying an extra one) is an error, not a skip.
                raise ValueError(
                    f"client {k} has keys {sorted(p)}, client 0 has {sorted(keys)}"
                )
        return {
            key: _weighted_mean([p[key] for p in client_params], w, name=key) for key in first
        }

    # Single-array case.
    if any(isinstance(p, Mapping) for p in client_params):
        raise TypeError("cannot mix dict and array client parameters")
    return _weighted_mean(client_params, w)
