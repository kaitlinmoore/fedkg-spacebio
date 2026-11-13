"""Numpy logistic regression and the in-process FedAvg protocol used by the FedAvg check.

The FLARE client (``jobs/fedavg_check/client.py``) and the in-process
reference (``jobs/fedavg_check/reference.py``) both call ``local_train`` from here. The local
training is literally the same code on both paths, so any difference between the FLARE result
and the reference has to come from aggregation or transport.

Parameter layout
----------------
One flat float64 vector ``theta = [w_1, ..., w_d, b]``, with the bias **last**. A single array,
not a dict, because FLARE's ``NPModelPersistor`` stores and saves exactly one array under
``"numpy_key"``.

Loss and gradient
-----------------
Mean binary cross-entropy over the n samples, with z = X·w + b and p = sigmoid(z):

    L(theta) = (1/n) Σ_i [ log(1 + e^{z_i}) − y_i · z_i ]
    ∂L/∂w    = (1/n) Xᵀ (p − y)
    ∂L/∂b    = (1/n) Σ_i (p_i − y_i)

Local training is plain full-batch gradient descent: ``theta ← theta − lr · ∇L``, repeated
``steps`` times. No minibatches, no randomness, no regularization: the same input always gives
the same output, so the check comes down to aggregation alone.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from fedkg_spacebio.federation.fedavg import fedavg
from fedkg_spacebio.federation.synthetic import SiteData


def _split(theta: np.ndarray) -> tuple[np.ndarray, float]:
    return theta[:-1], float(theta[-1])


def sigmoid(z: np.ndarray) -> np.ndarray:
    # 0.5·(1 + tanh(z/2)) equals 1/(1 + e^{−z}), but can't overflow for large |z|.
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def predict_proba(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    w, b = _split(theta)
    return sigmoid(X @ w + b)


def log_loss(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    w, b = _split(theta)
    z = X @ w + b
    # log(1 + e^z) computed stably as logaddexp(0, z).
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def gradient(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    residual = predict_proba(theta, X) - y  # (p − y), one entry per sample
    n = X.shape[0]
    grad_w = X.T @ residual / n
    grad_b = residual.sum() / n
    return np.append(grad_w, grad_b)


def local_train(
    theta: np.ndarray, X: np.ndarray, y: np.ndarray, lr: float, steps: int
) -> np.ndarray:
    """``steps`` full-batch gradient-descent steps from ``theta``. Returns a new float64 array."""
    theta = np.asarray(theta, dtype=np.float64).copy()
    for _ in range(steps):
        theta = theta - lr * gradient(theta, X, y)
    return theta


def accuracy(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((predict_proba(theta, X) >= 0.5) == (y == 1.0)))


def run_fedavg_inprocess(
    sites: Sequence[SiteData],
    theta0: np.ndarray,
    num_rounds: int,
    local_steps: int,
    lr: float,
    weighting: str = "samples",
) -> np.ndarray:
    """The FedAvg protocol with no FLARE: what the FLARE job should reproduce.

    Each round: every site starts from the current global theta, runs ``local_steps`` of
    ``local_train`` on its own data, and the server combines the results with our ``fedavg()``.
    ``weighting="samples"`` uses n_k (standard FedAvg). ``weighting="equal"`` uses 1 per site,
    which is what FLARE does when clients don't send a sample count.
    """
    if weighting not in ("samples", "equal"):
        raise ValueError(f"weighting must be 'samples' or 'equal', got {weighting!r}")
    weights = [s.n_samples if weighting == "samples" else 1.0 for s in sites]
    theta = np.asarray(theta0, dtype=np.float64).copy()
    for _ in range(num_rounds):
        updates = [local_train(theta, s.X, s.y, lr, local_steps) for s in sites]
        theta = fedavg(updates, weights)
    return theta


def centralized_gd(
    X: np.ndarray, y: np.ndarray, theta0: np.ndarray, steps: int, lr: float
) -> np.ndarray:
    """Plain gradient descent on pooled data: the budget-matched centralized comparison."""
    return local_train(theta0, X, y, lr, steps)
