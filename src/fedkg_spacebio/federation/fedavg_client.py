"""The FedAvg client result, with its sample count enforced (D-017, "now" stage).

Why this module exists
----------------------
FLARE's FedAvg weights each client's update by ``meta["NUM_STEPS_CURRENT_ROUND"]``. The Client
API doesn't fill that key in; the client script has to. If the key is missing or unusable,
FLARE 2.9.1 doesn't complain: ``base_fedavg._get_num_steps_weight`` returns 1.0, so sample
weighting silently turns into equal weighting. WS-02 showed this (FLARE without n_k matched the
equal-weight reference to 6.7e-16).

The cases FLARE turns into 1.0, read from that function in the installed 2.9.1:
1. the key is missing, or its value is ``None``;
2. the value is a bool (``True`` would otherwise count as 1);
3. ``float(value)`` fails (a non-numeric object; a string such as ``"12"`` is also refused
   here, see below);
4. the value is NaN or ±inf;
5. the value is ≤ 0.

``check_n_k`` raises in every one of those cases instead. Every FedAvg client builds its result
through ``fedavg_result``, so a result that FLARE would silently reweight can't be built.

Design choices
--------------
- **Strings are refused, even numeric ones.** FLARE would accept ``"12"`` via ``float()``. A
  sample count that arrives as text means something upstream is wrong (a CSV read without
  dtypes, a CLI argument passed through unparsed). Refusing it costs nothing.
- **Any positive finite real number passes, not only integers.** n_k is normally an integer,
  but D-017 allows a declared custom weight, and ``--weighting equal`` sends an explicit 1.
  numpy integer and float scalars are accepted, since that's what array code produces.
- **The server side** (WS-13): ``federation/fedavg_server.NkCheckedAggregator`` calls
  ``require_n_k`` on every received result and fails the round instead of letting FLARE fall
  back to 1.0 (CONTRACTS §6.2). The policy-side check of the job spec is separate (WS-12).
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType

N_K_KEY = FLMetaKey.NUM_STEPS_CURRENT_ROUND  # the string "NUM_STEPS_CURRENT_ROUND"


def check_n_k(value) -> float:
    """Return ``value`` as a float if it's a usable FedAvg weight; raise ``ValueError`` if not.

    Usable means a real number (not a bool, not a string), finite, and > 0. Those are exactly
    the cases in which FLARE 2.9.1 would use the value instead of silently replacing it with 1.0.
    """
    if value is None:
        raise ValueError(f"{N_K_KEY} is missing (None); FLARE would silently use 1.0 (D-017)")
    # bool is a subclass of int in Python, so it has to be excluded explicitly.
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{N_K_KEY} must be a real number, got {type(value).__name__} "
                         f"{value!r}; FLARE would silently use 1.0 (D-017)")
    weight = float(value)
    if not math.isfinite(weight):
        raise ValueError(f"{N_K_KEY} must be finite, got {value!r}; FLARE would silently "
                         "use 1.0 (D-017)")
    if weight <= 0:
        raise ValueError(f"{N_K_KEY} must be > 0, got {value!r}; FLARE would silently use 1.0 "
                         "(D-017)")
    return weight


def require_n_k(meta: Mapping | None) -> float:
    """The FedAvg weight in a client result's ``meta``; raise ``ValueError`` if absent or unusable.

    Used by the tests and by the server-side check (``fedavg_server.NkCheckedAggregator``).
    """
    if not meta or N_K_KEY not in meta:
        raise ValueError(f"client result has no {N_K_KEY}; FLARE would silently weight it as "
                         "1.0 (D-017)")
    return check_n_k(meta[N_K_KEY])


def fedavg_result(params: dict, n_k, *, current_round: int | None,
                  metrics: dict | None = None,
                  params_type: ParamsType = ParamsType.FULL) -> FLModel:
    """Build a FedAvg client's result with ``NUM_STEPS_CURRENT_ROUND`` = n_k always set.

    ``n_k`` is the client's training sample count for this round, or the explicit weight a job
    declared (``1`` for equal weighting). It's checked with ``check_n_k`` before the result
    exists, so an unusable weight fails on the client, loudly, instead of being reweighted by
    FLARE without a trace.
    """
    weight = check_n_k(n_k)
    # Sent as a plain Python int when it's a count (as FLARE's own sklearn executor sends it),
    # else as a float. Plain types also avoid numpy scalars in FLARE's message serialization.
    value = int(n_k) if isinstance(n_k, numbers.Integral) else weight
    return FLModel(params=params, params_type=params_type, metrics=metrics,
                   current_round=current_round, meta={N_K_KEY: value})


__all__ = ["N_K_KEY", "check_n_k", "fedavg_result", "require_n_k"]
