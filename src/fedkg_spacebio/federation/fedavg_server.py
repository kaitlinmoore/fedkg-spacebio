"""Server-side n_k check for FedAvg (D-017, D-036; CONTRACTS §6.2): fail the round, never fall back.

The client side (``fedavg_client.fedavg_result``) already refuses to build a result whose n_k
FLARE would silently replace with 1.0. This is the second half: the **server** checks every
received result with ``require_n_k`` before aggregating it, so a client that bypasses the helper
(``--demo_omit_n_k``, or any other client) makes the round fail instead of turning sample
weighting into equal weighting without a word.

How it plugs into FLARE 2.9.1
-----------------------------
``NumpyFedAvgRecipe(aggregator=...)`` accepts a ``ModelAggregator`` (FLARE's public extension
point for custom aggregation). FLARE's ``FedAvg`` controller calls ``accept_model`` for each
result as it arrives. If that raises, the controller marks the round failed and
``_get_aggregated_result`` refuses to produce a model ("aggregation failed; refusing to update or
save a partial model"), so the job ends with an error rather than a model. Read from the
installed ``nvflare/app_common/workflows/fedavg.py`` (2.9.1); the slow test checks it end to end.

The averaging is still FLARE's
------------------------------
After the check, the result goes to FLARE's own ``WeightedAggregationHelper`` with weight n_k,
the same helper and the same weight the built-in path uses when no aggregator is given
(``aggregation_weights`` default 1.0 × ``NUM_STEPS_CURRENT_ROUND``). So the WS-02 comparison of
FLARE's FedAvg against our own numpy ``fedavg()`` still compares two independent
implementations. Alternative considered: average in our own code inside the aggregator. That
would have made the FLARE-vs-reference check compare our code with itself.

Training metrics (``train_loss``) are averaged the same way, as the built-in path does.
"""

from __future__ import annotations

from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator
from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper

from fedkg_spacebio.federation.fedavg_client import require_n_k


def checked_weight(result: FLModel) -> float:
    """n_k of a received result, or ``ValueError`` naming the client (D-017)."""
    meta = result.meta or {}
    try:
        return require_n_k(meta)
    except ValueError as exc:
        raise ValueError(f"server refuses the result from {meta.get('client_name', '?')}: "
                         f"{exc}; the round fails instead of weighting it as 1.0") from None


class NkCheckedAggregator(ModelAggregator):
    """FedAvg aggregation that refuses any result without a usable n_k."""

    def __init__(self, exclude_vars: str | None = None):
        super().__init__()
        self.exclude_vars = exclude_vars
        self.reset_stats()

    def reset_stats(self):
        self._params = WeightedAggregationHelper(exclude_vars=self.exclude_vars)
        self._metrics = WeightedAggregationHelper()
        self._all_metrics = True
        self._params_type = None
        self._weights: dict[str, float] = {}

    def accept_model(self, model: FLModel):
        weight = checked_weight(model)  # raises → FLARE fails the round
        name = (model.meta or {}).get("client_name", f"client-{len(self._weights)}")
        if self._params_type is None:
            self._params_type = model.params_type
        self._params.add(data=model.params, weight=weight, contributor_name=name,
                         contribution_round=model.current_round)
        if model.metrics is None:
            self._all_metrics = False
        elif self._all_metrics:
            self._metrics.add(data=model.metrics, weight=weight, contributor_name=name,
                              contribution_round=model.current_round)
        self._weights[name] = weight

    def aggregate_model(self) -> FLModel:
        result = FLModel(params=self._params.get_result(), params_type=self._params_type,
                         metrics=(self._metrics.get_result() or None) if self._all_metrics
                         else None,
                         meta={"fedkg_n_k": dict(self._weights)})
        self.reset_stats()
        return result


__all__ = ["NkCheckedAggregator", "checked_weight"]
