"""FLARE client script for the FedAvg check: one copy runs per site.

AI-drafted (Claude, WS-02). The loop is the standard Client API pattern from FLARE 2.9:
``flare.init()``, then ``receive → train → send`` until the server stops the job.

Federation boundary: this script generates **only its own site's** data, from (seed, site name),
and the only thing it sends back is model parameters, its sample count and its local training
loss.

How FLARE weights this client
-----------------------------
FLARE's FedAvg weights each result by ``meta["NUM_STEPS_CURRENT_ROUND"]`` (times an optional
per-site ``aggregation_weights`` entry, default 1.0). The Client API does not fill that key in.
If it's missing, or not a positive finite number, FLARE silently uses 1.0, which means equal
weighting. So we set it to the site's sample count n_k, the way FLARE's own sklearn executor does.
``--weighting equal`` leaves it out on purpose, to show the silent fallback.
"""

import argparse

import numpy as np
import nvflare.client as flare
from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.np.constants import NPConstants

from fedkg_spacebio.federation.logreg import local_train, log_loss
from fedkg_spacebio.federation.synthetic import make_site_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--local_steps", type=int, required=True)
    parser.add_argument("--weighting", choices=["samples", "equal"], default="samples")
    args = parser.parse_args()

    flare.init()
    site_name = flare.get_site_name()  # "site-1", "site-2", ...
    data = make_site_data(site_name, seed=args.seed)  # this site's data and nothing else
    print(f"{site_name}: n={data.n_samples}, positive rate={data.y.mean():.3f}")

    while flare.is_running():
        global_model = flare.receive()
        received = global_model.params[NPConstants.NUMPY_KEY]
        # Round 0 arrives as float32 (NPModelPersistor casts the initial model). We train in
        # float64, so the reference and FLARE do identical arithmetic from then on.
        print(f"{site_name} round {global_model.current_round}: received dtype {received.dtype}")
        theta_in = np.asarray(received, dtype=np.float64)

        # local_train copies its input, so FLARE's in-process model object is never mutated.
        theta_out = local_train(theta_in, data.X, data.y, lr=args.lr, steps=args.local_steps)

        meta = {}
        if args.weighting == "samples":
            meta[FLMetaKey.NUM_STEPS_CURRENT_ROUND] = data.n_samples  # → FedAvg weight n_k

        flare.send(
            flare.FLModel(
                params={NPConstants.NUMPY_KEY: theta_out},
                params_type=flare.ParamsType.FULL,  # we send weights, not a weight difference
                metrics={"train_loss": log_loss(theta_out, data.X, data.y)},
                current_round=global_model.current_round,
                meta=meta,
            )
        )


if __name__ == "__main__":
    main()
