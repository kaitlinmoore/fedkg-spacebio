"""FLARE client script for the FedAvg check: one copy runs per site.

The loop is the standard Client API pattern from FLARE 2.9:
``flare.init()``, then ``receive → train → send`` until the server stops the job. The work of
one round is ``local_round``, a plain function, so the tests can call it without FLARE running.

Federation boundary: this script generates **only its own site's** data, from (seed, site name),
and the only thing it sends back is model parameters, its FedAvg weight and its local training
loss.

How FLARE weights this client (D-017)
-------------------------------------
FLARE's FedAvg weights each result by ``meta["NUM_STEPS_CURRENT_ROUND"]`` (times an optional
per-site ``aggregation_weights`` entry, default 1.0). The Client API does not fill that key in.
If it's missing, or not a positive finite number, FLARE silently uses 1.0. So every result is
built by ``fedkg_spacebio.federation.fedavg_client.fedavg_result``, which always sets the key
and refuses an unusable value:
- ``--weighting samples`` (default): the weight is the site's sample count n_k;
- ``--weighting equal``: the weight is an explicit 1 for every site. Same arithmetic as
  FLARE's fallback, but declared, the way D-017 allows equal weighting.

``--demo_omit_n_k`` is the one exception. It bypasses the helper and leaves the key out on
purpose, to show FLARE's silent fallback (WS-02's finding). It's off by default and warns
loudly. ``tests/test_fedavg.py`` runs it in the slow suite and checks that FLARE still falls
back to equal weighting, so a FLARE upgrade that changes the fallback gets noticed.
"""

import argparse
import sys
import warnings

import numpy as np
import nvflare.client as flare
from nvflare.app_common.np.constants import NPConstants

from fedkg_spacebio.federation.fedavg_client import fedavg_result
from fedkg_spacebio.federation.logreg import local_train, log_loss
from fedkg_spacebio.federation.synthetic import SiteData, make_site_data

OMIT_WARNING = (
    "\n" + "!" * 78 + "\n"
    "!! --demo_omit_n_k: this client sends NO NUM_STEPS_CURRENT_ROUND. FLARE will silently\n"
    "!! weight it as 1.0 (equal weighting). Demonstration only; never use for results (D-017).\n"
    + "!" * 78
)


def local_round(data: SiteData, received: np.ndarray, current_round: int, *, lr: float,
                local_steps: int, weighting: str = "samples",
                demo_omit_n_k: bool = False) -> flare.FLModel:
    """One round on this site: train from the received global model, return the result."""
    # Round 0 arrives as float32 (NPModelPersistor casts the initial model). We train in
    # float64, so the reference and FLARE do identical arithmetic from then on.
    theta_in = np.asarray(received, dtype=np.float64)
    # local_train copies its input, so FLARE's in-process model object is never mutated.
    theta_out = local_train(theta_in, data.X, data.y, lr=lr, steps=local_steps)
    params = {NPConstants.NUMPY_KEY: theta_out}
    metrics = {"train_loss": log_loss(theta_out, data.X, data.y)}

    if demo_omit_n_k:
        warnings.warn(OMIT_WARNING, stacklevel=2)
        # Built by hand on purpose: no weight key at all.
        return flare.FLModel(params=params, params_type=flare.ParamsType.FULL, metrics=metrics,
                             current_round=current_round, meta={})
    if weighting == "samples":
        weight = data.n_samples  # → FedAvg weight n_k
    elif weighting == "equal":
        weight = 1  # declared equal weighting
    else:
        raise ValueError(f"unknown weighting {weighting!r}")
    # We send weights, not a weight difference (params_type FULL, the helper's default).
    return fedavg_result(params, weight, current_round=current_round, metrics=metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--local_steps", type=int, required=True)
    parser.add_argument("--weighting", choices=["samples", "equal"], default="samples")
    parser.add_argument("--demo_omit_n_k", action="store_true",
                        help="demo only: send no weight, so FLARE silently uses 1.0")
    args = parser.parse_args()

    flare.init()
    site_name = flare.get_site_name()  # "site-1", "site-2", ...
    data = make_site_data(site_name, seed=args.seed)  # this site's data and nothing else
    print(f"{site_name}: n={data.n_samples}, positive rate={data.y.mean():.3f}")
    if args.demo_omit_n_k:
        print(OMIT_WARNING, file=sys.stderr, flush=True)

    while flare.is_running():
        global_model = flare.receive()
        received = global_model.params[NPConstants.NUMPY_KEY]
        print(f"{site_name} round {global_model.current_round}: received dtype {received.dtype}")
        flare.send(local_round(data, received, global_model.current_round, lr=args.lr,
                               local_steps=args.local_steps, weighting=args.weighting,
                               demo_omit_n_k=args.demo_omit_n_k))


if __name__ == "__main__":
    main()
