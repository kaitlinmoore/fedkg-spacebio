"""FLARE side of the FedAvg check: logistic regression on 3 synthetic sites with NumpyFedAvgRecipe.

Run from WSL after ``fedkg``:

    uv run python jobs/fedavg_check/job.py                      # clients send n_k (FedAvg)
    uv run python jobs/fedavg_check/job.py --weighting equal    # clients send an explicit 1
    uv run python jobs/fedavg_check/job.py --demo_omit_n_k      # demo: clients send no weight
    uv run python jobs/fedavg_check/job.py --demo_omit_n_k --server_check   # ...and it's refused
    uv run python jobs/fedavg_check/reference.py                # compare (see README.md)

Server-side n_k check (D-036): by default the server aggregates through
``fedkg_spacebio.federation.fedavg_server.NkCheckedAggregator``, which calls ``require_n_k`` on
every result and fails the round on a missing or unusable n_k. The averaging itself is still
FLARE's ``WeightedAggregationHelper``, so the comparison with ``reference.py`` still compares
FLARE's arithmetic with ours. ``--demo_omit_n_k`` on its own runs **stock** FLARE aggregation on
purpose: it is the canary that shows FLARE's silent fallback to 1.0. With ``--server_check`` too,
the same clients meet our server, and the job fails.

What the recipe wires up (server side):
- a FedAvg controller that runs NUM_ROUNDS rounds. ``min_clients=3`` means every round waits for
  all three sites, matching the reference, which always uses all sites;
- an NPModelPersistor holding the global model, a single vector of zeros. The persistor casts it
  to float32 on load; zeros are exact in float32, so FLARE and the reference start from the same
  point. Clients train in float64 and send float64, so later rounds stay in float64;
- a script runner that runs client.py in-process for each site, with the protocol settings as
  command-line arguments.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # so `import config` works however this script is launched

import config  # noqa: E402
from nvflare.app_common.np.recipes.fedavg import NumpyFedAvgRecipe  # noqa: E402
from nvflare.recipe import SimEnv  # noqa: E402

from fedkg_spacebio.federation.fedavg_server import NkCheckedAggregator  # noqa: E402
from fedkg_spacebio.federation.synthetic import N_FEATURES, SITE_NAMES  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weighting", choices=["samples", "equal"], default="samples")
    parser.add_argument("--demo_omit_n_k", action="store_true",
                        help="demo only: clients send no weight, so FLARE silently uses 1.0")
    parser.add_argument("--server_check", action="store_true",
                        help="with --demo_omit_n_k: use the n_k-checking server (job must fail)")
    args = parser.parse_args()
    if args.demo_omit_n_k and args.weighting != "samples":
        parser.error("--demo_omit_n_k sends no weight at all; don't combine it with --weighting")
    if args.server_check and not args.demo_omit_n_k:
        parser.error("the server check is already on by default; --server_check only changes "
                     "--demo_omit_n_k runs")
    mode = "omit_n_k" if args.demo_omit_n_k else args.weighting
    # Stock FLARE aggregation only for the canary; every other run has the server check.
    checked = not args.demo_omit_n_k or args.server_check

    recipe = NumpyFedAvgRecipe(
        name=config.job_name(mode) + ("_server_checked" if args.demo_omit_n_k and checked
                                      else ""),
        aggregator=NkCheckedAggregator() if checked else None,
        min_clients=len(SITE_NAMES),
        num_rounds=config.NUM_ROUNDS,
        model=[0.0] * (N_FEATURES + 1),  # theta = [w_1..w_d, b], all zeros
        train_script=str(HERE / "client.py"),
        train_args=(
            f"--seed {config.SEED} --lr {config.LR} --local_steps {config.LOCAL_STEPS} "
            f"--weighting {args.weighting}" + (" --demo_omit_n_k" if args.demo_omit_n_k else "")
        ),
        # Our clients report train_loss, not accuracy. Point best-model tracking at it so the
        # controller doesn't warn every round. Aggregation is unaffected.
        key_metric="train_loss",
        key_metric_mode="min",
    )
    env = SimEnv(clients=list(SITE_NAMES), workspace_root=str(config.sim_workspace_root()))
    run = recipe.execute(env)
    result = run.get_result()
    status = run.get_status()
    print()
    print("Result workspace:", result)
    print("Global model:    ", Path(result) / "server/simulate_job/models/server.npy")
    print("Job status:      ", status)
    if "COMPLETED" not in str(status):
        sys.exit(1)


if __name__ == "__main__":
    main()
