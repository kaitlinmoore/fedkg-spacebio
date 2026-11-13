"""FLARE side of the FedAvg check: logistic regression on 3 synthetic sites with NumpyFedAvgRecipe.

Run from WSL after ``fedkg``:

    uv run python jobs/fedavg_check/job.py                      # clients send n_k (FedAvg)
    uv run python jobs/fedavg_check/job.py --weighting equal    # clients omit n_k (demo)
    uv run python jobs/fedavg_check/reference.py                # compare (see README.md)

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

from fedkg_spacebio.federation.synthetic import N_FEATURES, SITE_NAMES  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weighting", choices=["samples", "equal"], default="samples")
    args = parser.parse_args()

    recipe = NumpyFedAvgRecipe(
        name=config.job_name(args.weighting),
        min_clients=len(SITE_NAMES),
        num_rounds=config.NUM_ROUNDS,
        model=[0.0] * (N_FEATURES + 1),  # theta = [w_1..w_d, b], all zeros
        train_script=str(HERE / "client.py"),
        train_args=(
            f"--seed {config.SEED} --lr {config.LR} --local_steps {config.LOCAL_STEPS} "
            f"--weighting {args.weighting}"
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
