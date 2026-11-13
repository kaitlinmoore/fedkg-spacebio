# FLARE jobs

Job definitions built with FLARE's Job Recipe API (FLARE 2.9.x). Run them in the FL simulator
from WSL:

```bash
uv run python jobs/<job_name>.py
```

Simulator workspaces go under `$NVFLARE_SIMULATOR_WORKSPACE_ROOT` / `$FEDKG_WORKDIR` (set by the `fedkg` shell function; defaults to
`~/fedkg-work` in WSL), not inside the repo. Writes to the Windows drive from WSL can fail.

## Jobs

### `fedavg_check/`: FLARE's FedAvg vs. our hand-written `fedavg()`
Trains logistic regression on 3 synthetic sites with `NumpyFedAvgRecipe` and checks that the
result equals the same protocol aggregated by `fedkg_spacebio.federation.fedavg` (owned core #3).
See `fedavg_check/README.md`.

```bash
uv run python jobs/fedavg_check/job.py                    # FLARE run (clients send n_k)
uv run python jobs/fedavg_check/job.py --weighting equal  # optional: clients omit n_k
uv run python jobs/fedavg_check/reference.py              # comparison; exit 0 = PASS
uv run pytest -m slow                                     # the same, as a test
```
