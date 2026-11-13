# fedavg_check: FLARE's FedAvg vs. our own `fedavg()`

It trains the same logistic regression on the same 3 synthetic sites two
ways and checks that the final global models agree:

1. **FLARE:** `NumpyFedAvgRecipe` in the FL simulator (`job.py` + `client.py`).
2. **Reference:** the same protocol in one process, aggregated by
   `fedkg_spacebio.federation.fedavg.fedavg` (`reference.py`).

`reference.py` also reports two centralized baselines on the pooled data: gradient descent with
the same step budget, and the unpenalized optimum (sklearn). Method and results are in
`docs/methodology.md` §5. Explanations are inline in the code.

## Run (WSL/Ubuntu)
```bash
fedkg                                                     # sets venv and work-dir env vars
uv run python jobs/fedavg_check/job.py                    # FLARE, clients send n_k   (~30 s)
uv run python jobs/fedavg_check/job.py --weighting equal  # FLARE, clients omit n_k   (~30 s)
uv run python jobs/fedavg_check/reference.py              # compare; exit 0 = PASS
```
Or run all of it as a test: `uv run pytest -m slow`. That runs the `samples` job and the
comparison. The comparison also re-checks the `equal` result if one exists.

## Files
| File | Role |
|---|---|
| `config.py` | Protocol settings (seed, rounds, local steps, lr) and tolerance, shared by both paths |
| `job.py` | Builds the recipe: 3 sites, `min_clients=3`, 20 rounds, zero init. Runs `SimEnv` |
| `client.py` | Per-site loop: generate own data, `receive → local_train → send`, with `NUM_STEPS_CURRENT_ROUND = n_k` |
| `reference.py` | Same protocol without FLARE; loads FLARE's model; prints the comparison; writes JSON |

Shared code in `src/fedkg_spacebio/federation/`:
- `fedavg.py`: the aggregation
- `logreg.py`: `local_train`, loss, and the in-process protocol
- `synthetic.py`: site data

## Outputs (never in the repo)
- FLARE workspaces: `$NVFLARE_SIMULATOR_WORKSPACE_ROOT/fedavg_check{,_equal}/`, with the final
  model at `server/simulate_job/models/server.npy`.
- Report: `$FEDKG_WORKDIR/results/fedavg_check.json`.

## The weighting trap
FLARE weights each client by `meta["NUM_STEPS_CURRENT_ROUND"]`. If a client doesn't send it,
FLARE **silently uses 1.0**, i.e. equal weighting. The `--weighting equal` run shows this: its
result matches our equal-weight reference, not the sample-weighted one.
