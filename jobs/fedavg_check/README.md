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
uv run python jobs/fedavg_check/job.py                    # FLARE, clients send n_k        (~30 s)
uv run python jobs/fedavg_check/job.py --weighting equal  # FLARE, explicit weight 1 each   (~30 s)
uv run python jobs/fedavg_check/job.py --demo_omit_n_k    # demo: clients send no weight   (~30 s)
uv run python jobs/fedavg_check/job.py --demo_omit_n_k --server_check  # same, our server: fails
uv run python jobs/fedavg_check/reference.py              # compare; exit 0 = PASS
```
Or run all of it as a test: `uv run pytest -m slow`. That runs all three jobs and
`reference.py --require-all`, and requires each FLARE result to match its reference. A second
slow test runs `--demo_omit_n_k --server_check` and requires the job to fail.

## Files
| File | Role |
|---|---|
| `config.py` | Protocol settings (seed, rounds, local steps, lr) and tolerance, shared by both paths |
| `job.py` | Builds the recipe: 3 sites, `min_clients=3`, 20 rounds, zero init, the n_k-checking aggregator (except the canary). Runs `SimEnv` |
| `client.py` | Per-site loop: generate own data, `receive → local_round → send`. `local_round` builds the result with `fedavg_client.fedavg_result`, so `NUM_STEPS_CURRENT_ROUND` is always set |
| `reference.py` | Same protocol without FLARE; loads FLARE's model; prints the comparison; writes JSON |

Shared code in `src/fedkg_spacebio/federation/`:
- `fedavg.py`: the aggregation
- `fedavg_client.py`: the client result with n_k enforced (D-017)
- `fedavg_server.py`: the server-side check (`NkCheckedAggregator`, D-036)
- `logreg.py`: `local_train`, loss, and the in-process protocol
- `synthetic.py`: site data

## Outputs (never in the repo)
- FLARE workspaces: `$NVFLARE_SIMULATOR_WORKSPACE_ROOT/fedavg_check{,_equal,_omit_n_k,_omit_n_k_server_checked}/`, with the final
  model at `server/simulate_job/models/server.npy`.
- Report: `$FEDKG_WORKDIR/results/fedavg_check.json`.

## The weighting trap (D-017)
FLARE weights each client by `meta["NUM_STEPS_CURRENT_ROUND"]`. If a client doesn't send it, or
sends something unusable (None, a bool, a non-number, NaN/inf, ≤ 0), FLARE **silently uses 1.0**,
i.e. equal weighting.

- **Normal runs can't fall into it.** Every result is built by
  `fedkg_spacebio.federation.fedavg_client.fedavg_result`, which refuses those values. Equal
  weighting (`--weighting equal`) is sent as an explicit 1 per client.
- **The server refuses it too** (D-036). Every run except the canary aggregates through
  `fedavg_server.NkCheckedAggregator` (FLARE's `ModelAggregator` extension point). It calls
  `require_n_k` on every result; a bad one makes FLARE mark the round failed and refuse to save a
  model. After the check, the averaging is FLARE's own `WeightedAggregationHelper` with weight
  n_k, the same helper and weight as the built-in path, so the comparison with `reference.py`
  still compares FLARE's arithmetic with ours.
- **`--demo_omit_n_k` shows the trap on purpose.** It bypasses the helper, sends no weight and
  prints a warning, and it runs FLARE's **stock** aggregation. Its result matches our
  equal-weight reference, not the sample-weighted one. The slow test asserts exactly that, so if
  a FLARE upgrade changes the fallback (for example, starts raising), the test fails and the
  D-017 notes get revisited. With `--server_check` the same clients meet our server and the job
  fails ("server refuses the result from site-N … NUM_STEPS_CURRENT_ROUND").
