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
uv run python jobs/fedavg_check/reference.py              # compare; exit 0 = PASS
```
Or run all of it as a test: `uv run pytest -m slow`. That runs all three jobs and
`reference.py --require-all`, and requires each FLARE result to match its reference.

## Files
| File | Role |
|---|---|
| `config.py` | Protocol settings (seed, rounds, local steps, lr) and tolerance, shared by both paths |
| `job.py` | Builds the recipe: 3 sites, `min_clients=3`, 20 rounds, zero init. Runs `SimEnv` |
| `client.py` | Per-site loop: generate own data, `receive → local_round → send`. `local_round` builds the result with `fedavg_client.fedavg_result`, so `NUM_STEPS_CURRENT_ROUND` is always set |
| `reference.py` | Same protocol without FLARE; loads FLARE's model; prints the comparison; writes JSON |

Shared code in `src/fedkg_spacebio/federation/`:
- `fedavg.py`: the aggregation
- `fedavg_client.py`: the client result with n_k enforced (D-017)
- `logreg.py`: `local_train`, loss, and the in-process protocol
- `synthetic.py`: site data

## Outputs (never in the repo)
- FLARE workspaces: `$NVFLARE_SIMULATOR_WORKSPACE_ROOT/fedavg_check{,_equal,_omit_n_k}/`, with the final
  model at `server/simulate_job/models/server.npy`.
- Report: `$FEDKG_WORKDIR/results/fedavg_check.json`.

## The weighting trap (D-017)
FLARE weights each client by `meta["NUM_STEPS_CURRENT_ROUND"]`. If a client doesn't send it, or
sends something unusable (None, a bool, a non-number, NaN/inf, ≤ 0), FLARE **silently uses 1.0**,
i.e. equal weighting.

- **Normal runs can't fall into it.** Every result is built by
  `fedkg_spacebio.federation.fedavg_client.fedavg_result`, which refuses those values. Equal
  weighting (`--weighting equal`) is sent as an explicit 1 per client.
- **`--demo_omit_n_k` shows the trap on purpose.** It bypasses the helper, sends no weight and
  prints a warning. Its result matches our equal-weight reference, not the sample-weighted one.
  The slow test asserts exactly that, so if a FLARE upgrade changes the fallback (for example,
  starts raising), the test fails and the D-017 notes get revisited.
- **Not yet:** a server-side check that fails a round when a result lacks n_k (CONTRACTS §6.2).
  It needs a custom aggregator and is planned with the masked-sum server path (WS-13).
