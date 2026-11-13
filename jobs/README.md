# FLARE jobs

Job definitions built with FLARE's Job Recipe API (FLARE 2.9.x). Run them in the FL simulator
from WSL:

```bash
uv run python jobs/<job_name>.py
```

Simulator workspaces go under `$NVFLARE_SIMULATOR_WORKSPACE_ROOT` / `$FEDKG_WORKDIR` (set by the `fedkg` shell function; defaults to
`~/fedkg-work` in WSL), not inside the repo. Writes to the Windows drive from WSL can fail.

## Jobs

### `fedavg_check/`: FLARE's FedAvg vs. our own `fedavg()`
Trains logistic regression on 3 synthetic sites with `NumpyFedAvgRecipe` and checks that the
result equals the same protocol aggregated by `fedkg_spacebio.federation.fedavg`.
See `fedavg_check/README.md`.

```bash
uv run python jobs/fedavg_check/job.py                    # FLARE run (clients send n_k)
uv run python jobs/fedavg_check/job.py --weighting equal  # optional: clients omit n_k
uv run python jobs/fedavg_check/reference.py              # comparison; exit 0 = PASS
uv run pytest -m slow                                     # the same, as a test
```

### `phase3a_meta/`: federated DE meta-analysis of liver (phase 3a, WS-07)
One FLARE client per (site, stratum) row of D-020. Each client runs pydeseq2 on its own
samples, maps its per-gene summary to NCBI Gene and sends only that. The server
(`MetaAnalysisController`) pools them: fixed effect, REML random effects and truncated HKSJ
intervals. Headline: FE BH q < 0.05 with a per-gene heterogeneity flag (Q p < 0.10); RE-HKSJ is
secondary. See [`docs/phase3a-liver-meta.md`](../docs/phase3a-liver-meta.md).

```bash
# WSL/Ubuntu, after `fedkg`
uv run python jobs/phase3a_meta/job.py --stratum iss                # FLARE (also writes manifests)
uv run python jobs/phase3a_meta/job.py --stratum earth
uv run python jobs/phase3a_meta/central.py --stratum iss            # Check 1: federated = central
uv run python jobs/phase3a_meta/central.py --stratum earth
uv run python jobs/phase3a_meta/check_genelab.py                    # Check 2: vs GeneLab DE
uv run python jobs/phase3a_meta/check_pooling.py --stratum iss      # Check 3: vs statsmodels
uv run python jobs/phase3a_meta/check_pooling.py --stratum earth
uv run python jobs/phase3a_meta/job.py --stratum iss --counts rrnarm    # sensitivity (+ central.py)
uv run python jobs/phase3a_meta/report.py --stratum iss             # per-gene tables, plots
uv run python jobs/phase3a_meta/report.py --stratum earth           # (repeat with --counts rrnarm)
uv run python jobs/phase3a_meta/report.py --sensitivity             # cross-run comparison
uv run pytest -m slow                                               # synthetic FLARE end to end
```

### `phase3a_compare/`: phase 3a method comparison and time-of-day checks (WS-08)
Not FLARE jobs: every method compared runs at the server on per-site summaries, so these read
WS-07's saved per-gene tables. See `docs/phase3a-method-comparison.md`.

```bash
uv run python jobs/phase3a_compare/simulate.py     # calibrated simulation (100 reps, parallel)
uv run python jobs/phase3a_compare/compare.py      # every method on the real per-gene tables
uv run python jobs/phase3a_compare/circadian.py    # Q-11 rhythmic-gene enrichment, core clock
uv run python jobs/phase3a_compare/report.py       # figures and tables
```
`time_of_day_audit.yaml` is the curated Q-11 metadata audit. Outputs:
`$FEDKG_WORKDIR/results/phase3a/method-comparison/`.

Inputs (phase3a_meta): `$FEDKG_WORKDIR/census/` (WS-06), count matrices and full DE tables in
`$FEDKG_WORKDIR/data/osdr/`, and the GeneLab annotation table in
`$FEDKG_WORKDIR/data/annotations/`. Outputs: `$FEDKG_WORKDIR/results/phase3a/<stratum>/<counts>/`
(`flare/`, `central/`, `check3.json`), `$FEDKG_WORKDIR/results/phase3a/check2/` and
`$FEDKG_WORKDIR/results/phase3a/sensitivity.csv`. The stable per-gene tables are
`flare/<variant>/per_gene.csv.gz` with `per_gene.schema.json` (CONTRACTS.md §4).
