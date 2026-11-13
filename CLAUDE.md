# CLAUDE.md — fedkg-spacebio (Federated KG Agent for Space Biology)

## What this is
An LLM agent that coordinates federated analysis of NASA OSDR space biology data across
simulated sites that can't pool their data. Knowledge graphs (SPOKE-GeneLab via mcp-genelab,
Monarch) provide shared semantics, NVIDIA FLARE runs the federated training, and per-site
policies are enforced by the agent before every job. Stretch: a FLUID-style store-and-forward
layer on FLARE for intermittent "orbital" links.

## Read first
- `reference/` is **local-only and gitignored**. Read it for background, but never commit it,
  quote it in public files, or copy its personal details (contacts, applications) into the repo.
  - `reference/HANDOFF.md`: the idea, prior work, takeaways (its open-decisions list is
    superseded by `DECISIONS.md`)
  - `reference/RESOURCES.md`: papers, data, KGs, MCP servers, FL frameworks
  - `reference/OUTLINE.md`: phased plan and concerns
  - `reference/DECISIONS.md`: decision log and open questions. Check it before proposing
    anything that touches a recorded decision. When Kaitlin makes a decision in the executive
    chat, add a dated entry (newest first); to reverse one, add a superseding entry rather than
    editing the old one.
  - `reference/WORKSTREAMS.md` and `reference/handoffs/`: how work is split. **If you were started
    for a specific workstream, read its handoff first, stay inside its scope, propose (don't record)
    decisions, and finish by writing `handoffs/WS-NN_return.md` and updating the workstream's row.**
    Only the executive chat, with Kaitlin, writes to `DECISIONS.md`.
  - `reference/learning/`: Kaitlin's learning notes. When a session teaches something
    (a walkthrough, an example run, a derivation), write it here as a file, with exact commands to
    rebuild any throwaway environment. Don't leave it only in the conversation. Polished pieces
    can be promoted to `docs/`.
  - `reference/literature/`: papers (PDFs). **Start from `reference/literature/INDEX.md`.**
    Don't open a PDF unless the index lacks a detail you need; if you do open one, add that
    detail to the paper's entry. If a PDF has no entry yet, read it once and add one using the
    template at the bottom of the index.
- `README.md` is the public description. Keep it consistent with the code; never claim more than exists.

## Standing principles (do not violate)
- **Defensible without AI assistance.** The owned core (below) must be something Kaitlin can
  rebuild from scratch and explain at a whiteboard. AI-assisted code is fine, but every
  non-trivial piece comes with a plain-language explanation in comments or `docs/methodology.md`.
  She wants to learn as we work, so explain choices, not just results.
- **Explain as you build.** For each transform, model, metric and protocol choice, record the
  alternatives considered and why this one won.
- **Honest framing.** Separate what was implemented from what a library does. Flag AI-assisted
  sections. Report nulls as nulls and say so when federated doesn't beat a baseline.
- **Plans before execution.** Propose a plan and wait for approval before large changes,
  restructures, cuts or deleting work. Kaitlin decides.
- **No time budgeting or feasibility scoping** unless she asks for it.
- **Tests where logic is testable.** pytest against committed synthetic fixtures. Test the seams
  we own, not FLARE/sklearn/xgboost internals.

## Owned core (whiteboard-defensible)
1. Ontology and gene-ID harmonization, including ortholog mapping and drop-rate accounting.
2. The leave-one-site-out split and its leakage guard.
3. FedAvg aggregation: implement once by hand (numpy) and check it against FLARE's result.
4. Policy evaluation: given a job spec and site policies, allow/deny with a reason.
5. The grounding checker: is every cited ID present in this trace's tool results?
6. (Stretch) The store-and-forward shim and its loss-of-signal model.

## Grounding and agent rules
- **Grounding is the product.** A claim is supported only if the ID it cites (graph edge, node,
  PMID) appeared in the agent's own tool results **for that trace**. Never fabricate IDs.
  Mocks use obviously fake IDs (e.g., `FAKE:0001`).
- Every agent run writes a trace in the format defined in `CONTRACTS.md`.
- The agent must make real decisions (plan, check site compatibility, enforce policy, schedule)
  rather than only wrap FLARE commands. Every decision is logged with its reason.

## Evaluation and leakage rules
- Headline evaluation is **leave-one-site-out** plus fixed, shared out-of-fold splits.
  A sample from a held-out site must NEVER appear in training. `splits.py` enforces this; test it.
- Fit every transform (normalization, gene filtering, feature selection, batch correction)
  **inside training folds only**. The leakage test covers transforms, not just sample membership.
- Labels (flight vs. ground) never become features, graph edges or prompt text.
- Each site must contain both flight and ground-control samples. A site with one arm or one
  tissue lets the model learn the site, not the biology.
- Always compare centralized, local-only and federated. Regularized baselines (logistic/elastic
  net) come first; justify any deeper model against them.
- Never tune on test data.

## ID, ontology and version hygiene
- IDs are CURIEs: `UBERON:`, `CL:`, `MONDO:`, `HP:`, `NCBIGene:` / `ENSEMBL:`, plus OSD accessions.
- Detect and record each dataset's gene-ID namespace and annotation version (GRCm39 vs.
  GRCm38/mm10) from metadata. Do not assume.
- Harmonize to one gene namespace, **NCBI Gene (Entrez)**, before merging (DECISIONS D-012: every
  KG gene node and ortholog edge uses it). Map Ensembl-indexed OSDR matrices once and log the drop rate. Never join on gene
  SYMBOL across datasets.
- Log the drop rate at every mapping step (gene IDs, orthologs, ontology terms). Silent loss is a bug.
- Pin knowledge-graph versions. mcp-genelab is **assumed** to be SPOKE-GeneLab v0.3.1+20260311 (the
  version isn't exposed live; see `docs/kg-schema.md`) (no OSDR data after
  2026-03-11). Keep the census and the KG version aligned. Pin a downloaded Monarch release.

## Federation rules
- Site code reads only its own site's data. The aggregator sees only model updates and allowed
  summaries, never raw site data. Keep that boundary visible in the code structure.
- The sites are **simulated** over public OSDR data. Frame it as a testbed with real
  between-study heterogeneity; never claim it solves an actual privacy problem.
- **FedAvg weighting (D-017):** weight sites by sample count n_k. Every FLARE client must set
  `FLMetaKey.NUM_STEPS_CURRENT_ROUND = n_k`. FLARE 2.9.1 silently treats a missing or invalid
  value as 1.0 (equal weighting). Equal or custom weights only via an explicit job-spec field.
- Federated learning is not automatically private. Any DP or secure-aggregation result is
  reported with its accuracy cost.
- FLUID-style work (phase 3b) adapts the design from Casaletto et al., 2025, and is credited as such.
  FLUID's orchestration scripts are public at nasa/AI4LS/FLUID (no LICENSE file, so treat them as
  reference only) and its shim code is on OpenFL's `fluid` / `fluid_rebase` branches (Apache-2.0).
  Any code adapted from OpenFL keeps its license notice and attribution.

## Governance / policy
- Each site declares a policy (allowed analyses, which outputs may leave, minimum aggregation
  size, DP requirement). Schema lives in `CONTRACTS.md`.
- The agent checks every job against all participating sites' policies **before** submission and
  logs allow/deny with the rule that decided it.
- Maintain a policy test suite of jobs that must be allowed and jobs that must be refused.

## Repo conventions
- **Contracts first:** `CONTRACTS.md` defines IDs, file columns, tool return shapes, trace
  format, site manifest, policy schema and job spec, with mocks so work proceeds without data.
- Keep raw and derived data, models and results out of git. Commit code, configs, synthetic
  fixtures (`tests/fixtures/`) and docs only.
- Cache every external API/KG call (gitignored cache dir). Secrets go in `.env` (gitignored).
- README carries exact data sources, OSD accessions, versions and access dates.
- `docs/methodology.md` explains approach, choices and evaluation.
- `.gitattributes` enforces LF line endings (`* text=auto eol=lf`).
- Don't edit the Patient-in-the-Graph repo (shared team repo). Carry over its conventions, not its code.

## Environment
- **Host:** Windows. The repo lives at `C:\dev\GitHub\fed-space\space-biology`, which is
  `/mnt/c/dev/GitHub/fed-space/space-biology` inside WSL.
- **Run Python and FLARE in WSL2, Ubuntu 24.04.** FLARE supports Linux/macOS, not native Windows.
- **Git runs from Git Bash only.** Never run git from inside WSL on this repo; mixing the two
  causes line-ending and file-mode noise.
- **Nothing generated is written into the repo from WSL.** Writes to `/mnt/c/...` from WSL fail
  for some tools ("Operation not permitted"). Kaitlin's `fedkg` shell function (in `~/.bashrc`) cds
  to the repo and sets: `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/fedkg-spacebio`,
  `PYTEST_ADDOPTS="-o cache_dir=$HOME/.cache/fedkg/pytest"`, `RUFF_CACHE_DIR=$HOME/.cache/fedkg/ruff`,
  `FEDKG_WORKDIR=$HOME/fedkg-work` and `NVFLARE_SIMULATOR_WORKSPACE_ROOT=$HOME/fedkg-work/flare-sim`
  (FLARE's `SimEnv` otherwise defaults to `/tmp/nvflare/simulation`). Never create `.venv` in the repo from WSL. Code that writes
  run outputs (FLARE simulator workspaces, caches, results) must default to `$FEDKG_WORKDIR`, not a
  path inside the repo.
- **uv for everything Python:** `uv add`, `uv sync`, `uv run …`, `uvx …`, `uv pip install …`.
  Never plain `pip`. Python version and deps are pinned in `pyproject.toml` + `uv.lock` (committed).
- **NVIDIA FLARE 2.9.x** (`nvflare[sklearn]`, pinned `>=2.9,<2.10`). Use the Job Recipe API
  (`nvflare.recipe`) and the FL simulator. The older Executor/ModelLearner examples were removed in
  2.7.2, and 2.9 removed `nvflare.fuel.utils.pipe` (including `FilePipe`); don't copy either from
  old tutorials. Check examples against the installed version.
- **Python 3.12** (`.python-version`); src layout, package `fedkg_spacebio`.
- **Network:** Claude's cloud workspace cannot reach OSDR, the KG endpoints or Monarch. Scaffold
  and test against mocks there; real data pulls and KG calls run locally in WSL (or through the
  mcp-genelab connector).
- When giving shell commands, say which shell they're for (Git Bash, WSL/Ubuntu, or PowerShell).

## Commands
```bash
# WSL/Ubuntu: run `fedkg` first (cds to the repo; sets venv, cache and work-dir paths)
uv sync                 # install locked deps into ~/.venvs/fedkg-spacebio
uv run pytest           # fast tests (synthetic fixtures only)
uv run pytest -m slow   # FLARE simulator end-to-end tests (skipped by default; D-018)
uv run ruff check .     # lint
uv add <pkg>            # add a dependency (updates pyproject.toml + uv.lock)
uv run python jobs/<job>.py   # run a FLARE job in the simulator
```
Git (Git Bash only): commit `pyproject.toml` and `uv.lock` together whenever deps change.
