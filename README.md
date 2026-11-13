# fedkg-spacebio — Federated KG Agent for Space Biology

An LLM agent that coordinates federated analysis of NASA OSDR space biology data across sites that can't pool their data. It uses knowledge graphs (SPOKE-GeneLab, Monarch) for shared semantics, NVIDIA FLARE for training, and site policies that the agent enforces before every job.

> **Status: early implementation.** Done so far: an OSDR liver census and site design, a FedAvg reference check against FLARE, and a federated differential-expression meta-analysis of mouse liver on FLARE (phase 3a, [`docs/phase3a-liver-meta.md`](docs/phase3a-liver-meta.md)). The agent, policy layer, ontology/ortholog harmonization and the federated classifiers are not built yet. Sections marked **TODO** are filled in as the work lands. Results are reported as they come out, including nulls and cases where federation does not beat a baseline.

---

## Why this project

Space biology data is scattered across missions, agencies and labs. It is small-n (few animals per mission), high-dimensional (tens of thousands of genes), and mostly from mice and rats. Some of it can't move freely: individual-level astronaut data in OSDR is controlled-access, and the datasets are large relative to Earth–space links.

Two lines of work set up the gap this repo targets:

- **The compute layer is proven.** NASA's FLUID work ran federated learning between Earth and the International Space Station (Casaletto et al., 2025), built on OpenFL with file-based "shim" relays for intermittent links.
- **The coordination layer isn't.** Recent work on AI-ready space life-sciences data proposes knowledge graphs as the semantic layer, agentic orchestration over MCP servers, and federated learning, but notes that governance and orchestration for a multi-agency or commercial federation have yet to be built.

Existing knowledge-graph agent tools (e.g., `mcp-genelab`) query open, centralized graphs. This project asks what an agent needs to do when the data must stay where it is: harmonize each site to a shared vocabulary, check what each site allows, run and monitor federated jobs, and explain the results with citations back to the graph.

## What it builds

```
                ┌──────────────────────────────────────────┐
  question ───► │  Agent                                   │ ───► grounded answer
                │  plan → check sites → submit → explain   │      (every claim cites
                └───┬───────────┬───────────┬──────────┬───┘       a graph ID or PMID)
                    │           │           │          │
             KG query      site registry  policy    FLARE job
           (mcp-genelab,   + manifests    check     submit / monitor
            Monarch)                                    │
                                                        ▼
                         ┌──────────┐  ┌──────────┐  ┌──────────┐
                         │ Site A   │  │ Site B   │  │ Site C   │  ← simulated sites over
                         │ harmonize│  │ harmonize│  │ harmonize│    OSDR datasets; data
                         │ + train  │  │ + train  │  │ + train  │    never leaves a site
                         └──────────┘  └──────────┘  └──────────┘
```

1. **Census and site design.** Enumerate OSDR rodent RNA-seq studies that contain both spaceflight and ground-control samples, then partition them into simulated sites (by mission, agency or lab), with the rationale written down.
2. **Local harmonization.** Each site maps its metadata to shared ontologies (CL, UBERON, Mondo, HPO), harmonizes gene IDs, and maps rodent genes to human orthologs through SPOKE-GeneLab or Monarch. Every step logs its drop rate, and mapping accuracy is evaluated against a hand-labeled gold set.
3. **Federated training.** Spaceflight vs. ground control from gene expression, run three ways: centralized (pooled), local-only (per site), and federated (FedAvg). The models are regularized logistic regression, then federated XGBoost, with a GNN over aligned subgraphs as a stretch goal.
4. **Agent and governance layer.** Each site declares a policy: which analyses it allows, which outputs may leave, a minimum aggregation size, and whether differential privacy is required. The agent checks every job against those policies before submitting it and logs the decision. Every claim it makes must cite a graph edge, node or PMID that appeared in its own tool results for that run.
5. **Stretch: FLUID-style intermittent links on FLARE.** One "orbital" site exchanges model updates through store-and-forward files under simulated loss-of-signal windows. The goal is to measure how federated training degrades and how the agent schedules around contact windows. This adapts FLUID's shim design (Casaletto et al., built on OpenFL) to FLARE, with credit.

## Evaluation

- **Model performance.** Centralized vs. local-only vs. federated, on fixed shared out-of-fold splits and **leave-one-site-out**. Normalization and feature selection are fit inside training folds only.
- **Harmonization.** Ontology and gene-ID mapping accuracy against a gold set; drop rate per site.
- **Grounding.** Supported-claim rate and fabricated-ID rate (target: zero).
- **Governance.** A test suite of jobs that should be allowed or refused under each site's policy.
- **Stretch.** Accuracy vs. rounds and vs. wall-clock time under different loss-of-signal schedules.

**Phase 3a (liver, federated DE meta-analysis):** 7 simulated sites in two strata (ISS-euthanized, k = 4; Earth-euthanized, k = 3). The pre-specified random-effects headline (REML + truncated Hartung–Knapp) found no gene at FDR < 0.05 in either stratum, which is expected with 3–4 sites. After seeing that, the headline was changed to fixed effect with a per-gene heterogeneity flag: 1,926 (ISS) and 1,722 (Earth) genes at FDR < 0.05, of which 971 and 870 show no detectable heterogeneity. Federated pooling reproduced central pooling exactly. Details, checks and limitations: [`docs/phase3a-liver-meta.md`](docs/phase3a-liver-meta.md). A follow-up comparison of empirical-Bayes random-effects methods against these (a simulation calibrated to the data, plus the real data), and a time-of-day check, are in [`docs/phase3a-method-comparison.md`](docs/phase3a-method-comparison.md); the headline method is still under review. No biological claims are made.

**TODO — classifier results tables and figures.**

## Data and knowledge sources

All data and graph releases are downloaded locally and **never committed**. Versions and access dates are recorded here as they're pulled.

| Source | Use | Access | Version / date |
|---|---|---|---|
| NASA Open Science Data Repository (OSDR), GeneLab | Spaceflight and ground-control omics + metadata | https://osdr.nasa.gov/ · Biological Data API: https://visualization.osdr.nasa.gov/biodata/api/ | Biological Data API v2; liver studies accessed 2026-11-12 (GeneLab processing GL-DPPD-7101-G, GRCm39, Ensembl 112; see [`docs/census-liver.md`](docs/census-liver.md)) |
| GeneLab reference annotations | Ensembl → NCBI Gene mapping for phase 3a | GL-DPPD-7110-A, Mus musculus Ensembl 112: `Mus_musculus.GRCm39.112-GL-annotations.tsv` (figshare file 48354460; https://github.com/nasa/GeneLab_Data_Processing) | Built 2024-08-11 (org.Mm.eg.db 3.19.1); downloaded 2026-11-13; SHA-256 `123a08990f1982b266115852d720e8d0482af6e873f8eca45c8449d3f54044e2` |
| Hughes et al. 2009, PLoS Genet 5:e1000442, Table S1 | Mouse-liver 24 h cycling probe sets for the phase 3a time-of-day check (Q-11) | https://journals.plos.org/plosgenetics/article/file?type=supplementary&id=10.1371/journal.pgen.1000442.s012 (CC-BY) | Downloaded 2026-11-13; `.xls` SHA-256 `8c213667b429af7884cfe5f91733a3223db56b05986d48cea0b03dbfdf68718c` |
| NCBI GEO GPL1261 annotation | Affymetrix Mouse 430 2.0 probe set → NCBI Gene, for the list above | https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL1nnn/GPL1261/annot/GPL1261.annot.gz | Annotation date 2016-08-09; downloaded 2026-11-13; SHA-256 `394c103cf4ca99500d06369e4fb2b9b9bffc2c10fd3cbeb0b0d64c51c3fc95f8` |
| SPOKE-GeneLab | Links OSDR experiments to a biomedical KG; human orthologs; CL/UBERON mapping | https://github.com/BaranziniLab/spoke_genelab | TODO |
| mcp-genelab | Natural-language → Cypher over the GeneLab KG | Hosted: `https://visualization.osdr.nasa.gov/kg/mcp` · https://github.com/sbl-sdsc/mcp-genelab | Assumed KG v0.3.1+20260311 (not exposed live; see [`docs/kg-schema.md`](docs/kg-schema.md)) |
| Monarch KG | Cross-species genes–phenotypes–diseases | https://data.monarchinitiative.org/monarch-kg | TODO (pinned release) |
| PrimeKG (optional) | Disease and drug context, shares Mondo IDs with Monarch | https://github.com/mims-harvard/PrimeKG | TODO |

**Selected OSD datasets and site assignments:** liver census done for OSD-47, -48, -137, -168, -173, -245, -379, -463 and -686 ([`docs/census-liver.md`](docs/census-liver.md)); the liver sites and ISS/Earth strata in [`SITES.md`](SITES.md) were accepted on 2026-11-12. Rebuild in WSL with `uv run python -m fedkg_spacebio.census.build`. Phase 3a used GeneLab's RSEM unnormalized count matrices (standard and rRNA-removed), SampleTables, contrasts and full differential-expression tables for OSD-47, -48, -137, -245, -379 and -463 (accessed 2026-11-12); no FASTQ or BAM files.

Each source has its own license and terms of use, which apply to the data independently of this repository's code license. Check each source before redistributing anything derived from it.

## Tech stack

- Python, managed with [uv](https://docs.astral.sh/uv/)
- [NVIDIA FLARE](https://nvflare.readthedocs.io/) 2.9.x (Job Recipe API, FL simulator)
- scikit-learn, XGBoost; PyTorch / PyTorch Geometric for the GNN stretch goal
- pydeseq2 (site differential expression), numpy/scipy (meta-analysis pooling), matplotlib; statsmodels as an independent comparator (tests and Check 3, dev dependency)
- Model Context Protocol (MCP) servers for knowledge-graph access
- pandas, pytest

## Repository layout

```
fedkg-spacebio/
├── CONTRACTS.md          # IDs, file schemas, tool return shapes, trace format, job spec
├── SITES.md              # site definitions and rationale (from the census)
├── src/fedkg_spacebio/
│   ├── census/           # OSDR enumeration
│   ├── harmonize/        # ontology + gene-ID mapping, ortholog mapping, drop-rate logs
│   ├── sites/            # site registry, manifests, policies
│   ├── evaluation/       # leave-one-site-out splits, leakage guard, metrics
│   ├── federation/       # FLARE job recipes, aggregation, intermittent-link shim (stretch)
│   ├── agent/            # tools, planner, grounding checker, trace logging
│   └── policy/           # policy-as-code evaluation
├── jobs/                 # FLARE job definitions
├── tests/                # pytest, against committed synthetic fixtures
│   └── fixtures/
├── docs/methodology.md   # choices, alternatives, evaluation design
└── notebooks/
```

## Setup

NVIDIA FLARE supports Linux and macOS, not native Windows. On Windows, run the project inside
**WSL2 (Ubuntu)**. Python 3.12 (pinned in `.python-version`; uv downloads it) and
`nvflare[sklearn]` 2.9.x are set in `pyproject.toml` and locked in `uv.lock`.

```bash
# inside WSL2 (Ubuntu), from the repo root
curl -LsSf https://astral.sh/uv/install.sh | sh   # one-time: install uv
# Keep the venv, tool caches and run outputs on the Linux filesystem: some writes to /mnt/c fail under WSL.
export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/fedkg-spacebio
export PYTEST_ADDOPTS="-o cache_dir=$HOME/.cache/fedkg/pytest" RUFF_CACHE_DIR=$HOME/.cache/fedkg/ruff
export FEDKG_WORKDIR=$HOME/fedkg-work    # FLARE simulator workspaces and run outputs
uv sync                      # creates the venv and installs locked dependencies
uv run pytest                # tests run against synthetic fixtures, no data needed
uv run ruff check .          # lint
```

Copy `.env.example` to `.env` (gitignored) for any API keys. All external API calls are cached.

## Conventions

- **Grounding is the product.** A claim counts as supported only if the ID it cites appeared in the agent's own tool results for that trace. IDs are never fabricated; mocks use obviously fake ones.
- **No label leakage.** Labels never become features, graph edges or prompt text. Folds are fixed and shared; nothing is tuned on test data.
- **Contracts first.** Interfaces are defined in `CONTRACTS.md`, with mocks, before real data arrives.
- **IDs as CURIEs** (`UBERON:`, `CL:`, `MONDO:`, `HP:`, `NCBIGene:`). Expression data is harmonized to NCBI Gene IDs, matching the KG; gene IDs are never joined on symbols across datasets.
- **Honest reporting.** Report it when a model doesn't beat a baseline, and report nulls as nulls.

## Limitations and scope

- **The sites are simulated.** Public OSDR data has no legal reason to stay in place. The federation is a testbed: the differences between studies (batch, platform, protocol) are real, and controlled-access human data motivates the design, but this repo does not claim to solve a real privacy problem.
- **Federated learning is not automatically private.** Model updates can leak information. Any differential-privacy or secure-aggregation results are reported with their accuracy cost.
- **Small n per site.** Few animals per mission means local-only models can be weak and every model can overfit; regularized baselines are reported throughout.
- **Rodent data, human context.** Cross-species claims go through ortholog and phenotype mappings and are only as good as those mappings.
- **The knowledge graph is a snapshot.** mcp-genelab reflects OSDR as of 2026-03-11; later studies have no graph nodes.

## Related work and acknowledgments

- Casaletto JA et al. *Bridging Earth and Space: A Flexible and Resilient Federated Learning Framework Deployed on the International Space Station.* bioRxiv, 2025. doi:10.1101/2025.01.14.633017
- *Building AI-Ready Data Systems for Space Life Sciences, Aerospace Medicine, and Deep Space Exploration.* arXiv:2606.28856, 2026.
- Rose PW et al. *mcp-proto-okn: Natural-language access to open scientific knowledge graphs through the Model Context Protocol.* 2026.
- SPOKE-GeneLab (Baranzini Lab, UCSF), mcp-genelab (SDSC), the Monarch Initiative, and NASA OSDR / GeneLab.

## AI assistance disclosure

Claude Code Opus 5.5 and Fable 5.2 used for coding assitance and audit.

## License

Code is licensed under the Apache License 2.0. See [`LICENSE`](LICENSE). Data and knowledge graphs are not included and remain under their own licenses.
