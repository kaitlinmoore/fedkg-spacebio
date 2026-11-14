# Methodology

Every choice gets recorded here with the alternatives considered and why this one won.
Sections fill in as each phase is built.

## 1. Census and site design

Liver is done (WS-06; numbers in [`census-liver.md`](census-liver.md)). Code:
`src/fedkg_spacebio/census/`, with a walkthrough in each module's docstring.

**Where the census rows come from.** OSDR's own sample metadata, one bulk `query/samples/` call
for all candidate studies. The KG is used only to pick candidates and to attach its comparison
IDs. *Alternative:* start from the KG's comparison nodes (WS-01's approach). *Rejected:* the KG's
labels are damaged for some studies (OSD-168 lacks spike-in labels on one side; labels are cut at
`(`; OSD-245's basal duration differs), so KG-first rows silently lose or mislabel comparisons.

**Groups and comparisons.** Samples with identical factor values form a group; every (flight
group, control group) pair in one assay is a comparison, flight → control only. The strict rule
(D-015) is "every non-spaceflight factor identical". Non-strict rows are kept, with the factors
that differ.

**n means animals.** n counts unique OSDR *source names* (ISA-Tab's subject), not samples.
*Why:* OSD-168 sequences 20 mice twice (±ERCC) and OSD-463 has technical replicates; sample counts
would inflate n by up to 63%.

**Overlap: two independent checks.**
- *Animals:* source names normalized (case, punctuation) and equal strain and sex.
- *Libraries:* identical count vectors, or r > 0.9999 on log1p counts with equal library size
  (Kaitlin's rule). *Why not r alone:* different mice from one tissue reach r ≈ 0.99. On the real
  data, the same mouse re-sequenced in a second study reached only 0.96, so r can't find animals
  either. *Why log1p:* on raw counts, a few very highly expressed liver genes dominate r.

**Which count matrix.** GeneLab publishes DE for both the standard and the rRNA-removed RSEM
matrix. We use the matrix behind the DE table the KG mirrors (Kaitlin's rule). Two checks decide
it:
- each DE table's per-sample block equals its own normalized-count file + 1;
- OSD-47's KG log2FC values equal the standard table's exactly.

Both point to the **standard** matrix.

**Big files are read partially.** HTTP Range reads take a DE table's header (to find `Stat_`) and a
few hundred rows (for the pairing check), instead of downloading up to 525 MB per table.

**Gate (proposed).** Counts published + a strict flight-vs-ground comparison with ≥ 3 independent
animals per arm; ≥ 3 such sites for phase 3a. *Why 3 per arm:* DESeq2's per-gene dispersion
estimate needs replicates within each arm; at 2 per arm it is mostly the fitted trend. *Why 3
sites:* random effects and I² need more than one degree of freedom between sites. Sensitivity
analyses vary the threshold (2–5), the strict rule (strict vs a location-aware variant) and the
stratum (flight euthanized on the ISS vs on Earth).

**Euthanasia location** is recorded per comparison, from the OSDR factor where it exists and
otherwise from quoted protocol text (`census/liver_curation.yaml`). Handling it (stratify vs pool)
is an open proposal; see `census-liver.md`.

## 2. Harmonization
Still to come: ontology mapping, ortholog mapping and gold-set evaluation. Gene-ID mapping for
phase 3a is below (WS-07, 2026-11-12; `src/fedkg_spacebio/harmonize/gene_map.py`).

### 2.1 Ensembl → NCBI Gene for per-gene summaries
**Target:** NCBI Gene (D-012).

**Source:** GeneLab's own annotation table for mouse, Ensembl 112 (GL-DPPD-7110-A,
figshare 48354460, checksum pinned in code). It's the table GeneLab annotated these DE files
with, so our mapping and theirs agree.

**When:** each site runs DE on its native Ensembl IDs and maps the **per-gene summary**, not the
counts.
- *Alternative:* map counts first and sum rows that share an Entrez ID. *Rejected:* summing
  counts of different loci creates a gene that was never measured, and it changes the
  dispersion estimate.

**Which pairs:** one-to-one only, decided over the **whole table**, so every site drops the
same IDs.
- *Alternatives:* keep the first Entrez ID of a one-to-many; keep the highest-expressed of a
  many-to-one. *Rejected:* both are guesses, and the second depends on site data, so sites
  could map differently.

**Accounting:** every Ensembl ID falls into one category (`one_to_one`, `no_entrez`,
`one_to_many`, `many_to_one`, `not_in_table`), counted over the table, over all of a site's
genes, and over its tested genes.
- Whole table: 32,945 of 57,186 Ensembl IDs are one-to-one. Most of the rest (23,910) have no
  Entrez ID; the remainder are 245 one-to-many and 86 many-to-one.
- Per-site drop rates on tested genes are in `docs/phase3a-liver-meta.md`.

## 3. Normalization and features
TODO: method and alternatives; how it is kept fold-safe.

## 4. Splits and leakage guard
TODO: leave-one-site-out and fixed out-of-fold splits; what the leakage tests check.

## 5. Federated training
Still to come: the models used on real OSDR sites, aggregation settings for them, and baselines.
What exists so far is the FedAvg reference check (WS-02, 2026-11-12) below.

### 5.1 FedAvg: what it computes
Each round, the server sends the current global parameters θ to every site. Site k trains locally
for a few steps on its own n_k samples and sends back θ_k. The server sets

    θ_global = Σ_k n_k · θ_k / Σ_k n_k

which is a weighted average of the site models, with weights proportional to sample counts.
Implementation: `src/fedkg_spacebio/federation/fedavg.py`. It takes a single array or a dict of
named arrays, works in float64, and raises on empty input, length or shape mismatches, mismatched
dict keys, or weights that are negative, non-finite or sum to zero.

**Why weight by n_k.** Write L_k for site k's mean loss. The pooled loss is
L = Σ_k (n_k/N) L_k, so its gradient is ∇L = Σ_k (n_k/N) ∇L_k. If each site takes **one**
full-batch gradient step from the same θ, then

    Σ_k (n_k/N)(θ − η∇L_k) = θ − η Σ_k (n_k/N)∇L_k = θ − η∇L

So one round of sample-weighted FedAvg is exactly one step of gradient descent on the pooled data.
That's the special case where federated equals centralized. The test suite checks it
(`test_one_local_step_fedavg_equals_centralized_gd`), and `reference.py` reports it
(max |Δθ| = 4.4e-16 over 20 rounds).

**Why it stops being equal.** The equivalence needs all of: one local step, full-batch gradients,
n_k weights, every site in every round, and the same step size. With several local steps, each
site walks toward *its own* optimum, and because ∇L_k depends nonlinearly on θ, averaging the
endpoints is not the endpoint of pooled gradient descent ("client drift"). The more the sites
differ (feature or label shift), the bigger the gap. Several local steps are still common: they
trade some accuracy for fewer communication rounds.

**Equal weights optimize a different objective.** With weights of 1, the same argument gives
gradient descent on (1/K) Σ_k L_k, which treats every site as equally important regardless of
size. That isn't "wrong", but it isn't the pooled-data model either: small sites get pulled up. We
use n_k weights by default (standard FedAvg; it reproduces pooled gradient descent). Equal or
other weightings stay available as an explicit choice.

### 5.2 How FLARE 2.9.1 weights clients
Read from the installed source (`nvflare/app_common/workflows/fedavg.py`, `base_fedavg.py`,
`aggregators/weighted_aggregation_helper.py`):

- weight_k = `aggregation_weights.get(site, 1.0)` × `meta["NUM_STEPS_CURRENT_ROUND"]`
- global = Σ weight_k · θ_k × (1 / Σ weight_k), with results added in **arrival order**
  (streaming "InTime" aggregation)
- If `NUM_STEPS_CURRENT_ROUND` is missing, a bool, non-numeric, non-finite or ≤ 0, FLARE uses
  **1.0 without a warning**. The Client API does not fill it in. A client that forgets to send
  its sample count therefore gets equal-weight averaging, silently.

Our client sends `meta[FLMetaKey.NUM_STEPS_CURRENT_ROUND] = n_k`, the same convention as FLARE's
own sklearn executor. Our `fedavg()` deliberately differs on bad input: it raises instead of
falling back to 1.0.

**Enforcement (D-017, client side).** Every FedAvg client result is built by
`federation/fedavg_client.fedavg_result`, which refuses, before the result exists, each value
FLARE would silently turn into 1.0 (missing/None, bool, non-numeric including numeric strings,
non-finite, ≤ 0). Equal weighting is sent as an explicit weight of 1 per client, so it is
declared rather than an accident. `tests/test_fedavg.py` has one test per refused case.

**Enforcement (D-036, server side; WS-13).** The server checks too: `federation/fedavg_server.py`
plugs a `ModelAggregator` (FLARE's extension point for custom aggregation) into the recipe. It
calls `require_n_k` on every result; a missing or unusable n_k makes FLARE mark the round failed
and refuse to save a model, instead of weighting that client 1.0. After the check, the averaging
is FLARE's own `WeightedAggregationHelper` with weight n_k, the same helper and weight as FLARE's
built-in path. *Alternative:* averaging in our own code inside the aggregator. Rejected because the
check in §5.3 would then compare our code with itself.

A `--demo_omit_n_k` flag bypasses the client helper to show the fallback, and runs FLARE's stock
aggregation on purpose; the slow test asserts FLARE still falls back, so a FLARE upgrade that
changes this is noticed. With `--server_check` the same clients meet our server, and the job
fails ("server refuses the result from site-3: client result has no NUM_STEPS_CURRENT_ROUND").

Two related FLARE details:
1. The NumPy recipe's persistor casts the initial model to float32 and saves a single array (key
   `numpy_key`). So the check uses one flat vector θ = [w, b] and a zero initialization (exact in
   float32). Clients train and send float64, so every later round stays float64.
2. FLARE's Newton-Raphson logistic-regression recipe (`FedAvgLrRecipe`, used by the `hello-lr`
   example) has its sites send gradient and Hessian **sums**, and then averages them with weights
   n_k again. So a site's effective weight is ∝ n_k². On the example's 4-site heart-disease data
   this lands 0.34 (max |Δθ|, with max |θ| = 0.80) away from pooled Newton-Raphson. That doesn't
   affect this check, but it's a reason to verify what any recipe sends and how it weights it
   before using it.

### 5.3 The check: FLARE vs. our own FedAvg
**Setup** (`jobs/fedavg_check/`; `src/fedkg_spacebio/federation/{synthetic,logreg}.py`):
- **Data.** 3 synthetic sites, sizes 400 / 150 / 50, 4 features with shared true coefficients.
  Each site has a different feature mean (covariate shift) and a different intercept, so positive
  rates are 52.5% / 82.0% / 12.0%. Each site's data comes from its own seeded RNG stream, so a
  FLARE client generates only its own site.
- **Model.** Unregularized logistic regression, full-batch gradient descent.
- **Protocol.** lr = 0.5, 5 local steps per round, 20 rounds, zero init, all 3 sites every round.
- **One shared trainer.** `local_train` is the same function on both paths, so a mismatch can
  only come from aggregation or transport.

**Alternatives considered.**
- sklearn `LogisticRegression` with warm starts on the sites: rejected. The solver's step count
  and internals would sit between us and the arithmetic we're checking.
- FLARE's `FedAvgLrRecipe`: rejected. It's a different algorithm (Newton-Raphson on aggregated
  statistics, not model averaging).
- A PyTorch model: rejected. It adds a dependency and nondeterminism, and buys nothing for a
  linear model.
- Equal-size, homogeneous sites: rejected. Every weighting gives nearly the same answer there, so
  a weighting bug would pass.

**Tolerance.** `np.allclose(rtol=1e-9, atol=1e-12)`. Both paths do the same float64 arithmetic
and differ only in summation order (FLARE adds results as they arrive), which accounts for a few
ulps (float64 eps ≈ 2.2e-16). Swapping n_k weights for equal weights moves θ by 7.0e-2. The
tolerance sits about 6 orders of magnitude above the expected noise and about 7 below a weighting
error.

**Results** (seed 0; training-set numbers on the pooled 600 samples):

| model | max \|Δθ\| vs FLARE (same weighting) | pooled log-loss | pooled accuracy | max \|Δθ\| to centralized optimum |
|---|---|---|---|---|
| FLARE FedAvg, clients send n_k | — | 0.4430 | 0.790 | 0.048 |
| reference `fedavg()`, n_k | **1.1e-15** (PASS) | 0.4430 | 0.790 | 0.048 |
| FLARE FedAvg, clients omit n_k (now `--demo_omit_n_k`) | — | 0.4435 | 0.785 | 0.069 |
| reference `fedavg()`, equal weights | **6.7e-16** (PASS) | 0.4435 | 0.785 | 0.069 |
| centralized GD, same 100-step budget | n/a | 0.4429 | 0.793 | 0.024 |
| centralized optimum (sklearn, C=∞) | n/a | 0.4429 | 0.793 | 0 |

**Reading it.**
- FLARE's FedAvg *is* sample-weighted FedAvg, and our numpy version reproduces it to
  machine precision, provided clients send n_k. Without n_k, FLARE reproduces our equal-weight
  version instead. That confirms the silent fallback directly.
- FedAvg with 5 local steps lands about twice as far from the pooled optimum as centralized
  gradient descent with the same number of gradient steps (0.048 vs. 0.024). That's client drift
  on heterogeneous sites. In loss terms the gap is tiny here (0.4430 vs. 0.4429), but it is not
  zero, and federated doesn't beat centralized.
- Equal weighting is further still (0.069), as expected: it optimizes the per-site average, not
  the pooled loss.

**Limits.** Synthetic data, training-set metrics only (this is an equivalence check, not a
generalization estimate), deterministic full-batch training, a simulator on one machine, and no
privacy mechanism. Federated learning by itself is not private.

**Reproduce.** `uv run python jobs/fedavg_check/job.py` (plus `--weighting equal` for explicit
equal weights and `--demo_omit_n_k` for the fallback demo), then
`uv run python jobs/fedavg_check/reference.py`. Or run `uv run pytest -m slow`, which runs all
three and requires each to match its reference.

### 5.4 Phase 3a: federated DE meta-analysis (liver)
Full write-up, checks and results: [`phase3a-liver-meta.md`](phase3a-liver-meta.md).

**Site DE.**
- pydeseq2 on each client's own flight and ground samples, plus covariates (`~ study +` for
  SpaceX-4, `~ age +` for SpaceX-16), on rounded RSEM counts.
- Unshrunk MLE log2FC and lfcSE. Shrinkage changes what the SE means.
- *Alternatives:*
  - an all-group fit like GeneLab's. *Rejected* (Kaitlin): it would couple a mission's ISS and
    Earth clients through shared dispersions.
  - edgeR or limma-voom. *Rejected:* pydeseq2 is pre-approved (D-016) and matches GeneLab's
    DESeq2.

**Inclusion.**
- Primary: genes with base mean ≥ 10 at **every** site of the stratum (a fixed, pre-specified
  filter; Kaitlin).
- Sensitivity runs: ≥ 3 sites; pydeseq2's independent filter; any finite p.
- *Why not independent filtering as primary:* its threshold is chosen per site from that site's
  own p-values, so the gene set would depend on each site's results.

**Pooling (D-021).**
- Inverse-variance FE.
- RE with REML τ² (fixed-point iteration from τ² = 0, truncated at 0, per-gene convergence flag).
- Hartung–Knapp–Sidik–Jonkman intervals with t(k−1), **truncated** (max(1, q); Kaitlin).
- Cochran's Q and I², and BH within each stratum and variant.
- *Alternatives:*
  - DerSimonian–Laird: biased low at small k.
  - Paule–Mandel: kept as the comparator (Check 3).
  - untruncated HKSJ: reported, since it can be narrower than the RE Wald interval when q < 1.

**Headline: changed after seeing results (D-026, 2026-11-13, Kaitlin).** The pre-specified headline
was RE with truncated HKSJ. It found **no gene at BH q < 0.05** in either stratum or any variant,
and structurally can't at k = 3–4: with t(2) or t(3), passing BH over ~13,000 genes needs
|t| ≥ 16 (Earth) or ≥ 8 (ISS) even if 1,000 genes passed together, and the largest observed
|t| was 6.7. The headline is now:
- **FE hit:** fixed-effect BH q < 0.05.
- **Heterogeneity flag:** Cochran's Q p < 0.10, with I² reported alongside. 0.10, not 0.05,
  because Q has little power with few sites.
- **Consistent FE hit:** FE hit without the flag, read as "a change consistent across *these*
  missions". It means no *detectable* heterogeneity, which with 3–4 sites is a weak statement:
  a wide-SE site can disagree without raising Q.
- RE-HKSJ stays as a secondary result, with the |t| floor reported.
- *What FE does and doesn't answer:* the precision-weighted average of the missions in the
  analysis. Not what a new mission would show (that's the RE question). Because it was chosen
  after seeing the results, every place that reports the headline says so.
- Definitions and counts live in one place (`federation/results.py`, `hit_counts`).

**Superseded by D-029 (2026-11-13, after the WS-08 comparison, §5.5).** The result is reported by
question:
- **"These missions":** FE hit, BH q < 0.05. Cochran's Q p and I² are annotations, not filters;
  the consistent/heterogeneous split above is kept as a description only. WS-08's simulation
  showed that "unflagged" doesn't make an FE hit generalize to a new mission.
- **"A new mission":** V3 (empirical-Bayes τ², §5.5) replaces truncated RE-HKSJ.
- **Sensitivity:** V2, for misestimated SEs.
- Every statement of the result carries D-029's fragility notes and D-035's caveat: consistent
  with a time-of-day mismatch of a few hours or a clock change; not separable (§5.6).
- The FLARE server computes V2 and V3 itself (`meta.pool`, per-gene schema 2). Both fit a prior
  across all genes of a variant, so the priors are saved with every pooled table and compared
  in Check 1.

**FLARE.** A one-round custom `ModelController` that waits for every client, validates each
summary and calls our pooling.
- *Rejected:*
  - `FedTaskRecipe`: its controller discards results;
  - `FedStatsRecipe`: named-feature statistics, with noise and min-count filters;
  - the FedAvg recipe: a model lifecycle we don't have.

**What federation does and doesn't add.** FE federated pooling equals central pooling of the
same summaries by construction, so Check 1 is a wiring check, not a benefit. RE, V2, V3 and the
heterogeneity flag need per-site estimates at the server, so they are not private. FE alone
would need only Σw·y and Σw per gene.

### 5.5 Phase 3a method comparison (WS-08)
Full write-up: [`phase3a-method-comparison.md`](phase3a-method-comparison.md). Its recommendation
was accepted as D-029 (§5.4, "Superseded by D-029").

**Two questions.** A per-gene hit can mean (A) "changed in these missions" (fixed effect's
question) or (B) "the mean across missions, the one a new mission would be drawn around, is
non-zero" (random effects'). Every method is labelled with the question it answers.

**Empirical-Bayes methods (server-side, on the per-site summaries WS-07 already sends).**
- **V2:** FE estimate with limma's moderation (Smyth 2004 §6.2) of φ = Q/(k−1). Under a
  multiplicative-heterogeneity model this is an exact scaled-χ² variance, so the conjugate
  theory holds. It answers A, robust to misestimated SEs.
- **V3:** a nonparametric prior over τ², fitted across all genes by EM on each gene's restricted
  likelihood. μ is averaged over each gene's τ² posterior. It answers B.
- *Rejected:* moderating the HKSJ q (the planned sketch, kept as "V1"). REML's τ̂² absorbs each
  gene's scatter, so q sits near 1. For Paule–Mandel q = 1 exactly whenever τ̂² > 0 (Röver, Knapp
  & Friede 2015). The limma fit then returns d₀ = ∞ and the test collapses to the plug-in RE
  Wald z.

**Calibrated simulation.**
- Copied from the real data: per-site SE vectors (resampled whole), V3's τ² prior, V2's φ
  prior, and the effect spread fitted by marginal likelihood.
- Swept rather than estimated: the τ² scale and the fraction of μ = 0 genes.
- Alternative generators: raw REML τ̂², multiplicative heterogeneity, noisy SEs (an approximate
  stress test), and the ≥3-site k pattern.
- 100 replicates per setting with fixed seeds; every method refits its hyperparameters on every
  replicate.
- Scored under both estimands: FDR, power, the call rate on μ = 0 heterogeneous genes, and
  interval coverage of μ and of the FE-weighted mission mean.

**Time of day (Q-11).**
- **Audit:** OSDR sample metadata, protocol text and publications, one source per cell.
- **Enrichment:** liver-cycling genes (Hughes et al. 2009, Table S1) mapped by probe set ID
  through GEO's GPL1261 annotation, never by symbol. Fisher and Mantel–Haenszel (base-mean
  deciles) tests, against a background of tested genes measurable on the array.
- These are diagnostics, not corrections: spaceflight may shift the clock itself.

### 5.6 Phase 3a time of day: circadian phase diagnostic (WS-09)
Full write-up: [`phase3a-time-of-day.md`](phase3a-time-of-day.md).
- **Reference:** a 24 h cosinor fitted per gene.
  - Peak times from GSE11923: hourly liver arrays, constant darkness, so times are CT.
  - Amplitudes from GSE54651: 6-hourly liver RNA-seq, giving the sites' log2(count + 1) scale.
  - Ten clock genes pass a rule fixed before any site data was opened.
- **Why not per-animal phases from z-scored genes:** centring within a site removes absolute
  time. Two arms then come out about 12 h apart for any true shift.
- **Clock plane (inside each site):**
  - the flight–ground difference **d** in the panel genes, adjusted for the site's DE
    covariates, is fitted as α**a** + β**b**, where a_g = A_g cos ωφ_g and b_g = A_g sin ωφ_g;
  - |Δ| = (24/π)·arcsin(|(α, β)|/2), reported as not identifiable when |(α, β)| ≥ 2;
  - R²_plane is the share of |d|² in the plane;
  - label permutation within covariate blocks;
  - bootstrap intervals paired with reference-bootstrap replicates, plus ±25% amplitude
    sensitivity.
- **Limits:** it can't tell a dissection-time mismatch from a clock shift, or a shift from an
  amplitude change. Direction needs a light-phase assumption.
- **Alternatives rejected:**
  - clock correlation distance (needs samples spread over the cycle and n ≈ 30);
  - ZeitZeiger and CYCLOPS (need more data);
  - VST (adds a fit; the panel genes are well expressed).
- **Exploratory absolute phase:** fitted against the RNA-seq gene baselines. It stopped at its
  pre-set ground-arm plausibility check.

## 6. Intermittent links (phase 3b, stretch)
TODO: shim design adapted from FLUID (Casaletto et al., 2025), loss-of-signal model, aggregation under stragglers.

## 7. Governance and policy enforcement
Code: `fedkg_spacebio.policy` (evaluator, release log, differencing) and
`fedkg_spacebio.federation.jobspec` / `gene_plan`. Contracts: CONTRACTS §5 (policies, decision
record, release log) and §6 (job spec). **The policies are invented for the testbed.** The real
missions declared no such rules; the policies exist so that enforcement can be tested on real
between-study structure. Nothing here claims to solve an actual privacy problem.

### 7.1 What is checked, and in what order
Before any job runs, the policy check takes the job spec, every relevant policy, the site
registry (manifests) and the release log, and returns **allow** or **deny** with the rule that
decided it. Order (CONTRACTS §5.4):
1. **Spec checks** (`spec#…`): is the spec coherent with the contract tables and the registry
   (known clients in the right stratum, methods of this task, outputs that meet each method's
   need, aggregation matching the outputs, the task's weighting, gene plan and exclusion files
   matching their hashes, manifests matching theirs)? If not, deny and stop: a policy's verdict on
   an incoherent spec would mean nothing.
2. **The project policy** (nothing per sample leaves a site; FedAvg weighting, D-017).
3. **Each site in the job**, in client order, rules in file order.
4. **Every other site in the stratum, differencing rules only.** A release that asks nothing of a
   site can still reveal it by subtraction.

Every policy defaults to deny, deny overrides allow, and the decision lists **every** failure, so
one answer tells the agent everything it would have to change. The deciding rule is the first
failure. The decision record names the spec hash, every policy file hash and the log head it was
checked against, so it can be recomputed later.

Two profiles share all files except SpaceX-16's: **open** (every site allows per-site summaries)
and **restricted** (SpaceX-16 allows only masked contributions to a secure sum, and adds a
differencing rule with `min_sites: 3`). Example A (CONTRACTS §5.5) is the point of the pair: the
full method set (FE, FE with heterogeneity, REML+HKSJ, V3) is allowed under open and refused
under restricted by `pol:SpaceX-16:2#release`, because the random-effects methods need each
site's own estimate. The re-planned job (FE and Cochran's Q/I², masked, secure sum; §7.7) is
allowed. That is the accuracy/privacy trade made visible: under the restricted profile the
analysis can answer question A ("changed in these missions", §5.5), with its heterogeneity
annotation, but not B. (Until WS-13, heterogeneity was refused too; D-039 moved it to masked
totals.)

### 7.2 Per-gene minimum k
In phase 3a the set of sites pooled for a gene varies by gene (inclusion thresholds, ≥ 3-site
variants, leave-one-out runs). So "at least 3 sites" is checked **per gene**: before the job,
the coordinator builds a **gene plan** (which clients contribute to which gene) from the
inclusion flags each site released as metadata, and the spec names it by hash. A site's
`min_aggregation` rule fails on every gene it contributes to that pools fewer than `min_sites`
sites, and on any of its clients with fewer than `min_n_per_arm` animals per arm (or no declared
count). k counts sites, whatever they send: a job asking one site for its own summary has k = 1.

### 7.3 Differencing: the check that needs a memory
Two releases can each pool ≥ 3 sites and still reveal one site when subtracted. Catching that
needs to know what was released before, so every release is written to an **append-only,
hash-chained release log** that records, per gene, which clients contributed (no output values).
The log can't be cleared: a "reset" starts a new log and records the switch, because a clearable
log would let the check be bypassed by clearing it between two requests.

**The test.** For one gene, write each logged release and the proposed one as a 0/1 row over the
stratum's clients (1 = contributed). A per-site summary adds its own single-client row (the
receiver reads that value directly); a masked contribution doesn't (only the sum is readable).
Anyone holding the results can form any combination cᵀR of the rows. Site i, protected with
`min_sites = m`, fails if some combination is non-zero at i and zero outside a group T that
contains i and covers fewer than m sites: the releases then give i's own value, or the total of
a small group including it.

The equivalent statement that is computed: **column i of R is not in the span of the columns
outside T.** (A combination zero outside T is a vector c orthogonal to every column outside T; it
can be non-zero at i exactly when column i has a part orthogonal to those columns.) Each test is
a tiny linear system, solved in exact rational arithmetic, because a policy decision shouldn't
depend on a floating-point tolerance. Exposure only grows with T, so only the largest allowed
groups need testing: SpaceX-16's own column plus the columns of one other site, for m = 3.
Genes with the same set of release rows share one answer; on 20,000 genes with 7 clients the
check takes a fraction of a second.

**Example B by hand** (CONTRACTS §5.5; columns A = SpaceX-4, B = SpaceX-8, C = SpaceX-13,
D = SpaceX-16; restricted profile; every release is a masked FE). Already released for gene
FAKE:0001: R1 (complete case, all four), R2 (≥ 3 sites, all four contribute to this gene) and
J3 (without SpaceX-8):

| release | A | B | C | D |
|---|---|---|---|---|
| R1 | 1 | 1 | 1 | 1 |
| R2 | 1 | 1 | 1 | 1 |
| J3 | 1 | 0 | 1 | 1 |

*J4: FE without SpaceX-16, complete case over A, B, C.* Proposed row 1 1 1 0. Then
R1 − J4 = 0 0 0 1: D's own contribution, a group of one site. In the span form: with T = {D},
the columns outside T are A = (1,1,1,1), B = (1,1,0,1), C = (1,1,1,1) (over the rows R1, R2, J3,
J4), and D's column is (1,1,1,0). Any combination a·A + b·B + c·C = (a+b+c, a+b+c, a+c, a+b+c)
has equal first and fourth entries, but D's are 1 and 0. So D's column is not in the span, and J4 is
refused by `pol:SpaceX-16:2#diff`, even though it asks nothing of SpaceX-16. On gene FAKE:0004
(SpaceX-16 never tested it) D's column is all zeros, so nothing about D can be revealed: the gene
passes, and resubmitting J4 without FAKE:0001 is allowed.

*J5: SpaceX-4's own per-site summary.* Proposed row 1 0 0 0. J3 − J5 = 0 0 1 1: the sum C + D,
a group of 2 sites < 3. With T = {C, D}, the columns outside T are A = (1,1,1,1) and
B = (1,1,0,0) (rows R1, R2, J3, J5); D's column is (1,1,1,0). a·A + b·B = (a+b, a+b, a, a) has
equal third and fourth entries; D's are 1 and 0, so D is exposed within {C, D}. J5 is refused,
but first by SpaceX-4's own minimum-k rule (k = 1), which comes earlier in evaluation order;
SpaceX-16's differencing failure is listed as well.

**What the log is and isn't.** It is tamper-evident: each entry hashes the previous one, every
contributor file is checked against its hash, and decision records name the head they saw, which
catches entries removed from the end. It is not tamper-proof: whoever can rewrite every file can
rebuild the chain. A real deployment would anchor the head outside the coordinator. A release is
logged only under the decision that allowed it and only if the log hasn't moved since that
decision; otherwise two checks against the same state could each pass and then combine.

Differencing compares across counts and inclusion variants (D-033): their per-site values are
nearly identical, so subtracting a standard-count release from an rRNA-removed one leaks
approximately. It doesn't compare across tasks (a clock-plane chord and a log2FC are different
quantities) or strata (different clients).

### 7.4 FedAvg weighting (D-017)
FLARE 2.9.1 silently weights a client 1.0 when its n_k is missing or unusable. The project
policy refuses a FedAvg job unless it declares `sample_count` weighting and every client's
manifest declares a usable n_k (the same check as the client helper, §5.2), or declares
`equal`/`custom` weights with a reason.

### 7.5 Alternatives considered
- **A general policy engine (OPA/Rego, Cedar).** Built for exactly this, but the rules here are
  six fixed kinds with numeric parameters, and the per-gene checks need exact linear algebra
  that the engine would call out to anyway. Small typed models keep each rule readable in one
  place.
- **Per-job minimum k only.** Misses per-gene variation (§7.2).
- **"No two releases differ by one site."** Misses combinations of three or more releases
  (J5's J3 − J5 = C + D) and per-site summaries mixed with masked sums.
- **Floating-point rank tests.** Right answer at this size, but a tolerance is the wrong kind of
  argument for a policy decision.
- **Masking the inclusion flags.** Would hide which genes pass each site's expression threshold,
  but then no per-gene check could run before submission (CONTRACTS §5.1).

### 7.6 What isn't built yet
The check decides; nothing yet refuses to *run* a job without an allow. That gate is
`job.submit` (CONTRACTS §7, with the agent tools). `jobs/phase3a_meta/job.py` (with `--profile`,
or `--spec` for a §6 job spec) keeps the profiles' outputs apart, but it runs whatever it is told.
No DP mechanism exists, so a policy requiring DP refuses every job.

### 7.7 Simulated secure sum: fixed effect and heterogeneity from masked totals (WS-13)

**What it is, and what it isn't.** Under the restricted profile SpaceX-16 lets only masked
contributions leave. This section is how a fixed effect, and Cochran's Q and I², are computed
without the server ever holding one site's estimate. It is a **simulation of the arithmetic**: the
pairwise mask seeds come from the job configuration (FLARE writes them into the job config in
clear text and its own secret scan warns about it), every client runs on one machine, and there
is no dropout recovery. It shows what a server would receive under such a policy. It is not a
security or privacy claim, and the data are public.

**Why totals are enough.** With w = 1/SE² per site and y = log2FC:
FE estimate = Σw·y / Σw, FE SE = 1/√Σw, and Q = Σw·(y − ȳ)² = Σw·y² − (Σw·y)²/Σw (expanding the
square). So each client sends three numbers per gene, w, w·y and w·y², and the server needs only
their sums over clients. Sending w·y² is extra disclosure, decided in D-039: three totals across
four sites don't identify a site. Random effects and V3 would need many more totals (a likelihood
over a τ² grid), from which single sites could be reconstructed, so they stay refused.

**Masking (`federation/secure_sum.py`).** Each value is turned into an integer
(round(x·2^40), "fixed point") and stored modulo 2^64; negatives wrap around (two's complement), so
−3 is 2^64 − 3. Each pair of clients shares a pseudo-random stream; the client earlier in sorted
order adds it, the later one subtracts it. One client's masked vector looks like noise (the tests check it
is uncorrelated with its true values, and that a sum over all but one client is noise too). Over
all clients every stream appears once with + and once with −, so the masks cancel exactly and
the server decodes the true integer totals. Streams come from SHAKE-256 (standard library, fixed
by its standard; numpy's generators can change their streams across versions).

**Choosing the bits.** On the real summaries the largest w at any site is 920 (SpaceX-16-ISST,
SE 0.033); the largest |w·y| is 119 and the largest w·y² 136. To keep every sum of K = 4 values
inside 2^63, each encoded value must stay below 2^61, i.e. |x| < 2.1 × 10⁶ at 40 fraction bits:
about 2,280 times the largest value seen, so a site with much smaller SEs still fits. Anything
larger is **refused, not wrapped**: a silent wrap would turn a large positive total into a large
negative one. Each value is off by at most 2^−41 ≈ 4.5e-13. *Alternative:* 32 fraction bits
(a 580,000× margin, 256× the error); both work, and 40 gives the tighter match.

**How close to plain pooling it must be.** Rounding moves each total by at most
ε = k·2^−41 (k sites for the gene). Then the estimate moves by at most ε(1 + |est|)/(Σw − ε), and
Q by at most ε + (2ε|est|Σw + ε² + est²Σw·ε)/(Σw − ε). These grow when Σw is small, so a fixed
tolerance of 2^−40 (what CONTRACTS §6.3 first said) can't hold for every gene; the per-gene bound
replaced it. Q is computed from the exact integer totals as a fraction, so the subtraction in
Σw·y² − (Σw·y)²/Σw, which loses digits badly in floating point when Q is small, loses none here.

**Result (`jobs/phase3a_meta/check_secure_sum.py`).** Masked against plain pooling of the same
per-site summaries, and against the existing schema-2 tables: both strata, both count matrices,
complete-case and ≥ 3-site plans. Every gene is inside its bound. The largest differences are
2.2e-12 for the estimate (0.72 of its bound) and 9.9e-12 for Q (0.88 of its bound). The `fe_hit`
and `het_flag` sets are identical everywhere. The nearest gene to the q = 0.05 line sits 1.4e-5
from it, and q moved by at most 5e-13. The restricted-profile FLARE runs (DE rerun at every site)
give the same: ISS 1,926 FE hits and 2,932 heterogeneity flags, Earth 1,722 and 3,986, identical
to the open-profile tables. So under the restricted profile the "these missions" answer and its
heterogeneity annotation are exactly the open-profile ones; what is lost is V3 ("a new mission")
and every per-site diagnostic, including leave-SpaceX-16-out (refused by differencing).

**Job specs.** `job.py --spec` runs a §6 spec. Clients check that their own inclusion flags are the
ones the gene plan was built from, and send values only for the plan's genes, minus any genes a
policy refused (those are left out of the vectors entirely, so the server never sums them). A
plain spec sends per-site summaries restricted to the plan's genes, and the released table
carries only the requested methods' columns, with `methods_run` and `aggregation` in its
metadata.

## 8. Agent and grounding
The agent answers one question by calling tools, records every call and decision in a trace, and
writes a report whose claims cite IDs from its own tool results. The grounding checker then grades
the report against the trace. §8.1 is the KG tool; §8.2–§8.5 are the trace, the grounding
checker, the policy and results tools, and the model interface (WS-15 part A). The agent loop and
the report builder come with `job.submit` (WS-15 part B).

### 8.1 KG explanation: the `kg.query` tool (WS-14; D-033, D-038)

**What it does.** For a gene the federated analysis reports, the tool can answer two questions:
- What are its human orthologs?
- In which other flight-vs-control comparisons was the gene reported changed?

It asks the hosted SPOKE-GeneLab knowledge graph, through the `mcp-genelab` server's read-only
`query` tool. We wrote the two Cypher templates and the code around them: the client,
validation, labels, cache and ID listing. The graph, its contents and the server belong to
SPOKE-GeneLab and SDSC. Contract: `CONTRACTS.md` §7.1.

**Named templates, not agent-written Cypher.** The agent picks a template and a gene; it can't
write a query. Every query that can run was designed and checked by hand, and its text is pinned
by a hash.
- *Alternative: let the model write Cypher, as the connector allows.* Rejected. Nobody would have
  reviewed what each query means: a confounded pair, or a mirror-image contrast counted twice.
- *Alternative: the connector's specialist tools.* Rejected. They all start from a comparison;
  both of our questions start from a gene.

**Strict comparisons only.** A comparison is listed only if:
- it is flight vs a ground, vivarium or basal control;
- it uses the same tissue on both sides;
- every other factor is identical.

Without this rule, one gene returned 38–128 comparisons, mostly mismatched pairs inside our own
input studies, e.g. flight at one age and tissue prep vs ground at another. The cost is that
some valid designs drop out: location-aware ISS-terminal contrasts and gravity-labelled studies.
They're counted (`n_nonstrict`), not hidden. Some comparisons also have control labels the scope
doesn't match (`Ground Control Rerun`, lower-case `Ground control`), so they're missed entirely.

**"Not our input study" doesn't mean "independent".** Each row is labelled with its relation to
our phase 3a inputs:
- an input study;
- OSD-168, which shares animals with two inputs;
- another study from an input mission;
- another mission;
- no mission recorded.

Rodent Research missions usually take many tissues from the same animals. For the six genes
tested in WS-14, 64 of 89 strict rows came from other tissues of our own missions, and the KG
can't show whether animals are shared. Only "another mission" rows are outside our inputs, and
even they mean "this comparison reported the gene changed": not a replication, and not an
effect size to pool. The KG keeps a DE edge only when adj_p < 0.1 and stores no standard error
(D-029).

**Liver is an explicit list of three UBERON terms:** liver, left lateral lobe and left lobe.
- *Alternative: reason over the UBERON hierarchy at query time.* Rejected. It needs another
  source, and a literal list can be audited; WS-14 checked it covers every liver material in the
  KG.

**Orthologs: all of them.** 1,607 mouse genes have more than one human ortholog (up to 26). The
KG gives no basis for ranking them, so the tool returns every one, sets `one_to_many`, and a
claim citing one should say it's one of k.

**Citing edges.** The KG has no edge IDs, so the tool builds a key from (type, start, end). WS-14
checked that no two edges of the same type join the same pair, for every type used, including
all 50.2 million DE edges.

**Version.** The graph exposes no release string. The tool records the assumed release plus a
fingerprint: nine graph counts and a checksum over comparison IDs. The fingerprint is part of
the cache key, so a changed graph misses the cache rather than serving stale rows. Its limit: a
rerun that changes edge values but keeps every count and ID would go unnoticed.

**Cache and offline replay.** Raw KG rows are cached on disk, keyed by (template, parameters
sent, Cypher hash, fingerprint). In offline mode the tool serves only from the cache, and a miss
is an error, so the demo can't silently depend on the endpoint being up. A pre-warm command
caches the top 25 ISS fixed-effect hits.
- *Alternative: `requests-cache`.* Rejected. It's a new dependency, and its store hides the
  per-request provenance we want to cite.
- *Alternative: the `mcp` Python SDK.* Rejected. It's a new dependency with an async stack, for
  four request types.

**Traps found on the wire (each has a test):**
- The server reports errors as prose with `isError: false`.
- The reply's first line, `total_rows:`, also contains `rows:`.
- A gene ID passed as an integer silently matches nothing.

### 8.2 The trace (CONTRACTS §8)
One JSON Lines file per run under `$FEDKG_WORKDIR/traces/`. In order: what was asked
(`run_start`, with the model and tool versions), every tool call and its result, every decision
with its reason and the calls it was based on, every model request (`model_call`: which model
answered, why it stopped, tokens; no prompts, no keys), the report, the grounding verdict, and
`run_end`.
- **Append-only, then sealed.** The writer never rewrites a line; after `run_end` it refuses
  appends and makes the file read-only. A crash leaves a valid prefix.
- **Hash chain.** Each record stores the previous record's hash and its own. Editing, removing or
  reordering a record breaks the chain, and the reader refuses the file. This matters because the
  grounding verdict is only as good as the file: if a tool result could be edited afterwards to
  contain a cited ID, an invented citation would pass. Like the release log, this is
  tamper-evident, not tamper-proof: whoever can rewrite the whole file can rebuild the chain.
- *Alternative: chmod only.* Nothing a later reader could check. *Alternative: one hash at the
  end.* An unfinished trace couldn't be checked at all.

### 8.3 Grounding: is every claim backed by this run's own tool results?
A claim has a kind (`result`, `kg_fact`, `policy`, `caveat`, `site`) and citations, each "this ID,
as returned by that call". The checker reads only the trace and applies five rules:
1. the cited ID is in the `ids` of the named call's result, recorded before the report;
2. that call's tool can support the claim's kind (a gene ID from the KG can't back a statement
   about the federated result, even though it's the same ID);
3. every ID-shaped token in the claim's text has a valid citation, so a claim can't cite one ID
   and slip in others;
4. IDs that appear in no tool result at all, cited or merely mentioned, are listed as fabricated;
5. a claim with no valid citation is unsupported.

**What counts as an ID in text.** Fixed forms (CURIEs, OSD accessions and comparison IDs, KG edge
keys, project run-time IDs, `note:` IDs) are matched by pattern, longest first, so an edge key
that contains two gene IDs is one token and a comparison ID isn't read as its `OSD-<n>` prefix.
Mission, site and client names have no reliable pattern (`SpaceX-16-ISST`, `Soyuz_TMA-14`), so
they are recognised two ways: names some tool listed in this trace, and any word of slug shape (a
capital letter, a `-` or `_` part, a digit), which catches a mission name no tool returned. Gene
symbols are names, not IDs: a slug-shaped word that is a gene symbol in the trace's results
(`Nkx2-1`) needs no citation and is never counted as fabricated. The cost of the slug shape is
that ordinary words of that shape count too, so the report writes "top 10" and states caveats in
words.
- *Alternative: patterns only, no slugs.* "The result holds without SpaceX-99" would pass.
- *Alternative: a fixed list of mission names.* It would have to come from outside the trace, and
  the checker's point is that it needs nothing but the trace.

**What it doesn't check:** whether the text says what the cited result says (a claim citing a
negative estimate while saying "higher"). That needs per-kind semantics; the report builder
writes claims from tool data, and the checker guards the IDs.

**Caveats are citable.** `results.read` returns the caveats that apply to the release it reads,
each a fixed sentence with an ID (`note:time-of-day`). A caveat claim cites that ID, so a dropped
or reworded caveat is visible in the trace. The phase 3a findings (the SpaceX-16 dependence, the
rhythmic-gene share, SpaceX-8 staying in) attach only to releases of that analysis, and the ones
that need per-site estimates only to plain releases. Under the restricted profile the report
says instead that those checks were refused, with the refusing decision.

### 8.4 Policy tools and the plan check
`sites.list` reports the registry (from the manifests) and recomputes each client's inclusion-flag
hash from the released flags; a mismatch is an error, because the per-gene checks and the gene
plan are built from those flags. `policy.check` wraps the evaluator of §7 unchanged and returns
the decision record plus the exact spec to submit.

**The plan check.** The release log is first come, first served, so which of two conflicting
releases the federation gets would otherwise depend on the order the agent happened to submit
them. `policy.check_plan` takes every release the agent intends to make and checks them in order
on a temporary copy of the log, as if each allowed one had been released; the real log is never
written. For each refusal it reports whether the spec would be allowed alone, which earlier specs
cause the refusal, and whether the conflict holds in both orders. The agent then records which
release it keeps as a `plan` decision. Example, restricted profile, ISS (fake genes, `tests/
test_tool_policy.py`): after the main result over all four clients and "without SpaceX-8",
"without SpaceX-16" is refused by SpaceX-16's differencing rule. It conflicts with **both**
earlier releases, each on its own: main − without-SpaceX-16 is SpaceX-16 alone, and
without-SpaceX-8 − without-SpaceX-16 is SpaceX-16 minus SpaceX-8, a two-site group, smaller than
the rule's minimum of three. Both conflicts are mutual.
- *Alternative: accept first come, first served as a limitation.* The decision would be made by
  arrival order, silently.
- *Alternative: an in-memory dry-run mode in the evaluator.* It would duplicate how the log builds
  contributor rows; copying the log and using the real append is exact by construction.

**Reading results.** `results.read` reads only released results, through a release record that
ties a release ID in the log to a table and its hash. It never returns per-site columns, whatever
the aggregation. A tool that could read any table on disk would make the policy layer optional.

### 8.5 The model interface
The agent talks to models through one small internal interface with two adapters: an
OpenAI-compatible adapter (NVIDIA NIM and most hosted providers) and Claude's native Messages API.
Both are plain HTTP. The interface carries three things, which each adapter translates: tool
definitions, the tool calls a model makes, and the results sent back.
- Our tool names contain dots, which neither API allows; the adapters map `.` to `_` and back.
- Every argument is validated by our own code before a tool runs, even where the provider promises
  schema-exact arguments (an OpenAI-compatible server may ignore that, or send arguments that
  aren't JSON). A bad call gets an error result; the tool doesn't run.
- With Claude, the assistant's turns are sent back exactly as received, because current models
  check that their earlier reasoning blocks come back unchanged; the conversation is append-only.
- Refusal fallbacks (the provider answering with another model) are off by default. A refusal is
  recorded as a model error, and the deterministic planner finishes the run. The model that
  actually answered is recorded for every request.
- API keys are read from environment variables at request time and never reach a trace, a log or
  an error message (tested with a fake key).
- *Alternative: one OpenAI-compatible client for both, using Anthropic's compatibility layer.*
  That layer is meant for quick checks and ignores strict schemas. *Alternative: the official
  SDKs.* The usual route; not used, to keep the runtime free of new dependencies.

## 9. Evaluation
TODO: metrics, uncertainty, what counts as federation "helping."

## 10. Limitations
TODO
