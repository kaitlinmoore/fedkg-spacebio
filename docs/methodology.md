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
declared rather than an accident. `tests/test_fedavg.py` has one test per refused case. The
server-side check that fails a round on a result without n_k needs a custom FLARE aggregator and
is left for the masked-sum server path (WS-13). A `--demo_omit_n_k` flag bypasses the helper to
show the fallback; the slow test asserts FLARE still falls back, so a FLARE upgrade that changes
this is noticed.

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
under restricted by `pol:SpaceX-16:2#release`, because heterogeneity methods need each site's own
estimate. The re-planned job (FE only, masked, secure sum) is allowed. That is the
accuracy/privacy trade made visible: under the restricted profile the analysis can answer
question A ("changed in these missions", §5.5) but not B.

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
`job.submit` (CONTRACTS §7, with the agent tools). The masked sum itself (§6.3) and the
server-side n_k check are WS-13. `jobs/phase3a_meta/job.py --profile` sets the manifests' policy
IDs and keeps the profiles' outputs apart, but it runs whatever it is told. No DP mechanism
exists, so a policy requiring DP refuses every job.

## 8. Agent and grounding
TODO: tools, planning, grounding rule and checker, trace format.

## 9. Evaluation
TODO: metrics, uncertainty, what counts as federation "helping."

## 10. Limitations
TODO
