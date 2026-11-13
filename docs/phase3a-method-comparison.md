# Phase 3a: method comparison (empirical-Bayes random effects vs fixed effect and HKSJ)

_WS-08, 2026-11-13. Builds on [`phase3a-liver-meta.md`](phase3a-liver-meta.md) (WS-07). No
biological claims are made here: gene names appear only as diagnostics for Q-11 (time of day)._

## 1. Why this comparison exists

WS-07's pre-specified headline, REML random effects with truncated Hartung–Knapp–Sidik–Jonkman
(HKSJ) intervals, found no gene at BH q < 0.05 in either stratum. With k = 3–4 missions its t
reference has 2–3 df, so BH over ~13,000 genes needs |t| ≈ 8–16. Fixed effect (FE) finds
~1,700–1,900 genes, but it answers a different question. D-026 made FE plus a heterogeneity
flag the interim headline and asked for this comparison: an empirical-Bayes (EB) method that
borrows information across genes, tested on a simulation calibrated to the real liver data and on
the real data.

**Two questions, kept apart throughout.** A per-gene hit can mean:
- **(A) these missions:** "the gene changed in the missions we analyzed": FE's question;
- **(B) a new mission:** "the mean effect across missions, the one a new mission would be drawn
  around, is not 0": the random-effects question.

A gene whose missions disagree but whose effects average to roughly 0 is a true finding under
(A) and a false one under (B).

## 2. Methods compared

| Key | Method | Question it answers |
|---|---|---|
| FE | inverse-variance fixed effect, z-test | A |
| FE + flag | FE hit and Cochran's Q p ≥ 0.10 (D-026 headline) | A, consistent genes only |
| HKSJ (trunc.) | REML τ², HKSJ variance × max(1, q), t(k−1) (D-025) | B |
| HKSJ (untrunc.) | as above without max(1, ·) | B |
| V1 | the planned sketch: limma moderation of the HKSJ q | B (in intent) |
| V2 | FE estimate, limma moderation of φ = Q/(k−1) | A, robust to SE misestimation |
| V3 | empirical-Bayes τ²: grid prior over τ² fitted across genes | B |

BH runs within each stratum and variant for every method. Code: `federation/moderation.py` (V1,
V2), `federation/eb_tau2.py` (V3), `evaluation/meta_methods.py` (one registry used by the
simulation and the real-data comparison).

### 2.1 Literature check
Read 2026-11-13; entries in the local literature index.
- **Moderating the HKSJ q, or shrinking τ²_g across genes in a cross-study meta-analysis:**
  nothing published found in a short search. That is not proof there's nothing.
- **Closest precedents:**
  - Yu et al. 2019 (BMC Bioinformatics 20:675) moderate a random-effect variance across genes,
    but inside one study.
  - Turner et al. 2012 (Int J Epidemiol 41:818) build log-normal τ² priors from 14,886
    Cochrane meta-analyses: borrowing across analyses rather than genes.
  - Marot et al. 2009 (Bioinformatics 25:2692) and Siangphoe & Archer 2017 (Brief Bioinform
    18:602) estimate τ²_g gene by gene. The latter report inflated FDR for a per-gene
    Paule–Mandel-type τ².
- **The machinery:** Smyth 2004 (Stat Appl Genet Mol Biol 3:Art. 3), §3 and §6.2: the
  scaled-F prior, its method-of-moments fit, which allows d_g to vary by gene, and the moderated
  t on d_g + d₀ df.
- **Not opened** (paywalled, so not relied on): Lund et al. 2012 (quasi-likelihood dispersions
  shrunk with Smyth's method) and Stanley & Doucouliagos 2015 (unrestricted weighted least
  squares meta-analysis). Both look like direct precedents for V2.

### 2.2 Why the sketch (V1) was not built as the EB method
Röver, Knapp & Friede 2015 (BMC Med Res Methodol 15:99, p. 3): the Paule–Mandel τ² is *defined*
by setting the HKSJ q to 1, so q = 1 whenever τ̂² > 0. REML is close to PM here (r = 0.994, WS-07
Check 3), and the real data behave the same way:

| | ISS (k = 4) | Earth (k = 3) |
|---|---|---|
| HKSJ q where τ̂² > 0: median (10th–90th pct) | 0.97 (0.76–1.17) | 0.98 (0.77–1.09) |
| HKSJ q where τ̂² = 0: median (10th–90th pct) | 0.51 (0.16–1.02) | 0.38 (0.06–0.89) |
| var(log q) vs χ² sampling variance ψ′(d/2) | 0.47 vs 0.94 | 0.89 vs 1.64 |
| Smyth §6.2 fit to q | d₀ = ∞, s₀² = 0.78 | d₀ = ∞, s₀² = 0.74 |

- REML's τ̂² has already absorbed each gene's scatter, so q varies *less* than a χ² variance
  estimate would. The prior fit then says the genes don't differ at all (d₀ = ∞).
- Every gene gets the same q̃, tested on ∞ df:
  - truncated, that is the RE Wald z with the plug-in τ̂²;
  - untruncated, it's narrower still.
- The RE Wald z is the test HKSJ was introduced to replace, and it treats the 47% (ISS) /
  42% (Earth) of genes with τ̂² = 0 as having no heterogeneity at all. V1 stays in the
  comparison to show this.

### 2.3 V2: moderated multiplicative heterogeneity
- **Model:** Var(y_i) = φ_g·v_i (site i's sampling variance v_i scaled by a per-gene factor).
- **Estimate:** φ̂_g = Q_g/(k−1), Cochran's Q over its df. Under this model,
  (k−1)·φ̂_g/φ_g ~ χ²(k−1) exactly, which is limma's assumption, so Smyth's conjugate theory
  applies as written.
- **The steps:**
  - estimate = the FE estimate;
  - φ̃_g = (d₀s₀² + d_gφ̂_g)/(d₀ + d_g);
  - SE² = φ̃_g / Σ(1/v_i);
  - t on d_g + d₀ df.
- **Limits:** d₀ → 0 is unrestricted weighted-least-squares meta-analysis (HKSJ with τ² = 0
  plugged in). d₀ → ∞ is FE with one common inflation. Tests check both limits.
- **Fitted priors (primary sets):** ISS d₀ = 24.9, s₀² = 1.35; Earth d₀ = 5.8, s₀² = 1.56.
  On average, sites scatter 35–56% more than their SEs predict.
- **The QQ check against the fitted scaled F** (`phi_qq.png`) is close to the line over the
  whole range. Earth's top tail is a little lighter than the fit, so the non-robust fit is not
  being dragged by outliers. The robust fit (Phipson et al. 2016) is deferred.
- **No trend with expression on the primary sets:** Spearman of log φ̂ with mean log base mean
  is 0.08 (ISS) / 0.04 (Earth), so a constant prior is used.
- **Truncation max(1, φ̃) never binds on the primary sets.** Even at φ̂ = 0,
  φ̃ ≥ d₀s₀²/(d₀ + d) = 1.21 (ISS) and 1.16 (Earth). No truncated V2 variant is reported, and a
  test covers the s₀² < 1 case where truncation would bind.
- **Exception: the "any finite p" sensitivity set.** It adds ~6,000 low-expression genes with
  φ̂ far below 1 (lowest base-mean decile median 0.14 / 0.08), so φ̂ trends with expression
  (Spearman 0.35). The single prior is then misspecified (ISS d₀ = 4.8, s₀² = 0.85), the smallest
  φ̃ is 0.52 / 0.46, and V2 finds 0 genes there with or without truncation. A prior that trends
  with expression (limma's covariate option) would be the fix. It's proposed, not built.
- **What V2 answers:** question A, with a variance that also absorbs any misestimation of the
  site SEs. It is not a "new mission" estimate: the weights stay 1/v_i, so a dominant site
  (SpaceX-16 carries ~40% of the FE weight) stays dominant.

### 2.4 V3: empirical-Bayes τ²
1. **A grid of τ² values:** 0 plus 59 log-spaced values, with the range set from the data.
2. **Each gene's restricted likelihood** at every grid value. This is p(y | τ²) with the mean
   integrated out under a flat prior; REML's τ̂² is its maximizer.
3. **A prior over the grid,** fitted to all genes at once by EM: a nonparametric maximum
   likelihood estimate of the τ² distribution, with no imposed shape.
4. **Each gene's posterior over the grid.**
5. **μ | y,** a mixture of normals N(μ̂(τ²_j), 1/Σw_j). Report:
   - the posterior mean;
   - an equal-tailed 95% interval;
   - p_EB = 2·min{P(μ > 0 | y), P(μ < 0 | y)}, with BH on p_EB.

**Exact reductions:** with a point-mass prior at τ² = 0, p_EB is the FE p-value; at τ² = c, the
RE Wald p-value at c. Tests check both, plus EM recovery of a known two-point τ² mixture,
stability under grid refinement and the credible-interval quantiles.

**The fitted prior differs sharply from per-gene REML** (`tau2_prior.png`). REML puts 47% /
42% of genes at exactly τ² = 0. The deconvolved prior puts 0.8% / 0.2% there, with median τ²
0.005 (ISS) / 0.017 (Earth). A gene whose τ̂² is 0 is therefore still averaged over plausible
τ² > 0 rather than being treated as FE.

**What V3 answers:** question B.

**Known weaknesses:**
- τ²_g is treated as exchangeable across genes and independent of μ_g. Genes with real effects
  are probably more heterogeneous, and their τ² gets shrunk toward the bulk. The simulation
  shows the cost: 95% intervals cover μ for 92–93% of heterogeneous and true-effect genes.
- v_i is treated as known.
- There is no exact small-sample theory.

**Federation.** All three run at the server on the per-site (log2FC, SE) summaries it already
receives under WS-07. No new data leaves a site. Like REML/HKSJ, they need each site's estimate,
not securely aggregated sums. They were computed from WS-07's saved per-gene tables; no site DE or
FLARE job was rerun.

## 3. Calibrated simulation

Code: `evaluation/meta_sim.py`; run: `jobs/phase3a_compare/simulate.py`.

**Copied from each stratum's real data:**
- the number of genes and each gene's per-site SE vector, resampled whole;
- V3's τ² prior;
- V2's φ prior;
- the spread σ_μ of non-zero mean effects, fitted by marginal likelihood for each assumed
  fraction of μ = 0 genes.

**Three kinds of gene:**
1. μ = 0 and no mission deviation;
2. μ = 0 with deviations (mission effects cancel on average);
3. μ ≠ 0.

Under (A) only kind 1 is null; under (B) kinds 1 and 2 are. Half of the μ = 0 genes are kind 1
(fixed; V3's prior has almost no mass at τ² = 0, so without this there would be no (A) nulls).

**Settings:**
- τ² scale ×0.5 / 1 / 2 × fraction μ = 0 of 0.80 / 0.90 / 0.95;
- at the central setting (×1, 0.90), four alternative generators:
  - τ² drawn from the raw REML τ̂² values (doesn't use V3's fit);
  - multiplicative heterogeneity from V2's prior (favours V2);
  - noisy reported SEs, v_obs = v·χ²(df)/df with df = n_flight + n_ground − 2. An
    **approximate stress test**: DESeq2 shrinks dispersions, so real SEs are less noisy than
    this;
  - (ISS) the ≥3-site k pattern.
- One setting added after calibration: the unconstrained marginal-likelihood fit put the fraction
  of μ = 0 genes at 0.37 (ISS) / 0.46 (Earth), far below the grid (many small effects rather
  than a few large ones), so a setting at that fit was added. It's reported as an addition to the
  pre-specified grid.

**Replicates and refits:** 100 replicates per setting with seeds
`SeedSequence([20261113, stratum, setting, replicate])`. Every method refits its own
hyperparameters on every simulated dataset.

**Scored:**
- empirical FDR at BH q < 0.05 (false calls / max(calls, 1), averaged over replicates) and power,
  under each estimand;
- the call rate on kind 2 genes;
- 95% interval coverage of μ (B's target) and of the FE-weighted mission mean (A's target).

_Figures are written to `$FEDKG_WORKDIR/results/phase3a/method-comparison/figures/` (not in git):
`sim_fdr.png`, `sim_power.png`, `phi_qq.png`, `tau2_prior.png`, `core_clock_iss.png`._

**Central setting (means of 100 replicates).** Power A counts kinds 2 and 3 as true; power B
counts kind 3 only.

| ISS | calls | FDR A | power A | FDR B | power B | kind-2 call rate | cover μ | cover FE mean |
|---|---|---|---|---|---|---|---|---|
| FE | 704 | 0.023 | 0.099 | 0.246 | 0.420 | 0.028 | 0.913 | 0.950 |
| FE + flag | 500 | 0.029 | 0.070 | 0.178 | 0.325 | 0.013 | 0.913 | 0.950 |
| HKSJ (trunc.) | 0 | 0 | 0 | 0 | 0 | 0 | 0.996 | 0.999 |
| HKSJ (untrunc.) | 0.7 | 0.056 | 0.000 | 0.071 | 0.000 | 0.000 | 0.946 | 0.957 |
| V1 (trunc.) | 447 | 0.015 | 0.063 | 0.110 | 0.314 | 0.008 | 0.946 | 0.966 |
| V1 (untrunc.) | 644 | 0.056 | 0.087 | 0.235 | 0.390 | 0.020 | 0.910 | 0.936 |
| V2 | 440 | 0.003 | 0.063 | 0.101 | 0.313 | 0.008 | 0.943 | 0.971 |
| V3 | 350 | 0.004 | 0.050 | 0.043 | 0.264 | 0.002 | 0.950 | 0.971 |

| Earth | calls | FDR A | power A | FDR B | power B | kind-2 call rate | cover μ | cover FE mean |
|---|---|---|---|---|---|---|---|---|
| FE | 826 | 0.023 | 0.114 | 0.582 | 0.267 | 0.080 | 0.869 | 0.950 |
| FE + flag | 407 | 0.042 | 0.055 | 0.477 | 0.164 | 0.030 | 0.869 | 0.950 |
| HKSJ (trunc.) | 0 | 0 | 0 | 0 | 0 | 0 | 0.997 | 1.000 |
| HKSJ (untrunc.) | 0.1 | 0.033 | 0.000 | 0.070 | 0.000 | 0.000 | 0.947 | 0.960 |
| V1 (trunc.) | 253 | 0.015 | 0.035 | 0.391 | 0.119 | 0.016 | 0.930 | 0.968 |
| V1 (untrunc.) | 528 | 0.075 | 0.069 | 0.521 | 0.195 | 0.041 | 0.884 | 0.935 |
| V2 | 20 | 0.000 | 0.003 | 0.391 | 0.009 | 0.001 | 0.943 | 0.986 |
| V3 | 0.07 | 0.000 | 0.000 | 0.050 | 0.000 | 0.000 | 0.950 | 0.984 |

**Worst case over all settings and generators** (max of the replicate-mean FDR):

| | FE | FE + flag | V1 (trunc.) | V2 | V3 |
|---|---|---|---|---|---|
| ISS FDR A | 0.075 (SE noise) | 0.080 (SE noise) | 0.039 | 0.011 | 0.008 |
| ISS FDR B | 0.497 | 0.381 | 0.232 | 0.146 | 0.049 |
| Earth FDR A | 0.058 (SE noise) | 0.091 (SE noise) | 0.041 | 0.000 | 0.000 |
| Earth FDR B | 0.793 | 0.722 | 0.635 | 0.437 | 0.100* |

\*Earth V3 makes ~0.07 calls per replicate, so its FDR is 0 in most replicates and 1 in a few.
The replicate SD is 0.22; the estimate is noise around a near-empty list.

**What the simulation says.**
1. **Question B (a new mission):**
   - Only V3 holds FDR near 0.05 across every setting, including the generators built not to
     favour it.
   - Truncated HKSJ is valid but has no power (coverage 0.99+). Untruncated HKSJ covers μ at
     ~0.945 and still makes no calls.
   - FE's FDR under B is 0.16–0.50 (ISS) and 0.51–0.79 (Earth). The heterogeneity flag lowers
     that only a little.
   - V1 behaves like the plug-in Wald test it reduces to (FDR B 0.11–0.64).
2. **Question A (these missions):**
   - FE is already calibrated when the SEs are right: FDR A 0.02–0.03, and 95% intervals cover
     the FE-weighted mission mean at exactly 0.95.
   - The flag doesn't help under A. It removes true heterogeneous changes from the list and
     keeps the false ones, so FE + flag has slightly *higher* FDR A and lower power than FE.
   - With noisy SEs (the stress test), FE drifts to 0.058–0.075. V2 stays ≤ 0.011 everywhere,
     at a power cost (ISS central 0.063 vs 0.099).
3. **V3's price:**
   - Power. ISS central power B is 0.26 vs FE's 0.42 (which comes with 25% false discoveries).
     In Earth (k = 3) it is effectively 0 in every setting.
   - Interval coverage for heterogeneous and true-effect genes is 0.92–0.93, below nominal.
     The overall 0.95 averages that with over-covered null genes.

## 4. Real data (standard counts)

Code: `evaluation/meta_compare.py`; run: `jobs/phase3a_compare/compare.py`. The registry
reproduces WS-07's FE and truncated-HKSJ q-values exactly (1,926 / 971 / 0 and 1,722 / 870 / 0).

**Hits at BH q < 0.05:**

| Run | FE | FE + flag | HKSJ tr. | HKSJ untr. | V1 tr. | V1 untr. | V2 | V3 |
|---|---|---|---|---|---|---|---|---|
| ISS primary | 1,926 | 971 | 0 | 0 | 304 | 740 | 439 | 31 |
| ISS ≥ 3 sites | 1,955 | 985 | 0 | 0 | 293 | 769 | 453 | 29 |
| ISS independent filter | 1,979 | 1,000 | 0 | 0 | 294 | 773 | 437 | 34 |
| ISS any finite p | 1,863 | 958 | 0 | 0 | 260 | 1,074 | 0† | 39 |
| ISS without SpaceX-8 | 2,765 | 1,928 | 0 | 0 | 1,325 | 2,495 | 915 | 575 |
| ISS rRNA-removed primary | 1,923 | 970 | 0 | 0 | 304 | 740 | 439 | 31 |
| Earth primary (= ≥ 3 sites) | 1,722 | 870 | 0 | 0 | 272 | 850 | 0 | 0 |
| Earth independent filter | 1,799 | 915 | 0 | 0 | 281 | 880 | 0 | 0 |
| Earth any finite p | 1,691 | 846 | 0 | 0 | 254 | 1,150 | 0† | 0 |
| Earth rRNA-removed primary | 1,726 | 875 | 0 | 0 | 272 | 842 | 0 | 0 |

† The constant prior is misspecified there (§2.3).

**Earth's zeros are not a bug.** The smallest V2 and V3 p-values are 4.9e-4 and 2.4e-4, against
the 3.9e-6 that BH needs for even one gene. FE's p ≈ 1e-22 genes are those where the three
missions scatter far more than their SEs.

**ISS primary, overlap and ranks.**
- **Jaccard:** FE vs V2 0.23, FE vs V3 0.02, V2 vs V3 0.07. All 31 V3 hits are FE hits.
- **Rank agreement:** Spearman of signed −log10 p is ≥ 0.99 between FE, V2 and V3, while the
  top-500 overlap is 0.60 (FE–V3) and 0.91 (FE–V2). The methods order genes alike; they differ
  in how far down the list they go.
- **Heterogeneity flag among hits:** FE 50%, V1 (trunc.) 9%, **V2 63%**, V3 45%. V2's strong
  moderation (d₀ = 25) pulls even very scattered genes toward the typical φ. For example,
  φ̂ = 10 becomes φ̃ = 2.3, so V2 keeps calling genes driven by the dominant site.

**Single-site influence, ISS** (full-data hits still hits when a site is dropped and BH rerun):

| Dropped | FE (1,926) | FE + flag (971) | V2 (439) | V3 (31) |
|---|---|---|---|---|
| SpaceX-13-ISST | 918 | 398 | 276 | 25 |
| SpaceX-16-ISST | 437 | 93 | **0** | **0** |
| SpaceX-4 | 1,441 | 513 | 177 | 7 |
| SpaceX-8 | 1,878 | 888 | 434 | 31 |

- Every V2 and V3 hit in ISS depends on SpaceX-16.
- Dropping SpaceX-8 *raises* the counts: FE 2,762, V2 916, V3 607.
- **Earth:** dropping SpaceX-21 leaves 189 of FE's 1,722 hits.

**Leave-one-mission-out sign agreement** (does the left-out mission's log2FC have the sign of
the pool of the others, over the full-data hits?). It's weak at k = 3–4: the left-out estimate is
one noisy mission, and the remaining pool has 2–3.
- **ISS, holding out SpaceX-13-ISST, SpaceX-16 or SpaceX-4:**
  - FE 0.91–0.98;
  - V2 0.995–0.998;
  - V3 1.00 (baseline over all genes 0.60–0.66).
- **ISS, holding out SpaceX-8:** FE 0.52, V2 0.52, V3 0.61 (baseline 0.47). SpaceX-8 doesn't
  share the others' direction on these genes.
- **Earth:** FE 0.78–0.89 (baseline ~0.54).

## 5. Time of day at dissection (Q-11)

**Audit** (`jobs/phase3a_compare/time_of_day_audit.yaml`, with a source for every cell):

| Site | Stratum | Flight and ground timing | Status |
|---|---|---|---|
| SpaceX-4 | ISS | ground ran 4 days behind flight on the same timeline (Choi et al. 2020); lights 06:00–18:00 GMT | days offset, clock time unknown |
| SpaceX-8 | ISS | duration only (Beheshti et al. 2019; OSDR) | unknown |
| SpaceX-13-ISST | ISS | flight on orbit 2018-02-06..09; ground "followed the same euthanasia timeline" (OSDR) | timeline matched, clock time unknown |
| SpaceX-16-ISST | ISS | on orbit after 22–23 days; ground in matched conditions (OSDR) | unknown |
| SpaceX-13-LAR | Earth | both arms processed 2018-01-16 at one facility (OSDR) | same day, clock time and order unknown |
| SpaceX-16-LAR | Earth | 2-day recovery then sacrifice (OSDR) | unknown |
| SpaceX-21 | Earth | flight 2021-01-14, habitat ground controls 2021-01-17 (OSDR) | days offset, clock time unknown |

No sample-level field in OSDR (characteristics, factor or parameter values), GeneLab SampleTables
or runsheets records a clock time or an euthanasia order.

**Rhythmic genes.**
- **Source:** Hughes et al. 2009 (PLoS Genet 5:e1000442), Table S1, 24 h cycling probe sets in
  mouse liver (COSOPT and Fisher's G FDR < 0.05, period 20–30 h; CC-BY).
- **Mapping to NCBI Gene:** by probe set ID through NCBI GEO's GPL1261 annotation (dated
  2016-08-09), never by symbol. Of 3,667 probe sets:
  - 164 have no unique Entrez ID (4.5% dropped);
  - 3,503 map to 2,640 genes;
  - 39 of those genes are absent from the pinned GeneLab annotation, leaving 2,601.
- **Background:** genes tested in the stratum that also have a uniquely mapped probe set on the
  array.

| Primary, standard | Hits on array | Rhythmic share of hits | Background share | OR | OR (Mantel–Haenszel, base-mean deciles) |
|---|---|---|---|---|---|
| ISS FE | 1,810 | 36% | 22% | 2.4 | 2.3 (p 2e-46) |
| ISS FE + flag | 919 | 31% | 22% | 1.7 | 1.7 (p 6e-12) |
| ISS V1 (trunc.) | 289 | 45% | 22% | 3.1 | 2.9 |
| ISS V2 | 412 | 55% | 22% | 4.7 | 4.1 (p 2e-42) |
| ISS V3 | 28 | 61% | 22% | 5.6 | 11.7 (p 2e-10) |
| Earth FE | 1,581 | 28% | 22% | 1.5 | 1.6 (p 1e-13) |
| Earth FE + flag | 801 | 24% | 22% | 1.2 | 1.4 (p 4e-4) |

- The stratified odds ratios show the enrichment isn't just "rhythmic genes are well expressed
  and so easier to detect".
- In ISS, the stricter the method, the larger the rhythmic share of its hits.
- **Without rhythmic genes** (dropping them / rerunning BH on the rest):
  - ISS FE 1,271 / 1,097;
  - ISS V2 213 / 102;
  - ISS V3 14 / 4;
  - Earth FE 1,283 / 1,241.

**Core clock genes, ISS** (`core_clock_iss.png`; symbols resolved to NCBI Gene in the pinned
annotation and cross-checked against GPL1261; Arntl appears as its current symbol Bmal1):
- **SpaceX-13-ISST, SpaceX-16-ISST and SpaceX-4 agree in direction:**
  - up in flight: Bmal1 (+1.0 to +3.1 log2), Npas2 (+2.5 to +2.8) and Nfil3;
  - down: Per3, Dbp (−1.6 to −3.1), Tef, Nr1d2, Bhlhe41 and Ciart.
- **SpaceX-8 shows none of it** (every core clock gene within ±0.4 log2, except Ciart +1.2).
- **Hit status:** V3 calls Bmal1, Npas2, Per3, Dbp, Tef, Bhlhe41 and Ciart.
- **Earth:** no consistent pattern. Only Tef (FE, FE + flag, untruncated V1) and Ciart (FE)
  are hits.

**Reading.**
- Genes that peak at opposite times of day move in opposite directions in three of the four ISS
  missions. That's what a difference in time of day between flight and ground euthanasia would
  produce. It's also what a spaceflight-shifted clock would produce, and these data can't tell
  the two apart.
- SpaceX-8 lacks the pattern and is also the site whose removal raises ISS hits most.
- This is a diagnostic. It doesn't establish an artifact, and no biological claim follows from it.

## 6. Recommendation (proposed; Kaitlin decides)

By question, not by hit count:
1. **Question A, these missions:** FE, without the flag as a hit filter.
   - The simulation shows FE is calibrated under A when the SEs are right.
   - The flag doesn't improve FDR under A or under B (§3); keep Q p and I² as per-gene
     annotations.
   - Report V2 as the sensitivity run that would catch understated SEs. On the real data the
     sites scatter 35–56% more than their SEs predict, and φ̂ rising in the highest-expression
     decile (§2.3) points to additive between-mission heterogeneity rather than SE error.
2. **Question B, a new mission:** V3, replacing truncated HKSJ as the random-effects result.
   It's the only method here that held FDR near 0.05 under B across all settings.
   - ISS: 31 genes.
   - Earth: 0. The simulation says Earth with k = 3 has essentially no power under B at these
     effect sizes; report that as a result.
   - Its ISS hits all depend on SpaceX-16, and 61% of them are rhythmic genes, so Q-11 has to
     be resolved before reading them.
3. **Q-11:** times are missing for every site, so the step-3 phase diagnostic is warranted.
   Proposed method in the WS-08 return note, not built. No time covariate was added to site DE.

**Other options considered:**
- keep D-026 (FE + flag) as the headline. The simulation shows "unflagged" does not make FE an
  answer to B (FDR B 0.18–0.48 at the central setting), so the "consistent" wording would need to
  say so;
- V2 as the only headline. It answers A conservatively, and its hits are 63% flagged in ISS.

## 7. Limitations
- The simulation is calibrated, not real:
  - effect sizes are normal;
  - τ² is independent of μ;
  - the kind-1/kind-2 split is fixed at 0.5;
  - the noisy-SE generator is only approximate.
- Every method treats the site SEs as known; V2 is the only one that absorbs misestimated SEs.
- V3's prior is treated as known (estimated from ~13,000 genes).
- The rhythmic list comes from one study's liver: one strain, males, 1-h sampling on arrays. Our
  sites use different strains, sexes and ages.
- No robust prior fit (Phipson 2016) and no expression-trended prior. The second matters only
  for the "any finite p" set.
- The figures use the reference palette's documented default colours. Its validator (Node) wasn't
  run, and every mark is also named in text.

## 8. Rebuild
```bash
# WSL/Ubuntu, after `fedkg`. Needs WS-07's per-gene tables. Rhythmic-gene inputs (downloaded
# 2026-11-13; checksums in circadian/rhythmic_mapping.json): Hughes 2009 Table S1 and GEO
# GPL1261.annot.gz under $FEDKG_WORKDIR/data/circadian/ (see jobs/phase3a_compare/circadian.py).
uv run python jobs/phase3a_compare/simulate.py      # ~14 min on 20 workers
uv run python jobs/phase3a_compare/compare.py
uv run python jobs/phase3a_compare/circadian.py
uv run python jobs/phase3a_compare/report.py
```
