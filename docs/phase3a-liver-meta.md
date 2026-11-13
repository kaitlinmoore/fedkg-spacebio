# Phase 3a: federated differential-expression meta-analysis of mouse liver

_WS-07, run 2026-11-12/13. Methods summary in [`methodology.md`](methodology.md) §2.1 and §5.4;
interfaces in [`CONTRACTS.md`](../CONTRACTS.md) §3–4; sites and strata in [`SITES.md`](../SITES.md)
(D-020) and [`census-liver.md`](census-liver.md)._

**Question.** For mouse liver, flight vs ground control, which genes change in the same direction
across missions, analyzed separately for mice euthanized on the ISS and mice returned alive and
euthanized on Earth? Each mission is a simulated site that runs its own DE and shares only per-gene
summaries.

**What this is not.** No biological claim is made here. Gene names below are results, not
findings; a literature check comes later. The sites are simulated over public data (see the
README's limitations).

## 1. Cohort

| Stratum | Client | Studies | n flight / ground | Design | Tested Ensembl genes¹ | Kept (1:1 NCBI Gene) | Dropped in mapping |
|---|---|---|---|---|---|---|---|
| ISS | SpaceX-4 | OSD-47, OSD-48 | 8 / 8 | `~ study + condition` | 15,076 | 13,434 | 10.9% |
| ISS | SpaceX-8 | OSD-137 | 6 / 6 | `~ condition` | 15,907 | 13,928 | 12.4% |
| ISS | SpaceX-13-ISST | OSD-245 | 10 / 10 | `~ condition` | 16,597 | 14,165 | 14.7% |
| ISS | SpaceX-16-ISST | OSD-379 | 20 / 20 | `~ age + condition` | 14,936 | 13,380 | 10.4% |
| Earth | SpaceX-13-LAR | OSD-245 | 10 / 9 | `~ condition` | 16,579 | 14,219 | 14.2% |
| Earth | SpaceX-16-LAR | OSD-379 | 15 / 15 | `~ age + condition` | 14,795 | 13,440 | 9.2% |
| Earth | SpaceX-21 | OSD-463 | 9 / 9 | `~ condition` | 15,873 | 13,921 | 12.3% |

¹ Base mean ≥ 10 and finite log2FC, SE and p (the primary inclusion rule), before mapping.

- **Mapping drops** are mostly genes with no Entrez ID (1,244–2,319 per site); one-to-many
  (67–79) and many-to-one (29–35) make up the rest. Source: GeneLab GL-DPPD-7110-A, Ensembl 112,
  SHA-256 `123a0899…f54044e2`.
- **SpaceX-13-LAR has 10 flight and 9 ground animals** because OSDR lists nine LAR ground
  controls (animal labels G2–G10; no G1 in the sample metadata, counts, runsheet or SampleTable).
  The metadata doesn't say why.
- **SpaceX-8's ISS euthanasia is confirmed** by the study's own paper (Beheshti et al. 2019,
  *Sci Rep* 9:19195: "these animals were sacrificed on-orbit"). It stays in the ISS stratum; an
  ISS-without-SpaceX-8 variant is still reported as a leave-one-out.
- Each client fits **only its own flight and ground samples** (Kaitlin, WS-07), so a mission's
  ISS and Earth clients share no data or dispersion estimates. No sample or animal is in two
  clients (checked when manifests are built).
- Counts are GeneLab's RSEM unnormalized counts, rounded half-to-even (4–7% of cells change,
  by at most 0.5). ERCC spike-ins are removed.

## 2. Methods in brief

1. **Site DE** (pydeseq2 0.5.4): Wald test, unshrunk MLE log2FC and `lfcSE`, contrast flight vs
   ground. Cook's refit at ≥ 7 replicates; otherwise defaults.
2. **Summary** (the federation boundary): per gene NCBI Gene ID, log2FC, SE, base mean and three
   inclusion flags. No counts, per-sample values or sample names. Validated on both sides.
3. **Inclusion.** Primary: base mean ≥ 10 at every site of the stratum (complete case). Genes
   whose p is NaN because of Cook's outlier filtering are also excluded (DESeq2 withholds their
   p-value because one sample may drive the estimate). Sensitivity variants: ≥ 3 sites; pydeseq2's independent filter; any finite
   p; ISS without SpaceX-8. Plus the rRNA-removed count matrix for everything.
4. **Pooling** at the FLARE server, per gene: inverse-variance fixed effect (FE); REML τ²; random
   effects (RE) with truncated Hartung–Knapp–Sidik–Jonkman (HKSJ) intervals on t(k−1); Cochran's
   Q and I². Benjamini–Hochberg (BH) within each stratum and variant.
5. **FLARE 2.9.1 simulator**, one round: a custom `ModelController` asks every client for its
   summary, checks the sender's identity and the schema, writes what it received, then pools.

## 3. The headline, and why it changed after the results

**Pre-specified:** RE with truncated HKSJ, BH q < 0.05. **Result: 0 genes in either stratum, in
every variant and with both count matrices.** This is structural, not a bug:

| | ISS (k = 4, t with 3 df) | Earth (k = 3, t with 2 df) |
|---|---|---|
| Largest \|t\| observed (truncated HKSJ) | 6.6 | 6.7 |
| \|t\| needed to pass BH as the single top gene | 82 | 509 |
| … if 100 genes passed together | 18 | 51 |
| … if 1,000 genes passed together | 8.1 | 16.0 |
| Smallest BH q, truncated / untruncated HKSJ | 0.78 / 0.38 | 0.97 / 0.49 |
| Genes with p < 0.05, truncated / untruncated | 278 / 1,062 | 40 / 909 |
| Genes where truncation applied (HKSJ q < 1) | 73% | 76% |

With 2–3 degrees of freedom the t distribution's tails are so heavy that no single gene's
p-value can survive correction over ~13,000 genes. Truncated HKSJ is also conservative here:
fewer than 5% of genes reach p < 0.05.

**Minimum detectable effect.** The smallest |log2FC| that passes BH when the effect is perfectly
consistent across sites (τ² = 0, so the truncated HKSJ SE equals the FE SE), at the median gene's
FE SE (ISS 0.065, Earth 0.092; 10th percentile in brackets). It's the effect at which the
expected test statistic just reaches the threshold, so roughly 50% power, and for HKSJ a best
case (any scatter between sites only raises it).

| Passing BH together with | ISS: FE | ISS: truncated HKSJ | Earth: FE | Earth: truncated HKSJ |
|---|---|---|---|---|
| 1 gene (rank 1) | 0.30 [0.21] | 5.4 [3.7] | 0.43 [0.29] | 47 [32] |
| 100 genes | 0.23 [0.16] | 1.15 [0.80] | 0.33 [0.22] | 4.7 [3.2] |
| 1,000 genes | 0.19 [0.13] | 0.53 [0.36] | 0.27 [0.18] | 1.5 [1.0] |

For scale, the median |log2FC| of the FE hits is 0.24 (ISS) and 0.34 (Earth). The FE thresholds
are those of a normal reference; the HKSJ ones are about 3× to 110× larger because of the t(k−1) quantile.

**New headline (D-026, decided 2026-11-13 by Kaitlin, after seeing the results above):**

- **FE hit:** fixed-effect BH q < 0.05.
- **Heterogeneity flag:** Cochran's Q p < 0.10, with I² alongside.
- **Consistent FE hit:** an FE hit without the flag, read as *a change consistent across these
  missions*. With 3–4 sites, "no flag" means "no detectable heterogeneity", which is weak: a
  site with a wide SE can point the other way without raising Q.
- RE-HKSJ remains a **secondary** result.

FE answers "what is the precision-weighted average change in these missions?". It does not
answer "what would a new mission show?" (the RE question), and a large, precise site can drive
it. Because this switch was made after seeing results, it should be read as exploratory.

## 4. Checks

**Check 1: federated = central.** For each stratum and count matrix, the same pipeline was run
in one process with no FLARE and compared with the FLARE run.

| Run | Transport (sent vs received) | Site DE re-run | Pooled tables |
|---|---|---|---|
| ISS / Earth × standard / rRNA-removed | bit-identical | bit-identical | identical (max \|Δ\| = 0, every column, every variant) |

This tests transport and wiring. FE pooling of the same summaries is the same computation either
way, so it says nothing about a *benefit* of federation.

**Check 2: site DE vs GeneLab's published DESeq2.** For each of the 10 source contrasts:
(2a) pydeseq2 with GeneLab's design (all samples in the study, GeneLab's groups); (2b) our
two-arm fit. GeneLab's SE is recovered as `Log2fc / Stat`.

- 2a, on well-estimated genes (both SEs < 2): r(log2FC) 0.92–0.99, r(SE) 0.97–0.99; median SE
  ratio 0.96–1.00; median |Δlog2FC| 0.016–0.034 and 90th percentile 0.10–0.20. pydeseq2
  reproduces DESeq2 closely.
- Pearson r over *all* genes is lower (0.50–0.98) because of a few near-separation genes (one
  arm almost all zeros: 1–164 per contrast), where pydeseq2's SE runs into the hundreds and
  DESeq2's stays at 4–9. These have huge SEs and get almost no weight in pooling.
- 2b differs from GeneLab more, as expected for a different design (r_well 0.59–0.98).
  2a vs 2b directly: median |Δlog2FC| 0.0006–0.013 per contrast; 92–99.9% of genes within 0.01,
  except OSD-379 LAR 32-week (22% within 0.01, 90% within 0.017). In OSD-379, 0.1–1.3% of genes
  differ by more than 2 log2 units: samples that the all-group fit treats differently (Cook's
  outlier handling and shared dispersions). Not investigated further.
- GeneLab's normalization varies by gene (consistent with tximport length offsets), which no
  counts-only fit can reproduce. This is a small, known source of difference.

**Check 3: pooling vs statsmodels on the real genes** (`statsmodels.stats.meta_analysis`, every
variant, standard counts):

| | ISS | Earth |
|---|---|---|
| FE estimate, FE variance, Cochran's Q, I² | agree to ≤ 5e-12 (relative) | same |
| HKSJ variance given statsmodels' PM τ² | agrees to 9e-16 | 7e-16 |
| τ² REML vs Paule–Mandel: Pearson / Spearman | 0.994 / 0.901 | 0.999 / 0.961 |
| … median / 90th pct \|Δτ²\| | 0.0008 / 0.013 | 0.0012 / 0.015 |
| τ² REML vs DerSimonian–Laird: Pearson / Spearman | 0.657 / 0.922 | 0.935 / 0.967 |
| Genes with τ² = 0: REML / PM / DL | 47% / 47% / 47% | 42% / 43% / 43% |
| RE-HKSJ hits with REML / PM / DL τ² | 0 / 0 / 0 | 0 / 0 / 0 |

- REML and PM put τ² at 0 for about the same share of genes but not always the same genes (they
  agree on 90–94%): PM and DL are at 0 exactly when Q ≤ k − 1, while REML's condition weights the
  sites differently.
- statsmodels' DL returns **untruncated** τ² (negative on 5,600–10,200 genes per variant) and an
  I² without the floor at 0. Both are truncated before comparing; the counts are in `check3.json`.
- REML converged for every gene (median 20 iterations; maximum 200 of 1,000 allowed). The REML
  root also matches a direct numerical maximization of the restricted likelihood
  (`tests/test_meta.py`).

## 5. Results (standard counts)

**Headline**

| | ISS (12,692 genes) | Earth (12,948 genes) |
|---|---|---|
| FE hits (BH q < 0.05) | **1,926** (1,167 up, 759 down) | **1,722** (925 up, 797 down) |
| … consistent (Q p ≥ 0.10) | **971** | **870** |
| … heterogeneous (Q p < 0.10) | 955 | 852 |
| All genes with the heterogeneity flag | 2,932 (23%) | 3,986 (31%) |
| RE-HKSJ hits (secondary) | 0 | 0 |
| τ²: median / 75th / 95th pct | 0.0012 / 0.019 / 0.11 | 0.0071 / 0.053 / 0.32 |
| I²: median / 75th / 95th pct | 8% / 50% / 76% | 20% / 65% / 88% |

"Up" means higher in flight. About half of the FE hits carry the heterogeneity flag, and the
genes with the smallest FE p-values mostly do (9 of the top 10 by FE p in ISS, 10 of 10 in Earth).
The strongest FE signals tend to come from genes where the missions disagree, often driven by one
precise site, so the consistent subset is the more conservative reading of the headline.

**Top 10 consistent FE hits by FE p** (`forest_top10_consistent.png`). Results only.

| ISS | log2FC (SE) | FE q | Q p | I² | | Earth | log2FC (SE) | FE q | Q p | I² |
|---|---|---|---|---|---|---|---|---|---|---|
| Chka | 1.38 (0.16) | 8.2e-15 | 0.12 | 49% | | Rnase6 | −0.78 (0.12) | 1.8e-08 | 0.42 | 0% |
| Tef | −1.20 (0.17) | 3.8e-10 | 0.22 | 32% | | Trim2 | 0.73 (0.11) | 3.1e-08 | 0.25 | 27% |
| 4931408D14Rik | −0.79 (0.12) | 4.6e-09 | 0.38 | 3% | | Cyp2a5 | −0.84 (0.14) | 1.2e-06 | 0.90 | 0% |
| Mcm10 | −0.58 (0.09) | 9.6e-09 | 0.11 | 51% | | Aprt | 0.67 (0.12) | 2.8e-06 | 0.13 | 51% |
| Per3 | −1.70 (0.26) | 1.1e-08 | 0.24 | 29% | | Mad2l2 | 0.61 (0.11) | 4.9e-06 | 0.38 | 0% |
| Bcl2l11 | −0.50 (0.08) | 1.7e-08 | 0.21 | 34% | | Porcn | 0.54 (0.10) | 5.7e-06 | 0.12 | 53% |
| Dnmt3b | 0.93 (0.14) | 1.8e-08 | 0.23 | 30% | | Tjp3 | 0.80 (0.14) | 6.2e-06 | 0.92 | 0% |
| Dbp | −1.90 (0.29) | 1.9e-08 | 0.12 | 49% | | Npnt | 1.30 (0.23) | 7.1e-06 | 0.20 | 37% |
| Per2 | −0.92 (0.15) | 1.9e-07 | 0.14 | 46% | | Ncf1 | −0.62 (0.11) | 7.1e-06 | 0.16 | 45% |
| Prob1 | 0.49 (0.08) | 2.8e-07 | 0.32 | 14% | | Col14a1 | −0.34 (0.06) | 1.1e-05 | 0.33 | 9% |

In the ISS forest plot, SpaceX-8 often doesn't follow the other three sites, but for these genes
its intervals are the widest, so Q doesn't flag it. That is what "no detectable
heterogeneity" means at k = 4.

**ISS vs Earth (descriptive).** On the 12,477 genes in both primary analyses, the FE estimates
correlate weakly (r = 0.20). 312 genes are FE hits in both strata (255 with the same sign); 80 are
consistent hits in both (72 with the same sign). The strata differ in euthanasia location and
also in mission, duration and handling, so this comparison isn't a test of anything.

**Sensitivity** (`results/phase3a/sensitivity.csv`; FE hits / consistent; overlap with the
primary's FE hits):

| Variant | ISS genes | ISS FE hits / consistent | shared with primary | Earth genes | Earth FE hits / consistent | shared |
|---|---|---|---|---|---|---|
| primary | 12,692 | 1,926 / 971 | – | 12,948 | 1,722 / 870 | – |
| ≥ 3 sites | 13,428 | 1,955 / 985 | 1,896 | 12,948 | 1,722 / 870 | 1,722 |
| independent filter | 13,593 | 1,979 / 1,000 | 1,896 | 14,255 | 1,799 / 915 | 1,699 |
| any finite p | 18,625 | 1,863 / 958 | 1,678 | 19,088 | 1,691 / 846 | 1,517 |
| ISS without SpaceX-8 | 12,714 | 2,765 / 1,928 | 1,878 | – | – | – |
| rRNA-removed counts (primary) | 12,692 | 1,923 / 970 | 1,923 | 12,948 | 1,726 / 875 | 1,722 |

- Inclusion rules and the count matrix barely matter: FE estimates correlate ≥ 0.9999 with the
  primary's, and most hits are shared. (In Earth, k = 3, "≥ 3 sites" is the primary.)
- Dropping SpaceX-8 raises the FE hit count to 2,765 and the consistent count to 1,928. With
  SpaceX-8 in, its disagreeing estimates flag many genes as heterogeneous; without it, k = 3
  and Q has even less power. More hits is not evidence that the smaller set is better.
- RE-HKSJ: 0 hits in every variant.

## 6. Limitations

- **Site = mission = everything.** Strain, sex, age, duration, hardware, diet, handling, time to
  dissection and processing batch all line up with site. τ², I² and the heterogeneity flag
  measure "everything that differs between missions", not variation in spaceflight's effect.
- **Few sites.** k = 4 (ISS) and 3 (Earth). τ² and I² are very imprecise, Q has little power,
  and the RE question (what would a new mission show?) can't be answered with BH-level
  confidence. The FE headline answers a narrower question.
- **The headline was chosen after seeing results**, so its hit counts are exploratory.
- **Time of day of dissection** isn't recorded as a covariate and may differ between flight and
  ground arms within a mission and between missions. Genes with strong daily expression rhythms
  could differ for that reason. No interpretation is made here.
- **FE and a dominant site.** SpaceX-16 (20 vs 20 and 15 vs 15) has the smallest median SE in
  each stratum. Its median share of the FE weight is 40% (ISS) and 39% (Earth), and it is the
  largest-weight site for 69% of ISS genes and 44% of Earth genes.
- **Near-separation genes** get huge SEs from pydeseq2 (Check 2). They carry almost no FE weight,
  but they're in the tables.
- **Cook's-flagged genes** at any site are excluded from the primary analysis (NaN p).
- **Mapping drops 9–15%** of each site's tested genes, mostly genes with no Entrez ID.
- **GeneLab's gene-length offsets** aren't reproduced (counts-only fits).
- **No privacy claim.** RE and the heterogeneity flag need per-site estimates at the server.
  Federated FE = central FE by construction (Check 1).

## 7. Outputs and how to rebuild

Under `$FEDKG_WORKDIR/results/phase3a/<stratum>/<counts>/`:
- `flare/sent/`, `flare/site_summaries/`: what each client sent and what the server received.
- `flare/<variant>/meta_genes.csv.gz`: the server's pooled table.
- `flare/<variant>/per_gene.csv.gz` + `per_gene.schema.json`: the **stable per-gene table**
  (schema `phase3a-per-gene/1`, CONTRACTS.md §4). It holds estimates, SEs, Q, I², τ², flags and
  each site's inputs, for comparing later methods (e.g. empirical Bayes) gene by gene.
- `flare/headline.json`, `flare/primary/{forest_top10,forest_top10_consistent,volcano,volcano_re}.png`.
- `central/check1.json` (Check 1), `check3.json` (Check 3).
- `results/phase3a/check2/` (Check 2) and `results/phase3a/sensitivity.csv`.

```bash
# WSL/Ubuntu, after `fedkg`. Data: count matrices, DE tables and metadata for OSD-47, -48, -137,
# -245, -379, -463 (OSDR, accessed 2026-11-12) and the GeneLab annotation table (2026-11-13).
for s in iss earth; do for c in standard rrnarm; do
  uv run python jobs/phase3a_meta/job.py --stratum $s --counts $c
  uv run python jobs/phase3a_meta/central.py --stratum $s --counts $c
  uv run python jobs/phase3a_meta/report.py --stratum $s --counts $c
done; uv run python jobs/phase3a_meta/check_pooling.py --stratum $s; done
uv run python jobs/phase3a_meta/check_genelab.py
uv run python jobs/phase3a_meta/report.py --sensitivity
uv run pytest -m slow     # synthetic FLARE end to end
```
