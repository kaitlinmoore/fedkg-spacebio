# Phase 3a: time of day and the liver clock (Q-11)

WS-09 follow-up to the time-of-day check in
[`phase3a-method-comparison.md`](phase3a-method-comparison.md) §5. Code:
- `src/fedkg_spacebio/evaluation/clock_reference.py` (reference fit);
- `src/fedkg_spacebio/federation/site_clock.py` (site side and boundary);
- `src/fedkg_spacebio/evaluation/clock_sim.py` and `clock_compare.py`;
- `jobs/phase3a_clock/`.

Results are written to `$FEDKG_WORKDIR/results/phase3a/circadian-phase/` and are not in git.

> **What this can't tell.** A flight–ground difference in clock time is consistent with
> euthanizing the two arms at different times of day, **or** with spaceflight shifting the
> clock. Nothing here separates the two. It also can't separate a shifted clock from one whose
> amplitude changed (§2.4). Actual euthanasia clock times and each ground facility's lighting
> schedule would be decisive. No biological claim is made.

## 1. Question

WS-08 found three things:
- ISS hits are enriched for liver-cycling genes;
- in three of four ISS missions, genes peaking at opposite times of day move in opposite
  directions in flight;
- no site records when animals were euthanized.

This diagnostic asks, per site:
1. how large any flight–ground difference in clock time is, in hours;
2. whether that difference has the shape of a clock-time difference;
3. whether SpaceX-8, the ISS site without the pattern, differs from the other three.

Only per-site aggregates leave a site.

## 2. Method

### 2.1 Why the planned estimator was replaced
The plan approved in D-030 was:
- z-score each clock gene within the site;
- give each animal the time t maximizing the correlation of its genes with cos(2π(t − φ_g)/24);
- compare the arms' circular means.

Checked before building, this can't measure the shift. Centring a gene within a site subtracts
its site mean, and that mean is what carries absolute time. For two arms at times t_f and t_g,
each arm's deviation from the site mean is ±½[cos(ω(t_f − φ)) − cos(ω(t_g − φ))]. That's a
cosine peaking 6 h from the midpoint, with opposite signs for the two arms. Template matching
then places the arms about 12 h apart **whatever the true shift**. The size of the shift ends
up in the deviation's amplitude instead.

`tests/test_site_clock.py::test_zscored_template_matching_puts_arms_12h_apart_whatever_the_shift`
reproduces this for true shifts of 1 h and 6 h.

### 2.2 The clock plane
Take each panel gene's peak time φ_g and amplitude A_g from a reference time course (§3), and
write a_g = A_g cos(ωφ_g), b_g = A_g sin(ωφ_g), with ω = 2π/24. A healthy clock at time t
gives, across genes,

    x_g(t) − m_g = cos(ωt)·a_g + sin(ωt)·b_g,

so every clock state is a point on the unit circle in the plane spanned by **a** and **b**.

Inside each site:
1. **Normalize:** log2(median-of-ratios normalized count + 1), with size factors from all genes
   in the site.
   - *Not VST:* the panel genes are well expressed, where VST ≈ log2. VST would add a
     dispersion fit, and the amplitudes have to be on the reference's log2(count + 1) scale.
2. **Difference:** per panel gene, OLS on condition plus the site's DE covariates (study for
   SpaceX-4, age for SpaceX-16). The flight coefficients form **d**.
3. **Plane fit:** d ≈ α**a** + β**b**. The **chord** (α, β) is flight's point on the circle minus
   ground's.
   - **Size:** L = |chord| = 2|sin(ωΔ/2)|, so |Δ| = (24/π)·arcsin(L/2). That assumes both arms
     have the reference amplitude. L ≥ 2 can't happen for two points on the unit circle, so
     it's reported as "not identifiable", never clipped.
   - **Midpoint:** the chord's angle gives the midpoint time, up to 12 h. "Flight later" and
     "flight earlier" are both reported.
     - A reading is "light-phase consistent" when both implied arm times fall in ZT −1 to 13:
       lights on, with 1 h of margin for the CT/ZT gap.
     - Using it assumes euthanasia happened in the light phase. §5.4 shows that assumption may
       fail where a facility lit on GMT.
   - **Coherence (R²_plane):** the share of |d|² that lies in the plane, i.e. how much of the
     clock-gene difference a clock-time difference explains.
4. **Test:** permute flight/ground labels within covariate blocks. It's exact when there are
   ≤ 20,000 relabelings (SpaceX-4: 5,040; SpaceX-8: 924); otherwise the observed labeling plus
   9,999 random ones, which makes the smallest attainable p 1e-4.
5. **Intervals:** 2,000 percentile-bootstrap resamples, stratified by arm × block. Each is paired
   with a residual-bootstrap replicate of the reference fit, so the hours carry the
   reference's calibration uncertainty. Also reported:
   - the sample-only interval;
   - |Δ| if every reference amplitude were 25% lower or higher.

### 2.3 Alternatives considered
- **Clock correlation distance** (Shilts et al. 2018, PeerJ 6:e4327) compares a group's
  clock-gene correlation matrix with a reference.
  - It needs samples "taken from multiple parts of the circadian cycle", and the authors
    suggest about 30.
  - Our arms are 6–20 animals euthanized in short windows. A healthy clock sampled that way
    scores as disrupted.
- **ZeitZeiger and CYCLOPS** need far more training data, or samples per site, than this.
- **The planned per-animal template match:** §2.1.

### 2.4 What each output can and can't show
- **L and its p:** whether flight and ground differ along the clock.
- **R²_plane:** whether the difference looks like clock movement. Validation (§4) shows real
  time differences in the reference data give R²_plane with median 0.82 (10th percentile 0.62).
  A synthetic shuffled clock gives about 0.62, so R²_plane **can't by itself separate a shifted
  clock from a disrupted one**. Reading bands, fixed before any site ran:
  - ≥ 0.73: clock-shaped;
  - 0.50–0.73: ambiguous;
  - < 0.50: not clock-shaped.
- **A shift vs a weaker clock:** if flight's clock is weaker at the same time (amplitude c < 1),
  the chord is radial, with length 1 − c. That reads as a shift of (24/π)·arcsin((1 − c)/2):
  - 50% weaker reads as about 1.9 h (confirmed in simulation);
  - a completely flat flight clock reads as 4 h.

  Only absolute phase (§6) could separate these.

## 3. Reference

**Sources:**

| Series | What | Design | Use |
|---|---|---|---|
| GSE11923 (Hughes et al. 2009) | Affymetrix Mouse 430 2.0, MAS5 | Liver; adult male C57BL/6J; LD 12:12, then constant darkness; hourly CT18–CT65; 48 arrays, each a pool of 3–5 mice | Peak times φ_g (log2 signal) |
| GSE54651 (Zhang et al. 2014) | RNA-seq, DESeq2-normalized counts (mm9, HTSeq) | Liver; C57BL/6; LD then constant darkness; every 6 h, CT22–CT64; 8 samples, no replicates | Amplitudes A_g and baselines on the log2(count + 1) scale |

- **Time frame:** circadian time (CT) of the references. The sites ran LD 12:12, i.e. zeitgeber
  time (ZT). After about 2 days of darkness the two frames differ by under 1 h. This isn't
  corrected.
- **Why two series:** GSE11923's hourly sampling pins the peak times. Arrays compress fold
  changes, though, and the hours are read off the amplitude scale, so amplitudes come from
  RNA-seq.
  - The RNA-seq/array amplitude ratio has median 1.0 (range 0.87–3.3; Npas2 3.3, Ciart 1.9).
- **Mapping:**
  - **GSE11923:** probe set → NCBI Gene through GPL1261, unique Entrez IDs only (38,664 of
    45,101 probe sets).
  - **GSE54651:** Ensembl → NCBI Gene with the pinned GeneLab table, 1:1 only.
    - 25,135 of 37,310 IDs kept; 6,428 dropped as no longer in Ensembl 112 (mm9-era IDs),
      plus 5,598 with no Entrez, 111 one-to-many and 38 many-to-one: a 32.6% drop rate.
    - All ten panel genes map.
- **Cosinor:** y = m + a·cos ωt + b·sin ωt by OLS, with an F test against a constant and BH q
  across all probe sets.

**Panel.** The rule was fixed before any site data was opened:
1. best probe set with q < 0.001, R² ≥ 0.5 and peak/trough ≥ 1.5-fold;
2. passing probe sets agree within 2 h;
3. rhythmic on GSE54651 (p < 0.05), with a peak within 3 h of GSE11923's.

| Gene | Kept | Peak (CT, GSE11923) | Array R² | Reason if dropped |
|---|---|---|---|---|
| Bmal1 (Arntl) | yes | 22.3 | 0.87 | |
| Npas2 | yes | 0.6 | 0.56 | |
| Nfil3 | yes | 22.8 | 0.76 | |
| Cry1 | yes | 19.8 | 0.88 | |
| Per2 | yes | 14.0 | 0.83 | |
| Per3 | yes | 11.4 | 0.62 | |
| Dbp | yes | 10.2 | 0.88 | |
| Nr1d1 | yes | 6.5 | 0.81 | |
| Nr1d2 | yes | 9.8 | 0.86 | |
| Ciart | yes | 11.1 | 0.67 | |
| Clock | no | | 0.76 | passing probe sets disagree on peak by 2.6 h |
| Tef | no | | 0.73 | passing probe sets disagree on peak by 2.6 h |
| Per1 | no | | 0.40 | R² < 0.5 |
| Hlf | no | | 0.36 | R² < 0.5 |
| Bhlhe40 | no | | 0.35 | R² < 0.5 |
| Bhlhe41 | no | | 0.21 | not rhythmic (q 0.015, 1.03-fold) |

All ten panel genes pass every site's expression rule (mean normalized count ≥ 10), so k = 10
everywhere.

## 4. Validation (before any site was run)

| Check | Result |
|---|---|
| GSE11923, absolute time of each array, leave-one-timepoint-out (gene baselines known; best case) | median error 0.64 h, 90th percentile 1.99 h |
| GSE11923, 3 vs 3 held-out arrays, true shift 2 / 4 / 6 h (clock plane, array amplitudes) | median \|Δ̂\| 1.9 / 4.0 / 5.7 h; median absolute error 0.6 / 1.1 / 1.3 h (90th percentile 1.2 / 2.2 / 2.4); midpoint error median 0.9–1.1 h; R²_plane median 0.77 / 0.81 / 0.87 |
| GSE54651 pairs 6 h apart (cross-platform: array peaks, RNA-seq amplitudes refitted without the pair) | median \|Δ̂\| 4.9 h; 14 of 16 identifiable; R²_plane median 0.87 |
| GSE54651 pairs 12 h apart (L = 2, the edge) | 4 of 8 reported not identifiable, as designed |
| GSE54651, absolute time, leave-one-out | median error 0.74 h, max 2.3 h |

**Synthetic site-like data** (`clock_sim.py`; 200 replicates per cell; n = 6, 10 or 20 per arm;
noise 0.15–0.6 log2; within-arm spread 0, 1 or 2 h; ground time uniform in ZT0–12):
- **Null (no shift):** permutation test rejects at 2–8.5% (nominal 5%).
- **Bias:** under 0.05 h without spread. |Δ| is underestimated by up to 0.3 h at 1 h spread and
  0.2–0.9 h at 2 h spread (more for larger shifts), because spread pulls each arm's centroid
  inside the circle.
- **95% interval for L:** covers 83–96% of the time with spread ≤ 1 h. At 2 h spread and 4–6 h
  shifts it falls to 0.79–0.87 at n = 10 and 0.61–0.88 at n = 20, where bias dominates. Read
  the intervals as approximate.
- **Damped flight clock (50%), no shift:** reads as 1.9 h (power 1.0, R²_plane 0.97–0.99).
- **Shuffled flight clock:** reads as 4.3–4.6 h, R²_plane 0.62.

**At each site's own n and noise** (spread 0, noise = the site's noise scalar):
- interval coverage 0.86–0.96;
- power for a 2 h shift: 0.98–1.0 at every site except SpaceX-8 (0.47);
- power for 4 h: 0.96 or more everywhere.

**Animals per arm for 80% power** (noise at each site's level):
- 2 h: 4–8 per arm, but 15 at SpaceX-8's noise;
- 4 h: 4 (6 at SpaceX-8's);
- 6 h: 4.

Three per arm can never reach p < 0.05: there are only 20 relabelings, so the smallest p is 0.05.

The noise scalar (the median within-arm residual SD over the panel: 0.75–1.68 log2) includes
clock variation among animals euthanized at different times. Treating all of it as noise makes
these power figures conservative.

## 5. Results

### 5.1 One row per client

|Δ| uses reference amplitudes. Intervals: joint (sample and reference bootstrap), then
sample-only.

| Client | Stratum | n flight / ground | \|Δ\| h (95% CI) | sample-only CI | amp ×0.75 / ×1.25 | p | R²_plane (perm. median) | Light-phase reading: flight / ground CT |
|---|---|---|---|---|---|---|---|---|
| SpaceX-4 | ISS | 8 / 8 | **2.5** (1.6–3.5) | 1.6–3.4 | 3.4 / 2.0 | 0.019 | 0.86 (0.77) | flight earlier: 3.6 / 6.1 |
| SpaceX-8 | ISS | 6 / 6 | **0.5** (0.3–4.1) | 0.3–4.0 | 0.7 / 0.4 | 0.66 | 0.76 (0.83) | flight later: 5.1 / 4.6 |
| SpaceX-13-ISST | ISS | 10 / 10 | **3.5** (2.3–5.1) | 2.4–4.8 | 4.9 / 2.8 | ≤ 1e-4 | 0.86 (0.66) | flight earlier: 5.4 / 8.9 |
| SpaceX-16-ISST | ISS | 20 / 20 | **4.1** (2.6–5.7) | 2.8–5.6 | 5.7 / 3.2 | ≤ 1e-4 | 0.95 (0.83) | flight earlier: 1.9 / 6.0 |
| SpaceX-13-LAR | Earth | 10 / 9 | **2.3** (0.8–3.9) | 0.9–3.7 | 3.1 / 1.8 | 0.012 | 0.81 (0.61) | flight earlier: 0.0 / 2.3 |
| SpaceX-16-LAR | Earth | 15 / 15 | **1.2** (0.4–2.4) | 0.4–2.4 | 1.6 / 1.0 | 0.042 | 0.56 (0.70) | flight earlier: 7.4 / 8.7 |
| SpaceX-21 | Earth | 9 / 9 | **2.1** (0.6–4.0) | 0.6–3.9 | 2.8 / 1.7 | 0.026 | 0.65 (0.70) | flight earlier: 10.4 / 12.5 |

- **Identifiability:** every shift and every interval is identifiable (L < 2).
- **The other reading:** each row's alternative reading is 12 h away on the midpoint and puts
  at least one arm in the dark phase. SpaceX-8's alternative is flight earlier at about CT 17.
- **Uncertainty sources:** the reference adds little width beyond the sample bootstrap
  (compare the two interval columns). A ±25% amplitude error moves |Δ| by about −20% to +40%.

### 5.2 Does SpaceX-8 differ from the other ISS sites?
**SpaceX-8 shows no detectable flight–ground clock difference. The data don't show that it
differs from the other three.**
- Its estimate is 0.5 h (p 0.66), but the interval reaches 4.1 h.
- Its chord against the pooled chord of the other three: χ²(2) = 3.7, **p = 0.16**.
- It has the smallest arms (6 / 6) and the highest noise of any site (1.68, vs 0.75–1.27).
  - At that n and noise, power is 0.47 for a 2 h shift and 0.96 for 4 h.
  - So a difference as large as SpaceX-16-ISST's would very likely have been seen. One as
    small as SpaceX-4's (2.5 h) could have been missed.
- Strain doesn't explain it. SpaceX-16 is BALB/c like SpaceX-8 (`SITES.md`), and it shows the
  largest shift.

The other three ISS sites also differ from each other: chord heterogeneity Q = 40.4, df 4,
p 4e-8. Their sizes are 2.5, 3.5 and 4.1 h, and their midpoints CT 4.9, 7.1 and 4.0. All four ISS
sites together: Q = 44.1, df 6, p 7e-8.

The χ² p-values rest on bootstrap covariances from 6–20 animals per arm, so treat them as
approximate.

## 5.3 Interpretation (what each pattern is consistent with)
- **SpaceX-4, -13-ISST and -16-ISST: a clock-shaped difference of 2.5–4.1 h.**
  - R²_plane is 0.86–0.95, all in the "clock-shaped" band.
  - Under the light-phase reading, all three point the same way: flight is earlier in the
    clock than ground.
  - That's consistent with any of:
    - ground animals euthanized 2.5–4 h later in their light cycle than flight animals;
    - a flight clock running that much earlier;
    - a flight clock that is about 64–100% weaker at about the same time (§2.4). A pure
      amplitude loss can mimic at most 4 h, so it only just reaches SpaceX-16-ISST's 4.1 h
      point estimate (interval 2.6–5.7 h).

    These can't be separated here.
  - Within-arm spread in euthanasia times would make the true size somewhat larger (§4).
  - This matches WS-08's gene-level pattern: Bmal1 and Npas2 up and Dbp, Per3 and Nr1d2 down in
    flight is what an earlier light-phase clock time looks like, since Dbp rises and Bmal1 falls
    through the light phase.
- **SpaceX-8: no detectable difference, with an interval that includes up to 4.1 h** (§5.2).
- **Earth stratum: smaller differences, 1.2–2.3 h, all with p < 0.05.**
  - SpaceX-13-LAR is clock-shaped (R² 0.81), although both arms were processed on the same day
    at one facility. That's consistent with processing order within the day, with a clock
    change after return, or with a flight amplitude change (a 50% loss reads as about 1.9 h).
  - SpaceX-16-LAR (0.56) and SpaceX-21 (0.65) are in the ambiguous band. Part of their
    clock-gene difference doesn't lie on the clock plane.
  - The three Earth sites differ from each other: Q = 17.9, df 4, p 0.001.
- **Reduced coherence in flight** (a per-arm question) can't be assessed. That needs absolute
  phase, which stopped at its pre-set check (§6). The plane test sees no off-plane
  ("disrupted") pattern at the ISS sites.

## 5.4 Housing and lighting: a hypothesis to compare, not a claim
`jobs/phase3a_compare/time_of_day_audit.yaml` records, per site, where the ground controls
were housed and the lighting schedule's time reference, with sources. The ISS keeps GMT, and
flight habitats were lit on GMT schedules where stated (RR-1: lights on 06:00–18:00 GMT).

Suppose a ground facility at Kennedy Space Center ran its lights on that GMT schedule but
euthanized animals in local working hours (Florida is GMT−4 or −5), while on-orbit work
followed GMT. Ground animals would then sit about 4–5 h later in their light cycle than flight
animals, so flight would look about 4–5 h earlier.

| Site | Ground housing (documented) | Lighting reference (documented) | Implied if GMT-lit and local-hours dissection | Measured (light-phase reading) |
|---|---|---|---|---|
| SpaceX-4 | ISS Environmental Simulator chamber, KSC | flight 06:00–18:00 GMT; ground schedule not stated | ~4 h, flight earlier (Oct 2014, EDT) | 2.5 h, flight earlier |
| SpaceX-8 | not stated | 07:00–19:00 GMT (record); ground not stated | ~4 h if at KSC; location undocumented | 0.5 h (not significant) |
| SpaceX-13-ISST | KSC | not recorded | ~5 h (Feb 2018, EST) | 3.5 h, flight earlier |
| SpaceX-16-ISST | flight hardware, matched conditions; KSC before launch | KSC facility "matched in timing to the light cycle … programmed for flight" (before launch) | ~5 h (Jan 2019, EST) | 4.1 h, flight earlier |
| SpaceX-13-LAR | both arms at one facility, same day | not recorded | none between arms | 2.3 h |
| SpaceX-16-LAR | not stated at euthanasia | not stated | undetermined | 1.2 h |
| SpaceX-21 | both arms at Texas A&M | lights on 07:00 GMT (= 01:00 CST) | none between arms; both late in the light phase or early dark | 2.1 h |

- **Where it fits:** the hypothesis matches the direction at all three ISS sites with a
  detectable shift, and roughly matches the size (measured sizes are 1–2 h smaller; within-arm
  spread biases them low).
- **Where it doesn't:** it doesn't explain SpaceX-8, or the Earth-stratum differences, where
  both arms were at one facility.
- **Status:** not a finding. Each site's ground-facility light schedule and euthanasia clock
  times would test it directly.

## 6. Exploratory absolute phase (B): stopped at the pre-set check
**What (B) estimates:** each animal's clock time is fitted on the uncentred panel against
GSE54651's gene baselines: x − m ≈ k + u**a** + v**b**, with k absorbing any library-wide
offset. (B) assumes the reference pipeline's baselines (mm9/HTSeq) match each site's
(GRCm39/RSEM) gene by gene. That can't be validated here.

**The rule, written before any site ran:** each site first sends only its ground arm's
aggregates. If any ground arm's circular mean falls outside the lights-on window (CT −1 to 13),
(B) stops and no flight-arm values are produced.

| Ground arm | Circular mean CT | Resultant length | Median template r | Median amplitude scale | In window |
|---|---|---|---|---|---|
| SpaceX-4 | 7.5 | 0.91 | 0.94 | 1.04 | yes |
| SpaceX-8 | 3.1 | 0.72 | 0.92 | 1.33 | yes |
| SpaceX-13-ISST | 10.2 | 0.98 | 0.80 | 0.66 | yes |
| SpaceX-16-ISST | 6.0 | 0.84 | 0.87 | 1.17 | yes |
| SpaceX-13-LAR | 12.6 | 0.67 | 0.68 | 0.61 | yes |
| SpaceX-16-LAR | 10.4 | 0.73 | 0.79 | 0.52 | yes |
| SpaceX-21 | **14.4** | 0.80 | 0.80 | 0.58 | **no** |

**SpaceX-21's ground arm falls outside the window, so (B) stopped.** No flight-arm or per-arm
coherence values from (B) exist.
- SpaceX-21's facility recorded lights on at 07:00 GMT, which is 01:00 local time (§5.4). If it
  dissected in local working hours, CT 14 is what you'd expect. That's noted, not used to
  rescue (B): widening the window after seeing the result would be choosing the answer.
- **ISS ground arms:** they land at CT 3–10, within 1.5 h of the ground times implied by the
  clock plane's light-phase reading (§5.1) at all four sites. Both estimates use the same
  reference peaks, so they aren't independent.
- **Earth ground arms:** their amplitude scales are low (0.52–0.61), which could reflect
  baseline mismatch rather than biology. Another reason (B) stays exploratory.

## 7. The federation boundary
- **Data access:** each client's data comes only through `sites/loader.py`.
- **What a site sends:**
  - arm sizes and the panel genes used, with drops;
  - the chord (2 numbers) and its 2 × 2 bootstrap covariance;
  - L and |Δ| with intervals and amplitude sensitivity;
  - both readings, R²_plane, permutation p-values, one noise scalar;
  - settings and versions.

  For (B): per-arm n, circular mean, resultant length, median template r and median amplitude
  scale, in two stages.
- **Enforcement:** `validate_clock_summary` and `validate_absolute_summary` check exact key
  whitelists. The server refuses anything else; that happened once during development, when an
  out-of-date summary was rejected.
- **Tests:**
  - an audit hook checks that a site opens only its own manifest and count file (another site's
    samples share that file);
  - no sample name appears in what leaves;
  - planted per-sample fields are rejected.
- **In-process, not FLARE:** one round, and the server only tabulates. FLARE would add only
  transport, and WS-07 already showed the FLARE path matches the in-process path.

## 8. Limitations
- **What can't be separated:** a dissection-time mismatch vs a real clock shift; a shift vs a
  change in amplitude.
- **Direction** depends on the light-phase assumption, which a GMT-lit facility could break.
- **The reference:**
  - male C57BL/6 kept in constant darkness;
  - most sites are female, two are BALB/c, all were on LD;
  - the CT/ZT gap is under 1 h and not corrected;
  - GSE54651 amplitudes come from 8 samples. Their uncertainty is propagated, and ±25% is
    shown.
- **R²_plane** separates clock-shaped from noise well, but shifted from disrupted only weakly.
- **Statistics:**
  - intervals cover about 85–95% rather than 95%, and less with wide within-arm spread;
  - heterogeneity p-values rest on small-sample bootstrap covariances;
  - power uses one noise value for all genes.
- **No phase covariate was added to site DE.** An arm-level shift is perfectly confounded with
  the flight indicator, so a covariate can't remove it. See the WS-09 return for the proposal.

## 9. Rebuild
```bash
# WSL/Ubuntu, after `fedkg`
uv run python jobs/phase3a_clock/reference.py   # ~5 s
uv run python jobs/phase3a_clock/validate.py    # ~2 min on 20 workers
uv run python jobs/phase3a_clock/run_sites.py   # ~10 s
uv run python jobs/phase3a_clock/report.py      # ~7 min (power simulations)
uv run python jobs/phase3a_clock/absolute.py    # (B), stage 1 and the stop rule
```
Inputs and checksums are in `jobs/phase3a_clock/common.py`. Sources and access dates are in the
README.
