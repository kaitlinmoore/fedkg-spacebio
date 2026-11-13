# Sites

How OSDR studies are partitioned into simulated sites, and why. Every site must contain both
spaceflight and ground-control samples, or the model learns the site instead of the biology.

Sites are missions (DECISIONS D-014). Tissue scope starts with liver (D-013). The numbers come from
the WS-06 census; see [`docs/census-liver.md`](docs/census-liver.md) for the evidence behind every
row.

## Liver

> **Status: Accepted 2026-11-12.** The mapping and rules below were proposed by WS-06 and decided in
> the project decision log: gate go; ISS/Earth strata with a location-aware flight contrast;
> REML + Hartung–Knapp for pooling; the site and data-handling rules listed under *Rules*.

n counts **animals** (unique OSDR source names) in the best **strict** flight-vs-ground-control
comparison (D-015). Every study listed has published per-sample RSEM counts (Ensembl IDs, GRCm39,
Ensembl 112, GL-DPPD-7101-G).

| Site ID | Basis | OSD accessions | Tissue | n flight | n ground | Flight euthanized | Platform | Notes |
|---|---|---|---|---|---|---|---|---|
| SpaceX-4 | Mission (RR-1) | OSD-47, OSD-48 | liver (`UBERON:0002107`) | 3 + 5 | 3 + 5 | ISS | Illumina HiSeq 3000 (OSD-47) | Two different RR-1 cohorts, no shared animals. OSD-47: C57BL/6NTac, 8 months, 21–22 d. OSD-48: C57BL/6J, 16 weeks, 37 d; use its `Carcass` comparison (its `Upon euthanasia` one has 2 per arm). |
| SpaceX-8 | Mission (RR-3) | OSD-137 | liver (`UBERON:0002107`) | 6 | 6 | ISS (confirmed, published: Beheshti et al. 2019, *Sci Rep* 9:19195) | Illumina HiSeq 4000 | BALB/c, 12 weeks, 39–41 d. |
| SpaceX-13 | Mission (RR-6) | OSD-245 | liver (`UBERON:0002107`) | 10 (LAR) / 10 (ISS-T) | 9 (LAR) / 10 (ISS-T) | Earth (LAR) **and** ISS (ISS-T), separate animals | Illumina NovaSeq 6000 | C57BL/6NTac, 36 weeks. Strict comparison: LAR ~30 d. ISS-T ~60 d passes only the location-aware rule. |
| SpaceX-16 | Mission (RRRM-1 / RR-8) | OSD-379 | liver left lateral lobe (`UBERON:0006727`) | 8 (LAR, 10–12 wk) / 10 (ISS-T, 10–12 wk) | 8 / 10 | Earth (LAR) **and** ISS (ISS-T), separate animals | Illumina NovaSeq | BALB/cAnNTac; also a 32-week age group (LAR 7 / 7, ISS-T 10 / 10). Pick one age group per analysis or model age. |
| SpaceX-21 | Mission (RR-23) | OSD-463 | liver left lateral lobe (`UBERON:0006727`) | 9 | 9 | Earth (LAR) | Illumina | C57BL/6J, **male** (the only male passing site), 38 d. 3 animals have technical replicates; use one sample per animal (GeneLab used `techrep1`). |
| STS-135 | Mission | OSD-173 | liver (`UBERON:0002107`) | 2 | 2 | Earth (3–5 h after landing) | Illumina | **Fails the n ≥ 3 gate.** Kept for completeness; excluded from phase 3a. |
| SpaceX-12 | Mission (MHU-2, JAXA) | OSD-686 | liver (`UBERON:0002107`) | 3 (`uG`) | 3 | Earth (≤ 36.5 h after splashdown) | Illumina NovaSeq 6000 | **No strict comparison** (gravity labels: `uG` vs `1G on Earth`). Kept with `strict_match = false`; excluded from phase 3a. |

**Excluded from every site: OSD-168** (RR-1 + RR-3 re-sequencing, SpaceX-4 and SpaceX-8). It spans
two missions, and its animals overlap the sites' own studies: 9 animals are OSD-48's and 12 are
OSD-137's (all its RR-3 animals). It also sequences 20 animals twice (with and without ERCC
spike-in).

**Rules (accepted):**
1. **One site per mission** (D-014). A study linked to exactly one mission in the KG belongs to that
   mission's site. OSDR's own `mission` field agrees for all nine liver candidates.
2. **Multi-mission studies are left out of cross-site work.** Keeping OSD-168 would either split it
   by its `space mission` factor (which is possible: each sample carries it) or assign it to one
   site. Either way its animals are already counted through OSD-48 and OSD-137, so it would count
   them twice. Leaving it out costs nothing.
3. **Overlap rule:** when the same animal appears in two studies, keep it in the single-mission
   study and drop the multi-mission one. If two *assigned* studies at different sites ever share
   animals, those animals are removed from both counts (`gate.independent_counts`). That didn't
   happen for liver.
4. **Within a site with several studies** (SpaceX-4), the studies are different cohorts. Model
   study as a covariate inside the site (`~ study + condition`) rather than pooling them blindly.
5. **Age within a site** (SpaceX-16): if both age groups appear in both arms, model age as a
   covariate (`~ age + condition`). Otherwise use the 10–12-week group only.
6. **ISS vs Earth euthanasia: analyze as separate strata.** Flight mice euthanized on the ISS
   (ISS-T) and flight mice returned alive and euthanized on Earth (LAR) are different contrasts,
   and the difference lines up with site. A **location-aware flight contrast** applies: `On ISS` vs
   `On Earth` counts as part of the flight condition (ground controls can't be on the ISS), as do
   durations within 3 days. The dissection method must still match. Rows matched this way carry
   `location_aware_match = true`; `strict_match` keeps its original meaning, so the strict set stays
   reproducible.
   - **ISS stratum (4 sites):** SpaceX-4, SpaceX-8 (ISS euthanasia confirmed, Beheshti et al. 2019), SpaceX-13 ISS-T,
     SpaceX-16 ISS-T.
   - **Earth stratum (3 sites):** SpaceX-13 LAR, SpaceX-16 LAR, SpaceX-21. No margin: losing one
     site leaves 2.
7. **Technical replicates:** one sample per animal, as GeneLab did (`techrep1`).
8. **Counts input:** the standard RSEM unnormalized matrix (the one GeneLab's DE used), rounded to
   integers (stated in every output), with the `_rRNArm` matrix as a sensitivity run.
9. **Pooling method (phase 3a):** inverse-variance fixed effect, plus random effects estimated by
   REML, with Hartung–Knapp(–Sidik–Jonkman) confidence intervals. τ² and I² are reported, flagged as
   imprecise with 3–4 sites. Strain, sex (SpaceX-21 is the only male site), age and duration vary
   with site, so heterogeneity can't be attributed to spaceflight alone.

**Gate (D-016) result: go** (accepted). 5 of 7 liver sites pass (SpaceX-4, -8, -13, -16, -21). Under the strict rule alone, the ISS stratum would have only 2 sites; the location-aware contrast
(rule 6) gives 4 ISS and 3 Earth sites.
