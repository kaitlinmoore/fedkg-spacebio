# Liver census (WS-06)

**Status:** findings from OSDR on 2026-11-12. The gate assessment and site mapping below were written
as *proposals*. **They were accepted on 2026-11-12, with additions;** the decided version is in
[`SITES.md`](../SITES.md) (ISS/Earth strata with a location-aware flight contrast, REML +
Hartung–Knapp pooling, and age and study covariates). AI-drafted (Claude Code, WS-06); every
number below comes from the build outputs listed under *How to rebuild*.

Scope: mouse liver RNA-seq, flight vs. control, for the nine candidate studies the GeneLab KG
returned (OSD-47, -48, -137, -168, -173, -245, -379, -463, -686). Data files are not in git; this
page holds only summaries of public OSDR metadata.

## Headline

1. **Every candidate publishes per-sample unnormalized counts** (RSEM, Ensembl gene IDs) and was
   processed with the **same pipeline**: GL-DPPD-7101-G, Ensembl release 112, GRCm39. The mix of
   old and new pipelines the handoff expected no longer exists for liver. Every published DE
   table has `Stat_` columns for every contrast (all 94 flight-vs-control comparisons).
2. **Overlap is real but confined to OSD-168.** It re-sequences 9 OSD-48 animals and 12 OSD-137
   animals as new libraries (no count vector is a near-copy). OSD-47 and OSD-48 share **no**
   animals: they are two different RR-1 cohorts (different strain, age and duration).
3. **The strict set is confirmed:** 9 strict flight-vs-ground comparisons in 7 studies (OSD-47, -48,
   -137, -173, -245, -379, -463), as WS-01 counted. OSD-168 also has strict comparisons in OSDR's own
   metadata (the KG's garbled labels hid them), but it spans two missions and is left unassigned.
4. **Proposed gate result: go.** 5 of 7 mission sites pass (≥ 3 independent animals per arm in a
   strict flight-vs-ground comparison, counts published); the verdict holds at thresholds 3, 4 and
   5. But the 5 sites mix **ISS-euthanized** (SpaceX-4, -8) and **Earth-euthanized** (SpaceX-13,
   -16, -21) flight mice, and within either stratum alone the strict rule leaves ≤ 3 sites. See
   *Gate assessment*.

## Studies

| Study | Site (mission) | Project | Strain, sex | Age at launch | Flight days | Groups in RNA-seq | Animals / samples | Flight euthanized |
|---|---|---|---|---|---|---|---|---|
| OSD-47 | SpaceX-4 | RR-1 (NASA validation) | C57BL/6NTac, F | 8 months | 21–22 | FLT, GC, BSL | 9 / 9 | ISS (stated) |
| OSD-48 | SpaceX-4 | RR-1 (CASIS) | C57BL/6J, F | 16 weeks | 37 | {FLT, GC} × {Carcass, Upon euthanasia} | 14 / 14 | ISS (stated) |
| OSD-137 | SpaceX-8 | RR-3 | BALB/c, F | 12 weeks | 39–41 | FLT, GC, BSL | 18 / 18 | ISS (inferred) |
| OSD-168 | *unassigned* (SpaceX-4 + SpaceX-8) | RR-1 + RR-3 | C57BL/6J + BALB/c, F | 12–16 weeks | 37–40 | mission × arm × ±ERCC | 32 / 52 | ISS |
| OSD-173 | STS-135 | BSP135 | C57BL/6NCrl, F | 9 weeks | ~13 (Shuttle) | FLT, GC, VIV | 6 / 6 | Earth, 3–5 h after landing |
| OSD-245 | SpaceX-13 | RR-6 | C57BL/6NTac, F | 36 weeks | ~30 (LAR) / ~60 (ISS-T) | arm × duration × location × dissection | 59 / 59 | both (factor) |
| OSD-379 | SpaceX-16 | RRRM-1 (RR-8) | BALB/cAnNTac, F | 10–12 or 32 weeks | 22–23 (ISS-T) / ~40 (LAR) | arm × age × duration × location × dissection | 141 / 141 | both (factor) |
| OSD-463 | SpaceX-21 | RR-23 | C57BL/6J, **M** | 16–17 weeks | 38 | FLT, GC, VIV | 27 / 31 | Earth (LAR) |
| OSD-686 | SpaceX-12 | MHU-2 (JAXA) | C57BL/6J, **M** | 9 weeks | 30 | FLT uG, FLT 1G-centrifuge, GC | 9 / 9 | Earth, ≤ 36.5 h after splashdown |

FLT flight, GC ground control, VIV vivarium, BSL basal; ISS-T ISS-terminal, LAR live animal
return. "Flight euthanized" comes from the OSDR factor `euthanasia location` (OSD-245, -379) or
from protocol text quoted in `src/fedkg_spacebio/census/liver_curation.yaml`. OSD-686's processed
files are named `GLDS-617_*`.

## Strict flight-vs-ground comparisons and n per arm

n counts **animals** (unique source names), not samples. "Indep." removes animals shared with an
assigned study in another site (none were). "In DE" is how many samples GeneLab's SampleTable
used.

| Study | Site | Comparison (flight vs ground; other factors) | n flight / ground | Indep. | In DE | `Stat_` | Flight euthanized | KG comparison | KG DE edges |
|---|---|---|---|---|---|---|---|---|---|
| OSD-47 | SpaceX-4 | — | 3 / 3 | 3 / 3 | 3 / 3 | yes | ISS | `OSD-47-65754b3ba1b48856d4afd330ef225e46` | 68 |
| OSD-48 | SpaceX-4 | Carcass | 5 / 5 | 5 / 5 | 5 / 5 | yes | ISS | `OSD-48-78339eb44e112247d0884d9dd11fbf66` | 845 |
| OSD-48 | SpaceX-4 | Upon euthanasia | 2 / 2 | 2 / 2 | 2 / 2 | yes | ISS | `OSD-48-e5367f6840912de790e4dce3ac16998c` | 80 |
| OSD-137 | SpaceX-8 | — | 6 / 6 | 6 / 6 | 6 / 6 | yes | ISS | `OSD-137-2d49c2d99e55a9e2c2729122a7521106` | 4 |
| OSD-173 | STS-135 | — | 2 / 2 | 2 / 2 | 2 / 2 | yes | Earth | `OSD-173-ee429909dcdb03b35f1f4b3841e43535` | 1,149 |
| OSD-245 | SpaceX-13 | Upon euthanasia, ~30 d, On Earth (LAR) | 10 / 9 | 10 / 9 | 10 / 9 | yes | Earth | `OSD-245-44db6dca1026abb90902ca4f0dff1440` | 136 |
| OSD-379 | SpaceX-16 | 10–12 wk, Upon euthanasia, ~40 d, On Earth (LAR) | 8 / 8 | 8 / 8 | 8 / 8 | yes | Earth | `OSD-379-7027731d151e43a51c7a8d6baa5959ba` | 301 |
| OSD-379 | SpaceX-16 | 32 wk, Upon euthanasia, ~40 d, On Earth (LAR) | 7 / 7 | 7 / 7 | 7 / 7 | yes | Earth | `OSD-379-f94068ab759fb5fcc2518fe1a5ff5fb5` | 517 |
| OSD-463 | SpaceX-21 | — | 9 / 9 | 9 / 9 | 9 / 9 | yes | Earth | `OSD-463-74371d7bb5afac495d23f0f48c8c9944` | 3,421 |

Also strict, but in the unassigned OSD-168: RR-1 with spike-in (5 / 5), RR-1 without spike-in
(5 / 5, **the same animals**), RR-3 with spike-in (4 / 4). OSD-686 has no strict comparison (its
flight arms are `uG` and `1G by centrifugation`, ground is `1G on Earth`).

ISS-terminal comparisons that the strict rule rejects only because location (and a 1–2 day
duration offset) is a recorded factor:

| Study | Comparison | n flight / ground | KG comparison | KG DE edges |
|---|---|---|---|---|
| OSD-245 | Carcass, ~60 d, On ISS vs On Earth | 10 / 10 | `OSD-245-e800bad5e8fe180307dada7f277c6a92` | 2,036 |
| OSD-379 | 10–12 wk, Carcass, 22–23 d On ISS vs 24 d On Earth | 10 / 10 | `OSD-379-d6b14f9ef8590e45a17dbc53c87dafd8` | 253 |
| OSD-379 | 32 wk, Carcass, 22–23 d On ISS vs 24 d On Earth | 10 / 10 | `OSD-379-6c2bd802431c520f3284f7497d4fb853` | 1,577 |

RR-1 and RR-3 flight mice were also euthanized on the ISS, but those studies don't record location
as a factor, so they pass the strict rule. The strict rule is therefore stricter on OSD-245/379
than on RR-1/RR-3 for the same design.

### Confirming the strict set

Built from OSDR's own sample metadata (not the KG): 94 flight → control comparisons, 23 strict.
Excluding OSD-168, the strict flight-vs-ground set is **9 comparisons in 7 studies**, matching
WS-01's KG count. OSD-168's strict comparisons were invisible to the KG query because the KG's
factor lists for OSD-168 are damaged (one side lacks the spike-in label; labels are cut at `(`).

## Overlap

**Animals (metadata).** Source names were normalized (upper-case, letters and digits only) and a
match also required equal strain and sex. No name collisions with different strain or sex.

| | OSD-47 | OSD-48 | OSD-137 | OSD-168 | others |
|---|---|---|---|---|---|
| OSD-47 | 9 | 0 | 0 | 0 | 0 |
| OSD-48 | 0 | 14 | 0 | **9** | 0 |
| OSD-137 | 0 | 0 | 18 | **12** | 0 |
| OSD-168 | 0 | **9** | **12** | 32 | 0 |

(Diagonal: unique animals per study. OSD-173, -245, -379, -463 and -686 share no animals with
anything.) OSD-168's 9 shared OSD-48 animals are the CASIS RR-1 flight and ground mice (e.g.
`RR1_FLT_M25`); its 12 shared OSD-137 animals are all of its RR-3 animals. OSD-168's other 11
RR-1 animals (basal, vivarium and some flight/ground) appear in no other liver study here.

**Within a study.** OSD-168: 20 animals sequenced twice (with and without ERCC spike-in).
OSD-463: 3 animals with technical replicates (`RR-23_F7` ×2, `RR-23_F8` ×3, `RR-23_GC1` ×2);
GeneLab's DE used only `techrep1`, so its n per arm equals the animal count (9 / 9).

**Libraries (counts).** Rule (Kaitlin, 2026-11-12): identical count vectors, or Pearson r > 0.9999
on log1p counts **with equal library size**. **No pair qualified.** The highest cross-study r was
0.990 (OSD-137 vs OSD-168, different animals). The same mouse in OSD-48 and OSD-168 (`M40`)
reached only r = 0.962, with OSD-48's library at 0.38× the size. So OSD-168 is a new sequencing
of the same animals, not a reprocessing of the same reads. Correlation can't identify an animal,
and here it doesn't even rank the true same-animal pair first, so animal identity must come from
metadata.

## Counts, gene IDs and annotation

| Study | Counts file (RSEM, standard) | Size | Genes | ERCC rows | Non-integer cells | DE table size | DE file created |
|---|---|---|---|---|---|---|---|
| OSD-47 | `GLDS-47_rna_seq_RSEM_Unnormalized_Counts_GLbulkRNAseq.csv` | 2.6 MB | 57,186 | 0 | 7.9% | 22.7 MB | 2025-10-07 |
| OSD-48 | `GLDS-48_rna_seq_RSEM_Unnormalized_Counts_GLbulkRNAseq.csv` | 3.4 MB | 57,186 | 0 | 7.2% | 35.9 MB | 2025-10-21 |
| OSD-137 | `GLDS-137_rna_seq_RSEM_Unnormalized_Counts_GLbulkRNAseq.csv` | 4.0 MB | 57,278 | 92 | 6.0% | 27.1 MB | 2026-09-01 |
| OSD-168 | `GLDS-168_rna_seq_RSEM_Unnormalized_Counts_GLbulkRNAseq.csv` | 9.6 MB | 57,278 | 92 | 6.6% | 268.8 MB | 2026-06-23 |
| OSD-173 | `GLDS-173_rna_seq_RSEM_Unnormalized_Counts_GLbulkRNAseq.csv` | 2.1 MB | 57,278 | 92 | 5.9% | 20.1 MB | 2026-02-06 |
| OSD-245 | `GLDS-245_rna_seq_RSEM_Unnormalized_Counts_GLbulkRNAseq.csv` | 10.5 MB | 57,278 | 92 | 5.9% | 95.6 MB | 2026-09-02 |
| OSD-379 | `GLDS-379_rna_seq_RSEM_Unnormalized_Counts_GLbulkRNAseq.csv` | 22.5 MB | 57,278 | 92 | 5.0% | 524.6 MB | 2025-09-09 |
| OSD-463 | `GLDS-463_rna_seq_RSEM_Unnormalized_Counts_GLbulkRNAseq.csv` | 6.0 MB | 57,278 | 92 | 4.3% | 31.1 MB | 2025-08-27 |
| OSD-686 | `GLDS-617_rna_seq_RSEM_Unnormalized_Counts_GLbulkRNAseq.csv` | 2.5 MB | 57,186 | 0 | 2.5% | 19.0 MB | 2025-12-03 |

- **Namespace:** Ensembl mouse gene IDs (`ENSMUSG…`, no version suffix) in every matrix, plus 92
  `ERCC-` spike-in rows in six studies. The DE tables add `SYMBOL` and `ENTREZID` columns
  (GeneLab's annotation table, GL-DPPD-7110-A). Nothing was converted (D-012 mapping comes later).
- **Annotation:** GRCm39, Ensembl release 112 (`Mus_musculus.GRCm39.112.gtf`) in every study's
  processing protocol; pipeline GL-DPPD-7101-G; NF_RCP 2.0.2 (OSD-47, -48, -379, -463, -686) or
  2.1.0 (OSD-137, -168, -173, -245). No study's text names two versions.
- **Each study also publishes an rRNA-removed variant** (`_rRNArm`; 354 rRNA genes dropped). Both
  were downloaded and recorded.
- **RSEM writes expected counts**, so 2.5–7.9% of cells are non-integer. DESeq2 and pydeseq2 need
  integers; phase 3a must round (as `DESeqDataSetFromTximport` does) and say so.

### Which count matrix GeneLab's DE used

GeneLab ran DE twice, once per matrix, and publishes both tables. Two checks, both from partial
reads:

1. **DE table ↔ normalized counts.** Each DE table's per-sample columns equal the matching
   normalized-count file **plus exactly 1** in every cell checked (a pseudocount; it's also why
   the KG's group means bottom out at 1.000). The standard DE table matches the standard
   normalized counts (deviation from the +1 offset ≤ 1×10⁻¹¹ in all nine studies) and not the
   rRNA-removed ones (deviations of 0.9 to 13). The same holds the other way round for `_rRNArm`.
2. **Which DE table the KG mirrors** (OSD-47 only). In the first 2 MB of each OSD-47 DE table, 6 of
   the KG's 68 flight-vs-ground genes appear. Their KG log2FC equals the **standard** table's
   exactly (max difference 0) and differs from the `_rRNArm` table's by up to 5.8×10⁻⁵.

So the KG's DE comes from the **standard** table, which is computed from the **standard RSEM
matrix**. Following Kaitlin's rule ("use whichever matrix GeneLab's own DE used"), phase 3a's
primary input is `*_RSEM_Unnormalized_Counts_GLbulkRNAseq.csv`, with `_rRNArm` recorded alongside.
**Caveat:** OSD-137, -168 and -245's DE files were created after the KG build (2026-03-11), so the
KG's edges for those studies may come from an earlier processing. Only OSD-47 was checked against
the KG.

## Pipeline and SE availability

All nine studies: GL-DPPD-7101-G, and every DE table (standard and `_rRNArm`) has a `Stat_` column
for every contrast. Each census comparison was mapped to GeneLab's contrast through the
SampleTable and contrasts files: all 94 have a contrast, and all 94 have `Stat_`. Under D-016 this
is a cross-check only: each site computes its own SE with pydeseq2.

## KG ↔ OSDR join

Each OSDR comparison was matched to a KG comparison ID on study, assay and group labels: 66
`exact`, 8 `partial`, 16 `ambiguous`, 4 `unmatched` (94 total). What blocked a clean join:
- **Assay names drifted:** the KG's OSD-245 name has a stray space; OSD-168's says `_illumina` where
  OSDR now says `_Illumina HiSeq 4000`.
- **OSD-168 labels are damaged in the KG:** the flight side lacks the spike-in label, so 16 KG
  comparisons fit two OSDR pairs each. No ID was assigned to those.
- **OSD-245 basal duration:** the KG says `1 day`, OSDR says `0 {day}`. The 4 basal comparisons stay
  unmatched.

Every strict flight-vs-ground comparison in an assigned study has a unique KG ID.

## Gate assessment for D-016 (proposed, not decided)

**Criterion (Kaitlin, 2026-11-12):** a mission site passes if it has published unnormalized
counts **and** a strict flight-vs-**ground-control** comparison with **≥ 3 independent animals in
each arm**. Phase 3a is a go if **≥ 3 sites** pass. `Stat_` is not part of the gate.

| Site | Studies | Best strict comparison | n flight / ground (indep.) | Flight euthanized | Passes | Reason |
|---|---|---|---|---|---|---|
| SpaceX-4 | OSD-47, OSD-48 | OSD-48 Carcass | 5 / 5 | ISS | yes | |
| SpaceX-8 | OSD-137 | OSD-137 | 6 / 6 | ISS | yes | |
| SpaceX-13 | OSD-245 | OSD-245 LAR | 10 / 9 | Earth | yes | |
| SpaceX-16 | OSD-379 | OSD-379 LAR 10–12 wk | 8 / 8 | Earth | yes | |
| SpaceX-21 | OSD-463 | OSD-463 | 9 / 9 | Earth | yes | |
| STS-135 | OSD-173 | OSD-173 | 2 / 2 | Earth | no | 2 animals per arm |
| SpaceX-12 | OSD-686 | — | — | Earth | no | no strict flight-vs-ground comparison |

**Verdict: go, 5 of 7 sites pass.** Sensitivity: thresholds 4 and 5 give the same 5 sites;
threshold 2 adds STS-135 (6 sites).

### The ISS vs Earth question (proposal for Kaitlin)

Flight mice euthanized on the ISS and mice returned alive and euthanized on Earth (hours to days
after splashdown, after re-entry and re-loading at 1 g) are different contrasts. In the pooled 5
sites that difference coincides with the site: SpaceX-4 and -8 are ISS, SpaceX-13, -16 and -21 are
Earth. A pooled random-effects estimate would read that difference as between-mission
heterogeneity.

| Rule | ISS stratum: passing sites | Earth stratum: passing sites |
|---|---|---|
| Strict (D-015) | SpaceX-4, -8 → **2: no-go** | SpaceX-13, -16, -21 → **3: go, no margin** |
| Location-aware (below) | SpaceX-4, -8, -13 (ISS-T), -16 (ISS-T) → **4: go** | SpaceX-13, -16, -21 → **3: go** |

*Location-aware* is a looser rule built for this comparison only: a pair is accepted if it is
strict, or if its only differences are flight `On ISS` vs control `On Earth` and durations within 3
days. It treats OSD-245/379's ISS-terminal arms the way RR-1/RR-3 are already treated. In OSD-245
and OSD-379 the ISS-T and LAR arms are different animals, so one mission can contribute to both
strata without double counting.

Options (recommendation first):
1. **Stratify:** primary analysis in the ISS stratum under the location-aware rule (4 sites), and
   the Earth/LAR stratum as a second analysis (3 sites). Each stratum is internally homogeneous in
   what "flight" means. This needs an amendment to D-015 for this use.
2. **Pool all 5 strict sites, with a sensitivity analysis** that drops the Earth sites (or adds
   euthanasia location as a meta-regression covariate). Simplest. But with 2 vs 3 sites, the
   covariate is barely estimable, and the strict rule leaves the ISS side at 2 sites.
3. **Pool under the location-aware rule** (still 5 sites, but SpaceX-13/-16 contribute their
   ISS-T arm). The pool is homogeneous except for SpaceX-21, the only male, Earth-only site.

Other site-level confounders the meta-analysis will meet whichever option is chosen: strain
(C57BL/6 substrains vs BALB/c), sex (SpaceX-21 is the only male site), age (9 weeks to 8 months)
and duration (21 to 60 days). They are all aligned with mission, which is the point of D-014, but
I² will absorb all of it.

## How to rebuild

WSL/Ubuntu, after `fedkg`:

```bash
uv run python -m fedkg_spacebio.census.build            # live; uses the cache when present
uv run python -m fedkg_spacebio.census.build --offline  # cache only; fails on a miss
```

It needs the KG result at `$FEDKG_WORKDIR/cache/kg/liver_comparisons.json`. The query is
`KG_QUERY` in `src/fedkg_spacebio/census/build.py`, run once on 2026-11-12 through the GeneLab KG
MCP `query` tool with `{"studies": [the nine OSD IDs], "liver": ["UBERON:0002107",
"UBERON:0006727"]}`. It returned 118 rows. Outputs go to `$FEDKG_WORKDIR/census/` (census, samples,
overlap matrix and pairs, library diagnostics, gate, sensitivity, strata, summary JSON), with
downloads in `$FEDKG_WORKDIR/data/osdr/` (121 MB) and 65 cached API responses (14 MB) in
`$FEDKG_WORKDIR/cache/osdr/`.

### Queries and URLs (accessed 2026-11-12)

- Samples, one call for all nine studies:
  `https://visualization.osdr.nasa.gov/biodata/api/v2/query/samples/?id.accession=OSD-47|OSD-48|OSD-137|OSD-168|OSD-173|OSD-245|OSD-379|OSD-463|OSD-686&study.source name&study.factor value&study.characteristics&study.parameter value&format=csv`
- Study metadata and protocol text: `https://visualization.osdr.nasa.gov/biodata/api/v2/dataset/OSD-<n>/`
- File listing: `https://osdr.nasa.gov/osdr/data/osd/files/<n>`
- Downloads: `https://osdr.nasa.gov/geode-py/ws/studies/OSD-<n>/download?source=datamanager&file=<file name>`.
  This redirects to a presigned S3 URL. DE tables, normalized counts and OSD-47's KG provenance
  check used HTTP `Range` reads (the first 64 KB, 256 KB, 1 MB or 2 MB). Only count matrices, SampleTable, contrasts and
  runsheet files were downloaded whole. No FASTQ or BAM.

## Limitations

- Flight euthanasia location for RR-3 (OSD-137) is **inferred** from "carcasses were stored at -80 C
  until … dissection on earth"; the text doesn't name the ISS.
- Only OSD-47's KG edges were checked against a DE table. Three studies were reprocessed after the
  KG build.
- The library check uses Kaitlin's near-exact rule. It would not detect the same reads processed
  with a different annotation (not an issue here: one annotation throughout).
- "Independent" means not shared with an assigned study at another site. Littermates, cage effects
  and shared ground-control cohorts can't be seen in OSDR metadata.
