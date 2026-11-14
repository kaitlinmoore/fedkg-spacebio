# GeneLab KG: confirmed schema, IDs and coverage

What the hosted SPOKE-GeneLab knowledge graph (served by `mcp-genelab`) actually contains, checked
live on **2026-11-12** against `https://visualization.osdr.nasa.gov/kg/mcp` (server `mcp-genelab`
1.0.0, 24 tools). Every number below comes from a tool call made that day; the Cypher for each
count is in the appendix so it can be re-run. The headline numbers (node counts, the strict
mouse flight-vs-ground set, OSD-47's stored DE genes) were re-checked through the claude.ai
GeneLab connector the same evening and matched.

## Version

- The graph does not expose a version string. `MetaNode` carries no version property, and
  `dbms.components()` is refused on the hosted endpoint (read-only, no `dbms.*`).
- We therefore **assume** SPOKE-GeneLab v0.3.1+20260311 (the version the mcp-genelab source
  documents) but **could not confirm it live**. The highest study accession is **OSD-937**
  (corrected 2026-11-14; an earlier note here said the OSD-77x range), and the latest mission
  start date is 2023-03-14 (SpaceX-27).
- Since nothing names the release, `kg.query` records a **fingerprint** instead: nine graph
  counts plus a checksum over comparison IDs, hashed (CONTRACTS §7.1). On 2026-11-14: 212
  studies (highest OSD-937), 13,146 comparisons, 138,736 `MGene`, 30,009 `Gene`, 42 missions,
  50,208,066 DE edges, 121,153 ortholog edges; hash `f0045ca84ae8…`.

## Node types

| Label | Count | `identifier` format | Other properties |
|---|---:|---|---|
| Study | 212 | `OSD-<n>` (all 212 match `^OSD-[0-9]+$`) | name, project_title, project_type, description, organism, taxonomy, host_organism, host_taxonomy, host_strain |
| Assay | 13,146 | `OSD-<n>-<32 hex>` (all match; the hex is an MD5 of the assay metadata) | name, technology, measurement, differential_analysis_method, factors_1/2 (lists), factor_space_1/2, material_1/2, material_name_1/2, material_id_1/2 |
| Mission | 42 | free-text slug, e.g. `SpaceX-21`, `STS-135`, `Expedition-14`, `Soyuz_TMA-14` | name, space_program, flight_program, start_date, end_date (Neo4j `DATE`) |
| MGene | 138,736 | bare Entrez Gene ID as a **string** (`"681355"`) | symbol, name, organism, taxonomy |
| Gene | 30,009 | bare Entrez Gene ID as a string | none |
| MethylationRegion | 8,003 | `<chrom>:<start>-<end>`, no `chr` prefix (`10:10433001-10434000`) | chromosome, start, end, in_promoter, in_exon, in_intron, dist_to_feature |
| Anatomy | 42 | `UBERON:NNNNNNN` (all 42) | none (no label/name on the node) |
| CellType | 7 | `CL:NNNNNNN` (all 7) | none |
| Organism | 46 | bare NCBI Taxonomy ID as a string (`"286"` = *Pseudomonas*) | name |

Notes:
- **MGene is not only "model organism" genes.** It holds genes for every studied organism, including
  29,026 *Homo sapiens* genes. By organism: human 29,026; mouse 27,415; *Arabidopsis* 25,426;
  *C. elegans* 17,460; zebrafish 17,142; *Drosophila* 15,933; rat 6,249; yeast 85.
- **Gene** has no organism property. Every human MGene has an `IS_ORTHOLOG` edge to the Gene
  with the identical ID (29,026 of 29,026), so Gene is the human reference set. The other
  983 Gene nodes were not characterized.
- **Organism** nodes are microbial taxa used only by the differential-abundance edges. They are
  not linked to Study; a study's organism is a string property on the Study node.
- **Anatomy and CellType nodes carry only an ID.** Human-readable tissue names live on the Assay
  (`material_name_1/2`).

## Relationship types

Relationship types carry a suffix (e.g. `_ASmMG`); queries must use the full name.

| Type | From → To | Count | Properties |
|---|---|---:|---|
| `MEASURED_DIFFERENTIAL_EXPRESSION_ASmMG` | Assay → MGene | 50,208,066 | log2fc, adj_p_value, group_mean_1/2, group_stdev_1/2 |
| `IS_ORTHOLOG_MGiG` | MGene → Gene | 121,153 | none |
| `MEASURED_DIFFERENTIAL_METHYLATION_ASmMR` | Assay → MethylationRegion | 27,454 | methylation_diff, q_value, group_mean_1/2, group_stdev_1/2 |
| `PERFORMED_SpAS` | Study → Assay | 13,146 | none |
| `METHYLATED_IN_MGmMR` | MGene → MethylationRegion | 8,270 | none |
| `INVESTIGATED_ASiA` | Assay → Anatomy | 7,214 | none |
| `INVESTIGATED_ASiCT` | Assay → CellType | 1,240 | none |
| `MEASURED_DIFFERENTIAL_ABUNDANCE_ASmO` | Assay → Organism | 193 | log2fc, lnfc, adj_p_value, q_value, group_mean_1/2, group_stdev_1/2 |
| `CONDUCTED_MIcS` | Mission → Study | 127 | none |

`get_neo4j_schema` omits the abundance edge from Assay's relationship list, but
`get_relationship_metadata` and a live count both confirm it.

## What an "Assay" node is

**An Assay node is one ordered pairwise comparison between two sample groups**, not an OSDR assay.
- OSD-48 has one RNA-seq OSDR assay with four groups ({Space Flight, Ground Control} ×
  {Carcass, Upon euthanasia}) and 12 RNA-seq Assay nodes in the KG: every ordered pair (4 × 3).
  The OSDR assay name is in `Assay.name`, which many comparison nodes share.
- **Both directions are stored.** For OSD-47 liver, the Space Flight → Ground Control comparison
  and its mirror share all 68 genes, with log2fc exactly negated and identical adj_p. Across all
  organisms there are 680 Space Flight → Ground Control and 664 Ground Control → Space Flight
  Assay nodes. **Count one direction only.**
- **log2fc is group 1 relative to group 2.** In OSD-48 liver (Space Flight, Carcass → Ground
  Control, Carcass), every checked gene had log2fc > 0 exactly when `group_mean_1 > group_mean_2`
  (e.g. *Ifit3*: 890.5 vs 376.4, log2fc +1.24). So in the Space Flight → control direction, a
  positive log2fc means higher in flight.
- Comparisons include confounded pairs (e.g. OSD-48 `Space Flight, Carcass` vs `Ground Control,
  Upon euthanasia`), so filtering on `factor_space` alone isn't enough.
- Factor lists don't always identify a comparison uniquely. OSD-462 (left kidney) has 3 OSDR
  RNA-seq assays, each with **two** Space Flight vs Ground Control comparisons whose factor lists
  are identical (`["Space Flight"]` vs `["Ground Control"]`) but whose results differ (55 vs 12 DE
  edges). Some design variable is missing from the factor lists.
  The same results also repeat across OSD-462's three OSDR assays: in the strict set, the 55-edge
  and 12-edge comparisons each appear three times with identical edge counts and adj_p ranges.
  They look like one dataset listed under three assays (not yet compared gene by gene), so count
  them once.

## Differential-expression edges are significance-filtered

**The KG stores DE edges only for genes with adj_p_value < 0.1.** It doesn't store full per-gene
summary statistics.
- In all 76 strict mouse RNA-seq flight-vs-ground comparisons (below), the largest stored adj_p
  is 0.0999996. Three comparisons have **zero** DE edges (OSD-246 spleen; OSD-562 hippocampus,
  two age groups). The median comparison has 136 edges, and 37 of the 76 have fewer than 100.
- Group means are floored: genes with no detectable expression in one arm show a group mean of
  exactly 1.000 with SD 0 (e.g. OSD-47 *Gm14419*, ground-control arm). That looks like a
  pseudocount or floor, so don't read the stored means as raw normalized counts.
- Consequence: a gene's absence from a comparison means "not significant at 0.1 or not measured";
  the KG can't tell those apart. Meta-analysis on these edges would only see a truncated,
  significance-selected sample.
- DE edges also carry **no standard error, raw p-value or per-group sample size**: only log2fc,
  adj_p_value and group means/SDs. Even without the 0.1 filter, inverse-variance weights couldn't
  be computed from the KG alone.

## Assays by organism and technology

Counts of Assay (comparison) nodes, all directions included, and the studies that contain them:

| Organism | Technology | Method label | Studies | Assay nodes |
|---|---|---|---:|---:|
| *Mus musculus* | RNA-Seq | DESeq2 | 91 | 4,004 |
| *Arabidopsis thaliana* | RNA-Seq | DESeq2 | 25 | 2,274 |
| *Drosophila melanogaster* | DNA microarray | DESeq2 | 6 | 1,720 |
| *Homo sapiens* | DNA microarray | DESeq2 | 21 | 1,376 |
| *Arabidopsis thaliana* | DNA microarray | DESeq2 | 15 | 1,136 |
| *Mus musculus* | DNA microarray | DESeq2 | 13 | 1,034 |
| *Homo sapiens* | RNA-Seq | DESeq2 | 22 | 566 |
| *C. elegans* | DNA microarray | DESeq2 | 4 | 562 |
| *Drosophila melanogaster* | RNA-Seq | DESeq2 | 7 | 354 |
| *Mus musculus* | WGBS | methylKit | 4 | 44 |
| *S. cerevisiae* | DNA microarray | DESeq2 | 1 | 30 |
| *Danio rerio* | RNA-Seq | DESeq2 | 1 | 12 |
| *Rattus norvegicus* | RNA-Seq | DESeq2 | 3 | 6 |
| Microbiota | 16S / ITS amplicon | DESeq2, ANCOM-BC | 1 | 24 |
| *Rattus norvegicus* | DNA microarray | DESeq2 | 2 | 4 |

Studies by organism: mouse 104, human 43, *Arabidopsis* 40, *Drosophila* 13, rat 5,
*C. elegans* 4, yeast 1, zebrafish 1, microbiota 1 (212 total).

Caveat: **every microarray assay is labeled `DESeq2`.** DESeq2 is a count-based method that isn't
normally used on microarrays. Either the label is a default, or the processing differs from what
the label says; check against OSDR before relying on it.

## Flight vs. ground: factor labels

The KG normalizes each side of a comparison into `factor_space_1/2` with four values: `Space
Flight`, `Ground Control`, `Vivarium Control`, `Basal Control`. Raw labels stay in `factors_1/2`.
- 8,344 of 13,146 Assay nodes have no `factor_space` on either side. Most are non-flight designs
  (e.g. hindlimb unloading, irradiation, sex comparisons).
- **The normalization misses some flight/ground labels.** Assays where `factor_space_1` is null but a
  raw factor mentions flight or ground include `Ground Control Rerun` (224), `Ground control`
  lower-case (109), `Ground Control` (84, unmapped on that side), `Suborbital Flight` (40),
  `Parabolic Flight` (16), and others. A census that relies only on `factor_space` will undercount.
- Within flight comparisons, gravity labels vary: `uG`, `μg`, `1G by centrifugation`,
  `1G with centrifugation`, `1G`, `1G on Earth`, `0.33G by centrifugation`. Some factor values are
  cut off at an opening parenthesis (`SpaceX-4 (RR1`, `Deep Underground Low Background laboratory (DULB-4900`).

## Mouse RNA-seq, flight vs. ground

- Any comparison with `factor_space_1 = Space Flight` and `factor_space_2 = Ground Control`:
  **221 Assay nodes in 68 studies** (one direction).
- **Strict** subset: same tissue on both sides (`material_id_1 = material_id_2`), and all
  non-flight factors identical. That gives **76 comparisons in 57 studies**. All 68 studies are
  linked to at least one mission.
- The same strict rule against the other two control types (mouse RNA-seq, one direction): Space
  Flight → Vivarium Control **30 comparisons in 18 studies**; Space Flight → Basal Control **18 in
  10**. All of those studies are already among the 57, so other controls add comparisons, not studies.
- The 11 studies with no strict comparison differ mostly in gravity or diet labels (OSD-168, 238,
  239, 253, 270, 288, 289, 426, 686, 758, 759). For example, OSD-288 compares
  `Space Flight, uG` with `Ground Control, 1G on Earth`, a valid flight-vs-ground contrast that the
  strict rule rejects. A looser rule that ignores gravity labels would add some of these.

### Tissue coverage (strict set)

UBERON terms in the KG are often lateral or sub-region terms, so the same organ appears under
several IDs. The grouping below is ours; the KG holds no UBERON hierarchy.

| Organ group (ours) | UBERON terms in the KG | Studies | Comparisons | Missions |
|---|---|---:|---:|---|
| Skeletal muscle | quadriceps femoris 0001377, gastrocnemius 0001388, soleus 0001389, tibialis anterior 0001385, extensor digitorum longus 0001386 | 11 | 11 | SpaceX-4, SpaceX-10, SpaceX-21 |
| Brain | hippocampus 0002961, left cerebral hemisphere 0002812, right hemisphere of cerebellum 0014890, cerebellum 0002037, brain 0000955 | 8 | 12 | SpaceX-8, SpaceX-18, SpaceX-21 |
| Liver | liver 0002107, liver left lateral lobe 0006727 | 7 | 9 | SpaceX-4, SpaceX-8, SpaceX-13, SpaceX-16, SpaceX-21, STS-135 |
| Kidney | left kidney 0004538, right kidney 0004539 | 5 | 11 | SpaceX-4, SpaceX-8, SpaceX-18, SpaceX-21 |
| Eye | retina 0000966, eye 0000970, left eye 0004548 | 5 | 5 | SpaceX-4, SpaceX-8, SpaceX-9, SpaceX-12 |
| Skin | skin 0001003 | 4 | 7 | SpaceX-11, SpaceX-13, SpaceX-15 |
| Thymus | 0002370 | 3 | 3 | SpaceX-12, SpaceX-13, SpaceX-21 |
| Spleen | 0002106 | 3 | 3 | SpaceX-12, SpaceX-13, SpaceX-21 |
| Adrenal gland | 0002369 | 3 | 3 | SpaceX-4, SpaceX-8, SpaceX-21 |
| Heart | heart 0000948, heart right ventricle 0002080 | 2 | 3 | NG-11, SpaceX-18 |
| Colon | colon 0001155, descending colon 0001158 | 2 | 3 | SpaceX-13, SpaceX-21 |
| Lung | left lung 0002168 | 2 | 2 | SpaceX-13, SpaceX-21 |
| Mammary gland | 0001911 | 1 | 2 | SpaceX-16 |
| Bone marrow | 0002371 | 1 | 2 | SpaceX-14 |
| **Total** | | **57** | **76** |  |

(UBERON prefixes omitted for space. The group counts were re-checked with a single Cypher query that applies this mapping; every term mapped.)

## Missions and programs

- 42 Mission nodes; 119 of 212 studies are linked to at least one (112 to one, 6 to two, 1 to
  three). **93 studies have no mission**, including 33 mouse and 29 human studies.
- `space_program` is the sponsoring agency and is sometimes a comma-joined list (`ESA, NASA`,
  `JAXA, NASA`, `NASA, NIH`). Studies per value: NASA 94, JAXA 12, ESA 7, null 5, and one study each
  for CNSA, ASI, `ESA, NASA`, `JAXA, NASA` and `NASA, NIH`.
- `flight_program` is inconsistent: `International Space Station (ISS)` (92 studies), `ISS` (9),
  the typo `Internation Space Station (ISS)` (2), several Shuttle/STS variants, `SJ-10`, null.
- Mission IDs mix separators (`Soyuz-TMA-16M` vs `Soyuz_TMA-14`).
- Some studies belong to two missions (e.g. OSD-168: SpaceX-4 and SpaceX-8; OSD-162: SpaceX-8
  and SpaceX-9; OSD-289: SpaceX-9 and SpaceX-12).

Mouse RNA-seq flight-vs-ground studies by mission (68 studies; the 3 two-mission studies are
counted under both): SpaceX-21 16, SpaceX-4 13, SpaceX-8 8, SpaceX-12 7, SpaceX-13 6,
SpaceX-18 5, SpaceX-9 3, SpaceX-11 2, SpaceX-15 2, SpaceX-16 2, SpaceX-27 2, and one each for
SpaceX-10, SpaceX-14, SpaceX-17, NG-11 and STS-135.

## Ortholog coverage

| Organism | MGenes | With ≥1 `IS_ORTHOLOG` edge | With >1 |
|---|---:|---:|---:|
| *Homo sapiens* | 29,026 | 29,026 (self) | 0 |
| *Mus musculus* | 27,415 | 20,098 (73.3%) | 1,607 |
| *Arabidopsis thaliana* | 25,426 | 0 | 0 |
| *C. elegans* | 17,460 | 6,128 | 2,773 |
| *Danio rerio* | 17,142 | 14,745 | 1,447 |
| *Drosophila melanogaster* | 15,933 | 6,913 | 2,581 |
| *Rattus norvegicus* | 6,249 | 5,985 | 297 |
| *S. cerevisiae* | 85 | 41 | 14 |

These are fractions of all MGene nodes, not of genes measured in a given assay. In one spot
check (OSD-47 liver, flight vs. ground), 61 of the 68 stored DE genes had an ortholog (6 had
more than one).

## Other gaps

- 4,800 Assay nodes have neither an Anatomy nor a CellType edge: *Arabidopsis* 3,404,
  *Drosophila* 1,048, human 298, mouse 24 (2 studies), microbiota 24, *C. elegans* 2.
- 108 Assay nodes have two Anatomy edges; the rest have one.
- `Study.host_strain` and `Study.host_organism` are **null for all 104 mouse studies**. Strain, where
  it appears at all, is in the raw factor lists (e.g. OSD-253 `C57BL/6J`, `C3H/HeJ`); otherwise it
  has to come from OSDR. `project_type` has a case variant (`Spaceflight Study` 71 mouse studies,
  `Spaceflight study` 1); 32 mouse studies are `Ground Study`.
- `MGene.symbol` sometimes joins several symbols with `|` (e.g. Entrez 108143 `Taf9|Ak6`):
  mouse 291, human 678, *Arabidopsis* 6,686. One more reason never to join on symbols.

## How this was queried

The specialist tools (`get_study_info`, `select_assays`) were used for single-study checks (OSD-48).
The census-wide counts aren't covered by any specialist tool, so they used the read-only `query`
tool (Cypher; 1,000-row and 60 s caps). Key queries:

```cypher
// node and relationship counts
MATCH (n) WHERE NOT n:MetaNode RETURN labels(n)[0] AS label, count(*) AS n
MATCH ()-[r]->() WHERE type(r) <> 'MetaRelationship' RETURN type(r), count(*)

// assays by organism x technology
MATCH (s:Study)-[:PERFORMED_SpAS]->(a:Assay)
RETURN s.organism, a.technology, a.measurement, a.differential_analysis_method,
       count(DISTINCT s) AS studies, count(a) AS assays

// strict mouse RNA-seq flight-vs-ground comparisons
MATCH (s:Study {organism:'Mus musculus'})-[:PERFORMED_SpAS]->(a:Assay {technology:'RNA Sequencing (RNA-Seq)'})
WHERE a.factor_space_1 = 'Space Flight' AND a.factor_space_2 = 'Ground Control'
  AND a.material_id_1 = a.material_id_2
  AND [x IN a.factors_1 WHERE x <> 'Space Flight'] = [x IN a.factors_2 WHERE x <> 'Ground Control']
CALL { WITH a MATCH (a)-[r:MEASURED_DIFFERENTIAL_EXPRESSION_ASmMG]->()
       RETURN count(r) AS n, max(r.adj_p_value) AS maxp }
RETURN s.identifier, a.identifier, a.material_id_1, a.material_name_1, a.factors_1, a.factors_2, n, maxp

// ortholog coverage
MATCH (g:MGene) OPTIONAL MATCH (g)-[:IS_ORTHOLOG_MGiG]->(h:Gene)
WITH g, count(h) AS k
RETURN g.organism, count(g), sum(CASE WHEN k>0 THEN 1 ELSE 0 END), sum(CASE WHEN k>1 THEN 1 ELSE 0 END)
```
