# Contracts

The interfaces every part of the system agrees on. Defined **before** real data arrives, with
mocks, so census, harmonization, federation, policy and agent work can proceed independently.
Change a contract here first, then the code and fixtures that depend on it.

> **Status:** §1 (identifiers) is filled in from WS-01. §2 (census) is filled in from WS-01 and
> WS-06 (OSDR columns, 2026-11-12). §3–§8 are skeleton.

## 1. Identifiers

Source formats confirmed live against the GeneLab KG on 2026-11-12 (see `docs/kg-schema.md`).
Every ID we store or emit is a CURIE or an OSD accession. The KG returns some IDs bare; the
conversion rule says how to make them CURIEs.

| Entity | Our form | KG form | Conversion |
|---|---|---|---|
| Study | `OSD-<n>` | `Study.identifier`, `OSD-<n>` | none |
| Comparison (KG "Assay") | `OSD-<n>-<32 hex>` | `Assay.identifier` | none. An ordered two-group comparison, **not** an OSDR assay. The OSDR assay name is `Assay.name`. |
| Mission | KG slug, e.g. `SpaceX-21` | `Mission.identifier` | none. Treat as opaque; the separators are inconsistent (`Soyuz_TMA-14`). |
| Gene, mouse/other organism | `NCBIGene:<n>` | `MGene.identifier`, bare digits as a string | prefix `NCBIGene:` |
| Gene, human reference | `NCBIGene:<n>` | `Gene.identifier`, bare digits as a string | prefix `NCBIGene:` |
| Tissue | `UBERON:NNNNNNN` | `Anatomy.identifier` / `Assay.material_id_*` | none |
| Cell type | `CL:NNNNNNN` | `CellType.identifier` | none |
| Organism / taxon | `NCBITaxon:<n>` | `Organism.identifier`, `Study.taxonomy`, `MGene.taxonomy`, bare digits | prefix `NCBITaxon:` |
| Methylation region | `<chrom>:<start>-<end>` (KG form) | `MethylationRegion.identifier` | none; no `chr` prefix, 1 kb bins |

- **Gene namespace target: NCBI Gene (Entrez)** (decided 2026-11-12). The KG is NCBIGene-only.
  OSDR liver matrices use unversioned `ENSMUSG` IDs (Ensembl 112, GRCm39; see
  `docs/census-liver.md`). They are mapped to NCBIGene once, with the drop rate logged. Every row
  also keeps its source namespace.
- Never join on gene symbols across datasets.
- Mocks and fixtures use obviously fake IDs: `FAKE:NNNN`, `OSD-FAKE-N`, comparison
  `OSD-FAKE-N-00000000000000000000000000000000`.
- TODO: site IDs, job IDs, trace IDs, run IDs (format and who mints them).

## 2. Census table (`census/`)

**Status: filled for liver (WS-06, 2026-11-12).** Built by `fedkg_spacebio.census.build`; written to
`$FEDKG_WORKDIR/census/liver_census.csv` (list columns as JSON strings). Evidence and numbers:
`docs/census-liver.md`.

**Grain (changed in WS-06):** one row per **OSDR flight → control group pair** within one OSDR
assay, built from OSDR sample metadata. The flight → control direction only. Each row carries the
KG comparison ID when one KG comparison matches it uniquely; otherwise `comparison_id` is null and
`kg_match` says why. *Was:* one row per KG comparison node. *Why changed:* the KG's labels are
damaged for some studies (OSD-168, OSD-245 basal), so KG-first rows would silently lose real
comparisons; OSDR-first rows keep them and expose the join quality.

| Column | Type | Source | Allowed values / notes |
|---|---|---|---|
| `osd_id` | str | OSDR `id.accession` | `OSD-<n>` |
| `comparison_id` | str \| null | KG `Assay.identifier` | `OSD-<n>-<32 hex>`; null unless `kg_match` is `exact` or `partial` |
| `kg_match` | str | derived | `exact` · `partial` (KG labels are a subset; still unique) · `ambiguous` (several OSDR pairs fit) · `unmatched` **(new)** |
| `kg_assay_match` | str \| null | derived | `exact` · `normalized` (case/whitespace) · `only_assay` (one RNA-seq assay each side) **(new)**; KG assay names lag OSDR's |
| `osdr_assay_name` | str | OSDR `id.assay name` | *was* KG `Assay.name`, which can differ from OSDR's current name |
| `mission_ids` | list[str] | KG `Mission`←`CONDUCTED_MIcS` | may be empty or have several entries; OSDR `mission.name` agreed for all liver studies |
| `site` | str \| null | proposed rule | the mission if `mission_ids` has exactly one entry, else null **(new; proposal, see SITES.md)** |
| `space_program` | list[str] | KG `Mission.space_program` | split the comma-joined KG strings; may be empty |
| `organism_taxon` | str | KG `Study.taxonomy` | `NCBITaxon:<n>` |
| `host_strain` | str \| null | *OSDR* `study.characteristics.strain` | `; `-joined if the two arms differ |
| `tissue_uberon` | str | KG `Assay.material_id_1` | from the matched KG comparison, else the study's single KG tissue term |
| `tissue_label` | str | KG `Assay.material_name_1` | |
| `organ_group` | str | ours | roll-up of lateral/sub-region terms (e.g. liver left lateral lobe → liver) |
| `control_type` | str | *OSDR* spaceflight factor | `Ground Control` · `Vivarium Control` · `Basal Control` |
| `factors_flight`, `factors_control` | list[str] | *OSDR* factor values | arm label first, then the other factors sorted by factor name; units without braces (`~60 day`) |
| `factors_flight_kg`, `factors_control_kg` | list[str] \| null | KG `Assay.factors_1/2` | the raw KG labels of the matched comparison **(new)** |
| `strict_match` | bool | derived | true if every non-spaceflight factor is identical on both sides (D-015) |
| `differing_factors` | list[str] | derived | factor names that differ; empty iff `strict_match` **(new)** |
| `strict_location_aware` | bool | derived | **proposed, sensitivity only:** strict, or differs only by flight `On ISS` vs control `On Earth` and durations ≤ 3 days apart **(new)** |
| `technology` | str | constant | `RNA Sequencing (RNA-Seq)` |
| `de_method` | str \| null | KG `Assay.differential_analysis_method` | `DESeq2` for all liver rows, consistent with OSDR's protocol text |
| `n_de_edges_kg` | int \| null | derived | KG DE edges stored (adj_p < 0.1 only) |
| `n_flight`, `n_control` | int | *OSDR* | **animals** (unique normalized source names) per arm. *Was:* "sample counts per arm"; changed because one animal can give several samples (spike-in duplicates, technical replicates) |
| `n_flight_samples`, `n_control_samples` | int | *OSDR* | samples per arm **(new)** |
| `n_flight_indep`, `n_control_indep` | int | derived | animals not shared with an assigned study at another site **(new)** |
| `n_flight_in_de`, `n_control_in_de` | int | *OSDR* SampleTable | samples GeneLab's DE used **(new)** |
| `flight_animals`, `control_animals` | list[str] | *OSDR* | normalized source keys **(new)**; inputs to the overlap check |
| `sex` | str | *OSDR* `study.characteristics.sex` | `male` · `female` · `mixed` · `unknown` |
| `euthanasia_location_flight` | str \| null | *OSDR* factor, else curated protocol text | `ISS` · `Earth` **(new)** |
| `euthanasia_location_flight_source` | str | derived | `OSDR factor` · `curated (stated)` · `curated (inferred)` **(new)** |
| `euthanasia_location_control` | str | derived | `Earth` for all control arms **(new)** |
| `counts_file`, `counts_file_rrnarm` | str \| null | *OSDR* file listing | `*_RSEM_Unnormalized_Counts[_rRNArm]_GLbulkRNAseq.csv`; `counts_file` is the one GeneLab's (and the KG's) DE used **(new)** |
| `de_file` | str \| null | *OSDR* file listing | the standard DE table **(new)** |
| `genelab_contrast` | str \| null | *OSDR* SampleTable + contrasts | e.g. `(Space Flight)v(Ground Control)` **(new)** |
| `has_stat` | bool \| null | *OSDR* DE header | `Stat_<genelab_contrast>` present; cross-check only (D-016) **(new)** |
| `gene_id_namespace` | str | *OSDR* counts matrix | `ENSEMBL` · `NCBIGene` · `SYMBOL` · `mixed`; all liver rows `ENSEMBL` (unversioned) |
| `annotation_version` | str | *OSDR* protocol text | `<genome build>, Ensembl <release>`, e.g. `GRCm39, Ensembl 112` |
| `pipeline_version`, `nf_rcp_version` | str | *OSDR* protocol text | e.g. `GL-DPPD-7101-G`, `2.0.2` **(new)** |
| `kg_version` | str | pinned | `SPOKE-GeneLab v0.3.1+20260311` (assumed; not exposed live) |
| `access_date` | date | ours | date of the OSDR pull (the KG result is cached with its own date) |

Rows failing the strict rule are **kept** with `strict_match = false` (D-015); this resolves the old
open question. Site definition is D-014; the liver mapping is proposed in `SITES.md`.

## 3. Site manifest (`sites/`)
TODO: which studies and samples belong to a site, where its data lives locally, its gene-ID
namespace, and a pointer to its policy.

## 4. Harmonization outputs (`harmonize/`)
TODO: harmonized expression matrix layout, mapping tables (source ID → target CURIE, method,
confidence), and the drop-rate log format.

## 5. Site policy schema (`policy/`)
TODO: allowed analyses, outputs that may leave the site, minimum aggregation size, DP
requirement, and the shape of an allow/deny decision (including the rule that decided it).

## 6. Federated job spec (`federation/`)
TODO: task, model, participating sites, rounds, aggregation strategy, split definition, and
(for phase 3b) each site's contact-window schedule.

## 7. Agent tool return shapes (`agent/`)
TODO: one entry per tool (KG query, site registry, policy check, job submit, job status,
results reader). Every tool result that carries evidence returns the IDs it found.

## 8. Agent trace format (`agent/`)
TODO: one record per step (tool calls, inputs, outputs, decisions and reasons). The grounding
checker reads this to decide whether each cited ID appeared in the trace's own tool results.
