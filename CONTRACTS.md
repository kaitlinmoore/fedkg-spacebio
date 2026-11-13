# Contracts

The interfaces every part of the system agrees on. Defined **before** real data arrives, with
mocks, so census, harmonization, federation, policy and agent work can proceed independently.
Change a contract here first, then the code and fixtures that depend on it.

> **Status:** §1 (identifiers) and §2 (census, draft) are filled in from WS-01; §3–§8 are skeleton.

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

- **Gene namespace target: Proposed, not decided** (see the WS-01 return note). The KG is
  NCBIGene-only, so KG joins need NCBIGene. CLAUDE.md currently says to prefer Ensembl. Until
  Kaitlin decides, store the source namespace on every row and don't convert between them.
- Never join on gene symbols across datasets.
- Mocks and fixtures use obviously fake IDs: `FAKE:NNNN`, `OSD-FAKE-N`, comparison
  `OSD-FAKE-N-00000000000000000000000000000000`.
- TODO: site IDs, job IDs, trace IDs, run IDs (format and who mints them).

## 2. Census table (`census/`)

**Status: draft columns.** Grain is one row per **flight-vs-control comparison**: one
KG comparison node in the flight → control direction only (the KG stores both directions;
see `docs/kg-schema.md`). Columns marked *OSDR* aren't in the KG and need the OSDR Biological
Data API.

| Column | Type | Source | Allowed values / notes |
|---|---|---|---|
| `osd_id` | str | KG `Study.identifier` | `OSD-<n>` |
| `comparison_id` | str | KG `Assay.identifier` | `OSD-<n>-<32 hex>` |
| `osdr_assay_name` | str | KG `Assay.name` | shared by several comparisons |
| `mission_ids` | list[str] | KG `Mission`←`CONDUCTED_MIcS` | may be empty (93 of 212 studies) or have several entries |
| `space_program` | list[str] | KG `Mission.space_program` | split the comma-joined KG strings; may be empty |
| `organism_taxon` | str | KG `Study.taxonomy` | `NCBITaxon:<n>` |
| `host_strain` | str \| null | raw factors, else *OSDR* | KG `Study.host_strain` is null for all 104 mouse studies |
| `tissue_uberon` | str | KG `Assay.material_id_1` | must equal `material_id_2` for a row to be valid |
| `tissue_label` | str | KG `Assay.material_name_1` | |
| `organ_group` | str | ours | roll-up of lateral/sub-region terms (e.g. left kidney → kidney); the mapping table lives in `harmonize/` |
| `control_type` | str | KG `Assay.factor_space_2` | `Ground Control` · `Vivarium Control` · `Basal Control` |
| `factors_flight` | list[str] | KG `Assay.factors_1` | raw labels |
| `factors_control` | list[str] | KG `Assay.factors_2` | raw labels |
| `strict_match` | bool | derived | true if the non-flight factors are identical on both sides |
| `technology` | str | KG `Assay.technology` | e.g. `RNA Sequencing (RNA-Seq)` |
| `de_method` | str | KG `Assay.differential_analysis_method` | KG labels microarrays `DESeq2` too; verify against OSDR |
| `n_de_edges_kg` | int | derived | KG DE edges stored (adj_p < 0.1 only) |
| `n_flight`, `n_control` | int | *OSDR* | sample counts per arm |
| `sex` | str | *OSDR* (sometimes in factors) | `male` · `female` · `mixed` · `unknown` |
| `gene_id_namespace` | str | *OSDR* | e.g. `ENSEMBL`, `NCBIGene` for the per-sample matrix |
| `annotation_version` | str | *OSDR* | e.g. `GRCm39`, `GRCm38`; record it, don't assume it |
| `kg_version` | str | pinned | `SPOKE-GeneLab v0.3.1+20260311` (assumed; not exposed live) |
| `access_date` | date | ours | date of the KG/OSDR pull |

Open: whether rows that fail the strict match (e.g. `uG` vs `1G on Earth`) are kept with
`strict_match = false` or dropped. Tissue scope and site definition are Q-2.

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
