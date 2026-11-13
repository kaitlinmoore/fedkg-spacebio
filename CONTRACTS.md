# Contracts

The interfaces every part of the system agrees on. Defined **before** real data arrives, with
mocks, so census, harmonization, federation, policy and agent work can proceed independently.
Change a contract here first, then the code and fixtures that depend on it.

> **Status:** §1 (identifiers) is filled in from WS-01. §2 (census) is filled in from WS-01 and
> WS-06 (OSDR columns, 2026-11-12). §3 (site manifest) and §4 (gene mapping, site → server
> summary) are filled for phase 3a (WS-07). §5–§8 are skeleton.

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
| `location_aware_match` | bool | derived | **D-020:** strict, or differs only by flight `On ISS` vs control `On Earth` and durations ≤ 3 days apart. Selects the ISS-stratum comparisons. *Was* `strict_location_aware` (WS-06 proposal); renamed in WS-07 |
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

**Status: filled for phase 3a (WS-07, 2026-11-12; results table 2026-11-13).** One JSON file per FLARE client (a
(site, stratum) row of D-020), built by `fedkg_spacebio.sites.manifest.write_manifests` from the
census and written to `$FEDKG_WORKDIR/phase3a/manifests/<counts_variant>/<client>.json`. In a
real federation each site would declare its own. Here the coordinator builds them from public
metadata and never opens a count file. Still TODO: the gene-ID namespace field and a pointer to
the site's policy (phase 4).

| Field | Type | Notes |
|---|---|---|
| `client` | str | FLARE client name, e.g. `SpaceX-13-ISST`. Must equal the file name. The loader refuses a mismatch |
| `site` | str | mission (D-014), e.g. `SpaceX-13` |
| `stratum` | str | `iss` · `earth` (D-020) |
| `design` | str | pydeseq2 formula: `~ condition`, `~ study + condition` (SpaceX-4), `~ age + condition` (SpaceX-16) |
| `contrast` | list[str] | `["condition", "flight", "ground"]` (positive = higher in flight) |
| `counts_variant` | str | `standard` · `rrnarm` |
| `studies` | list[str] | OSD accessions this client may read. The loader rejects any other |
| `sources[]` | list | one per study: `osd_id`, `counts_file` (relative to the data root, inside `<osd_id>/`), `samples[]` |
| `sources[].samples[]` | list | `sample` (count-matrix column), `animal` (`<osd>:<source key>`), `arm` (`flight` · `ground`), covariates (`study`, `age`) |
| `census_rows` | list | the census rows used: `comparison_id`, `genelab_contrast`, factors, `strict_match`, `location_aware_match`, euthanasia location and source, `de_file` |
| `n_flight`, `n_ground` | int | animals per arm (one sample per animal, D-022) |

Rules enforced in code and tests:
- Each spec matches exactly one location-aware Ground-Control census row.
- One sample per animal (GeneLab's SampleTable picks `techrep1`).
- No sample or animal belongs to two clients (`check_disjoint`).
- The loader opens only its own manifest and the files it names, and reads only the listed
  sample columns.

## 4. Harmonization outputs and the site → server summary

**Status: filled for phase 3a (WS-07, 2026-11-12).** Still TODO: a harmonized expression
matrix, for the classifier phase.

**Gene mapping** (`fedkg_spacebio.harmonize.gene_map`):
- Source: GeneLab GL-DPPD-7110-A, mouse, Ensembl 112
  (`Mus_musculus.GRCm39.112-GL-annotations.tsv`, figshare file 48354460, SHA-256
  `123a0899…f54044e2`, checked on load).
- **One-to-one only, decided over the whole table:** `one_to_one` kept; `no_entrez`,
  `one_to_many` and `many_to_one` dropped (all claimants of a shared Entrez ID).
- Site DE runs on Ensembl IDs, and the **per-gene summaries** are mapped before they leave the
  site (D-012 as read in WS-07). Counts are never summed across IDs.
- Drop log: `{all_genes: {category: n}, tested_genes: {category: n}, source, sha256}`.

**Site → server summary** (`fedkg_spacebio.federation.site_de.SiteSummary`, checked by
`validate_summary` on both sides). This is the federation boundary: **no counts, no per-sample
values, no sample names.**

| Per gene (1-D arrays, same length) | Type | Notes |
|---|---|---|
| `ncbi_gene` | int64 | NCBI Gene ID (digits; CURIE `NCBIGene:<n>`), unique |
| `log2fc` | float64 | unshrunk MLE, flight vs ground |
| `se` | float64 | pydeseq2 `lfcSE`, > 0 |
| `base_mean` | float64 | mean of normalized counts over the site's samples |
| `tested_mean10` | bool | primary inclusion: base_mean ≥ 10, finite log2fc/se/p (Kaitlin, WS-07) |
| `tested_indep_filter` | bool | sensitivity (a): padj not NaN |
| `tested_finite_p` | bool | sensitivity (b): finite p |

| Site metadata (FLARE `meta["fedkg_site_summary"]`) | Notes |
|---|---|
| `client`, `site`, `stratum` | must match the FLARE sender (`meta["client_name"]`) |
| `n_flight`, `n_ground` | int |
| `design`, `counts_variant`, `de_settings` | DE settings include captured pydeseq2 warnings |
| `rounding` | rule, share of cells changed, max change |
| `n_genes_de`, `n_ercc_dropped`, `mapping` | the drop log above, plus counts sent |
| `versions`, `census_rows` | package versions; `comparison_id`, `genelab_contrast` |

**Governance note (for phase 4 policy):** random effects (REML) need each site's per-gene
estimate and SE at the server, so per-site summaries leave the site. A fixed-effect-only
analysis would need only Σw·y and Σw per gene (w = 1/SE²), which secure aggregation could
deliver without revealing any single site's estimate. A site policy that forbids releasing
per-site estimates is therefore compatible with FE but not with RE. The heterogeneity flag
(Cochran's Q) also needs per-site estimates.

**Phase 3a per-gene results table** (schema `phase3a-per-gene/1`,
`fedkg_spacebio.federation.results`; written by `jobs/phase3a_meta/report.py`). One file per
(stratum, counts, variant): `results/phase3a/<stratum>/<counts>/flare/<variant>/per_gene.csv.gz`
plus `per_gene.schema.json` (column dtypes and definitions, thresholds, provenance: clients,
inclusion flag, n per arm, package versions, annotation checksum). The stable interface for
comparing later pooling methods against these outputs.
- Format: gzip CSV, UTF-8, header, no index; floats `%.17g` (exact round trip); missing = empty;
  booleans `True`/`False`. Key: `ncbi_gene`. Any column change bumps the schema version.

| Column(s) | Notes |
|---|---|
| `ncbi_gene`, `symbol` | key; symbol is a display label from the annotation table |
| `k` | sites pooled for this gene |
| `fe_est`, `fe_se`, `fe_p`, `fe_q` | fixed effect; `fe_q` is BH within stratum and variant |
| `re_est`, `hksj_se`, `hksj_ci_low`, `hksj_ci_high`, `re_p`, `re_q` | REML RE, truncated HKSJ |
| `hksj_multiplier`, `re_p_untruncated` | HKSJ q before truncation; untruncated p |
| `tau2`, `reml_converged`, `i2`, `q`, `q_p` | heterogeneity (I² Q-based) |
| `fe_hit` | `fe_q` < 0.05: the headline (changed after seeing results, 2026-11-13) |
| `het_flag` | `q_p` < 0.10 |
| `fe_hit_consistent`, `fe_hit_heterogeneous` | `fe_hit` split by `het_flag` |
| `re_hit` | `re_q` < 0.05 (secondary) |
| `log2fc__<client>`, `se__<client>`, `base_mean__<client>`, `included__<client>` | per-site inputs; `included` = entered pooling for this gene |
| `n_flight__<client>`, `n_ground__<client>` | animals per arm in that site's fit (constant per column) |

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
