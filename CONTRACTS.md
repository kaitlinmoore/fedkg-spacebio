# Contracts

The interfaces every part of the system agrees on. Defined **before** real data arrives, with
mocks, so census, harmonization, federation, policy and agent work can proceed independently.
Change a contract here first, then the code and fixtures that depend on it.

> **Status:** §1 (identifiers) is filled in from WS-01, with run-time IDs added in WS-10. §2
> (census) is filled in from WS-01 and WS-06 (OSDR columns, 2026-11-12). §3 (site manifest) and §4
> (gene mapping, site → server summary) are filled for phase 3a (WS-07). §5 (site policy) and §6
> (job spec) are drafted, and §7 (tool returns) and §8 (trace) are sketched for the governance demo
> (WS-10, 2026-11-13). None of §5–§8 is implemented yet; the code changes they imply are listed
> for WS-03's code part.

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

**Run-time IDs** (WS-10, 2026-11-13; implemented with the policy and agent layers). Times are UTC.
Mocks put `FAKE` in place of the timestamp and random part (`job:FAKE-0001`).

| Entity | Form | Example | Minted by |
|---|---|---|---|
| Site | mission slug (D-014) | `SpaceX-16` | census (`site` column, §2) |
| Client | site, plus a stratum suffix when a site has clients in both strata (`-ISST`, `-LAR`) | `SpaceX-16-ISST` | manifest builder (§3) |
| Policy | `pol:<site>:<version>`; the project policy is `pol:project:<version>` | `pol:SpaceX-16:2` | the site's policy file (§5) |
| Rule | `<policy id>#<rule key>` | `pol:SpaceX-16:2#diff` | the policy file |
| Job | `job:<yyyymmddThhmmssZ>-<6 hex>` | `job:20261116T142301Z-a1b2c3` | the policy check, on first seeing a draft spec (§6) |
| Run | `run:<job suffix>-<attempt>`; FLARE's own job UUID is stored alongside | `run:20261116T142301Z-a1b2c3-1` | `job.submit` (§7) |
| Release | `rel:<job suffix>-<n>` | `rel:20261116T142301Z-a1b2c3-1` | the server's release log, when it releases a result (§5.3) |
| Trace | `trace:<yyyymmddThhmmssZ>-<8 hex>` | `trace:20261120T150000Z-0f3c9a12` | the agent runtime, at the start of a run (§8) |
| Tool call | `<trace id>/call/<n>` | `trace:…/call/7` | the agent runtime |
| Decision | `<trace id>/dec/<n>`; `dec:<job suffix>-<n>` when the policy check runs outside an agent | `trace:…/dec/2` | the policy check, or the agent for its own decisions (§8) |
| KG edge (composite key) | `kgedge:<REL_TYPE>\|<start id>\|<end id>`, with our CURIE forms | `kgedge:IS_ORTHOLOG_MGiG\|NCBIGene:<n>\|NCBIGene:<n>` | the `kg.query` tool (§7) |

The KG has no stable edge IDs, so edges are cited by this composite key. It's unique only if the
KG holds at most one edge of a type between two nodes. Check that before relying on it (WS-14).

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
| `tested_mean10` | bool | primary inclusion: base_mean ≥ 10, finite log2fc/se/p (D-024, D-027) |
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
(Cochran's Q) also needs per-site estimates. §5 turns this into policy rules; §6.3 describes the
masked sum.

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
| `fe_hit` | `fe_q` < 0.05. Per D-029 this is the "these missions" result, with Q and I² as annotations, not filters; V3 answers "a new mission". Its V3 columns and this description's code-side change are pending (WS-03 code part) |
| `het_flag` | `q_p` < 0.10 |
| `fe_hit_consistent`, `fe_hit_heterogeneous` | `fe_hit` split by `het_flag` (the D-026 headline; descriptive only under D-029) |
| `re_hit` | `re_q` < 0.05 (secondary) |
| `log2fc__<client>`, `se__<client>`, `base_mean__<client>`, `included__<client>` | per-site inputs; `included` = entered pooling for this gene |
| `n_flight__<client>`, `n_ground__<client>` | animals per arm in that site's fit (constant per column) |

## 5. Site policy schema (`policy/`)

**Status: drafted (WS-10, 2026-11-13); not implemented.** The evaluator, the release log and the
policy test suite are planned. **The policies in the examples are invented for the testbed.** The
real missions declared no such rules; the policies exist to test enforcement.

### 5.1 What a policy says

One YAML file per site (`tests/fixtures/policies/<profile>/<site>.yaml` for fixtures). A site's
policy covers all of its clients (e.g. `SpaceX-16-ISST` and `SpaceX-16-LAR`). A project policy,
`pol:project:<n>`, holds the rules every job must pass whatever the sites allow.

| Field | Type | Notes |
|---|---|---|
| `schema` | str | `site-policy/1` |
| `policy_id` | str | `pol:<site>:<version>`, or `pol:project:<version>` (§1) |
| `site` | str \| null | §1 site ID; null for the project policy |
| `effective` | date | |
| `default` | str | always `deny`: anything no rule allows is refused |
| `rules[]` | list | evaluated in file order. Each has `key` (unique in the file; the rule ID is `<policy_id>#<key>`), `kind`, its parameters, and `reason` (the text reported when the rule decides) |

**Rule kinds**

| `kind` | Parameters | Passes when |
|---|---|---|
| `analyses` | `tasks: [..]`; `methods: [..]` or `"*"` | every task and method in the job is listed |
| `release` | `outputs: [..]`, output kinds below | every output the job asks of this site's clients is listed |
| `min_aggregation` | `min_sites` (int), `min_n_per_arm` (int) | every gene this site contributes to pools at least `min_sites` sites, **counted per gene** (§5.3), and each of the site's clients in the job has at least `min_n_per_arm` animals per arm |
| `differencing` | `min_sites` (int, ≥ 2) | no combination of this release with logged ones isolates this site's contribution to any gene, or the sum of a group of fewer than `min_sites` sites that includes it (§5.3) |
| `dp` | `required` (bool); mechanism parameters later | `required: false`, or the job declares a DP mechanism. None exists yet, so `required: true` denies every job |
| `weighting` | (project policy only) | a `fedavg` job declares `weighting`; `sample_count` needs every client to declare n_k; `equal` or `custom` needs a `reason` (D-017) |

**Output kinds** (what may leave a client)

| Kind | Contents | What the receiver can read |
|---|---|---|
| `site_metadata` | n per arm, design, per-gene inclusion flags (§4 `tested_*`) | the coordinator, before any job (site registry, §7) |
| `masked_contribution` | per gene in the job's gene plan: w·y and w (w = 1/SE²), masked so that only the sum over clients is readable (§6.3) | per-gene sums only |
| `per_site_summary` | the §4 site → server summary | each site's per-gene estimate and SE |
| `model_update` | FedAvg weights plus n_k (phase 3) | each site's update |
| `per_sample` | anything per sample or per animal | never allowed (project policy) |

Inclusion flags are `site_metadata` on purpose. The per-gene checks in §5.3 have to know, before a
job runs, which sites will contribute to each gene. The cost: the coordinator learns which genes
pass the expression threshold at each site. The alternative, masked flags so that only k per gene
is learned, would make per-gene checks impossible before submission.

### 5.2 What each method needs

| Method key | Task | Needs from every client | Why |
|---|---|---|---|
| `fe` | `de_meta` | `masked_contribution` or `per_site_summary` | the FE estimate and SE are functions of Σw·y and Σw per gene |
| `fe_het` | `de_meta` | `per_site_summary` | Cochran's Q and I² need each site's estimate |
| `reml_hksj` | `de_meta` | `per_site_summary` | REML τ² and the HKSJ scale use each site's (y, SE) |
| `v2` | `de_meta` | `per_site_summary` | moderates φ = Q/(k−1) |
| `v3` | `de_meta` | `per_site_summary` | each gene's restricted likelihood over the τ² grid |
| `clock_plane` | `clock_plane` | `per_site_summary` | D-031's per-site outputs (chord, shift, R²) |
| `fedavg` | `fedavg_classifier` | `model_update` | n_k required (D-017) |

A leave-one-mission-out run is the same method on fewer clients. It needs nothing new from any
client, but it is a new release, so §5.3 checks it against the release log.

### 5.3 Per-gene checks and the release log

In phase 3a the set of sites contributing to a gene varies by gene: complete-case vs ≥ 3-site
inclusion, other gene-set variants, leave-one-out runs. So minimum k and differencing are checked
**gene by gene**, not only per job.

**Release log.** Every result the server releases gets a release ID (§1). For each gene it records
the set of clients that contributed. A `per_site_summary` or `model_update` from client c counts as
a release over {c} alone. Releases are compared within a *data key* = (stratum, clients),
**whatever their counts variant, inclusion variant or DE settings**. Those variants give
near-identical per-site values (FE estimates correlate ≥ 0.9999 across count matrices in WS-07),
so subtracting across variants leaks approximately. The log lives under `$FEDKG_WORKDIR`, one per
policy profile. Genes are keyed by NCBI Gene ID.

**The log is append-only.** No entry is ever edited or removed. A "reset" (e.g. between demo
runs) means starting a new log under a new profile ID, and that switch is itself recorded. If
the log could be cleared, the differencing check could be bypassed by resetting between two
requests.

**Minimum k, per gene.** For a site i with `min_aggregation.min_sites = m`, every gene in the
job's gene plan that i contributes to must have k_g ≥ m. A gene that i doesn't contribute to isn't
checked against i's rule.

**Differencing, per gene.** For gene g, write each logged release, and the proposed one, as a 0/1
row over the stratum's clients (1 = contributed). Call the stacked rows R. A site i with
`differencing.min_sites = m` fails on gene g if some linear combination of the rows, cᵀR, is
non-zero at i and zero at every client outside a set T, where i ∈ T and |T| < m. In words: no set
of releases may let anyone compute i's own contribution (|T| = 1), or the sum of a group smaller
than m that includes i. With at most 7 clients per stratum, it is exact and cheap: enumerate each
T (at most 2⁶ per site and gene) and solve a tiny linear system.

The check runs for **every** proposed release, including one asked of another site. A
`per_site_summary` from SpaceX-4 can be refused by SpaceX-16's differencing rule (example B, step J5).

**Per-gene failures.** Any failing gene makes the job `deny`. The decision lists the failing genes
per rule. The agent may resubmit with those genes excluded (`genes_exclude` in §6, pointing to the
decision). Leaving a gene out of a release reveals nothing about its values.

### 5.4 The decision record

Produced by the policy check (§7 `policy.check`) for one exact job spec and the release log as it
stands.

| Field | Type | Notes |
|---|---|---|
| `decision_id` | str | §1 |
| `job_spec_sha256` | str | hash of the spec without its `policy_decision` field (§6) |
| `release_log_head` | str \| null | last release ID in the log when checked (null if empty) |
| `outcome` | str | `allow` · `deny` |
| `deciding_rule` | str \| null | the first failing rule in evaluation order; null when allowed |
| `reason` | str | that rule's `reason`, plus specifics (method, output, gene count) |
| `evaluations[]` | list | every rule evaluated: `rule_id`, `site`, `scope` (`job` · `gene`), `result` (`pass` · `fail` · `not_applicable`), `detail`; for gene-scope failures `genes_failed` (count) and `genes_failed_ref` + `sha256` (the list, under `$FEDKG_WORKDIR`) |
| `policies` | list | `policy_id` and `sha256` of every policy file used |
| `evaluator_version` | str | |

**Combining rules.** Deny overrides allow, and anything not allowed is denied. Evaluation order is
the project policy, then sites in the order of the job's `clients`, then rules in file order.
`deciding_rule` is the first failure in that order; every failure is listed.

### 5.5 Worked examples (testbed policies)

Two profiles for the same seven liver clients. In the **open** profile every site allows
per-site summaries. In the **restricted** profile SpaceX-16 allows only masked contributions;
the other sites keep their open policies. Fake gene IDs; real client names.

```yaml
# pol:project:1 (both profiles)
schema: site-policy/1
policy_id: "pol:project:1"
site: null
effective: 2026-11-16
default: deny
rules:
  - key: no-per-sample
    kind: release
    outputs: [site_metadata, masked_contribution, per_site_summary, model_update]
    reason: "Nothing per sample or per animal leaves a site."
  - key: fedavg-weighting
    kind: weighting
    reason: "FedAvg weights by n_k unless the job declares equal or custom weights with a reason (D-017)."
```

```yaml
# Open profile: pol:SpaceX-16:1 (every site's open policy has this shape)
schema: site-policy/1
policy_id: "pol:SpaceX-16:1"
site: SpaceX-16
effective: 2026-11-16
default: deny
rules:
  - key: analyses
    kind: analyses
    tasks: [de_meta, clock_plane]
    methods: "*"
    reason: "Differential-expression meta-analysis and the clock diagnostic only."
  - key: release
    kind: release
    outputs: [site_metadata, masked_contribution, per_site_summary]
    reason: "Per-gene summaries may leave the site; nothing per sample."
  - key: min-k
    kind: min_aggregation
    min_sites: 3
    min_n_per_arm: 5
    reason: "Pooled results must combine at least 3 sites; at least 5 animals per arm."
  - key: dp
    kind: dp
    required: false
    reason: "No differential-privacy requirement."
```

```yaml
# Restricted profile: pol:SpaceX-16:2
schema: site-policy/1
policy_id: "pol:SpaceX-16:2"
site: SpaceX-16
effective: 2026-11-16
default: deny
rules:
  - key: analyses
    kind: analyses
    tasks: [de_meta]
    methods: "*"
    reason: "Differential-expression meta-analysis only."
  - key: release
    kind: release
    outputs: [site_metadata, masked_contribution]
    reason: "Per-site estimates may not leave SpaceX-16; it contributes only masked values to a secure sum."
  - key: min-k
    kind: min_aggregation
    min_sites: 3
    min_n_per_arm: 5
    reason: "Pooled results must combine at least 3 sites; at least 5 animals per arm."
  - key: diff
    kind: differencing
    min_sites: 3
    reason: "No combination of releases may isolate SpaceX-16's contribution, or a group of fewer than 3 sites that includes it."
  - key: dp
    kind: dp
    required: false
    reason: "No differential-privacy requirement."
```

**Example A: fixed effect allowed, random effects refused (job level, restricted profile).**
The agent's first plan for "which genes change with flight across ISS missions" asks for the full
phase 3a method set.

```json
{
  "decision_id": "trace:FAKE-0001/dec/2",
  "job_spec_sha256": "FAKE-sha256-0001",
  "release_log_head": null,
  "outcome": "deny",
  "deciding_rule": "pol:SpaceX-16:2#release",
  "reason": "Per-site estimates may not leave SpaceX-16; it contributes only masked values to a secure sum. Methods fe_het, reml_hksj, v3 need per_site_summary from SpaceX-16-ISST.",
  "evaluations": [
    {"rule_id": "pol:project:1#no-per-sample", "site": null, "scope": "job", "result": "pass", "detail": ""},
    {"rule_id": "pol:project:1#fedavg-weighting", "site": null, "scope": "job", "result": "not_applicable", "detail": "task de_meta"},
    {"rule_id": "pol:SpaceX-4:1#release", "site": "SpaceX-4", "scope": "job", "result": "pass", "detail": ""},
    {"rule_id": "pol:SpaceX-16:2#analyses", "site": "SpaceX-16", "scope": "job", "result": "pass", "detail": ""},
    {"rule_id": "pol:SpaceX-16:2#release", "site": "SpaceX-16", "scope": "job", "result": "fail", "detail": "per_site_summary requested from SpaceX-16-ISST"},
    {"rule_id": "pol:SpaceX-16:2#min-k", "site": "SpaceX-16", "scope": "gene", "result": "pass", "detail": "k = 4 for every gene in the plan"},
    {"rule_id": "pol:SpaceX-16:2#diff", "site": "SpaceX-16", "scope": "gene", "result": "pass", "detail": "release log empty"}
  ],
  "policies": [{"policy_id": "pol:SpaceX-16:2", "sha256": "FAKE-sha256-0002"}],
  "evaluator_version": "0.0.0-mock"
}
```
(Evaluations for the other sites are elided.) The agent then plans `methods: [fe]`,
`outputs: masked_contribution` for every client and `aggregation: secure_sum` (§6.4). That spec
gets `outcome: allow`. The agent reports that heterogeneity, the "new mission" estimate (V3) and
REML/HKSJ were not run, and names `pol:SpaceX-16:2#release` as the reason.

**Example B: per-gene minimum k and differencing (restricted profile, ISS stratum).**
Clients in column order: A = SpaceX-4, B = SpaceX-8, C = SpaceX-13-ISST, D = SpaceX-16-ISST.
Rows are contributing-client sets. R1, R2 and J3 were released and logged; J4 and J5 were refused,
so they add nothing to the log.

| Step | Request | `FAKE:0001` | `FAKE:0003` | `FAKE:0004` | Decision |
|---|---|---|---|---|---|
| R1 | FE secure sum, complete case | {A,B,C,D} | – (not tested at B) | – | allow. k = 4; the log is empty |
| R2 | FE secure sum, ≥ 3 sites | {A,B,C,D} | {A,C,D} | {A,B,C} | allow. FAKE:0001 repeats R1's row. FAKE:0003 has k = 3, and its only row involving D is this one. FAKE:0004 doesn't involve D |
| J3 | FE without SpaceX-8, ≥ 3 sites | {A,C,D} | {A,C,D} | – (k = 2 without B) | allow. For FAKE:0001, rows 1111 and 1011: the difference isolates B, whose open policy allows that. Any combination non-zero at D is also non-zero at A and C, so \|T\| ≥ 3 |
| J4 | FE without SpaceX-16, complete case over A,B,C | {A,B,C} | – | {A,B,C} | **deny**, `pol:SpaceX-16:2#diff`. For FAKE:0001, 1111 − 1110 = 0001 isolates D. FAKE:0004 passes: same row as R2 and no D. The agent may resubmit without the failing genes, which leaves only genes D never contributed to |
| J5 | `per_site_summary` from SpaceX-4 only | {A} | {A} | {A} | **deny**, `pol:SpaceX-16:2#diff`. For FAKE:0001, after R1 and J3: 1011 − 1000 = 0011, the sum of C and D, a group of 2 < 3. FAKE:0003 the same (R2's 1011). The request asks nothing of SpaceX-16, but SpaceX-16's rule refuses it |

**Example C: minimum k per gene in the Earth stratum (either profile).** The Earth stratum has
three clients (SpaceX-13-LAR, SpaceX-16-LAR, SpaceX-21). Any leave-one-out run gives k = 2 for
every gene, so it fails `#min-k` at both remaining sites in either profile. In the restricted
profile, dropping SpaceX-16-LAR also fails `pol:SpaceX-16:2#diff`. Under `min_sites_per_gene: all`
every Earth gene has k = 3 and passes.

## 6. Federated job spec (`federation/`)

**Status: drafted (WS-10, 2026-11-13); not implemented.** Today's phase 3a jobs take their
settings as command-line flags (`jobs/phase3a_meta/`). This spec is what the agent submits and
the policy check reads. Format: YAML or JSON, one document per job.

### 6.1 Fields

| Field | Type | Notes |
|---|---|---|
| `schema` | str | `job-spec/1` |
| `job_id` | str \| null | §1. Null in a draft; minted when the policy check first sees the spec |
| `task` | str | `de_meta` (phase 3a) · `clock_plane` (D-031) · `fedavg_classifier` (phase 3) |
| `question` | str | free text, kept for the record; never used in a decision |
| `stratum` | str | `iss` · `earth` |
| `clients` | list[str] | §1 client IDs; every one must be in the site registry |
| `methods` | list[str] | §5.2 keys |
| `counts_variant` | str | `standard` · `rrnarm` |
| `inclusion` | object | `rule`: `mean10_finite` (D-024/D-027) · `indep_filter` · `finite_p`; `min_sites_per_gene`: `all` (complete case) or an int |
| `gene_plan` | object | `ref` + `sha256` of a table `ncbi_gene` → contributing clients. Built from the sites' inclusion flags before the policy check (§5.3). A client sends values only for its genes in the plan |
| `genes_exclude` | object \| null | `{decision_id, ref, sha256}`: genes removed after a per-gene deny (§5.3) |
| `outputs` | map | client → output kind (§5.1). Must provide what every method needs (§5.2) |
| `aggregation` | object | `strategy`: `plain` · `secure_sum`; for `secure_sum`, `fixed_point_bits` and `modulus_bits` (§6.3) |
| `weighting` | object | **required.** `de_meta`: `{scheme: inverse_variance}` (fixed by the method). `fedavg_classifier`: `{scheme: sample_count}` by default (D-017); `equal`, or `custom` with `weights` (client → number), only with a `reason` |
| `rounds` | int | 1 for `de_meta` and `clock_plane` |
| `split` | object \| null | classifier only: `{scheme: loso, folds_ref, sha256}` (fixed shared folds; leave-one-site-out). Null otherwise |
| `contact_windows` | object \| null | phase 3b placeholder: per client a list of `{start, end}` (UTC), plus `deadline`. Null means always connected |
| `provenance` | object | `kg_version`, `census_access_date`, `annotation_sha256`, and each client's `manifest_sha256` (§3) |
| `policy_decision` | str \| null | the decision ID that allowed **this exact spec** (`job_spec_sha256` matches). `job.submit` refuses a spec without one |

### 6.2 Site → client contract additions

- **`de_meta` with `per_site_summary`:** unchanged, §4.
- **`de_meta` with `masked_contribution`:** for each gene the client has in `gene_plan`, the masked
  fixed-point values of w·y and w (w = 1/SE², y = log2FC), over the plan's gene order. Genes the
  client isn't assigned are sent as masked zeros, so every client's vector has the same length.
  `n_flight`, `n_ground` and the §4 site metadata go unmasked. The server learns per-gene sums only.
- **`fedavg_classifier`:** every client result carries `FLMetaKey.NUM_STEPS_CURRENT_ROUND = n_k`
  (training samples this round). The client also declares n_k in its manifest. The server-side check
  fails the round if a result lacks it, rather than letting FLARE fall back to 1.0 (D-017).
- **`clock_plane`:** D-031's whitelist of per-site outputs, as `per_site_summary`.

### 6.3 `secure_sum`: simulated secure aggregation

- Each pair of clients (i, j) shares a seed. Client i adds the pseudo-random mask stream for each
  pair with j > i and subtracts it for j < i, mod 2^`modulus_bits`. The masks cancel in the sum
  over all clients.
- Values are fixed-point integers (round(x · 2^`fixed_point_bits`)), so the sum is exact modulo
  the rounding.
- **This is a simulation.** The pairwise seeds come from the job config, not a key exchange. In the
  FLARE simulator every client runs in one process. A missing client makes the job fail (no
  dropout recovery). It shows what the server could learn under this policy; it is not a
  security claim.
- **Check:** FE from masked sums equals FE from plain per-site summaries within 2^−`fixed_point_bits`
  per gene, on synthetic data and on the real liver summaries (the same kind of check as WS-07
  Check 1).

### 6.4 Examples

```yaml
# The restricted-profile job allowed after example A (§5.5)
schema: job-spec/1
job_id: "job:FAKE-0002"
task: de_meta
question: "Which liver genes change with spaceflight across ISS missions?"
stratum: iss
clients: [SpaceX-4, SpaceX-8, SpaceX-13-ISST, SpaceX-16-ISST]
methods: [fe]
counts_variant: standard
inclusion: {rule: mean10_finite, min_sites_per_gene: all}
gene_plan: {ref: "plans/job-FAKE-0002/gene_plan.csv", sha256: "FAKE-sha256-0003"}
genes_exclude: null
outputs:
  SpaceX-4: masked_contribution
  SpaceX-8: masked_contribution
  SpaceX-13-ISST: masked_contribution
  SpaceX-16-ISST: masked_contribution
aggregation: {strategy: secure_sum, fixed_point_bits: 32, modulus_bits: 64}
weighting: {scheme: inverse_variance}
rounds: 1
split: null
contact_windows: null
provenance:
  kg_version: "SPOKE-GeneLab v0.3.1+20260311"
  census_access_date: 2026-11-12
  annotation_sha256: "FAKE-sha256-0004"
  manifest_sha256: {SpaceX-4: "FAKE-sha256-0005", SpaceX-8: "FAKE-sha256-0006",
                    SpaceX-13-ISST: "FAKE-sha256-0007", SpaceX-16-ISST: "FAKE-sha256-0008"}
policy_decision: "trace:FAKE-0001/dec/3"
```

```yaml
# A FedAvg job (phase 3; synthetic sites). Declares n_k weighting explicitly.
schema: job-spec/1
job_id: "job:FAKE-0003"
task: fedavg_classifier
question: "Flight vs ground from liver expression, leave-one-site-out."
stratum: iss
clients: [FAKE-SITE-1, FAKE-SITE-2, FAKE-SITE-3]
methods: [fedavg]
counts_variant: standard
inclusion: {rule: mean10_finite, min_sites_per_gene: all}
gene_plan: {ref: "plans/job-FAKE-0003/gene_plan.csv", sha256: "FAKE-sha256-0009"}
genes_exclude: null
outputs: {FAKE-SITE-1: model_update, FAKE-SITE-2: model_update, FAKE-SITE-3: model_update}
aggregation: {strategy: plain}
weighting: {scheme: sample_count}      # each client sends NUM_STEPS_CURRENT_ROUND = n_k
rounds: 20
split: {scheme: loso, folds_ref: "splits/FAKE-folds.json", sha256: "FAKE-sha256-0010"}
contact_windows: null
provenance: {kg_version: "SPOKE-GeneLab v0.3.1+20260311", census_access_date: 2026-11-12,
             annotation_sha256: "FAKE-sha256-0004", manifest_sha256: {}}
policy_decision: null                  # not yet checked
```
A spec with `weighting: {scheme: equal}` and no `reason` fails `pol:project:1#fedavg-weighting`.

## 7. Agent tool return shapes (`agent/`)

**Status: sketched for the demo (WS-10, 2026-11-13).** Every tool returns the same envelope. The
`ids` list is what the grounding checker (§8) trusts, so each tool must list every ID in `data`
that a claim could cite. A test scans each tool's `data` for ID patterns and checks they are all
in `ids`.

```json
{
  "call_id": "trace:FAKE-0001/call/7",
  "tool": "kg.query",
  "tool_version": "0.1.0",
  "ok": true,
  "error": null,
  "data": {},
  "ids": [],
  "cache": {"hit": true, "key": "sha256:FAKE", "fetched": "2026-11-20T15:02:11Z"}
}
```
`error` is `{code, message}` when `ok` is false. `cache` is null for tools that make no external
call. The trace records the version of every tool used (§8 `run_start`), so the set of tools a
run used can be audited.

| Tool | Input | `data` | `ids` |
|---|---|---|---|
| `sites.list` | `stratum?`, `tissue?` | per client: `client`, `site`, `stratum`, `tissue_uberon`, `n_flight`, `n_ground`, `studies`, `design`, `policy_id`, `inclusion_flags_sha256` | client and site IDs, OSD accessions, UBERON terms, policy IDs |
| `policy.check` | a job spec (§6) | the decision record (§5.4) | decision ID, rule IDs, policy IDs |
| `job.submit` | job spec with `policy_decision`; `mode`: `replay` · `live` | `job_id`, `run_id`, `mode`, `status` | job and run IDs |
| `job.status` | `job_id` | `status` (`queued` · `running` · `done` · `failed`), `release_ids`, `started`, `finished`, `log_tail` | job, run and release IDs |
| `results.read` | `release_id`; `select`: `top` (with `sort_by`, `n`) or `genes` | `method`, `stratum`, `k_by_gene`, rows (`ncbi_gene`, `symbol`, `estimate`, `se`, `p`, `q`), `methods_run`, `methods_refused` (method → decision ID), `notes` (the D-029 caveats that apply to this release) | NCBI Gene CURIEs, release ID, decision IDs |
| `kg.query` | `template` (named, from an allow-list) and its parameters | `rows` of nodes and edges, `cypher_sha256`, `kg_version` | node CURIEs, composite edge keys (§1) |

- **`job.submit`** refuses a spec whose `policy_decision` is missing, isn't `allow`, or doesn't
  match the spec's hash. In `replay` mode it returns the recorded release for an identical spec,
  and refuses if none exists.
- **`results.read`** never returns per-site columns from a release made under `secure_sum`.
- **`kg.query`** runs only named templates, not free Cypher. Demo templates:
  `gene_human_orthologs` (MGene → `IS_ORTHOLOG_MGiG` → Gene) and `gene_de_comparisons` (comparisons
  with a DE edge to the gene, with their study and tissue). Calls go to the hosted mcp-genelab
  endpoint and are cached.
- Symbols are display labels and never count as evidence.

## 8. Agent trace format (`agent/`)

**Status: sketched for the demo (WS-10, 2026-11-13).** One JSON Lines file per run,
`$FEDKG_WORKDIR/traces/<trace id>.jsonl`. Every record has `trace_id`, `seq` (0, 1, 2, …), `ts`
(UTC) and `type`.

| `type` | Fields |
|---|---|
| `run_start` | `question`, `policy_profile`, `agent_version`, `model` (open; recorded when chosen), `kg_version`, `tools` (name → version) |
| `tool_call` | `call_id`, `tool`, `input` |
| `tool_result` | the §7 envelope |
| `decision` | `decision_id`, `kind` (`compatibility` · `policy` · `plan` · `schedule`), `choice`, `reason`, `based_on` (call IDs) |
| `report` | `claims[]`: `claim_id`, `text`, `kind` (`result` · `kg_fact` · `policy` · `caveat`), `citations[]` of `{id, call_id}` |
| `grounding` | `checker_version`, per claim `verdict` (`supported` · `unsupported`) and the failing citations, `supported_rate`, `fabricated_ids` |
| `run_end` | `status`, `error` |

**Grounding rules** (the checker reads only this file):
1. Each citation's `id` must be in the `ids` of the `tool_result` with that `call_id`, **in this
   trace**.
2. The citing call must be a tool that can support the claim's kind: `result` → `results.read`;
   `kg_fact` → `kg.query`; `policy` → `policy.check`; `caveat` → `results.read` or `policy.check`.
3. Any ID-shaped token in a claim's text (a CURIE, OSD accession, edge key or project ID) must be
   cited. An uncited ID makes the claim unsupported.
4. `fabricated_ids` lists cited IDs found in no tool result of the trace. The target is zero.

**Example** (records abbreviated):
```json
{"trace_id": "trace:FAKE-0001", "seq": 0, "ts": "2026-11-20T15:00:00Z", "type": "run_start", "question": "Which liver genes change with spaceflight across ISS missions?", "policy_profile": "restricted", "agent_version": "0.0.0-mock", "model": null, "kg_version": "SPOKE-GeneLab v0.3.1+20260311", "tools": {"results.read": "0.1.0", "kg.query": "0.1.0"}}
{"trace_id": "trace:FAKE-0001", "seq": 9, "ts": "2026-11-20T15:03:10Z", "type": "tool_result", "call_id": "trace:FAKE-0001/call/5", "tool": "results.read", "ok": true, "data": {"method": "fe", "rows": [{"ncbi_gene": "FAKE:0001", "symbol": "Fake1", "estimate": -1.2, "q": 1e-6}]}, "ids": ["FAKE:0001", "rel:FAKE-0001-1"]}
{"trace_id": "trace:FAKE-0001", "seq": 11, "ts": "2026-11-20T15:03:40Z", "type": "tool_result", "call_id": "trace:FAKE-0001/call/6", "tool": "kg.query", "ok": true, "data": {"rows": [{"mgene": "FAKE:0001", "gene": "FAKE:9001", "edge": "kgedge:IS_ORTHOLOG_MGiG|FAKE:0001|FAKE:9001"}]}, "ids": ["FAKE:0001", "FAKE:9001", "kgedge:IS_ORTHOLOG_MGiG|FAKE:0001|FAKE:9001"]}
{"trace_id": "trace:FAKE-0001", "seq": 12, "ts": "2026-11-20T15:04:00Z", "type": "report", "claims": [
  {"claim_id": "c1", "kind": "result", "text": "FAKE:0001 is lower in flight across these missions (FE).", "citations": [{"id": "FAKE:0001", "call_id": "trace:FAKE-0001/call/5"}]},
  {"claim_id": "c2", "kind": "kg_fact", "text": "Its human ortholog is FAKE:9001.", "citations": [{"id": "kgedge:IS_ORTHOLOG_MGiG|FAKE:0001|FAKE:9001", "call_id": "trace:FAKE-0001/call/6"}, {"id": "FAKE:9001", "call_id": "trace:FAKE-0001/call/6"}]},
  {"claim_id": "c3", "kind": "kg_fact", "text": "FAKE:0001 is also changed in OSD-FAKE-7.", "citations": [{"id": "OSD-FAKE-7", "call_id": "trace:FAKE-0001/call/6"}]}]}
```
`c1` and `c2` are supported. `c3` is unsupported: `OSD-FAKE-7` isn't in call 6's `ids` (rule 1),
and `FAKE:0001` appears in the text without a citation (rule 3). `OSD-FAKE-7` is then listed in
`fabricated_ids`.
