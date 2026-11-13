# Contracts

The interfaces every part of the system agrees on. Defined **before** real data arrives, with
mocks, so census, harmonization, federation, policy and agent work can proceed independently.
Change a contract here first, then the code and fixtures that depend on it.

> **Status:** skeleton. Each section lists what it must define; the fields are TODO.

## 1. Identifiers
- All biological and ontology IDs are CURIEs: `UBERON:`, `CL:`, `MONDO:`, `HP:`, `NCBIGene:`,
  `ENSEMBL:`. Studies use OSD accessions (`OSD-NNN`).
- Mocks and fixtures use obviously fake IDs: `FAKE:NNNN`, `OSD-FAKE-N`.
- TODO: site IDs, job IDs, trace IDs, run IDs (format and who mints them).

## 2. Census table (`census/`)
TODO: columns, types and allowed values. At minimum: OSD accession, mission, agency, organism,
tissue (UBERON), strain, sex, n flight, n ground control, platform, gene-ID namespace,
annotation version, access date.

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
