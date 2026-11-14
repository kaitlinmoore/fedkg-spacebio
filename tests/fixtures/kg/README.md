# Recorded GeneLab KG responses (`kg.query` fixtures)

These files are the one exception to the synthetic-only rule in `../README.md`: they hold **real, public
rows** from the hosted SPOKE-GeneLab knowledge graph (`mcp-genelab`), recorded on 2026-11-14 so the
`kg.query` tests can run offline. They are small (one gene per file) and are not a KG dump.

- The Cypher for each named template and its `cypher_sha256` were recorded here as `templates.json`
  and now live in `src/fedkg_spacebio/agent/kg_templates.json`, the tool's allow-list (same
  strings, same hashes).
- `<template>__<bare NCBI Gene ID>.json`: one recorded run. `rows` is the parsed `rows:` array that
  the `query` tool returned for `kg_params`; `rows_canonical_sha256` is SHA-256 over the rows with
  keys sorted and no whitespace (ECMAScript number formatting).
- `kg_fingerprint.json`: the graph-level counts used as the version signal, and its SHA-256.

Cases covered:

| Case | Files |
|---|---|
| one human ortholog | `gene_human_orthologs__{12660,21685,70024,18628,12125,13436,13170,18627,381148}` |
| no ortholog edge | `gene_human_orthologs__77059` |
| one-to-many orthologs (19 and 14 rows; some human symbols null or pipe-joined) | `gene_human_orthologs__{99412,319158}` |
| gene ID not in the KG (0 rows) | `*__999999999` |
| non-mouse gene (human, taxon 9606) | `*__7157` |
| strict DE comparisons, including phase 3a input studies | `gene_de_comparisons__{13436,77059,99412,70024,12660}` |
| more rows than the default cap (35) | `gene_de_comparisons__13170` |
| repeated results under three comparison IDs (OSD-462) | `gene_de_comparisons__{77059,99412,12660,13170}` |
| study linked to two missions (OSD-162) | `gene_de_comparisons__13170` |
| mouse gene in the KG with no flight-vs-control DE edge | `gene_de_comparisons__11470` |
| tissue given only as a CL term, no Anatomy edge | `gene_de_comparisons__7157` |

Symbols in these rows are display labels and never evidence. A DE row means the comparison
reported the gene at adj_p < 0.1 (the KG's storage filter); it is not an effect size to pool.
