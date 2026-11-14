# Synthetic test fixtures

Small, hand-made files that let tests run without real data or network access. They ARE
committed (the data rules in `.gitignore` deliberately exclude this folder).

Rules:
- Synthetic only, except `tests/fixtures/kg/`: small recorded responses from public KGs, with
  source and date (see its README). No real OSDR samples and no knowledge-graph dumps.
- Use obviously fake IDs (e.g., `FAKE:0001`, `OSD-FAKE-1`) so a fixture can never be mistaken
  for, or cited as, real data.
- Cover the hard cases on purpose: mixed gene-ID namespaces, a site with only one arm, a
  one-to-many ortholog, a policy that should deny a job.
- Keep each fixture tiny and describe what it exercises in the test that uses it.
