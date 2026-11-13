"""Census of OSDR studies (phase 1).

Builds flight-vs-control comparisons from OSDR sample metadata (Biological Data API v2), joins them
to the GeneLab KG's comparison IDs, checks animal and library overlap, inspects the published count
matrices and DE tables, and assesses the D-016 gate. Output columns are defined in CONTRACTS.md §2.
Real API calls run locally (WSL) through the cached ``OsdrClient``; tests use synthetic fixtures.

Modules: ``osdr_client`` (HTTP + cache), ``samples`` (tidy samples), ``comparisons`` (groups,
strict rule, KG join), ``overlap``, ``files`` (processed files and headers), ``gate`` (sites and
the gate), ``build`` (the CLI that runs it all). See ``docs/census-liver.md``.
"""
