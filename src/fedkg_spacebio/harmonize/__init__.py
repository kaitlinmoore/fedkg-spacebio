"""Local harmonization (phase 2). OWNED CORE.

Per site: map metadata to shared ontologies (CL, UBERON, Mondo, HPO), harmonize gene IDs to a
single namespace (prefer Ensembl), and map rodent genes to human orthologs. Every mapping step
logs its drop rate. Never join on gene symbols across datasets. IDs are CURIEs.
"""
