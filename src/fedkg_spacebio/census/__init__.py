"""Census of OSDR studies (phase 1).

Enumerates rodent RNA-seq studies through the OSDR Biological Data API and keeps those with
both spaceflight and ground-control samples in the same study. Output columns are defined in
CONTRACTS.md. Real API calls run locally (WSL); tests use synthetic fixtures. Cache every call.
"""
