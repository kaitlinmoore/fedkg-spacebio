"""Pre-warm the KG cache for the demo's genes, so the demo can run offline.

    # WSL/Ubuntu, after `fedkg`
    uv run python -m fedkg_spacebio.agent.kg_prewarm                 # top 25 ISS FE hits
    uv run python -m fedkg_spacebio.agent.kg_prewarm --top 50
    uv run python -m fedkg_spacebio.agent.kg_prewarm --genes NCBIGene:13170 NCBIGene:12660
    uv run python -m fedkg_spacebio.agent.kg_prewarm --recordings tests/fixtures/kg

What it does:
1. Picks genes from the phase 3a ISS fixed-effect results table: rows with ``fe_hit``, sorted by
   (``fe_q``, ``ncbi_gene``), the first ``--top``. Genes come from the ``ncbi_gene`` column,
   never from symbols (symbols aren't unique and aren't evidence). ``--genes`` overrides the
   table with explicit NCBI Gene CURIEs.
2. Takes the KG fingerprint live (recorded as current in the cache manifest), then runs both
   templates for each gene through ``kg_query``, so the cache holds exactly what the tool will
   ask for offline.
3. Prints, and writes to ``<cache>/prewarm_report.json``: each call's outcome, cache hit or miss,
   live latency, the cache size, and (with ``--recordings``) any gene whose live rows differ from
   a part 1 recording.

The default ``--top 25`` is generous on purpose: in the offline demo a cache miss is an error,
so the cache should cover more genes than the report will mention. Ranking by ``fe_q`` only
chooses which genes to cache; it isn't a claim about them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from fedkg_spacebio.agent import kg_query as kq
from fedkg_spacebio.agent.kg_cache import KgCache, rows_sha256, utc_now

DEFAULT_RESULTS = Path("results/phase3a/iss/standard/flare/primary/per_gene.csv.gz")


def default_results_path() -> Path:
    root = Path(os.environ.get("FEDKG_WORKDIR", str(Path.home() / "fedkg-work")))
    return root / DEFAULT_RESULTS


def genes_from_results(path: Path, top: int) -> list[str]:
    """The ``top`` FE hits by (``fe_q``, ``ncbi_gene``), as NCBI Gene CURIEs."""
    t = pd.read_csv(path, usecols=["ncbi_gene", "fe_hit", "fe_q"])
    hits = t[t["fe_hit"].astype(bool)].sort_values(["fe_q", "ncbi_gene"], kind="mergesort")
    return [f"NCBIGene:{int(g)}" for g in hits["ncbi_gene"].head(top)]


def compare_with_recordings(cache: KgCache, recordings: Path) -> list[dict[str, Any]]:
    """Cached entries whose (template, gene) has a recording with different rows."""
    recs = {}
    for p in recordings.glob("*__*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        recs[(r["template"], r["kg_params"].get("gene"))] = r
    diffs = []
    for p in cache.entries():
        e = json.loads(p.read_text(encoding="utf-8"))
        r = recs.get((e["template"], e["kg_params"].get("gene")))
        if r is None:
            continue
        a, b = e["rows"], r["rows"]
        if e["template"] == "gene_de_comparisons" and a and b:
            # collect() order isn't guaranteed; compare after the tool's sort.
            a = [{**a[0], "rows": sorted(a[0]["rows"], key=kq._sort_key)}]
            b = [{**b[0], "rows": sorted(b[0]["rows"], key=kq._sort_key)}]
        if rows_sha256(a) != rows_sha256(b):
            diffs.append({"template": e["template"], "gene": e["kg_params"].get("gene"),
                          "cached": p.name, "recorded": r.get("fetched")})
    return diffs


def prewarm(genes: list[str], store: kq.KgStore) -> list[dict[str, Any]]:
    calls = []
    for g in genes:
        for template in kq.agent_templates():
            store.last_elapsed_ms = None
            t0 = time.perf_counter()
            res = kq.kg_query(template, {"gene": g}, store=store)
            calls.append({
                "gene": g,
                "template": template,
                "ok": res["ok"],
                "error": res["error"],
                "cache_hit": (res["cache"] or {}).get("hit"),
                "kg_ms": None if store.last_elapsed_ms is None else round(store.last_elapsed_ms, 1),
                "total_ms": round((time.perf_counter() - t0) * 1000, 1),
                "n_ids": len(res["ids"]),
            })
    return calls


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", type=Path, default=None,
                    help="per-gene results table (default: ISS standard primary FLARE run)")
    ap.add_argument("--top", type=int, default=25, help="number of FE hits to cache (default 25)")
    ap.add_argument("--genes", nargs="+", help="explicit NCBIGene:<n> CURIEs instead of --results")
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--recordings", type=Path, default=None,
                    help="directory of part 1 recordings to compare live rows with")
    args = ap.parse_args(argv)

    if args.genes:
        genes = list(dict.fromkeys(args.genes))
        source = "--genes"
    else:
        path = args.results or default_results_path()
        genes = genes_from_results(path, args.top)
        source = str(path)
    for g in genes:
        kq.gene_to_kg(g)  # fail early on anything that isn't NCBIGene:<n>

    cache = KgCache(args.cache_dir) if args.cache_dir else KgCache()
    t0 = time.perf_counter()
    with kq.KgStore(cache, offline=False) as store:
        fp = store.fingerprint()
        calls = prewarm(genes, store)
        kg_version = store.kg_version()
    report = {
        "run": utc_now(),
        "gene_source": source,
        "top": None if args.genes else args.top,
        "genes": genes,
        "kg_version": kg_version,
        "fingerprint_ms": fp.get("elapsed_ms"),
        "wall_s": round(time.perf_counter() - t0, 2),
        "calls": calls,
        "n_errors": sum(not c["ok"] and c["error"]["code"] not in ("not_found", "wrong_taxon")
                        for c in calls),
        "cache_dir": str(cache.root),
        "cache_entries": len(cache.entries()),
        "cache_bytes": cache.size_bytes(),
        "differs_from_recordings": (
            compare_with_recordings(cache, args.recordings) if args.recordings else None
        ),
    }
    (cache.root / "prewarm_report.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="utf-8")

    live = [c["kg_ms"] for c in calls if c["kg_ms"] is not None]
    print(f"{len(genes)} genes, {len(calls)} calls, {report['n_errors']} errors, "
          f"{sum(c['cache_hit'] is False for c in calls)} live")
    if live:
        live.sort()
        print(f"KG latency ms: min {live[0]}, median {live[len(live) // 2]}, max {live[-1]}")
    print(f"fingerprint {fp['fingerprint_sha256'][:12]} ({report['fingerprint_ms']} ms); "
          f"wall {report['wall_s']} s")
    print(f"cache: {report['cache_entries']} entries, {report['cache_bytes']} bytes "
          f"in {cache.root}")
    if report["differs_from_recordings"]:
        print(f"DIFFERS FROM RECORDINGS: {report['differs_from_recordings']}")
    for c in calls:
        if not c["ok"]:
            print(f"  {c['gene']} {c['template']}: {c['error']['code']}")
    return 1 if report["n_errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
