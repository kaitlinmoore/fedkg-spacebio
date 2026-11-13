"""Q-11 rhythmic-gene check for the phase 3a methods (WS-08).

    uv run python jobs/phase3a_compare/circadian.py

Run after ``compare.py`` (it reads that job's per-gene ``calls.csv.gz``). Inputs, under
``$FEDKG_WORKDIR/data/circadian/`` (downloaded once, checksums written to the output):

- ``hughes2009_plosgen/TableS1_24h_cycling.xls``: Hughes et al. 2009, PLoS Genet 5:e1000442,
  Table S1 (24 h cycling probe sets in mouse liver; CC-BY), from
  https://journals.plos.org/plosgenetics/article/file?type=supplementary&id=10.1371/journal.pgen.1000442.s012
  and its CSV conversion ``TableS1_24h_cycling.csv``. Legacy .xls needs xlrd, which isn't a
  project dependency, so the conversion ran once in a throwaway environment:
  ``uvx --with xlrd --with pandas python -I xls2csv.py <in>.xls <out>.csv``
  (xls2csv.py: ``pd.read_excel(src, header=None)`` on the first sheet, written with
  ``to_csv(index=False, header=False)``).
- ``gpl1261/GPL1261.annot.gz``: NCBI GEO platform annotation for the Affymetrix Mouse Genome
  430 2.0 array (annotation date in the file), from
  https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL1nnn/GPL1261/annot/GPL1261.annot.gz

Writes ``$FEDKG_WORKDIR/results/phase3a/method-comparison/circadian/``:
``rhythmic_mapping.json`` (drop counts, checksums), ``core_clock_ids.csv`` (symbol resolution),
``enrichment.csv``, ``without_rhythmic.csv``, ``core_clock.csv``.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase3a_meta.config import annotation_path, results_dir, workdir  # noqa: E402

from fedkg_spacebio.evaluation import circadian, meta_methods  # noqa: E402

RUNS = [("iss", "primary"), ("iss", "primary_without_SpaceX-8"), ("iss", "min3_sites"),
        ("earth", "primary")]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    data = workdir() / "data" / "circadian"
    s1_xls = data / "hughes2009_plosgen" / "TableS1_24h_cycling.xls"
    s1_csv = data / "hughes2009_plosgen" / "TableS1_24h_cycling.csv"
    annot_p = data / "gpl1261" / "GPL1261.annot.gz"
    comp = workdir() / "results" / "phase3a" / "method-comparison"
    od = comp / "circadian"
    od.mkdir(parents=True, exist_ok=True)

    annot = circadian.read_geo_annot(annot_p)
    pmap, plog = circadian.probe_to_entrez(annot)
    genelab = pd.read_csv(annotation_path(), sep="\t", dtype=str)
    known = {int(x) for v in genelab["ENTREZID"].dropna() for x in str(v).split("|") if x}
    s1 = pd.read_csv(s1_csv)
    rhythmic, rlog = circadian.rhythmic_genes(s1, pmap, annot["ID"], known_ids=known)
    measurable = set(pmap.to_numpy().tolist()) & known
    clock_ids = circadian.resolve_symbols(circadian.CORE_CLOCK, genelab, annot)
    clock_ids.to_csv(od / "core_clock_ids.csv", index=False)
    (od / "rhythmic_mapping.json").write_text(json.dumps({
        "source": "Hughes et al. 2009 PLoS Genet 5:e1000442, Table S1 (24 h cycling, liver)",
        "selection": "COSOPT FDR < 0.05 and Fisher's G FDR < 0.05, COSOPT period 20-30 h",
        "files": {p.name: sha256(p) for p in (s1_xls, s1_csv, annot_p)},
        "platform_annotation_date": next((ln for ln in gzip_head(annot_p)
                                          if ln.startswith("!Annotation_date")), None),
        "platform_mapping": plog, "rhythmic_mapping": rlog,
        "measurable_genes_on_array": len(measurable),
        "core_clock_resolution": clock_ids["status"].value_counts().to_dict(),
    }, indent=1))

    enr, wo, core = [], [], []
    ok_clock = clock_ids[clock_ids["status"] == "ok"]
    for stratum, variant in RUNS:
        calls = pd.read_csv(comp / "real" / stratum / "standard" / variant / "calls.csv.gz")
        genes = calls["ncbi_gene"].to_numpy(dtype=np.int64)
        is_r = np.isin(genes, list(rhythmic))
        is_m = np.isin(genes, list(measurable))
        # Expression strata for the Mantel–Haenszel check: deciles of the mean over sites of
        # log10 base mean, from WS-07's per-gene table (same gene order as calls).
        pg = pd.read_csv(results_dir(stratum, "standard", "flare") / variant / "per_gene.csv.gz")
        if not np.array_equal(pg["ncbi_gene"].to_numpy(dtype=np.int64), genes):
            raise ValueError("per-gene table and calls are not in the same gene order")
        bm = pg[[c for c in pg.columns if c.startswith("base_mean__")]].to_numpy(dtype=float)
        with np.errstate(divide="ignore"):
            lbm = np.nanmean(np.log10(np.where(bm > 0, bm, np.nan)), axis=1)
        dec = pd.qcut(lbm, 10, labels=False).astype(float)
        for m in meta_methods.METHODS:
            hit = calls[f"{m}__hit"].to_numpy(dtype=bool)
            enr.append({"stratum": stratum, "variant": variant, "method": m,
                        **circadian.enrichment(hit, is_r, is_m),
                        **circadian.enrichment_stratified(hit, is_r, is_m, dec)})
            wo.append({"stratum": stratum, "variant": variant, "method": m,
                       **circadian.without_rhythmic(calls, m, is_r)})
        if variant != "primary":
            continue
        table = pd.read_csv(results_dir(stratum, "standard", "flare") / variant
                            / "per_gene.csv.gz").set_index("ncbi_gene")
        site_cols = [c for c in table.columns if c.startswith("log2fc__")]
        for _, r in ok_clock.iterrows():
            g = int(r["ncbi_gene"])
            row = {"stratum": stratum, "symbol": r["symbol"], "ncbi_gene": g,
                   "tested": bool(g in set(genes)), "rhythmic_hughes2009": g in rhythmic}
            if row["tested"]:
                c = calls.loc[calls["ncbi_gene"] == g].iloc[0]
                row["het_flag"] = bool(c["het_flag"])
                for m in meta_methods.METHODS:
                    row[f"{m}__est"] = c[f"{m}__est"]
                    row[f"{m}__q"] = c[f"{m}__q"]
                    row[f"{m}__hit"] = bool(c[f"{m}__hit"])
                for sc in site_cols:
                    row[sc] = table.loc[g, sc]
            core.append(row)

    pd.DataFrame(enr).to_csv(od / "enrichment.csv", index=False, float_format="%.6g")
    pd.DataFrame(wo).to_csv(od / "without_rhythmic.csv", index=False)
    pd.DataFrame(core).to_csv(od / "core_clock.csv", index=False, float_format="%.6g")
    print(json.dumps(rlog), flush=True)
    cols = ["stratum", "method", "hits_measurable", "rhythmic_hits", "share_hits",
            "share_background", "odds_ratio", "p_one_sided", "or_mh", "p_cmh_one_sided"]
    print(pd.DataFrame(enr).query("variant == 'primary'")[cols].to_string(index=False),
          flush=True)
    return 0


def gzip_head(p: Path, n: int = 10) -> list[str]:
    import gzip
    with gzip.open(p, "rt", errors="replace") as fh:
        return [next(fh).rstrip("\n") for _ in range(n)]


if __name__ == "__main__":
    raise SystemExit(main())
