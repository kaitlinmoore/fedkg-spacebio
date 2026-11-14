"""Paths and inputs shared by the Q-11 clock jobs (WS-09). Outputs go under $FEDKG_WORKDIR."""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase3a_meta.config import annotation_path, workdir  # noqa: E402

from fedkg_spacebio.evaluation import clock_reference as cref  # noqa: E402

SOURCES = {
    "GSE11923": {
        "file": "gse11923/GSE11923_series_matrix.txt.gz",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE11nnn/GSE11923/matrix/"
               "GSE11923_series_matrix.txt.gz",
        "sha256": "2dd9f79ce8562155e003e43984e49b1ed8d190f4ac197f6d612274d21eab39ab",
        "accessed": "2026-11-13",
        "citation": "Hughes ME et al. 2009, PLoS Genet 5:e1000442 (PMID 19343201)",
        "design": "liver, adult male C57BL/6J, entrained LD 12:12 then constant darkness; "
                  "hourly CT18-CT65 (48 arrays, each a pool of 3-5 mice); Affymetrix Mouse "
                  "430 2.0 (GPL1261); MAS5 signal",
    },
    "GSE54651": {
        "file": "gse54651/GSE54651_Liv_quant.txt.gz",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE54nnn/GSE54651/suppl/"
               "GSE54651_Liv_quant.txt.gz",
        "sha256": "70f8e01c9e88a783f3090407f6dc1c64960e3a7904aa6570da3ef844ebe69e0a",
        "accessed": "2026-11-13",
        "citation": "Zhang R et al. 2014, PNAS 111:16219 (PMID 25349387)",
        "design": "liver, C57BL/6, entrained LD 12:12 then constant darkness; every 6 h "
                  "CT22-CT64 (8 samples, no replicates); RNA-seq, STAR (mm9) + HTSeq, "
                  "DESeq2-normalized gene counts, Ensembl gene IDs",
    },
    "GPL1261": {"file": "gpl1261/GPL1261.annot.gz",
                "url": "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL1nnn/GPL1261/annot/"
                       "GPL1261.annot.gz",
                "accessed": "2026-11-13 (WS-08)"},
}


def data_dir() -> Path:
    return workdir() / "data" / "circadian"


def out_dir(*parts: str) -> Path:
    p = workdir() / "results" / "phase3a" / "circadian-phase"
    p = p.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_sources() -> dict:
    out = {}
    for name, s in SOURCES.items():
        path = data_dir() / s["file"]
        digest = sha256(path)
        if "sha256" in s and digest != s["sha256"]:
            raise ValueError(f"{name}: SHA-256 {digest} != pinned {s['sha256']}")
        out[name] = {k: v for k, v in s.items() if k != "file"} | {"sha256": digest}
    return out


def load_array() -> tuple[pd.DataFrame, np.ndarray]:
    values, samples = cref.read_series_matrix(data_dir() / SOURCES["GSE11923"]["file"])
    ct = cref.ct_from_titles(samples.loc[values.columns, "title"])
    return cref.log2_mas5(values), ct


def genelab_annotation() -> pd.DataFrame:
    return pd.read_csv(annotation_path(), sep="\t", dtype=str)


def load_reference() -> tuple[cref.ClockBasis, list[cref.ClockBasis], pd.DataFrame]:
    """The saved basis (peaks GSE11923, amplitudes GSE54651) and its bootstrap replicates."""
    d = out_dir("reference")
    basis = cref.ClockBasis.from_frame(pd.read_csv(d / "basis_rnaseq.csv"),
                                       "peaks GSE11923, amplitudes GSE54651")
    z = np.load(d / "reference_bootstrap.npz")
    boots = [cref.ClockBasis(basis.ncbi_gene, basis.symbol, z["peak"][i], z["amplitude"][i],
                             z["mesor"][i], "bootstrap") for i in range(len(z["peak"]))]
    return basis, boots, pd.read_csv(d / "basis_array.csv")
