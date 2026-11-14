"""Q-11 reference fit (WS-09): peak times from GSE11923, amplitudes from GSE54651, the panel.

    uv run python jobs/phase3a_clock/reference.py

Inputs (under ``$FEDKG_WORKDIR/data/circadian/``, checksums pinned in ``common.SOURCES``):
GSE11923 series matrix, GSE54651 liver counts, GPL1261 platform annotation (from WS-08).

Writes ``$FEDKG_WORKDIR/results/phase3a/circadian-phase/reference/``:
- ``array_cosinor_panel.csv``: the cosinor for every probe set of every candidate gene;
- ``rnaseq_cosinor_panel.csv``: the GSE54651 cosinor for the candidates;
- ``panel_decisions.csv``: every candidate, kept or dropped, with its reason;
- ``basis_rnaseq.csv``, ``basis_array.csv``: the clock plane (peak, amplitude, mesor);
- ``reference_bootstrap.npz``: 2,000 residual-bootstrap replicates of the basis;
- ``reference.json``: sources, design, mapping drop log, rule, panel.
"""

import json

import numpy as np
from common import annotation_path, check_sources, data_dir, genelab_annotation, load_array, out_dir

from fedkg_spacebio.evaluation import circadian
from fedkg_spacebio.evaluation import clock_reference as cref
from fedkg_spacebio.harmonize.gene_map import load_map, map_summary

N_BOOT = 2000
SEED = 20261113


def main() -> int:
    od = out_dir("reference")
    sources = check_sources()

    # Array: log2 MAS5, cosinor on every probe set (BH needs them all), probe → NCBI Gene.
    arr, ct_a = load_array()
    afit = cref.array_fit_with_q(arr, ct_a)
    annot = circadian.read_geo_annot(data_dir() / "gpl1261" / "GPL1261.annot.gz")
    pmap, plog = circadian.probe_to_entrez(annot)
    genelab = genelab_annotation()
    ids = circadian.resolve_symbols(cref.PANEL, genelab, annot)

    # RNA-seq: Ensembl (mm9 era) → NCBI Gene with the pinned GeneLab table, 1:1 only.
    rq, ct_r = cref.read_zhang_quant(data_dir() / "gse54651" / "GSE54651_Liv_quant.txt.gz")
    gmap = load_map(annotation_path())
    mapped, rlog = map_summary(rq.assign(tested=True), gmap, tested_col=None)
    rlog_all = rlog["all_genes"]
    rnaseq = cref.log2_counts(mapped[rq.columns])
    rfit = cref.cosinor(rnaseq.to_numpy(), ct_r)
    rfit.index = rnaseq.index

    kept, decisions = cref.select_panel(afit, pmap, ids, rfit)
    decisions.to_csv(od / "panel_decisions.csv", index=False)
    cand = set(ids["ncbi_gene"].dropna().astype(int))
    afit[afit.index.isin(pmap.index[pmap.isin(cand)])].assign(
        ncbi_gene=lambda f: pmap.reindex(f.index).to_numpy()).to_csv(
        od / "array_cosinor_panel.csv")
    rfit[rfit.index.isin(cand)].to_csv(od / "rnaseq_cosinor_panel.csv")

    basis = cref.basis_from_panel(kept, "rnaseq")
    basis.to_frame().to_csv(od / "basis_rnaseq.csv", index=False)
    cref.basis_from_panel(kept, "array").to_frame().to_csv(od / "basis_array.csv", index=False)

    boots = cref.bootstrap_bases(kept, arr, ct_a, rnaseq, ct_r, N_BOOT, SEED)
    np.savez_compressed(od / "reference_bootstrap.npz",
                        peak=np.stack([b.peak for b in boots]),
                        amplitude=np.stack([b.amplitude for b in boots]),
                        mesor=np.stack([b.mesor for b in boots]))
    amp_ratio = kept["amplitude_rnaseq"] / kept["amplitude_array"]

    (od / "reference.json").write_text(json.dumps({
        "sources": sources,
        "time_frame": "circadian time (CT) of the reference series (constant darkness); sites "
                      "ran LD 12:12 (ZT). CT ~ ZT within ~1 h over the first days in darkness; "
                      "not corrected.",
        "array": {"n_arrays": int(arr.shape[1]), "ct_range": [float(ct_a.min()),
                                                             float(ct_a.max())],
                  "transform": "log2(MAS5 signal), floored at 1",
                  "platform_mapping": plog},
        "rnaseq": {"n_samples": int(rq.shape[1]), "ct": ct_r.tolist(),
                   "transform": "log2(DESeq2-normalized count + 1)",
                   "ensembl_to_ncbi": rlog_all,
                   "ensembl_to_ncbi_drop_rate": round(
                       1 - rlog_all["one_to_one"] / max(sum(rlog_all.values()), 1), 4)},
        "rule": cref.PANEL_RULE,
        "panel_symbol_resolution": ids.to_dict(orient="records"),
        "panel_kept": kept["symbol"].tolist(),
        "panel_dropped": decisions.loc[~decisions["kept"], ["symbol", "reason"]
                                       ].to_dict(orient="records"),
        "amplitude_ratio_rnaseq_over_array": {
            "median": float(amp_ratio.median()), "min": float(amp_ratio.min()),
            "max": float(amp_ratio.max())},
        "bootstrap": {"n": N_BOOT, "seed": SEED,
                      "method": "residual bootstrap of both cosinor fits per gene"},
    }, indent=1, default=str))
    print(decisions[["symbol", "kept", "reason", "array_r2", "array_peak_to_trough_fold",
                     "array_peak"]].to_string())
    print(kept[["symbol", "peak", "amplitude_array", "amplitude_rnaseq", "peak_rnaseq",
                "p_rnaseq"]].round(3).to_string())
    print("RNA-seq/array amplitude ratio:", amp_ratio.round(2).tolist())
    print("Ensembl→NCBI (GSE54651):", rlog_all)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
