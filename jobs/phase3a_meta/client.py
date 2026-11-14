"""FLARE client for phase 3a: one copy runs per (site, stratum) client.

Client API loop from FLARE 2.9: ``flare.init()``, then
``receive → work → send``.

Federation boundary: the client
- reads only its own manifest and the count file it names (``load_site`` enforces it);
- runs DE locally (pydeseq2);
- maps its per-gene summary to NCBI Gene;
- sends only what its mode allows (each mode is validated before sending).

Modes
-----
- **Flag-driven** (``--sent_dir``; the original phase 3a runs): sends the full §4 summary and
  keeps an exact copy in ``sent/<client>.npz``. Check 1 compares it, bit for bit, with what the
  server received.
- **Job spec** (``--spec``, CONTRACTS §6). The client reads the spec, the gene plan and the
  exclusion list (hashes checked; ``spec_run.plan_rows``), and first checks that its own flags
  are the ones the plan was built from (its ``inclusion_flags_sha256`` must equal the spec's
  provenance value). Then:
  - ``per_site_summary`` outputs: sends the §4 summary restricted to the genes the plan assigns
    it, so the server can read only what the policy check counted;
  - ``masked_contribution`` outputs: builds w, w·y (and w·y² when ``fe_het`` is requested) over
    the plan's rows, zeros for unassigned genes, encodes them to fixed point (refusing overflow),
    masks them with the pairwise streams, and sends only the masked vectors plus the unmasked
    site metadata (``secure_sum.py``).

  It keeps two records under ``--sites_dir/<client>/``:
  - ``sent/``: exactly what left the client (masked vectors, or the restricted summary);
  - ``local/``: the full plain summary the site keeps for itself. The server never reads
    ``sites/``; in a real federation this would be the site's own disk.

``--mask_seed`` is the run's root seed for the pairwise masks. It comes from the job config, which
the server process could also read: this simulates the arithmetic of masking, not key exchange
(CONTRACTS §6.3).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import nvflare.client as flare

from fedkg_spacebio.federation import secure_sum as ss
from fedkg_spacebio.federation.gene_plan import INCLUSION_FLAG
from fedkg_spacebio.federation.jobspec import load_job_spec
from fedkg_spacebio.federation.meta_flare import MASKED_META_KEY, SUMMARY_META_KEY
from fedkg_spacebio.federation.site_de import SiteSummary, run_de, summarize, validate_summary
from fedkg_spacebio.federation.spec_run import plan_rows
from fedkg_spacebio.harmonize.gene_map import load_map
from fedkg_spacebio.sites.inclusion import inclusion_flags_sha256
from fedkg_spacebio.sites.loader import load_site


def _save(folder: Path, name: str, arrays: dict, meta: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / f"{name}.npz", **arrays)
    (folder / f"{name}.meta.json").write_text(json.dumps(meta, indent=1))


def restricted_summary(summary: SiteSummary, rows, client: str) -> SiteSummary:
    """The §4 summary holding only the genes the plan assigns this client (plain spec runs)."""
    assigned = set(rows.index[rows[client].to_numpy(dtype=bool)].tolist())
    keep = np.isin(summary.genes["ncbi_gene"], list(assigned))
    if int(keep.sum()) != len(assigned):
        raise ValueError(f"{client}: the plan assigns {len(assigned)} genes, the summary has "
                         f"{int(keep.sum())} of them")
    genes = {k: v[keep] for k, v in summary.genes.items()}
    meta = dict(summary.meta)
    meta["mapping"] = {**meta["mapping"], "n_sent": int(keep.sum()),
                       "n_tested_mean10_sent": int(genes["tested_mean10"].sum())}
    meta["inclusion_flags_sha256"] = inclusion_flags_sha256(genes)  # of the flags actually sent
    validate_summary(genes, meta)
    return SiteSummary(genes, meta)


def spec_payload(summary: SiteSummary, spec, client: str, policy_root: str | None,
                 mask_seed: bytes | None) -> tuple[dict, dict, str]:
    """What a spec-mode client sends: (arrays, metadata, FLARE meta key)."""
    want = spec.provenance.inclusion_flags_sha256.get(client)
    if summary.meta["inclusion_flags_sha256"] != want:
        raise ValueError(f"{client}: this site's inclusion flags (sha256 "
                         f"{summary.meta['inclusion_flags_sha256'][:12]}…) aren't the ones the "
                         "job's gene plan was built from; refusing to send")
    rows, plan_sha, excl_sha = plan_rows(spec, policy_root)
    kinds = {spec.outputs[c] for c in spec.clients}
    if kinds == {"per_site_summary"}:
        s = restricted_summary(summary, rows, client)
        return s.genes, s.meta, SUMMARY_META_KEY
    if kinds != {"masked_contribution"} or spec.aggregation.strategy != "secure_sum":
        raise ValueError("outputs must be all per_site_summary or all masked_contribution "
                         "with secure_sum")
    if mask_seed is None:
        raise ValueError("secure_sum needs --mask_seed")
    params = ss.Params(spec.aggregation.fixed_point_bits, spec.aggregation.modulus_bits)
    quantities = ss.quantities_for(spec.methods)
    flag = INCLUSION_FLAG[spec.inclusion.rule]
    values = ss.contribution_values(summary.genes, rows, client, flag, quantities)
    arrays = ss.masked_contribution(values, client, sorted(spec.clients), root_seed=mask_seed,
                                    job_id=spec.job_id, params=params)
    meta = {**summary.meta, "job_id": spec.job_id, "gene_plan_sha256": plan_sha,
            "genes_exclude_sha256": excl_sha, "n_rows": int(len(rows)),
            **params.to_dict(), "quantities": list(quantities), "mask_scheme": ss.MASK_SCHEME}
    ss.validate_contribution(arrays, meta)
    return arrays, meta, MASKED_META_KEY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_dir", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--sent_dir", help="flag-driven runs: where to keep the copy sent")
    parser.add_argument("--spec", help="job-spec runs: the spec file (CONTRACTS §6)")
    parser.add_argument("--policy_root", help="job-spec runs: root that spec refs resolve against")
    parser.add_argument("--sites_dir", help="job-spec runs: <client>/sent and <client>/local")
    parser.add_argument("--mask_seed", help="secure_sum: the run's root seed (hex)")
    parser.add_argument("--n_cpus", type=int, default=4)
    parser.add_argument("--no_verify_annotation", action="store_true")  # synthetic tests only
    args = parser.parse_args()
    if bool(args.spec) == bool(args.sent_dir):
        parser.error("give exactly one of --sent_dir (flag-driven) or --spec (job spec)")
    spec = load_job_spec(args.spec) if args.spec else None
    seed = bytes.fromhex(args.mask_seed) if args.mask_seed else None

    flare.init()
    client = flare.get_site_name()
    while flare.is_running():
        task = flare.receive()
        site = load_site(client, Path(args.manifest_dir), Path(args.data_root))
        print(f"{client}: {site.provenance['n_flight']} flight / "
              f"{site.provenance['n_ground']} ground, {site.provenance['n_genes']} genes")
        res = run_de(site.counts, site.metadata, site.manifest["design"], n_cpus=args.n_cpus)
        gene_map = load_map(Path(args.annotation), verify=not args.no_verify_annotation)
        summary = summarize(site, res, gene_map)

        if spec is None:
            arrays, meta, key = summary.genes, summary.meta, SUMMARY_META_KEY
            _save(Path(args.sent_dir), client, arrays, meta)
        else:
            mine = Path(args.sites_dir) / client
            _save(mine / "local", "summary", summary.genes, summary.meta)  # stays at the site
            arrays, meta, key = spec_payload(summary, spec, client, args.policy_root, seed)
            _save(mine / "sent", client, arrays, meta)                     # exactly what leaves

        flare.send(flare.FLModel(
            params=arrays,
            params_type=flare.ParamsType.FULL,
            meta={key: meta},
            current_round=task.current_round,
        ))


if __name__ == "__main__":
    main()
