"""FLARE client for phase 3a: one copy runs per (site, stratum) client.

Client API loop from FLARE 2.9: ``flare.init()``, then
``receive → work → send``.

Federation boundary: the client
- reads only its own manifest and the count file it names (``load_site`` enforces it);
- runs DE locally (pydeseq2);
- maps its per-gene summary to NCBI Gene;
- sends only that summary (``validate_summary`` runs before sending).

It also keeps an exact copy of what it sent in its own output folder (``sent/<client>.npz``).
In a real federation that copy would sit on the site's own disk. Check 1 compares it, bit for
bit, with what the server received.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import nvflare.client as flare

from fedkg_spacebio.federation.meta_flare import SUMMARY_META_KEY
from fedkg_spacebio.federation.site_de import run_de, summarize
from fedkg_spacebio.harmonize.gene_map import load_map
from fedkg_spacebio.sites.loader import load_site


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_dir", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--sent_dir", required=True)
    parser.add_argument("--n_cpus", type=int, default=4)
    parser.add_argument("--no_verify_annotation", action="store_true")  # synthetic tests only
    args = parser.parse_args()

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

        sent = Path(args.sent_dir)
        sent.mkdir(parents=True, exist_ok=True)
        np.savez(sent / f"{client}.npz", **summary.genes)
        (sent / f"{client}.meta.json").write_text(json.dumps(summary.meta, indent=1))

        flare.send(flare.FLModel(
            params=summary.genes,
            params_type=flare.ParamsType.FULL,
            meta={SUMMARY_META_KEY: summary.meta},
            current_round=task.current_round,
        ))


if __name__ == "__main__":
    main()
