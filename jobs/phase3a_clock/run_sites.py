"""Q-11 site runs (WS-09): each client computes its clock summary from its own data only.

    uv run python jobs/phase3a_clock/run_sites.py

Run after ``reference.py`` and ``validate.py``. In-process, not a FLARE job (WS-09 plan §4):
one round, and the server only tabulates.

The boundary is kept in the code:
- each client's data comes only through ``load_site`` (its own manifest and files);
- the site function returns a summary that ``validate_clock_summary`` checks against exact
  whitelists;
- only that summary is written to ``summaries/<client>.json``, which the server step
  (``report.py``) reads.

Uses the WS-07 standard manifests (``$FEDKG_WORKDIR/phase3a/manifests/standard``).
"""

import json
import sys

from common import load_reference, out_dir
from phase3a_meta.config import annotation_path, data_root, manifest_dir

from fedkg_spacebio.federation.site_clock import site_clock_summary
from fedkg_spacebio.harmonize.gene_map import load_map
from fedkg_spacebio.sites.loader import load_site

CLIENTS = ["SpaceX-4", "SpaceX-8", "SpaceX-13-ISST", "SpaceX-16-ISST",
           "SpaceX-13-LAR", "SpaceX-16-LAR", "SpaceX-21"]
SEED = 20261113


def main() -> int:
    basis, boots, _ = load_reference()
    gmap = load_map(annotation_path())
    od = out_dir("summaries")
    for i, client in enumerate(CLIENTS):
        site = load_site(client, manifest_dir("standard"), data_root())  # site-side from here
        summary = site_clock_summary(site, gmap, basis, boots, seed=SEED + i)
        del site  # nothing but the validated summary leaves this loop body
        (od / f"{client}.json").write_text(json.dumps(summary, indent=1))
        r = summary["result"]
        print(f"{client:15s} k={r['k']:2d} L={r['L']:.3f} delta={r['delta_h']} "
              f"ci={r['delta_h_ci']} p={r['p_perm_L']:.4f} R2={r['r2_plane']:.3f} "
              f"noise={r['noise_sd']:.3f} {r['identifiability']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
