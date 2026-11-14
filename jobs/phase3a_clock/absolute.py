"""Q-11 (B), exploratory (WS-09): absolute clock time per arm, with a stop rule.

    uv run python jobs/phase3a_clock/absolute.py

Run only after the clock-plane method passed validation (``validate.py``).

Stage 1: every site sends **only its ground arm's** aggregates. The server checks each ground
arm's circular mean against the lights-on window, CT −1..13 (ZT0–12 with a 1 h CT/ZT margin).
That rule was written down before any site ran (``validation/interpretation_rules.json``).
- If any ground arm falls outside the window, every ground placement is reported and (B)
  **stops**: no flight-arm values are requested.
- Otherwise, stage 2 requests both arms.

Why the stop rule: (B) assumes the reference's gene baselines (mm9/HTSeq pipeline) match each
site's (GRCm39/RSEM) gene by gene, and that can't be checked directly. Ground controls taken
at night would mean those baselines are off, or that the ground arm was dissected at an
unexpected time. Either way, (B)'s flight values couldn't be trusted.

Writes ``$FEDKG_WORKDIR/results/phase3a/circadian-phase/absolute/``: ``stage1/<client>.json``,
``stage1_check.json`` and, if reached, ``stage2/<client>.json``.
"""

import json

from common import load_reference, out_dir
from phase3a_meta.config import annotation_path, data_root, manifest_dir
from run_sites import CLIENTS

from fedkg_spacebio.federation.site_clock import in_window, site_absolute_summary
from fedkg_spacebio.harmonize.gene_map import load_map
from fedkg_spacebio.sites.loader import load_site


def run_stage(stage: str, basis, gmap) -> dict:
    od = out_dir("absolute", "stage1" if stage == "ground" else "stage2")
    out = {}
    for client in CLIENTS:
        site = load_site(client, manifest_dir("standard"), data_root())
        s = site_absolute_summary(site, gmap, basis, stage)
        del site
        (od / f"{client}.json").write_text(json.dumps(s, indent=1))
        out[client] = s
    return out


def main() -> int:
    basis, _, _ = load_reference()
    gmap = load_map(annotation_path())
    od = out_dir("absolute")
    s1 = run_stage("ground", basis, gmap)
    check = {c: {"ground_circ_mean_ct": s["arms"][0]["circ_mean_ct"],
                 "resultant_length": s["arms"][0]["resultant_length"],
                 "median_template_r": s["arms"][0]["median_template_r"],
                 "median_scale": s["arms"][0]["median_scale"],
                 "in_lights_on_window": in_window(s["arms"][0]["circ_mean_ct"])}
             for c, s in s1.items()}
    passed = all(v["in_lights_on_window"] for v in check.values())
    (od / "stage1_check.json").write_text(json.dumps(
        {"window_ct": [-1, 13], "all_ground_arms_in_window": passed, "sites": check},
        indent=1))
    print(json.dumps(check, indent=1))
    if not passed:
        print("STOP: at least one ground arm is outside the lights-on window; (B) ends here.")
        return 0
    s2 = run_stage("both", basis, gmap)
    for c, s in s2.items():
        print(c, s["arms"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
