"""Build the liver census from OSDR (+ one cached KG query). WS-06.

Run (WSL/Ubuntu, after ``fedkg``)::

    uv run python -m fedkg_spacebio.census.build

Inputs:
- OSDR, through ``OsdrClient`` (cached under ``$FEDKG_WORKDIR/cache/osdr/``).
- The KG comparisons for the candidate studies, from one read-only GeneLab-KG query saved to
  ``$FEDKG_WORKDIR/cache/kg/liver_comparisons.json`` (query text: ``KG_QUERY`` below).
- ``liver_curation.yaml`` (flight euthanasia location where OSDR doesn't record it as a factor).

Outputs, all under ``$FEDKG_WORKDIR/census/`` (never in the repo):
``liver_census.csv``, ``liver_samples.csv``, ``liver_overlap_matrix.csv``,
``liver_overlap_pairs.csv``, ``liver_within_study_duplicates.csv``,
``liver_library_matches.csv``, ``liver_library_best.csv``, ``liver_kg_unmatched.csv``,
``liver_studies.csv``, ``liver_gate.csv``, ``liver_gate_sensitivity.csv``,
``liver_gate_strata.csv``, ``liver_overlap_name_collisions.csv``, ``liver_summary.json``.
Count matrices and small DE-side tables go to ``$FEDKG_WORKDIR/data/osdr/<OSD>/``.
"""

from __future__ import annotations

import argparse
import io
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from fedkg_spacebio.census import comparisons as cmp
from fedkg_spacebio.census import files as fl
from fedkg_spacebio.census import gate as gt
from fedkg_spacebio.census import overlap as ov
from fedkg_spacebio.census.osdr_client import OsdrClient, default_workdir
from fedkg_spacebio.census.samples import SAMPLE_FIELDS, parse_samples

LIVER_CANDIDATES = ["OSD-47", "OSD-48", "OSD-137", "OSD-168", "OSD-173", "OSD-245", "OSD-379",
                    "OSD-463", "OSD-686"]
LIVER_UBERON = ["UBERON:0002107", "UBERON:0006727"]
KG_VERSION = "SPOKE-GeneLab v0.3.1+20260311 (assumed; not exposed live)"
HEAD_BYTES = 65536
PROVENANCE_BYTES = 2_000_000  # partial read of OSD-47's two DE tables for the KG provenance check

# The one read-only KG query (GeneLab KG MCP `query` tool, 2026-11-12), with
# params {"studies": LIVER_CANDIDATES, "liver": LIVER_UBERON}.
KG_QUERY = """\
MATCH (s:Study)-[:PERFORMED_SpAS]->(a:Assay)
WHERE s.identifier IN $studies
  AND a.technology = 'RNA Sequencing (RNA-Seq)'
  AND a.factor_space_1 = 'Space Flight'
  AND a.material_id_1 IN $liver AND a.material_id_2 IN $liver
OPTIONAL MATCH (m:Mission)-[:CONDUCTED_MIcS]->(s)
WITH s, a, collect(DISTINCT m.identifier) AS mission_ids, collect(DISTINCT m.space_program) AS space_program
CALL {
  WITH s, a
  OPTIONAL MATCH (a)-[r:MEASURED_DIFFERENTIAL_EXPRESSION_ASmMG]->(g:MGene)
  RETURN count(r) AS n_de_edges_kg, max(r.adj_p_value) AS max_adj_p,
         collect(CASE WHEN s.identifier = 'OSD-47' AND a.factor_space_2 = 'Ground Control'
                      THEN [g.identifier, toString(r.log2fc), toString(r.adj_p_value)] END) AS osd47_edges
}
RETURN s.identifier AS osd_id, s.taxonomy AS taxonomy, mission_ids, space_program,
       a.identifier AS comparison_id, a.name AS osdr_assay_name,
       a.material_id_1 AS material_id_1, a.material_id_2 AS material_id_2,
       a.material_name_1 AS material_name_1,
       a.factor_space_1 AS factor_space_1, a.factor_space_2 AS factor_space_2,
       a.factors_1 AS factors_1, a.factors_2 AS factors_2,
       a.differential_analysis_method AS de_method,
       n_de_edges_kg, max_adj_p, osd47_edges
ORDER BY osd_id, comparison_id
"""  # noqa: E501  (kept verbatim: this is the exact query text that was run)

LOCATION_BY_FACTOR = {"On ISS": "ISS", "On Earth": "Earth"}


# ------------------------------------------------------------------ euthanasia location

def load_curation(path: Path | None = None) -> dict:
    path = path or Path(__file__).with_name("liver_curation.yaml")
    return yaml.safe_load(path.read_text())


def euthanasia_location(factors: dict, arm_is_flight: bool, osd: str, curation: dict) -> tuple:
    """(location, source) for one side of a comparison: ``ISS`` / ``Earth`` / None."""
    if "euthanasia location" in factors:
        return LOCATION_BY_FACTOR.get(factors["euthanasia location"]), "OSDR factor"
    if not arm_is_flight:
        return "Earth", "control arm"
    cur = curation.get(osd)
    if cur:
        return cur["flight_euthanasia_location"], f"curated ({cur['confidence']})"
    return None, "unknown"


def _duration_days(label: str | None) -> float | None:
    """``"22 to 23 day"`` → 22.5, ``"~60 day"`` → 60, ``"24 day"`` → 24; else None."""
    if not label or "day" not in label:
        return None
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", label)]
    return sum(nums) / len(nums) if nums else None


def location_aware_match(row: pd.Series, max_day_gap: float = 3.0) -> bool:
    """The location-aware flight contrast (D-020; amends D-015 for the ISS/Earth strata).

    Written to the census as ``location_aware_match`` (was ``strict_location_aware`` in WS-06,
    when the rule was still a proposal). ``strict_match`` keeps its D-015 meaning.

    True if the pair is strict, or the only differences are those an ISS-terminal design forces:
    flight euthanized ``On ISS`` vs control ``On Earth``, and durations within ``max_day_gap``
    days (ground controls run on a few days' offset). This treats OSD-245/379 ISS-T arms the way
    RR-1/RR-3 are already treated, where location isn't a recorded factor.
    """
    if row["strict_match"]:
        return True
    diff = set(row["differing_factors"])
    if not diff or not diff <= {"euthanasia location", "duration"}:
        return False
    f, c = row["flight_factors"], row["control_factors"]
    if "euthanasia location" in diff and not (
        f.get("euthanasia location") == "On ISS" and c.get("euthanasia location") == "On Earth"
    ):
        return False
    if "duration" in diff:
        df, dc = _duration_days(f.get("duration")), _duration_days(c.get("duration"))
        if df is None or dc is None or abs(df - dc) > max_day_gap:
            return False
    return True


# ------------------------------------------------------------------ per-study files

def read_head(client: OsdrClient, osd: str, file_name: str, start: int = HEAD_BYTES,
              limit: int = 8 * HEAD_BYTES * 16) -> bytes:
    """First bytes of a file, enough for the header line plus some rows.

    Starts at 64 KB and quadruples until the header line is complete *and* at least a few rows
    follow (OSD-379's DE header alone is longer than 64 KB: 16 groups give 240 contrasts).
    """
    n = start
    while True:
        head = client.head_bytes(osd, file_name, n)
        if head.count(b"\n") >= 20 or n >= limit or len(head) < n:
            return head
        n *= 4


def inspect_study_files(client: OsdrClient, osd: str) -> dict:
    """Download counts and DE-side tables, read DE/normalized headers. Returns a study record."""
    listing = client.file_listing(osd)
    files = fl.select_rnaseq_files(listing)
    rec: dict = {"osd_id": osd, "files": files}

    for role in ("counts_rsem", "counts_rsem_rrnarm", "sample_table", "contrasts", "runsheet"):
        if role in files:
            f = files[role]
            rec[f"path_{role}"] = client.download(osd, f["file_name"], f.get("file_size"))

    for role in ("de", "de_rrnarm", "normalized", "normalized_rrnarm"):
        if role in files:
            rec[f"head_{role}"] = read_head(client, osd, files[role]["file_name"])

    # Namespace + gene counts from the full RSEM matrices.
    for role in ("counts_rsem", "counts_rsem_rrnarm"):
        if f"path_{role}" in rec:
            m = pd.read_csv(rec[f"path_{role}"], index_col=0)
            rec[f"matrix_{role}"] = m
            rec[f"namespace_{role}"] = fl.detect_namespace(m.index.astype(str))
            rec[f"n_genes_{role}"] = len(m)
            rec[f"n_samples_{role}"] = m.shape[1]
            # RSEM writes *expected* counts, which can be fractional; DESeq2 needs integers.
            rec[f"frac_noninteger_{role}"] = float((m.to_numpy() % 1 != 0).mean())

    if "head_de" in rec:
        rec["de_info"] = fl.de_header_info(fl.header_columns(rec["head_de"]))
    if "head_de_rrnarm" in rec:
        rec["de_rrnarm_info"] = fl.de_header_info(fl.header_columns(rec["head_de_rrnarm"]))

    # Which normalized-count file does each DE table's per-sample block reproduce?
    rec["pairing"] = {}
    for de_role in ("de", "de_rrnarm"):
        if f"head_{de_role}" not in rec:
            continue
        de = fl.parse_partial_csv(rec[f"head_{de_role}"])
        for norm_role in ("normalized", "normalized_rrnarm"):
            if f"head_{norm_role}" not in rec:
                continue
            norm = fl.parse_partial_csv(rec[f"head_{norm_role}"])
            norm = norm.rename(columns={norm.columns[0]: "ENSEMBL"})
            sample_cols = [c for c in norm.columns[1:] if c in de.columns]
            rec["pairing"][f"{de_role}~{norm_role}"] = fl.compare_partial_tables(
                de, norm, "ENSEMBL", sample_cols)
    return rec


def kg_provenance_osd47(client: OsdrClient, study: dict, kg_rows: list[dict]) -> dict:
    """Which of OSD-47's two DE tables does the KG's flight→ground comparison mirror?

    Reads the first ``PROVENANCE_BYTES`` of each DE table (a partial read, not a download) and
    compares log2FC for the KG's stored genes that fall in that range, joined on ENTREZID.
    """
    edges = next((r["osd47_edges"] for r in kg_rows if r["osd_id"] == "OSD-47"
                  and r["factor_space_2"] == "Ground Control" and r["osd47_edges"]), [])
    kg = {str(g): float(l2) for g, l2, _ in edges}
    out: dict = {"n_kg_edges": len(kg)}
    for role in ("de", "de_rrnarm"):
        name = study["files"][role]["file_name"]
        part = fl.parse_partial_csv(client.head_bytes("OSD-47", name, PROVENANCE_BYTES))
        col = "Log2fc_(Space Flight)v(Ground Control)"
        hits = part[part["ENTREZID"].isin(kg.keys())]
        diffs = [abs(float(v) - kg[e]) for e, v in zip(hits["ENTREZID"], hits[col], strict=True)
                 if v not in ("", "NA")]
        out[role] = {"rows_read": len(part), "n_shared_genes": len(diffs),
                     "max_abs_log2fc_diff": max(diffs) if diffs else None}
    return out


# ------------------------------------------------------------------ main build

def build(workdir: Path | None = None, offline: bool = False) -> dict:
    work = workdir or default_workdir()
    out_dir = work / "census"
    out_dir.mkdir(parents=True, exist_ok=True)
    client = OsdrClient(cache_dir=work / "cache" / "osdr", data_dir=work / "data" / "osdr",
                        offline=offline)
    kg_rows = json.loads((work / "cache" / "kg" / "liver_comparisons.json").read_text())
    curation = load_curation()
    access_date = date.today().isoformat()

    # 1. Samples (one bulk call) → groups → comparisons → KG join.
    samples = parse_samples(client.query_samples(LIVER_CANDIDATES, SAMPLE_FIELDS))
    not_liver = samples[~samples["material_type"].fillna("").str.contains("liver", case=False)]
    groups = cmp.build_groups(samples)
    comps = cmp.build_comparisons(groups)
    comps, kg_unmatched = cmp.match_kg(comps, kg_rows)

    # 2. Study-level metadata: missions from the KG (cross-checked with OSDR), processing text.
    study_kg = {}
    for r in kg_rows:
        study_kg.setdefault(r["osd_id"], r)
    studies = []
    for osd in LIVER_CANDIDATES:
        meta = client.dataset(osd)["metadata"]
        mission = meta.get("mission", {}).get("name")
        proc = fl.processing_info(meta.get("study protocol description") or [])
        kg = study_kg.get(osd, {})
        studies.append({
            "osd_id": osd,
            "glds_ids": [i for i in (meta.get("identifiers") or []) if str(i).startswith("GLDS")]
            if isinstance(meta.get("identifiers"), list) else [meta.get("identifiers")],
            "project_identifier": meta.get("project identifier"),
            "osdr_mission": mission if isinstance(mission, list) else [mission],
            "mission_ids": sorted(kg.get("mission_ids", [])),
            "space_program": sorted({p.strip() for s in kg.get("space_program", []) if s
                                     for p in s.split(",")}),
            "organism_taxon": f"NCBITaxon:{kg['taxonomy']}" if kg.get("taxonomy") else None,
            **proc,
        })
    studies_df = pd.DataFrame(studies).set_index("osd_id")

    # 3. Files: counts, DE headers, pairing, Stat_, n in DE.
    file_recs = {osd: inspect_study_files(client, osd) for osd in LIVER_CANDIDATES}
    provenance = kg_provenance_osd47(client, file_recs["OSD-47"], kg_rows)

    # 4. Overlap: animals (metadata) and libraries (counts).
    matrix, pairs, collisions = ov.animal_overlap(samples)
    dups = ov.within_study_duplicates(samples)
    lib_matches, lib_best = ov.library_overlap(
        {osd: r["matrix_counts_rsem"] for osd, r in file_recs.items() if "matrix_counts_rsem" in r})

    # 5. Census columns per comparison.
    def per_row(row: pd.Series) -> dict:
        osd = row["osd_id"]
        st, fr = studies_df.loc[osd], file_recs[osd]
        loc_f, src_f = euthanasia_location(row["flight_factors"], True, osd, curation)
        loc_c, src_c = euthanasia_location(row["control_factors"], False, osd, curation)
        res: dict = {
            "mission_ids": st["mission_ids"],
            "space_program": st["space_program"],
            "organism_taxon": st["organism_taxon"],
            "organ_group": "liver",
            "technology": "RNA Sequencing (RNA-Seq)",
            "euthanasia_location_flight": loc_f,
            "euthanasia_location_flight_source": src_f,
            "euthanasia_location_control": loc_c,
            "counts_file": fr["files"].get("counts_rsem", {}).get("file_name"),
            "counts_file_rrnarm": fr["files"].get("counts_rsem_rrnarm", {}).get("file_name"),
            "de_file": fr["files"].get("de", {}).get("file_name"),
            "gene_id_namespace": fr.get("namespace_counts_rsem", {}).get("namespace"),
            "annotation_version": (f"{st['genome_build']}, Ensembl {st['ensembl_release']}"
                                   if st["genome_build"] else None),
            "pipeline_version": st["pipeline_version"],
            "nf_rcp_version": st["nf_rcp_version"],
            "kg_version": KG_VERSION,
            "access_date": access_date,
        }
        if "path_sample_table" in fr and "path_contrasts" in fr:
            ct = fl.contrast_for(row["flight_samples"], row["control_samples"],
                                 pd.read_csv(fr["path_sample_table"]),
                                 pd.read_csv(fr["path_contrasts"]))
            res.update({k: ct[k] for k in ("n_flight_in_de", "n_control_in_de")})
            res["genelab_contrast"] = ct["contrast"]
            res["has_stat"] = (ct["contrast"] in fr["de_info"]["stat_contrasts"]
                               if ct["contrast"] and "de_info" in fr else None)
        # Fall back to a study-level tissue term when this pair has no unique KG match.
        if row["tissue_uberon"] is None:
            same = [r for r in kg_rows if r["osd_id"] == osd]
            if same and len({r["material_id_1"] for r in same}) == 1:
                res["tissue_uberon"] = same[0]["material_id_1"]
                res["tissue_label"] = same[0]["material_name_1"]
        return res

    extra = pd.DataFrame([per_row(r) for _, r in comps.iterrows()], index=comps.index)
    for c in ("tissue_uberon", "tissue_label"):
        if c in extra:
            comps[c] = comps[c].where(comps[c].notna(), extra.pop(c))
    census = pd.concat([comps, extra], axis=1)
    census["location_aware_match"] = census.apply(location_aware_match, axis=1)

    # 6. Sites, independent animals, gate (+ sensitivity).
    sites = gt.assign_sites({osd: studies_df.loc[osd, "mission_ids"] for osd in LIVER_CANDIDATES})
    census["site"] = census["osd_id"].map(sites)
    census = gt.independent_counts(census, pairs, sites)
    counts_available = {osd: "counts_rsem" in r["files"] for osd, r in file_recs.items()}
    gate_table, verdict = gt.assess_gate(census, sites, counts_available)
    sens = gt.sensitivity(census, sites, counts_available,
                          strict_columns=("strict_match", "location_aware_match"))
    strata_table = gt.strata(census, sites, counts_available,
                             strict_columns=("strict_match", "location_aware_match"))

    # 7. Study table for the docs.
    study_rows = []
    for osd in LIVER_CANDIDATES:
        st, fr = studies_df.loc[osd], file_recs[osd]
        ns = fr.get("namespace_counts_rsem", {})
        study_rows.append({
            "osd_id": osd, "site": sites[osd], "mission_ids": st["mission_ids"],
            "osdr_mission": st["osdr_mission"], "project": st["project_identifier"],
            "glds_ids": st["glds_ids"],
            "pipeline_version": st["pipeline_version"], "nf_rcp_version": st["nf_rcp_version"],
            "ensembl_release": st["ensembl_release"], "genome_build": st["genome_build"],
            "gtf": st["gtf"], "protocol_conflict": st["conflict"],
            "counts_file": fr["files"].get("counts_rsem", {}).get("file_name"),
            "counts_bytes": fr["files"].get("counts_rsem", {}).get("file_size"),
            "counts_rrnarm_file": fr["files"].get("counts_rsem_rrnarm", {}).get("file_name"),
            "counts_rrnarm_bytes": fr["files"].get("counts_rsem_rrnarm", {}).get("file_size"),
            "counts_frac_noninteger": fr.get("frac_noninteger_counts_rsem"),
            "n_genes": fr.get("n_genes_counts_rsem"),
            "n_genes_rrnarm": fr.get("n_genes_counts_rsem_rrnarm"),
            "n_samples_in_counts": fr.get("n_samples_counts_rsem"),
            "namespace": ns.get("namespace"), "n_ercc_rows": ns.get("n_ercc"),
            "ensembl_versioned": ns.get("versioned"),
            "de_file": fr["files"].get("de", {}).get("file_name"),
            "de_bytes": fr["files"].get("de", {}).get("file_size"),
            "de_has_stat": fr.get("de_info", {}).get("has_any_stat"),
            "de_n_contrasts": len(fr.get("de_info", {}).get("contrasts", [])),
            "de_rrnarm_has_stat": fr.get("de_rrnarm_info", {}).get("has_any_stat"),
            "de_id_columns": fr.get("de_info", {}).get("id_columns"),
            "pairing": fr["pairing"],
            "files_date_created": fr["files"].get("de", {}).get("date_created"),
        })
    study_table = pd.DataFrame(study_rows)

    # 8. Write.
    list_cols = ["mission_ids", "space_program", "factors_flight", "factors_control",
                 "differing_factors", "factors_flight_kg", "factors_control_kg"]
    census_out = census.drop(columns=["flight_samples", "control_samples", "flight_factors",
                                      "control_factors"])
    for c in list_cols + ["flight_animals", "control_animals"]:
        census_out[c] = census_out[c].map(lambda v: json.dumps(v) if isinstance(v, list) else v)
    census_out.to_csv(out_dir / "liver_census.csv", index=False)
    s_out = samples.drop(columns=["factors", "group_key"])
    s_out.to_csv(out_dir / "liver_samples.csv", index=False)
    matrix.to_csv(out_dir / "liver_overlap_matrix.csv")
    pairs.to_csv(out_dir / "liver_overlap_pairs.csv", index=False)
    collisions.to_csv(out_dir / "liver_overlap_name_collisions.csv", index=False)
    dups.to_csv(out_dir / "liver_within_study_duplicates.csv", index=False)
    lib_matches.to_csv(out_dir / "liver_library_matches.csv", index=False)
    lib_best.to_csv(out_dir / "liver_library_best.csv", index=False)
    pd.DataFrame(kg_unmatched).to_csv(out_dir / "liver_kg_unmatched.csv", index=False)
    study_table.to_csv(out_dir / "liver_studies.csv", index=False)
    gate_table.to_csv(out_dir / "liver_gate.csv", index=False)
    sens.to_csv(out_dir / "liver_gate_sensitivity.csv", index=False)
    strata_table.to_csv(out_dir / "liver_gate_strata.csv", index=False)
    summary = {
        "access_date": access_date,
        "n_samples": len(samples),
        "n_samples_not_liver": len(not_liver),
        "n_comparisons": len(census),
        "n_strict": int(census["strict_match"].sum()),
        "strict_studies": sorted(census.loc[census["strict_match"], "osd_id"].unique().tolist()),
        "strict_ground_studies": sorted(census.loc[census["strict_match"] & (
            census["control_type"] == "Ground Control"), "osd_id"].unique().tolist()),
        "kg_match_counts": census["kg_match"].value_counts().to_dict(),
        "n_kg_unmatched": len(kg_unmatched),
        "gate": verdict,
        "kg_provenance_osd47": provenance,
        "live_requests": client.live_requests,
    }
    (out_dir / "liver_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return {"summary": summary, "census": census, "gate": gate_table, "sensitivity": sens,
            "strata": strata_table,
            "studies": study_table, "matrix": matrix, "pairs": pairs, "dups": dups,
            "lib_matches": lib_matches, "lib_best": lib_best, "kg_unmatched": kg_unmatched,
            "collisions": collisions}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--offline", action="store_true", help="fail on cache misses (no network)")
    args = ap.parse_args(argv)
    res = build(offline=args.offline)
    buf = io.StringIO()
    json.dump(res["summary"], buf, indent=1, default=str)
    print(buf.getvalue())
    print(res["gate"].to_string())


if __name__ == "__main__":
    main()
