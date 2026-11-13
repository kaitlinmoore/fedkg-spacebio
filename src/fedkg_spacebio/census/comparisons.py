"""Flight-vs-control comparisons from OSDR sample groups, and their join to the KG (WS-06).

AI-drafted (Claude, WS-06).

From samples to comparisons
---------------------------
1. **Groups.** Within one OSDR assay, samples with identical factor values form a group (see
   ``samples.py``). OSD-48 liver RNA-seq has four: {Space Flight, Ground Control} × {Carcass, Upon
   euthanasia}.
2. **Comparisons.** Every (flight group, control group) pair in the same assay is one
   flight-vs-control comparison, in that direction only. The KG stores both directions
   (``docs/kg-schema.md``); we keep flight → control, so a positive log2FC means higher in flight.
   Flight-vs-flight pairs (e.g. ``uG`` vs ``1G by centrifugation``) aren't flight-vs-control and are
   left out.
3. **Strict rule (D-015, as WS-01 applied it to the KG).** A comparison is ``strict_match`` when
   every factor other than the spaceflight factor has the same value on both sides. A confounded
   pair such as ``Space Flight, Carcass`` vs ``Ground Control, Upon euthanasia`` fails, as does
   ``uG`` vs ``1G on Earth``. Non-strict rows are kept with ``strict_match = False`` and the list of
   factors that differ, so the looser set can be re-derived without redoing the census.
4. **n per arm** counts unique **animals** (normalized source names); the number of samples is
   kept alongside. They differ when one mouse gave several libraries.

Joining to the KG
-----------------
The KG's comparison IDs (``Assay.identifier``) are what the rest of the project cites, so each
OSDR comparison is matched to a KG comparison on (study, assay name, labels on each side). Labels
don't always agree exactly:
- the KG drops OSDR's unit braces (``~60 {day}`` → ``~60 day``); ``samples.normalize_label`` does
  the same on our side;
- the KG cuts some labels at an opening parenthesis (``SpaceX-4 (RR1``), and in OSD-168 drops one
  side's spike-in label entirely;
- the KG occasionally has a different value (OSD-245 basal ``1 day`` vs OSDR ``0 {day}``).

The KG's assay names can also lag OSDR's: OSD-245's KG name has a stray space after the last
underscore, and OSD-168's says ``_illumina`` where OSDR now says ``_Illumina HiSeq 4000`` (the KG
was built from an older metadata revision). So the assay is matched first, graded as
``kg_assay_match``: ``exact``, ``normalized`` (case and whitespace ignored) or ``only_assay`` (the
study has exactly one RNA-seq assay on each side, so they must be the same).

Then the labels are matched, again graded, and nothing is forced: ``exact`` (every label on both
sides matches, same number of labels), ``partial`` (every KG label matches but OSDR has extra
labels; still a unique pair), ``ambiguous`` (several OSDR pairs fit one KG comparison) or
``unmatched``. Only ``exact`` and ``partial`` matches get a ``comparison_id``. KG comparisons
that match nothing are returned separately so they can be reported.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd

from fedkg_spacebio.census.samples import CONTROL_TYPE_BY_ARM, normalize_label


def build_groups(samples: pd.DataFrame) -> pd.DataFrame:
    """One row per (study, assay, group) with its arm, factors, samples and animals."""
    rows = []
    for (osd, assay, key), g in samples.groupby(["osd_id", "assay_name", "group_key"], sort=True):
        first = g.iloc[0]
        rows.append({
            "osd_id": osd,
            "assay_name": assay,
            "group_key": key,
            "arm": first["arm"],
            "spaceflight_label": first["spaceflight_label"],
            "factors": first["factors"],
            "labels": [first["spaceflight_label"]]
            + [v for _, v in sorted(first["factors"].items())],
            "samples": sorted(g["sample_name"]),
            "animals": sorted(set(g["source_key"])),
            "n_samples": len(g),
            "n_animals": g["source_key"].nunique(),
            "sex": sorted(set(g["sex"].dropna())),
            "strain": sorted(set(g["strain"].dropna())),
        })
    return pd.DataFrame(rows)


def differing_factors(a: dict[str, str], b: dict[str, str]) -> list[str]:
    """Names of the non-spaceflight factors whose values differ (or exist on one side only)."""
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def build_comparisons(groups: pd.DataFrame) -> pd.DataFrame:
    """All flight → control comparisons within each assay, with the strict flag and n per arm."""
    rows = []
    for (osd, assay), g in groups.groupby(["osd_id", "assay_name"], sort=True):
        flights = g[g["arm"] == "flight"]
        controls = g[g["arm"].isin(CONTROL_TYPE_BY_ARM)]
        for _, f in flights.iterrows():
            for _, c in controls.iterrows():
                diff = differing_factors(f["factors"], c["factors"])
                rows.append({
                    "osd_id": osd,
                    "osdr_assay_name": assay,
                    "control_type": CONTROL_TYPE_BY_ARM[c["arm"]],
                    "factors_flight": f["labels"],
                    "factors_control": c["labels"],
                    "strict_match": not diff,
                    "differing_factors": diff,
                    "n_flight": f["n_animals"],
                    "n_control": c["n_animals"],
                    "n_flight_samples": f["n_samples"],
                    "n_control_samples": c["n_samples"],
                    "flight_animals": f["animals"],
                    "control_animals": c["animals"],
                    "flight_samples": f["samples"],
                    "control_samples": c["samples"],
                    "flight_factors": f["factors"],
                    "control_factors": c["factors"],
                    "sex": _sex_summary(f["sex"] + c["sex"]),
                    "host_strain": "; ".join(sorted(set(f["strain"] + c["strain"]))) or None,
                })
    return pd.DataFrame(rows)


def _sex_summary(values: Sequence[str]) -> str:
    """CONTRACTS §2 vocabulary: ``male`` · ``female`` · ``mixed`` · ``unknown``."""
    found = {v.strip().lower() for v in values if v}
    if not found:
        return "unknown"
    return found.pop() if len(found) == 1 and found <= {"male", "female"} else "mixed"


# ------------------------------------------------------------------ KG join

def label_matches(kg_label: str, osdr_label: str) -> bool:
    """True if a KG label denotes the same value as an (already normalized) OSDR label.

    Equal after whitespace normalization, or the KG label is a cut-off prefix of the OSDR one at a
    parenthesis: ``SpaceX-4 (RR1`` and ``SpaceX-4`` both match ``SpaceX-4 (RR1)``.
    """
    k, o = normalize_label(kg_label), normalize_label(osdr_label)
    if k == o:
        return True
    return o.startswith(k) and (o[len(k):].startswith(" (") or o[len(k):].startswith(")")
                                or ("(" in k and ")" not in k))


def match_side(kg_labels: Sequence[str], osdr_labels: Sequence[str]) -> str | None:
    """``"exact"``, ``"partial"`` or None for one side of a comparison.

    Each KG label must claim a distinct OSDR label (greedy; label lists are 1 to 5 long).
    """
    unused = list(osdr_labels)
    for k in kg_labels:
        hit = next((o for o in unused if label_matches(k, o)), None)
        if hit is None:
            return None
        unused.remove(hit)
    return "exact" if not unused else "partial"


def _norm_assay(name: str) -> str:
    return "".join(name.lower().split())


def resolve_assay(kg_assay: str, osdr_assays: Sequence[str]) -> tuple[str | None, str | None]:
    """The OSDR assay a KG assay name refers to, and how sure we are (see module docstring)."""
    if kg_assay in osdr_assays:
        return kg_assay, "exact"
    norm = [a for a in osdr_assays if _norm_assay(a) == _norm_assay(kg_assay)]
    if len(norm) == 1:
        return norm[0], "normalized"
    if len(osdr_assays) == 1:
        return osdr_assays[0], "only_assay"
    return None, None


def match_kg(comparisons: pd.DataFrame, kg_rows: Iterable[dict]) -> tuple[pd.DataFrame, list[dict]]:
    """Attach KG comparison IDs to OSDR comparisons.

    Returns ``(comparisons_with_kg, kg_unmatched)``. Added columns: ``comparison_id`` (None unless
    the match is exact or partial and unique), ``kg_match`` and the KG's ``n_de_edges_kg``,
    ``de_method``, ``tissue_uberon``, ``tissue_label``, ``factors_flight_kg``,
    ``factors_control_kg``.
    """
    out = comparisons.copy()
    for col in ("comparison_id", "n_de_edges_kg", "de_method", "tissue_uberon", "tissue_label",
                "factors_flight_kg", "factors_control_kg"):
        out[col] = None
    out["kg_match"] = "unmatched"
    out["kg_candidates"] = 0
    out["kg_assay_match"] = None
    assays_by_study = out.groupby("osd_id")["osdr_assay_name"].unique().to_dict()

    kg_unmatched = []
    for kg in kg_rows:
        if kg.get("factor_space_2") not in CONTROL_TYPE_BY_ARM.values():
            continue  # flight-vs-flight: not part of the census
        hits = []
        assay, assay_grade = resolve_assay(kg["osdr_assay_name"],
                                           list(assays_by_study.get(kg["osd_id"], [])))
        if assay is None:
            kg_unmatched.append({**_kg_brief(kg), "reason": "no OSDR assay fits the KG assay name"})
            continue
        sel = out[(out["osd_id"] == kg["osd_id"]) & (out["osdr_assay_name"] == assay)
                  & (out["control_type"] == kg["factor_space_2"])]
        for idx, row in sel.iterrows():
            s1 = match_side(kg["factors_1"], row["factors_flight"])
            s2 = match_side(kg["factors_2"], row["factors_control"])
            if s1 and s2:
                hits.append((idx, "exact" if s1 == s2 == "exact" else "partial"))
        if len(hits) == 1:
            idx, grade = hits[0]
            if out.at[idx, "comparison_id"] is not None:
                # Two KG comparisons claim one OSDR pair: keep neither ID, flag it.
                out.at[idx, "kg_match"] = "ambiguous"
                out.at[idx, "comparison_id"] = None
                continue
            out.at[idx, "comparison_id"] = kg["comparison_id"]
            out.at[idx, "kg_match"] = grade
            out.at[idx, "kg_assay_match"] = assay_grade
            out.at[idx, "n_de_edges_kg"] = kg.get("n_de_edges_kg")
            out.at[idx, "de_method"] = kg.get("de_method")
            out.at[idx, "tissue_uberon"] = kg.get("material_id_1")
            out.at[idx, "tissue_label"] = kg.get("material_name_1")
            out.at[idx, "factors_flight_kg"] = kg.get("factors_1")
            out.at[idx, "factors_control_kg"] = kg.get("factors_2")
        elif len(hits) > 1:
            for idx, _ in hits:
                if out.at[idx, "comparison_id"] is None:
                    out.at[idx, "kg_match"] = "ambiguous"
                out.at[idx, "kg_candidates"] = max(out.at[idx, "kg_candidates"], len(hits))
            kg_unmatched.append({**_kg_brief(kg), "kg_assay_match": assay_grade,
                                 "reason": f"ambiguous: {len(hits)} OSDR pairs fit"})
        else:
            kg_unmatched.append({**_kg_brief(kg), "kg_assay_match": assay_grade,
                                 "reason": "no OSDR pair fits the labels"})
    return out, kg_unmatched


def _kg_brief(kg: dict) -> dict:
    return {k: kg.get(k) for k in ("osd_id", "comparison_id", "factors_1", "factors_2",
                                   "factor_space_2", "n_de_edges_kg")}
