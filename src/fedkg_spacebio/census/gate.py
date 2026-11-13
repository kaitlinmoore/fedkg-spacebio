"""Site assignment and the D-016 gate assessment (WS-06). Proposals, not decisions.

AI-drafted (Claude, WS-06).

Site rule (proposed, following D-014)
-------------------------------------
- A study linked to exactly one mission belongs to that mission's site.
- A study linked to several missions (OSD-168: SpaceX-4 and SpaceX-8) is **not assigned** and is
  kept out of cross-site work. Its animals already appear in single-mission studies (see the
  overlap check), so assigning it would double-count them.

Independent animals
-------------------
An animal counts toward a comparison's ``n_*_indep`` unless it also appears in an *assigned* study
of a **different** site. Animals shared with an unassigned study (OSD-168) don't reduce n: that
study is the one left out. Animals shared by two studies of the same site would be counted once at
the site level (none were found for liver).

Gate (Kaitlin's criterion, 2026-11-12)
--------------------------------------
A mission site **passes** if it has
1. published per-sample unnormalized counts, and
2. at least one **strict** flight-vs-**ground-control** comparison with at least ``min_animals``
   (default 3) independent animals in each arm.

Phase 3a is a **go** if at least ``min_sites`` (default 3) sites pass. ``Stat_`` availability is
reported as a cross-check only: under D-016 each site computes its own SE with pydeseq2.

Why 3 animals per arm: DESeq2 estimates a per-gene dispersion from the replicates within each
arm, then shrinks it towards a trend fitted across genes. With 2 per arm that per-gene estimate
rests on 2 residual degrees of freedom in total and the result is mostly the trend; 3 is the usual
floor for a two-group design. Why 3 sites: a random-effects pool and I² estimate between-site
variance from the spread of site estimates; with 2 sites that is one degree of freedom.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def assign_sites(study_missions: dict[str, list[str]]) -> dict[str, str | None]:
    """``OSD → site``: the mission if there is exactly one, else None (unassigned)."""
    return {osd: (m[0] if len(m) == 1 else None) for osd, m in study_missions.items()}


def independent_counts(census: pd.DataFrame, overlap_pairs: pd.DataFrame,
                       sites: dict[str, str | None]) -> pd.DataFrame:
    """Add ``n_flight_indep`` / ``n_control_indep`` and the animals removed, per comparison."""
    # source_key sets per study that are shared with an assigned study in another site.
    shared: dict[str, set[str]] = {osd: set() for osd in sites}
    if not overlap_pairs.empty:
        from fedkg_spacebio.census.samples import normalize_source

        for _, p in overlap_pairs.iterrows():
            a, b = p["study_a"], p["study_b"]
            sa, sb = sites.get(a), sites.get(b)
            if sa is not None and sb is not None and sa != sb:
                shared[a].add(normalize_source(p["source_a"]))
                shared[b].add(normalize_source(p["source_b"]))
    out = census.copy()
    out["n_flight_indep"] = [len(set(an) - shared.get(o, set()))
                             for o, an in zip(out["osd_id"], out["flight_animals"], strict=True)]
    out["n_control_indep"] = [len(set(an) - shared.get(o, set()))
                              for o, an in zip(out["osd_id"], out["control_animals"], strict=True)]
    return out


def assess_gate(census: pd.DataFrame, sites: dict[str, str | None],
                counts_available: dict[str, bool], min_animals: int = 3, min_sites: int = 3,
                strict_column: str = "strict_match") -> tuple[pd.DataFrame, dict]:
    """Per-site gate table, and the overall go/no-go, for one threshold and one strict rule.

    ``strict_column`` selects which comparisons count as strict, so the same code runs the
    sensitivity analyses (e.g. a location-aware rule) without changing the gate logic.
    """
    rows = []
    site_ids = sorted({s for s in sites.values() if s is not None})
    for site in site_ids:
        studies = sorted(o for o, s in sites.items() if s == site)
        cand = census[(census["osd_id"].isin(studies)) & census[strict_column]
                      & (census["control_type"] == "Ground Control")].copy()
        cand["n_min_indep"] = cand[["n_flight_indep", "n_control_indep"]].min(axis=1)
        has_counts = all(counts_available.get(o, False) for o in studies)
        best = cand.sort_values("n_min_indep", ascending=False).head(1)
        n_best = int(best["n_min_indep"].iloc[0]) if len(best) else 0
        rows.append({
            "site": site,
            "studies": studies,
            "counts_published": has_counts,
            "n_strict_ground_comparisons": len(cand),
            "best_comparison_study": best["osd_id"].iloc[0] if len(best) else None,
            "best_comparison_id": best["comparison_id"].iloc[0] if len(best) else None,
            "best_n_flight_indep": int(best["n_flight_indep"].iloc[0]) if len(best) else 0,
            "best_n_control_indep": int(best["n_control_indep"].iloc[0]) if len(best) else 0,
            "best_euthanasia_location_flight": (
                best["euthanasia_location_flight"].iloc[0]
                if len(best) and "euthanasia_location_flight" in best else None),
            "passes": bool(has_counts and n_best >= min_animals),
            "reason": _reason(has_counts, len(cand), n_best, min_animals),
        })
    table = pd.DataFrame(rows)
    n_pass = int(table["passes"].sum()) if len(table) else 0
    verdict = {
        "min_animals": min_animals,
        "min_sites": min_sites,
        "strict_rule": strict_column,
        "n_sites": len(table),
        "n_sites_passing": n_pass,
        "passing_sites": table.loc[table["passes"], "site"].tolist() if len(table) else [],
        "go": n_pass >= min_sites,
    }
    return table, verdict


def _reason(has_counts: bool, n_cand: int, n_best: int, min_animals: int) -> str:
    if not has_counts:
        return "no published unnormalized counts"
    if n_cand == 0:
        return "no strict flight-vs-ground comparison"
    if n_best < min_animals:
        return (f"best strict comparison has {n_best} independent animals in its smaller arm "
                f"(< {min_animals})")
    return f"strict comparison with ≥ {min_animals} independent animals per arm"


def sensitivity(census: pd.DataFrame, sites: dict[str, str | None],
                counts_available: dict[str, bool], thresholds: Iterable[int] = (2, 3, 4, 5),
                strict_columns: Iterable[str] = ("strict_match",),
                min_sites: int = 3) -> pd.DataFrame:
    """The gate verdict across thresholds and strict rules, one row per combination."""
    rows = []
    for col in strict_columns:
        for t in thresholds:
            _, v = assess_gate(census, sites, counts_available, min_animals=t, min_sites=min_sites,
                               strict_column=col)
            rows.append({"strict_rule": col, "min_animals": t,
                         "n_sites_passing": v["n_sites_passing"],
                         "passing_sites": ", ".join(v["passing_sites"]), "go": v["go"]})
    return pd.DataFrame(rows)


def strata(census: pd.DataFrame, sites: dict[str, str | None], counts_available: dict[str, bool],
           column: str = "euthanasia_location_flight", min_animals: int = 3, min_sites: int = 3,
           strict_columns: Iterable[str] = ("strict_match",)) -> pd.DataFrame:
    """The gate run separately within each stratum (e.g. flight euthanized on the ISS vs on Earth).

    One row per (strict rule, stratum, site). A site can appear in two strata when it has
    separate cohorts for each (OSD-245, OSD-379: ISS-terminal and live-animal-return mice).
    """
    rows = []
    for col in strict_columns:
        for value in sorted(census[column].dropna().unique()):
            sub = census[census[column] == value]
            table, verdict = assess_gate(sub, sites, counts_available, min_animals=min_animals,
                                         min_sites=min_sites, strict_column=col)
            for _, r in table.iterrows():
                rows.append({"strict_rule": col, "stratum": value, "stratum_go": verdict["go"],
                             "stratum_sites_passing": verdict["n_sites_passing"],
                             **{k: r[k] for k in ("site", "best_comparison_study",
                                                  "best_comparison_id", "best_n_flight_indep",
                                                  "best_n_control_indep", "passes", "reason")}})
    return pd.DataFrame(rows)
