"""Turn OSDR's bulk sample query into one tidy row per RNA-seq sample (WS-06).

AI-drafted (Claude, WS-06).

Input is the CSV from ``OsdrClient.query_samples`` with these field prefixes requested (see
``SAMPLE_FIELDS``): ``study.source name``, ``study.factor value``, ``study.characteristics`` and
``study.parameter value``. OSDR returns one column per field that exists in *any* requested
study, and the literal string ``NaN`` where a study doesn't have it.

Two OSDR facts this module leans on:
- **``source name`` is the animal.** In ISA-Tab (the format OSDR metadata is stored in) a *source*
  is the biological subject and a *sample* is material taken from it. One mouse can give several
  samples: two libraries with and without ERCC spike-ins (OSD-168), or technical replicates
  (OSD-463). n per arm must therefore count unique sources, not rows.
- **The arm is the ``spaceflight`` factor value.** OSDR labels it ``Space Flight``, ``Ground
  Control``, ``Vivarium Control`` or ``Basal Control``. Anything else maps to ``other`` and is kept,
  not dropped, so a surprise label is visible in the output.

Every other factor (dissection condition, duration, euthanasia location, age, altered gravity,
spike-in, space mission ...) becomes part of the sample's **group**. Two samples are in the same
group exactly when all of their factor values match.
"""

from __future__ import annotations

import io
import re

import pandas as pd

SAMPLE_FIELDS = (
    "study.source name",
    "study.factor value",
    "study.characteristics",
    "study.parameter value",
)

ARM_BY_LABEL = {
    "space flight": "flight",
    "ground control": "ground",
    "vivarium control": "vivarium",
    "basal control": "basal",
}
CONTROL_TYPE_BY_ARM = {
    "ground": "Ground Control",
    "vivarium": "Vivarium Control",
    "basal": "Basal Control",
}

FACTOR_PREFIX = "study.factor value."
SPACEFLIGHT_FACTOR = FACTOR_PREFIX + "spaceflight"
MISSING = {"", "nan", "none", "not applicable", "n/a"}


def _clean(value: object) -> str | None:
    """OSDR's ``NaN``/empty/``Not Applicable`` → None; everything else stripped."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return None if text.lower() in MISSING else text


def normalize_label(value: str) -> str:
    """Make an OSDR value comparable with the KG's form of the same label.

    OSDR writes units in braces (``~60 {day}``); the KG drops them (``~60 day``). Whitespace is
    collapsed. Case is kept: labels are compared case-sensitively elsewhere.
    """
    return re.sub(r"\s+", " ", value.replace("{", "").replace("}", "")).strip()


def normalize_source(name: str) -> str:
    """Normalize a source (animal) name for cross-study matching.

    Upper-case and drop everything that isn't a letter or digit, so ``RR-1_FLT_M1``,
    ``RR1_FLT_M1`` and ``rr1 flt m1`` all become ``RR1FLTM1``. This is deliberately loose; the
    overlap check requires strain and sex to agree as well before calling two sources the same
    animal (see ``overlap.py``).
    """
    return re.sub(r"[^A-Z0-9]", "", name.upper())


def parse_samples(csv_text: str, assay_pattern: str = r"rna-sequencing") -> pd.DataFrame:
    """Parse the bulk sample CSV into one row per sample of the matching assays.

    Columns: ``osd_id, assay_name, sample_name, source_name, source_key, arm, spaceflight_label,
    group_key, factors`` (a dict of the non-spaceflight factors), and the characteristics we
    report: ``sex, strain, age_at_launch, material_type, launch_mission``. Factor values and the
    parameters we report (``duration_param``, ``euthanasia_method``) are kept as ``factor:<name>``
    and plain columns.
    """
    raw = pd.read_csv(io.StringIO(csv_text), dtype=str, keep_default_na=False)
    raw = raw[raw["id.assay name"].str.contains(assay_pattern, case=False, regex=True)]
    factor_cols = sorted(c for c in raw.columns if c.startswith(FACTOR_PREFIX))
    if SPACEFLIGHT_FACTOR not in factor_cols:
        raise ValueError("sample query has no 'study.factor value.spaceflight' column")
    other_factors = [c for c in factor_cols if c != SPACEFLIGHT_FACTOR]

    def col(row: pd.Series, name: str) -> str | None:
        return _clean(row[name]) if name in row.index else None

    records = []
    for _, row in raw.iterrows():
        label = col(row, SPACEFLIGHT_FACTOR)
        factors = {
            c.removeprefix(FACTOR_PREFIX): normalize_label(v)
            for c in other_factors
            if (v := col(row, c)) is not None
        }
        source = col(row, "study.source name") or ""
        rec = {
            "osd_id": row["id.accession"],
            "assay_name": row["id.assay name"],
            "sample_name": row["id.sample name"],
            "source_name": source,
            "source_key": normalize_source(source),
            "spaceflight_label": label,
            "arm": ARM_BY_LABEL.get((label or "").lower(), "other"),
            "factors": factors,
            # Group = arm label + every other factor value. Sorted so the key is order-free.
            "group_key": (label or "",) + tuple(sorted(factors.items())),
            "sex": col(row, "study.characteristics.sex"),
            "strain": col(row, "study.characteristics.strain"),
            "age_at_launch": col(row, "study.characteristics.age at launch"),
            "material_type": col(row, "study.characteristics.material type"),
            "launch_mission": col(row, "study.characteristics.launch mission"),
            "duration_param": col(row, "study.parameter value.duration"),
            "euthanasia_method": col(row, "study.parameter value.euthanasia method"),
        }
        for name, value in factors.items():
            rec[f"factor:{name}"] = value
        records.append(rec)
    if not records:
        return pd.DataFrame(columns=["osd_id", "assay_name", "sample_name", "source_name"])
    return pd.DataFrame.from_records(records)
