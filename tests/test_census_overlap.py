"""Animal- and library-level overlap checks (synthetic)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fedkg_spacebio.census import overlap as ov
from fedkg_spacebio.census.samples import parse_samples

FIX = Path(__file__).parent / "fixtures" / "census"


def test_animal_overlap_needs_name_and_strain_and_sex():
    samples = parse_samples((FIX / "samples_query.csv").read_text())
    matrix, pairs, collisions = ov.animal_overlap(samples)
    # FAKE-2's FAKE1_FLT_M1/M2 are FAKE-1's FAKE-1_FLT_M1/M2 (same strain, same sex).
    assert matrix.loc["OSD-FAKE-1", "OSD-FAKE-2"] == 2
    assert set(pairs["source_b"]) == {"FAKE1_FLT_M1", "FAKE1_FLT_M2"}
    # FAKE-1_GC_M4 appears in both, but with a different strain: a collision, not an overlap.
    assert collisions["source_a"].tolist() == ["FAKE-1_GC_M4"]
    assert matrix.loc["OSD-FAKE-1", "OSD-FAKE-1"] == 7  # unique animals on the diagonal


def test_within_study_duplicates_report_technical_replicates():
    samples = parse_samples((FIX / "samples_query.csv").read_text())
    dups = ov.within_study_duplicates(samples)
    assert dups["source_name"].tolist() == ["FAKE-1_FLT_M3"]
    assert dups["factors_that_differ"].iloc[0] == ["none (technical replicate)"]


def _counts(rng, base, n_samples=4, prefix="S"):
    """Poisson counts around one shared expression profile: different 'mice', same tissue."""
    data = {f"{prefix}{i}": rng.poisson(base * rng.uniform(0.8, 1.2)) for i in range(n_samples)}
    return pd.DataFrame(data, index=[f"FAKE:{g:05d}" for g in range(len(base))])


def test_library_overlap_flags_identical_and_near_exact_only():
    rng = np.random.default_rng(0)
    base = rng.lognormal(mean=4, sigma=2, size=2000)
    a = _counts(rng, base, prefix="A")
    b = _counts(rng, base, prefix="B")  # different "mice" from the same tissue profile
    b["B_copy"] = a["A0"]  # an exact reprocessed copy
    near = a["A1"].copy()
    near.iloc[0] += 1  # one read moved between genes: r ≈ 1 and the same library size
    near.iloc[1] -= 1
    b["B_near"] = near
    b["B_scaled"] = a["A2"] * 2  # r on log1p stays very high, but the library is twice the size
    matches, best = ov.library_overlap({"OSD-FAKE-1": a, "OSD-FAKE-2": b})
    found = set(zip(matches["sample_a"], matches["sample_b"], strict=True))
    assert ("A0", "B_copy") in found
    assert ("A1", "B_near") in found
    assert ("A2", "B_scaled") not in found
    # Different mice correlate highly but are never flagged.
    assert not any(sb.startswith("B") and sb[1:].isdigit() for _, sb in found)
    different = best[(best["study"] == "OSD-FAKE-2") & (best["sample"] == "B0")]
    assert different["r_log1p"].iloc[0] > 0.9  # high r alone ...
    assert ("A0", "B0") not in found  # ... is not a match


def test_pearson_columns_matches_numpy():
    rng = np.random.default_rng(1)
    X, Y = rng.normal(size=(50, 3)), rng.normal(size=(50, 2))
    r = ov._pearson_columns(X, Y)
    assert np.allclose(r[1, 0], np.corrcoef(X[:, 1], Y[:, 0])[0, 1])
