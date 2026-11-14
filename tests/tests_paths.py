"""Fixture paths shared by tests that don't need the policy helpers."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPECS = REPO / "tests" / "fixtures" / "jobspecs"
