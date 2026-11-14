"""CONTRACTS.md stays in step with the code's whitelists.

The boundary objects are enforced in code by exact key whitelists. CONTRACTS.md is where people
(and the policy layer) read what crosses the boundary. These tests fail if a field is added to a
whitelist without being documented in its contract section.
"""

import re
from pathlib import Path

import pytest

from fedkg_spacebio.federation import site_clock, site_de

CONTRACTS = Path(__file__).resolve().parents[1] / "CONTRACTS.md"


def section(title_start: str) -> str:
    """Text from the heading that starts with ``title_start`` to the next heading of any level."""
    text = CONTRACTS.read_text(encoding="utf-8")
    m = re.search(rf"^#+ {re.escape(title_start)}.*?$(.*?)(?=^#+ )", text, re.M | re.S)
    assert m, f"no section {title_start!r} in CONTRACTS.md"
    return m.group(1)


def backticked(text: str) -> set[str]:
    return set(re.findall(r"`([^`]+)`", text))


@pytest.mark.parametrize("name", ["SUMMARY_FIELDS", "RESULT_FIELDS", "READING_FIELDS",
                                  "ABSOLUTE_SUMMARY_FIELDS", "ABSOLUTE_ARM_FIELDS"])
def test_clock_whitelists_are_documented_in_4_1(name):
    documented = backticked(section("4.1 Clock-plane summaries"))
    missing = [f for f in getattr(site_clock, name) if f not in documented]
    assert not missing, f"site_clock.{name} fields missing from CONTRACTS §4.1: {missing}"


def test_site_summary_metadata_is_documented_in_4():
    documented = backticked(section("4. Harmonization outputs"))
    missing = [f for f in site_de.META_FIELDS if f not in documented]
    assert not missing, f"site_de.META_FIELDS missing from CONTRACTS §4: {missing}"
