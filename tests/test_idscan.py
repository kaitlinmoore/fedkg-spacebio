"""ID-shaped tokens in text (CONTRACTS §8 rule 3): patterns, listed slugs, slug shape, symbols."""

from __future__ import annotations

import pytest

from fedkg_spacebio.agent.idscan import data_tokens, is_pattern_id, scan

CMP = "OSD-379-d6b14f9ef8590e45a17dbc53c87dafd8"
EDGE = f"kgedge:MEASURED_DIFFERENTIAL_EXPRESSION_ASmMG|{CMP}|NCBIGene:13170"


@pytest.mark.parametrize("text, want", [
    ("FAKE:0001 is lower in flight.", ["FAKE:0001"]),
    ("Seen in OSD-48, and OSD-47.", ["OSD-48", "OSD-47"]),
    (f"Comparison {CMP} reported it.", [CMP]),                    # not OSD-379
    (f"Edge {EDGE}.", [EDGE]),                                     # one token, not three
    ("(UBERON:0006727)", ["UBERON:0006727"]),
    ("Denied by pol:SpaceX-16:2#diff.", ["pol:SpaceX-16:2#diff"]),  # not SpaceX-16
    ("Refused at spec#clients-known; see rel:FAKE-0001-1", ["spec#clients-known",
                                                            "rel:FAKE-0001-1"]),
    ("trace:FAKE-0001/dec/3 chose it", ["trace:FAKE-0001/dec/3"]),
    ("job:20261116T142301Z-a1b2c3 ran", ["job:20261116T142301Z-a1b2c3"]),
    ("See note:time-of-day.", ["note:time-of-day"]),
    ("OSD-FAKE-7 too", ["OSD-FAKE-7"]),
])
def test_pattern_ids(text, want):
    assert scan(text) == want


def test_plain_words_are_not_ids():
    text = ("Fixed effect, RNA-Seq, q < 0.05, flight vs ground, ISS, GRCm39, BH-adjusted, "
            "2-sided, k = 3.")
    assert scan(text) == []


def test_slug_shape_catches_unlisted_missions():
    assert scan("without SpaceX-99 or STS-200") == ["SpaceX-99", "STS-200"]
    assert scan("on Soyuz_TMA-14") == ["Soyuz_TMA-14"]


def test_listed_slug_longest_first():
    vocab = ["SpaceX-16", "SpaceX-16-ISST"]
    assert scan("SpaceX-16-ISST and SpaceX-16's other client", vocab) == [
        "SpaceX-16-ISST", "SpaceX-16"]


def test_listed_lowercase_ids_found_only_when_listed():
    assert scan("logged in restricted-1") == []
    assert scan("logged in restricted-1", ["restricted-1"]) == ["restricted-1"]


def test_site_slug_alone_is_not_the_client():
    """A site name followed by a stratum suffix the tools never listed is its own token."""
    assert scan("SpaceX-16-LAR", ["SpaceX-16"]) == ["SpaceX-16-LAR"]


def test_gene_symbols_are_names():
    text = "Nkx2-1 and its human ortholog NKX2-5 changed"
    assert scan(text) == ["Nkx2-1", "NKX2-5"]                     # slug-shaped by default
    assert scan(text, symbols=["Nkx2-1", "NKX2-5"]) == []
    assert scan(text, symbols=["Nkx2-1"]) == ["NKX2-5"]


def test_listed_id_beats_symbol_of_same_spelling():
    assert scan("SpaceX-4", vocabulary=["SpaceX-4"], symbols=["SpaceX-4"]) == ["SpaceX-4"]


def test_is_pattern_id():
    assert is_pattern_id("NCBIGene:13170")
    assert is_pattern_id(CMP)
    assert not is_pattern_id("SpaceX-16")
    assert not is_pattern_id("FAKE:0001.")


def test_data_tokens_skip_symbol_fields():
    data = {"rows": [{"ncbi_gene": "NCBIGene:1", "symbol": "Nkx2-1"}],
            "orthologs": [{"gene": "NCBIGene:2", "gene_symbol": "NKX2-5"}],
            "notes": ["All hits vanish without SpaceX-16."]}
    assert data_tokens(data) == {"NCBIGene:1", "NCBIGene:2", "SpaceX-16"}
