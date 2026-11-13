"""Site loader: the federation boundary at the data end (``fedkg_spacebio.sites.loader``).

Fixtures (``tests/fixtures/phase3a/``, fake IDs only):
- ``OSD-FAKE-1`` holds SITE-A's and SITE-B's samples in **one** count file, the way OSD-245 and
  OSD-379 serve an ISS client and an Earth client;
- ``OSD-FAKE-2`` holds SITE-C's, with a technical replicate column SITE-C doesn't use;
- an ERCC row and non-integer RSEM-style counts.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from fedkg_spacebio.sites.loader import BoundaryError, load_site

FIX = Path(__file__).parent / "fixtures" / "phase3a"
DATA, MANIFESTS = FIX / "data", FIX / "manifests"

_OPENED: list[str] | None = None


def _audit(event, args):
    if _OPENED is not None and event == "open" and isinstance(args[0], (str, Path)):
        _OPENED.append(str(Path(args[0]).resolve()))


sys.addaudithook(_audit)  # audit hooks can't be removed; it records only while _OPENED is a list


def opened_during(fn):
    global _OPENED
    _OPENED = []
    try:
        result = fn()
        return result, list(_OPENED)
    finally:
        _OPENED = None


def test_site_a_gets_exactly_its_own_samples_from_a_shared_file():
    site = load_site("SITE-A", MANIFESTS, DATA)
    assert list(site.counts.columns) == ["A_FLT_1", "A_FLT_2", "A_GC_1", "A_GC_2"]
    assert not any(c.startswith("B_") for c in site.counts.columns)
    assert site.metadata["condition"].tolist() == ["flight", "flight", "ground", "ground"]
    assert site.provenance["n_flight"] == 2 and site.provenance["n_ground"] == 2


def test_site_a_opens_only_its_manifest_and_its_count_file():
    _, opened = opened_during(lambda: load_site("SITE-A", MANIFESTS, DATA))
    allowed = {str((MANIFESTS / "SITE-A.json").resolve()),
               str((DATA / "OSD-FAKE-1" / "GLDS-FAKE-1_rna_seq_RSEM_Unnormalized_Counts_"
                    "GLbulkRNAseq.csv").resolve())}
    ours = {p for p in opened if p.startswith(str(FIX.resolve()))}
    assert ours and ours <= allowed, ours - allowed
    assert not any("SITE-B" in p or "SITE-C" in p or "OSD-FAKE-2" in p for p in opened)


def _tampered(tmp_path, edit):
    mdir = tmp_path / "manifests"
    shutil.copytree(MANIFESTS, mdir)
    m = json.loads((mdir / "SITE-A.json").read_text())
    edit(m)
    (mdir / "SITE-A.json").write_text(json.dumps(m))
    return mdir


def test_manifest_for_someone_else_is_refused(tmp_path):
    mdir = _tampered(tmp_path, lambda m: m.update(client="SITE-B"))
    with pytest.raises(BoundaryError, match="belongs to"):
        load_site("SITE-A", mdir, DATA)


def test_path_outside_the_site_study_is_refused(tmp_path):
    def edit(m):
        m["sources"][0]["counts_file"] = ("OSD-FAKE-1/../OSD-FAKE-2/GLDS-FAKE-2_rna_seq_RSEM_"
                                          "Unnormalized_Counts_GLbulkRNAseq.csv")
    with pytest.raises(BoundaryError, match="outside"):
        load_site("SITE-A", _tampered(tmp_path, edit), DATA)


def test_study_not_in_manifest_is_refused(tmp_path):
    def edit(m):
        m["sources"][0]["osd_id"] = "OSD-FAKE-2"
        m["sources"][0]["counts_file"] = ("OSD-FAKE-2/GLDS-FAKE-2_rna_seq_RSEM_Unnormalized_"
                                          "Counts_GLbulkRNAseq.csv")
    with pytest.raises(BoundaryError, match="not in this site's manifest"):
        load_site("SITE-A", _tampered(tmp_path, edit), DATA)


def test_one_arm_site_is_refused(tmp_path):
    def edit(m):
        m["sources"][0]["samples"] = [s for s in m["sources"][0]["samples"] if s["arm"] == "flight"]
    with pytest.raises(ValueError, match="both arms"):
        load_site("SITE-A", _tampered(tmp_path, edit), DATA)


def test_two_samples_from_one_animal_are_refused(tmp_path):
    def edit(m):
        m["sources"][0]["samples"][1]["animal"] = m["sources"][0]["samples"][0]["animal"]
    with pytest.raises(ValueError, match="one sample per animal"):
        load_site("SITE-A", _tampered(tmp_path, edit), DATA)


def test_rounding_half_to_even_and_ercc_dropped():
    site = load_site("SITE-A", MANIFESTS, DATA)
    assert "ERCC-00002" not in site.counts.index
    assert site.provenance["n_ercc_dropped"] == 1
    assert site.counts.dtypes.unique().tolist() == [np.dtype("int64")]
    # 10.5 → 10 and 11.5 → 12 (half to even); 100.49 → 100, 99.51 → 100; 2.5 → 2, 3.5 → 4
    assert site.counts.loc["FAKEENSG0001", ["A_FLT_1", "A_FLT_2"]].tolist() == [10, 12]
    assert site.counts.loc["FAKEENSG0003", ["A_FLT_1", "A_FLT_2"]].tolist() == [100, 100]
    assert site.counts.loc["FAKEENSG0004", ["A_FLT_1", "A_FLT_2"]].tolist() == [2, 4]
    assert site.provenance["rounding"]["fraction_cells_changed"] == pytest.approx(6 / 16)
    assert site.provenance["rounding"]["max_abs_change"] == pytest.approx(0.5)


def test_unused_technical_replicate_is_not_loaded():
    site = load_site("SITE-C", MANIFESTS, DATA)
    assert "C_GC_1_techrep2" not in site.counts.columns
    assert len(site.counts.columns) == 4
