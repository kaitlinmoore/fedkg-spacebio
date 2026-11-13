"""Site data loader: one client's own counts and sample metadata, nothing else.

The federation boundary, at the data end
----------------------------------------
``load_site(client, manifest_dir, data_root)`` is the only way site code gets data. It:

1. opens exactly one manifest, ``<manifest_dir>/<client>.json``. It never lists the directory,
   so it never sees another client's manifest. It refuses a manifest whose ``client`` field
   names someone else;
2. opens only the count files the manifest names. Each must resolve inside
   ``<data_root>/<osd>/`` for a study in the manifest's ``studies``. That stops ``../`` and
   absolute paths;
3. reads only the manifest's sample columns (``usecols``), so another client's samples in the
   same file (OSD-245 and OSD-379 serve an ISS client and an Earth client) are never loaded.
   It then checks that the columns it got are exactly the manifest's samples.

``tests/test_sites_loader.py`` checks all three, including with a Python audit hook that records
every file the loader opens.

What it does to the data (D-022), each recorded in ``SiteData.provenance``
-------------------------------------------------------------------------
- **ERCC spike-in rows are dropped.** They are synthetic RNAs added in the lab, not mouse genes,
  and GeneLab's standard DE doesn't use them.
- **Counts are rounded to integers**, half to even (``numpy.rint``, the same rule as R's
  ``round``, which ``DESeqDataSetFromTximport`` applies). RSEM writes *expected* counts, which
  are 2.5–7.9% non-integer in these studies. DESeq2's negative binomial needs integers. The
  share of cells changed and the largest change are logged.
- **Several studies in one site** (SpaceX-4: OSD-47 + OSD-48) are joined column-wise. Their
  gene index must be identical (same annotation). It's an error otherwise, since a silent outer
  join would invent zeros.
- **Metadata:** ``condition`` is ``flight`` or ``ground``, plus any covariates the design uses
  (``study``, ``age``). Labels never become features or prompt text; they exist only to define
  the contrast.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ERCC = re.compile(r"^ERCC-\d{5}$")


@dataclass
class SiteData:
    client: str
    counts: pd.DataFrame  # genes × samples, int64, Ensembl gene IDs as the index
    metadata: pd.DataFrame  # samples × (condition, covariates), same order as counts columns
    manifest: dict
    provenance: dict


class BoundaryError(PermissionError):
    """A site tried to read something outside its own manifest."""


def read_manifest(client: str, manifest_dir: Path) -> dict:
    path = Path(manifest_dir) / f"{client}.json"
    manifest = json.loads(path.read_text())
    if manifest.get("client") != client:
        raise BoundaryError(f"manifest {path.name} belongs to {manifest.get('client')!r}, "
                            f"not {client!r}")
    return manifest


def _resolve_inside(data_root: Path, osd: str, rel: str, allowed: set[str]) -> Path:
    if osd not in allowed:
        raise BoundaryError(f"study {osd} is not in this site's manifest")
    base = (Path(data_root) / osd).resolve()
    path = (Path(data_root) / rel).resolve()
    if not path.is_relative_to(base):
        raise BoundaryError(f"{rel} is outside {osd}/")
    return path


def load_site(client: str, manifest_dir: Path, data_root: Path) -> SiteData:
    manifest = read_manifest(client, manifest_dir)
    allowed = set(manifest["studies"])
    blocks, meta_rows, sources_prov = [], [], []
    for src in manifest["sources"]:
        path = _resolve_inside(data_root, src["osd_id"], src["counts_file"], allowed)
        wanted = [s["sample"] for s in src["samples"]]
        header = pd.read_csv(path, nrows=0).columns
        id_col = header[0]
        missing = set(wanted) - set(header)
        if missing:
            raise ValueError(f"{client}: samples not in {path.name}: {sorted(missing)}")
        block = pd.read_csv(path, usecols=[id_col, *wanted], index_col=0)
        block.index.name = "ensembl"
        if list(block.columns) != wanted:  # usecols keeps file order; restore manifest order
            block = block[wanted]
        if set(block.columns) != set(wanted):
            raise BoundaryError(f"{client}: loader returned columns outside the manifest")
        blocks.append(block)
        sources_prov.append({"osd_id": src["osd_id"], "file": path.name,
                             "n_samples": len(wanted), "n_rows": int(block.shape[0])})
        for s in src["samples"]:
            row = {"sample": s["sample"], "condition": s["arm"], "animal": s["animal"]}
            row.update({k: v for k, v in s.items() if k not in ("sample", "arm", "animal")})
            meta_rows.append(row)

    first = blocks[0].index
    for b in blocks[1:]:
        if not b.index.equals(first):
            raise ValueError(f"{client}: studies have different gene indexes; refusing to join")
    raw = pd.concat(blocks, axis=1)

    is_ercc = raw.index.to_series().str.match(ERCC).to_numpy()
    raw = raw.loc[~is_ercc]

    values = raw.to_numpy(dtype=np.float64)
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError(f"{client}: counts must be finite and non-negative")
    rounded = np.rint(values)
    changed = rounded != values
    counts = pd.DataFrame(rounded.astype(np.int64), index=raw.index, columns=raw.columns)

    metadata = pd.DataFrame(meta_rows).set_index("sample").loc[counts.columns]
    if metadata["animal"].duplicated().any():
        raise ValueError(f"{client}: more than one sample per animal")
    arms = set(metadata["condition"])
    if arms != {"flight", "ground"}:
        raise ValueError(f"{client}: a site needs both arms, got {sorted(arms)}")

    provenance = {
        "client": client, "sources": sources_prov,
        "n_genes_in_file": int(raw.shape[0] + is_ercc.sum()),
        "n_ercc_dropped": int(is_ercc.sum()), "n_genes": int(counts.shape[0]),
        "n_samples": int(counts.shape[1]),
        "n_flight": int((metadata["condition"] == "flight").sum()),
        "n_ground": int((metadata["condition"] == "ground").sum()),
        "rounding": {"rule": "half to even (numpy.rint)",
                     "fraction_cells_changed": float(changed.mean()),
                     "max_abs_change": float(np.abs(rounded - values).max(initial=0.0))},
    }
    return SiteData(client, counts, metadata.drop(columns="animal"), manifest, provenance)
