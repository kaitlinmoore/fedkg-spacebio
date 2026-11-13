"""Paths and settings shared by job.py (FLARE), client.py and central.py (in-process).

Both paths import from here, so they can't drift apart.
"""

import os
from pathlib import Path

# Check 1 tolerance (federated vs central pooling). Both paths run the same float64 code on the
# same summaries. The only differences possible are float rounding in serialization (none
# expected: FLARE moves numpy arrays as binary) and summation order (none: matrices are
# assembled in sorted client order). rtol 1e-12 is about 4,500 ulps, generous for rounding and
# far below any real difference (a single swapped gene moves estimates by ~1e-1).
RTOL = 1e-12
ATOL = 1e-14

N_CPUS_PER_CLIENT = 4  # pydeseq2 workers per site; 4 sites × 4 = 16 of 24 cores


def workdir() -> Path:
    """Root for run outputs: $FEDKG_WORKDIR, default ~/fedkg-work. Never inside the repo."""
    return Path(os.environ.get("FEDKG_WORKDIR", Path.home() / "fedkg-work"))


def sim_workspace_root() -> Path:
    env = os.environ.get("NVFLARE_SIMULATOR_WORKSPACE_ROOT")
    return Path(env) if env else workdir() / "flare-sim"


def manifest_dir(counts: str, root: Path | None = None) -> Path:
    return (root or workdir()) / "phase3a" / "manifests" / counts


def data_root(root: Path | None = None) -> Path:
    return (root or workdir()) / "data" / "osdr"


def annotation_path(root: Path | None = None) -> Path:
    return ((root or workdir()) / "data" / "annotations"
            / "Mus_musculus.GRCm39.112-GL-annotations.tsv")


def results_dir(stratum: str, counts: str, path: str, root: Path | None = None) -> Path:
    """``path`` is ``flare`` or ``central``."""
    return (root or workdir()) / "results" / "phase3a" / stratum / counts / path


def job_name(stratum: str, counts: str) -> str:
    return f"phase3a_meta_{stratum}_{counts}"
