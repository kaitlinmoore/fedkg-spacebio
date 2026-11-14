"""Paths and settings shared by job.py (FLARE), client.py and central.py (in-process).

Both paths import from here, so they can't drift apart.
"""

import os
from pathlib import Path

from fedkg_spacebio.policy.schema import load_profile

REPO = Path(__file__).resolve().parents[2]
POLICIES = REPO / "policies"

# Policy profiles (D-036; CONTRACTS §5.5). The two differ only in SpaceX-16's policy. The policy
# files are the single source of each site's policy_id; the manifests copy it from them.
# Paths: the open profile keeps the paths every earlier run used, so existing manifests and
# results stay where they are. Any other profile adds profiles/<profile>/, so two profiles never
# overwrite each other's manifests or results. (Alternative: <profile>/ in every path. It is
# symmetric, but it would move every existing output.)
PROFILES = ("open", "restricted")

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


def _profile_part(profile: str) -> tuple[str, ...]:
    if profile not in PROFILES:
        raise ValueError(f"unknown policy profile {profile!r}; expected one of {PROFILES}")
    return () if profile == "open" else ("profiles", profile)


def policy_ids(profile: str) -> dict[str, str]:
    """site → policy_id, read from ``policies/<profile>/`` (what the manifests must carry)."""
    _profile_part(profile)
    return load_profile(POLICIES / profile).site_policy_ids()


def manifest_dir(counts: str, root: Path | None = None, profile: str = "open") -> Path:
    return (root or workdir()).joinpath("phase3a", "manifests", *_profile_part(profile), counts)


def data_root(root: Path | None = None) -> Path:
    return (root or workdir()) / "data" / "osdr"


def annotation_path(root: Path | None = None) -> Path:
    return ((root or workdir()) / "data" / "annotations"
            / "Mus_musculus.GRCm39.112-GL-annotations.tsv")


def results_dir(stratum: str, counts: str, path: str, root: Path | None = None,
                profile: str = "open") -> Path:
    """``path`` is ``flare`` or ``central``."""
    return (root or workdir()).joinpath("results", "phase3a", *_profile_part(profile), stratum,
                                        counts, path)


def spec_results_dir(stratum: str, counts: str, slug: str, root: Path | None = None,
                     profile: str = "open") -> Path:
    """A job-spec run's folder: ``…/<stratum>/<counts>/jobs/<job slug>/`` (``server/`` and
    ``sites/`` inside). Separate from ``flare/`` and ``central/``, so a spec run never overwrites
    the flag-driven results."""
    return results_dir(stratum, counts, "jobs", root, profile) / slug


def job_name(stratum: str, counts: str) -> str:
    return f"phase3a_meta_{stratum}_{counts}"
