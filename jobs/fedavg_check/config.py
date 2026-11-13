"""Protocol settings shared by job.py (FLARE) and reference.py (in-process). Change them here only.

If the two paths used different settings, the comparison would be meaningless, so both import
from this one place. client.py gets the values as command-line arguments from job.py, because
FLARE runs it from a copied job folder.
"""

import os
from pathlib import Path

SEED = 0  # synthetic-data seed (fedkg_spacebio.federation.synthetic)
NUM_ROUNDS = 20  # FedAvg rounds
LOCAL_STEPS = 5  # full-batch gradient steps per site per round (>1, so client drift can occur)
LR = 0.5  # gradient-descent learning rate (features are ~unit scale, so 0.5 is stable)

# Comparison tolerance for FLARE vs. reference. Both paths do float64 arithmetic on identical
# inputs. They differ only in summation order: FLARE adds client results in arrival order, the
# reference in site order. That costs a few ulps per round (float64 eps ≈ 2.2e-16), so ~1e-15 is
# expected. A wrong weighting moves theta by ~1e-2. rtol=1e-9 sits orders of magnitude above the
# first and below the second.
RTOL = 1e-9
ATOL = 1e-12


def workdir() -> Path:
    """Root for run outputs: $FEDKG_WORKDIR, default ~/fedkg-work. Never inside the repo."""
    return Path(os.environ.get("FEDKG_WORKDIR", Path.home() / "fedkg-work"))


def sim_workspace_root() -> Path:
    """FLARE simulator workspace root. NVFLARE_SIMULATOR_WORKSPACE_ROOT wins (as in SimEnv)."""
    env = os.environ.get("NVFLARE_SIMULATOR_WORKSPACE_ROOT")
    return Path(env) if env else workdir() / "flare-sim"


def job_name(weighting: str) -> str:
    return "fedavg_check" if weighting == "samples" else "fedavg_check_equal"
