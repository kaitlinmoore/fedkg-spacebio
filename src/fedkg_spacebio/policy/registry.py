"""The site registry as the policy check sees it: one record per client, from its manifest.

The policy check needs a few facts about each client: its site and stratum (which policy
governs it, which clients are compared), animals per arm (minimum n), n_k (FedAvg weighting,
D-017), which policy its manifest declares, and the hash of the manifest (so a decision says
exactly which manifests it was checked against, CONTRACTS §6.1 ``provenance``). This module reads
just those fields. Sample lists and file paths are never needed here.

``n_flight`` and ``n_ground`` are read as None when a manifest doesn't declare them (the
synthetic FedAvg sites, whose per-arm counts are random). A ``min_aggregation`` rule then fails
for that client: an undeclared count can't show it meets a minimum.

Two deliberate non-checks:
- **n_k isn't validated on load.** A missing or unusable n_k is a reason to *deny* a FedAvg job
  by name (``pol:project:1#fedavg-weighting``), not a reason the registry can't be read.
- **The manifest's ``policy_id`` isn't compared with the loaded policies on load.** That is
  ``check_against_profile``, which the policy check calls before evaluating. A mismatch (say an
  open-profile manifest checked under the restricted profile) is an error, never a silent choice
  of one or the other (CONTRACTS §5.4).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from fedkg_spacebio.sites.manifest import STRATA, check_policy_id


class PolicyMismatchError(ValueError):
    """A manifest names a policy that isn't the loaded one for its site."""


@dataclass(frozen=True)
class ClientRecord:
    client: str
    site: str
    stratum: str
    policy_id: str
    n_flight: int | None  # None: not declared
    n_ground: int | None
    n_k: object  # as declared (may be absent or unusable; the weighting rule judges it)
    manifest_sha256: str


def _count(m: dict, field: str, path: Path) -> int | None:
    v = m.get(field)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise ValueError(f"{path.name}: {field} must be a non-negative integer, got {v!r}")
    return v


def read_client(path: str | Path) -> ClientRecord:
    """One manifest file → its record. The client must equal the file name."""
    path = Path(path)
    data = path.read_bytes()
    m = json.loads(data)
    missing = [f for f in ("client", "site", "stratum", "policy_id") if f not in m]
    if missing:
        raise ValueError(f"{path.name}: manifest lacks {missing}")
    if m["client"] != path.stem:
        raise ValueError(f"{path.name}: client {m['client']!r} doesn't match the file name")
    if m["stratum"] not in STRATA:
        raise ValueError(f"{path.name}: unknown stratum {m['stratum']!r}")
    check_policy_id(m["policy_id"], m["site"])
    return ClientRecord(
        client=m["client"], site=m["site"], stratum=m["stratum"], policy_id=m["policy_id"],
        n_flight=_count(m, "n_flight", path), n_ground=_count(m, "n_ground", path),
        n_k=m.get("n_k"), manifest_sha256=hashlib.sha256(data).hexdigest())


def load_registry(manifest_dir: str | Path,
                  clients: Iterable[str] | None = None) -> dict[str, ClientRecord]:
    """Client → record, in the order given (or sorted by name if ``clients`` is None).

    The order matters: it is the "registry order" in which sites outside a job are evaluated
    (CONTRACTS §5.4 step 4).
    """
    manifest_dir = Path(manifest_dir)
    names = list(clients) if clients is not None else sorted(
        p.stem for p in manifest_dir.glob("*.json"))
    registry = {}
    for name in names:
        path = manifest_dir / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"no manifest for client {name!r} in {manifest_dir}")
        registry[name] = read_client(path)
    return registry


def check_against_profile(registry: dict[str, ClientRecord], profile) -> None:
    """Raise ``PolicyMismatchError`` unless every manifest names its site's loaded policy."""
    problems = []
    for rec in registry.values():
        loaded = profile.sites.get(rec.site)
        if loaded is None:
            problems.append(f"{rec.client}: no policy for site {rec.site} in profile "
                            f"{profile.name!r}")
        elif loaded.policy.policy_id != rec.policy_id:
            problems.append(f"{rec.client}: manifest names {rec.policy_id}, but profile "
                            f"{profile.name!r} has {loaded.policy.policy_id} for site "
                            f"{rec.site}")
    if problems:
        raise PolicyMismatchError("manifests don't match the loaded policies:\n  "
                                  + "\n  ".join(problems))
