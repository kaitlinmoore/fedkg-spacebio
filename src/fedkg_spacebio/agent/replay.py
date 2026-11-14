"""The replay key: which recorded release an identical spec may replay (``job.submit``, Part B).

``replay`` mode returns the recorded result of an identical spec instead of running FLARE. The
spec hash (CONTRACTS §6.1) can't be the key, because it covers fields that differ on every run
even when the computation is the same. The replay key is the SHA-256 of the canonical spec
(same canonical JSON as the spec hash) **without exactly these fields** (``REPLAY_DROPS``):

- ``job_id``: minted fresh by every policy check;
- ``policy_decision``: names this run's decision;
- ``gene_plan.ref``: where this run wrote its gene plan (a path); ``gene_plan.sha256`` stays, so
  the genes and contributors are pinned;
- ``genes_exclude.ref`` and ``genes_exclude.decision_id``: where the exclusion list was written and
  which of this run's decisions refused those genes; ``genes_exclude.sha256`` stays.

Everything else stays in the key: task, question, stratum, clients, methods, counts variant,
inclusion rule, outputs, aggregation and its bits, weighting, rounds, split, contact windows and
all of provenance (KG version, census date, annotation, manifest and inclusion-flag hashes).
Any change to those is a different computation (or a different description of it) and can't
replay a recording.
"""

from __future__ import annotations

import copy
import hashlib
import json

from fedkg_spacebio.federation.jobspec import JobSpec

REPLAY_DROPS: tuple[str, ...] = ("job_id", "policy_decision", "gene_plan.ref",
                                 "genes_exclude.ref", "genes_exclude.decision_id")


def replay_payload(spec: JobSpec) -> dict:
    """The spec as hashed for replay: canonical fields, nulls left out, ``REPLAY_DROPS`` removed."""
    d = copy.deepcopy(spec.model_dump(mode="json", by_alias=True, exclude_none=True))
    for path in REPLAY_DROPS:
        *parents, leaf = path.split(".")
        node = d
        for p in parents:
            node = node.get(p) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(leaf, None)
    return d


def replay_key(spec: JobSpec) -> str:
    data = json.dumps(replay_payload(spec), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()
