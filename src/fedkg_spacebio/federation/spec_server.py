"""Server side of a job-spec run (CONTRACTS §6): pool what arrived, write the released table.

Plain functions with no FLARE import. The two FLARE controllers for spec runs
(``meta_flare.SecureSumController`` and ``meta_flare.SpecPlainController``) call them, and the
tests call them directly, including the audit test that shows the secure-sum path never handles a
per-site value.

Two modes, one table shape (``results.spec_per_gene_table``):
- **secure_sum:** each client's masked vectors are validated, summed (masks cancel only over the
  full set of clients) and decoded to per-gene totals; FE (and, with w·y², Q, its p and I²) is
  computed from the totals (``secure_sum.fe_from_totals``). The table has no per-site estimates.
- **plain:** each client's §4 summary must hold **exactly** the genes the plan assigns it; the
  rows are pooled with ``meta.pool``, the same function as the flag-driven runs, so a plan equal
  to a flag-driven variant's gene set gives the same numbers.

In both, the table keeps only the columns of the methods in the spec (approved WS-13 plan); the
schema file lists ``methods_run`` and ``aggregation``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from fedkg_spacebio.federation import meta
from fedkg_spacebio.federation import results as rs
from fedkg_spacebio.federation import secure_sum as ss
from fedkg_spacebio.federation.site_de import SiteSummary


class SpecPoolError(ValueError):
    """What arrived doesn't match the job: refuse rather than pool something else."""


# ------------------------------------------------------------------ secure sum


def pool_secure_sum(contributions: Mapping[str, tuple[Mapping, Mapping]], rows: pd.DataFrame,
                    params: ss.Params, expected: Mapping) -> tuple[pd.DataFrame, dict]:
    """Validate every client's masked contribution, sum, decode, and compute FE (+ Q).

    ``contributions``: client → (arrays, metadata) exactly as received. ``expected``: the job's
    own ``job_id``, plan and exclusion hashes, f, m, ``n_rows`` and quantities; every client must
    report the same. Returns the pooled frame (``ncbi_gene``, ``k``, ``fe_*``, and ``q``,
    ``q_p``, ``i2`` when w·y² was sent) and run info (bounds, clamped Q count).
    """
    for client, (arrays, m) in contributions.items():
        if m.get("client") != client:
            raise SpecPoolError(f"contribution claims client {m.get('client')!r} but came from "
                                f"{client!r}")
        ss.validate_contribution(arrays, m, expected)
    totals = ss.aggregate({c: a for c, (a, _) in contributions.items()}, rows, params)
    pooled = ss.fe_from_totals(totals)
    use = pooled["fe_est"].notna()
    b = ss.error_bound(pooled["fe_est"], 1 / pooled["fe_se"] ** 2, pooled["k"],
                       params.fixed_point_bits)
    info = {
        "n_rows": int(len(rows)),
        "n_q_clamped_to_zero": int(pooled.attrs.get("n_q_clamped", 0)),
        "error_bound_max": {k: float(np.nanmax(v[use])) if use.any() else None
                            for k, v in b.items() if k != "q" or "q" in pooled},
        "error_bound_note": "worst-case |difference| from plain pooling due to rounding to "
                            "fixed_point_bits, per gene: eps(1+|est|)/(sum_w - eps) for the "
                            "estimate, eps = k * 2^-(f+1) (secure_sum.error_bound); evaluated "
                            "at the decoded values",
    }
    return pooled, info


# ------------------------------------------------------------------ plain


def pool_plain_spec(summaries: Mapping[str, SiteSummary], rows: pd.DataFrame, flag: str,
                    methods) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Pool the plan's rows from per-site summaries that hold exactly their assigned genes."""
    clients = sorted(rows.columns)
    if sorted(summaries) != clients:
        raise SpecPoolError(f"summaries from {sorted(summaries)}, the job's clients are "
                            f"{clients}; refusing to pool a partial stratum")
    frames = {c: summaries[c].to_frame() for c in clients}
    Y = np.full((len(rows), len(clients)), np.nan)
    V = np.full_like(Y, np.nan)
    for j, c in enumerate(clients):
        f = frames[c]
        assigned = rows.index[rows[c].to_numpy(dtype=bool)]
        if not np.array_equal(np.sort(f.index.to_numpy()), np.sort(assigned.to_numpy())):
            raise SpecPoolError(f"{c} sent {len(f)} genes; the plan assigns it {len(assigned)}. "
                                "A per-site summary must hold exactly its plan genes")
        if not f[flag].all():
            raise SpecPoolError(f"{c} sent a plan gene it didn't flag under {flag}")
        pos = rows.index.get_indexer(f.index)
        Y[pos, j] = f["log2fc"].to_numpy(dtype=np.float64)
        V[pos, j] = f["se"].to_numpy(dtype=np.float64) ** 2
    pooled = meta.pool(Y, V, eb=bool({"v2", "v3"} & set(methods)))
    pooled.insert(0, "ncbi_gene", rows.index.to_numpy(dtype=np.int64))
    return pooled.reset_index(drop=True), frames


# ------------------------------------------------------------------ outputs


def write_outputs(server_dir: Path, pooled: pd.DataFrame, *, methods, rows: pd.DataFrame,
                  site_n: Mapping[str, tuple[int, int]], aggregation: dict, provenance: dict,
                  run_info: dict, frames: Mapping[str, pd.DataFrame] | None = None,
                  symbols: pd.Series | None = None) -> pd.DataFrame:
    """``per_gene.csv.gz``, ``per_gene.schema.json`` and ``run.json`` under ``server_dir``."""
    server_dir = Path(server_dir)
    server_dir.mkdir(parents=True, exist_ok=True)
    table = rs.spec_per_gene_table(pooled, methods, rows, site_n, frames, symbols)
    table.to_csv(server_dir / "per_gene.csv.gz", index=False, float_format="%.17g")
    prov = dict(provenance)
    if "eb_priors" in pooled.attrs:
        prov["eb_priors"] = pooled.attrs["eb_priors"]
    sch = rs.schema(table, prov, methods_run=list(methods), aggregation=aggregation)
    (server_dir / "per_gene.schema.json").write_text(json.dumps(sch, indent=1))
    counts = {"n_genes": int(len(table))}
    for flag in ("fe_hit", "het_flag", "v3_hit", "v2_hit", "re_hit"):
        if flag in table:
            counts[flag] = int(table[flag].sum())
    (server_dir / "run.json").write_text(json.dumps(
        {**run_info, "methods_run": list(methods), "aggregation": aggregation,
         "hits": counts}, indent=1))
    return table


def write_received(received_dir: Path, client: str, arrays: Mapping, meta_: Mapping) -> None:
    """What arrived from one client, exactly: arrays (``.npz``) and metadata."""
    received_dir = Path(received_dir)
    received_dir.mkdir(parents=True, exist_ok=True)
    np.savez(received_dir / f"{client}.npz", **{k: np.asarray(v) for k, v in arrays.items()})
    (received_dir / f"{client}.meta.json").write_text(json.dumps(meta_, indent=1))


__all__ = ["SpecPoolError", "pool_plain_spec", "pool_secure_sum", "write_outputs",
           "write_received"]
