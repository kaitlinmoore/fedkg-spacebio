"""``results.read`` (CONTRACTS §7): read a released phase 3a result, and nothing else.

Release records
---------------
A result is readable only once it has been **released**: allowed by a policy decision and
appended to the release log. The log stores no output values (only who contributed to which
gene), so something has to connect a release ID to its table. That is the release record,
``$FEDKG_WORKDIR/agent/releases/<release id as a file stem>.json``, written by ``job.submit``
right after the log append:

    {release_id, job_id, job_spec_sha256, profile, log_id, stratum, counts_variant,
     inclusion: {rule, min_sites_per_gene}, clients,
     table: {path, sha256}, schema: {path, sha256},   # paths relative to $FEDKG_WORKDIR
     methods_refused: {method: decision ID}}

Before reading, the tool checks that the release is really in the named log, with the same job and
spec hash, and that the table and schema files still hash to the recorded values. A record that
fails either check is refused (``release_not_in_log``, ``table_changed``).

Why not read the existing phase 3a tables directly: they were never released, and a tool that
could read any table on disk would make the policy layer optional.

What it returns
---------------
For one method of the release (default ``fe``): pooled rows only, never a per-site column, whatever
the aggregation (a plain release's table has per-site estimates; the tool doesn't pass them on).
``select`` is ``{"top": {"sort_by", "n"}}`` (the n smallest q, smallest p, or largest |estimate|,
ties by gene ID) or ``{"genes": [NCBIGene CURIEs]}``. At most 50 rows.

Rows: ``ncbi_gene`` (as a CURIE), ``symbol`` (a display label, never evidence), ``k``,
``estimate``, ``se``, ``p``, ``q`` and ``hit`` for estimate methods; ``cochran_q``, ``q_p``, ``i2``
and ``het_flag`` for ``fe_het``. Missing numbers are null. Plus ``n_tested`` and ``n_hits``
over the whole release, ``methods_run``, ``methods_refused`` (method → the decision that refused
it), and ``notes``: the caveats that apply (``agent/notes.py``), each with its ``note:`` ID.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from fedkg_spacebio.agent.envelope import ToolError, envelope
from fedkg_spacebio.agent.idscan import SLUG_SHAPE_RE
from fedkg_spacebio.agent.notes import ReleaseInfo, notes_for
from fedkg_spacebio.agent.tools import Tool, ToolContext
from fedkg_spacebio.ids import RELEASE_ID_RE, file_stem
from fedkg_spacebio.policy.release_log import LogIntegrityError, ReleaseLog

NAME = "results.read"
VERSION = "0.1.0"
MAX_ROWS = 50
GENE_RE = r"^NCBIGene:[1-9][0-9]*$"

# method → (estimate, se, p, q, hit) columns of the schema-2 per-gene table.
ESTIMATE_COLUMNS: dict[str, tuple[str, str, str, str, str]] = {
    "fe": ("fe_est", "fe_se", "fe_p", "fe_q", "fe_hit"),
    "v2": ("fe_est", "v2_se", "v2_p", "v2_q", "v2_hit"),
    "v3": ("v3_est", "v3_sd", "v3_p", "v3_q", "v3_hit"),
    "reml_hksj": ("re_est", "hksj_se", "re_p", "re_q", "re_hit"),
}
HET_COLUMNS = {"cochran_q": "q", "q_p": "q_p", "i2": "i2", "het_flag": "het_flag"}
RECORD_FIELDS = ("release_id", "job_id", "job_spec_sha256", "profile", "log_id", "stratum",
                 "counts_variant", "inclusion", "clients", "table", "schema", "methods_refused")


class Top(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sort_by: Literal["q", "p", "abs_estimate"] = "q"
    n: int = Field(default=10, ge=1, le=MAX_ROWS)


class Select(BaseModel):
    model_config = ConfigDict(extra="forbid")
    top: Top | None = None
    genes: list[str] | None = Field(default=None, min_length=1, max_length=MAX_ROWS)

    @model_validator(mode="after")
    def _one(self):
        if (self.top is None) == (self.genes is None):
            raise ValueError("select exactly one of top or genes")
        if self.genes is not None:
            import re
            bad = [g for g in self.genes if not re.fullmatch(GENE_RE, g)]
            if bad:
                raise ValueError(f"genes must be NCBIGene CURIEs; got {bad[:3]}")
        return self


class ResultsReadInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    release_id: str = Field(pattern=RELEASE_ID_RE.pattern)
    method: Literal["fe", "fe_het", "v2", "v3", "reml_hksj"] = "fe"
    select: Select = Field(default_factory=lambda: Select(top=Top()))


# ------------------------------------------------------------------ release records


def release_record_path(ctx: ToolContext, release_id: str) -> Path:
    return Path(ctx.releases_dir) / f"{file_stem(release_id)}.json"


def write_release_record(ctx: ToolContext, record: dict) -> Path:
    """Write a release record once (``job.submit`` calls this after the log append)."""
    missing = [f for f in RECORD_FIELDS if f not in record]
    if missing:
        raise ValueError(f"release record lacks {missing}")
    path = release_record_path(ctx, record["release_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(record, f, indent=1, sort_keys=True)
        f.write("\n")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_release(ctx: ToolContext, release_id: str) -> tuple[dict, pd.DataFrame, dict]:
    path = release_record_path(ctx, release_id)
    if not path.is_file():
        raise ToolError("unknown_release", f"no release record for {release_id}")
    rec = json.loads(path.read_text())
    if rec.get("release_id") != release_id or any(f not in rec for f in RECORD_FIELDS):
        raise ToolError("unknown_release", f"{path.name} isn't a complete record of "
                                           f"{release_id}")
    log = ReleaseLog(rec["profile"], rec["log_id"], ctx.policy_root)
    try:
        entry = next((e for e in log.entries() if e["release_id"] == release_id), None)
    except LogIntegrityError as exc:
        raise ToolError("release_not_in_log", f"{rec['log_id']} doesn't verify: {exc}") from None
    if entry is None or entry["job_id"] != rec["job_id"] or \
            entry["job_spec_sha256"] != rec["job_spec_sha256"]:
        raise ToolError("release_not_in_log", f"{release_id} isn't in {rec['log_id']} as "
                                              "recorded")
    files = {}
    for key in ("table", "schema"):
        p = Path(ctx.workdir) / rec[key]["path"]
        if not p.is_file() or _sha(p) != rec[key]["sha256"]:
            raise ToolError("table_changed", f"{release_id}: {key} file is missing or changed")
        files[key] = p
    schema = json.loads(files["schema"].read_text())
    if "methods_run" not in schema:
        raise ToolError("table_changed", f"{release_id}: the table's schema lists no "
                                         "methods_run (not a job-spec table)")
    table = pd.read_csv(files["table"], compression="gzip")
    return rec, table, schema


# ------------------------------------------------------------------ rows


def _num(x):
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def _bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return {"True": True, "False": False}.get(x)
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return bool(x)


def _row(r: pd.Series, method: str) -> dict:
    sym = r.get("symbol")
    out = {"ncbi_gene": f"NCBIGene:{int(r['ncbi_gene'])}",
           "symbol": sym if isinstance(sym, str) else None,
           "k": int(r["k"])}
    if method == "fe_het":
        for name, col in HET_COLUMNS.items():
            out[name] = _bool(r.get(col)) if name == "het_flag" else _num(r.get(col))
    else:
        est, se, p, q, hit = ESTIMATE_COLUMNS[method]
        out.update(estimate=_num(r.get(est)), se=_num(r.get(se)), p=_num(r.get(p)),
                   q=_num(r.get(q)), hit=_bool(r.get(hit)))
    return out


def _select(table: pd.DataFrame, method: str, sel: Select) -> tuple[pd.DataFrame, list[str]]:
    if sel.genes is not None:
        wanted = [int(g.split(":")[1]) for g in sel.genes]
        present = table[table["ncbi_gene"].isin(wanted)]
        order = {g: i for i, g in enumerate(wanted)}
        present = present.sort_values("ncbi_gene", key=lambda s: s.map(order))
        missing = [f"NCBIGene:{g}" for g in wanted if g not in set(present["ncbi_gene"])]
        return present, missing
    top = sel.top
    if method == "fe_het":
        if top.sort_by != "p":
            raise ToolError("invalid_param", "fe_het rows sort by p (Cochran's Q p) only")
        key = table["q_p"]
        asc = True
    else:
        est, _, p, q, _ = ESTIMATE_COLUMNS[method]
        key = {"q": table[q], "p": table[p], "abs_estimate": table[est].abs()}[top.sort_by]
        asc = top.sort_by != "abs_estimate"
    t = table.assign(_key=key).dropna(subset=["_key"])
    t = t.sort_values(["_key", "ncbi_gene"], ascending=[asc, True], kind="mergesort")
    return t.head(top.n), []


def results_read(inp: ResultsReadInput, ctx: ToolContext, call_id: str | None) -> dict:
    rec, table, schema = _load_release(ctx, inp.release_id)
    methods_run = list(schema["methods_run"])
    refused = dict(rec.get("methods_refused") or {})
    if inp.method not in methods_run:
        why = (f"; refused by {refused[inp.method]}" if inp.method in refused else "")
        raise ToolError("method_not_run", f"{inp.method} wasn't run in {inp.release_id} "
                                          f"(ran: {methods_run}){why}")
    rows_df, not_in_release = _select(table, inp.method, inp.select)
    rows = [_row(r, inp.method) for _, r in rows_df.iterrows()]

    if inp.method == "fe_het":
        tested = table["q_p"].notna()
        hits = table["het_flag"].map(_bool).fillna(False)
    else:
        _, _, p, _, hit = ESTIMATE_COLUMNS[inp.method]
        tested = table[p].notna()
        hits = table[hit].map(_bool).fillna(False)

    agg = schema.get("aggregation") or {"strategy": "plain"}
    info = ReleaseInfo(
        profile=rec["profile"], stratum=rec["stratum"], counts_variant=rec["counts_variant"],
        inclusion_rule=rec["inclusion"]["rule"],
        min_sites_per_gene=rec["inclusion"]["min_sites_per_gene"],
        clients=frozenset(rec["clients"]), aggregation=agg.get("strategy", "plain"),
        method=inp.method, methods_run=frozenset(methods_run))
    notes = [{"note_id": n.note_id, "text": n.text} for n in notes_for(info)]

    data = {
        "release_id": rec["release_id"], "job_id": rec["job_id"], "profile": rec["profile"],
        "log_id": rec["log_id"], "method": inp.method, "stratum": rec["stratum"],
        "counts_variant": rec["counts_variant"], "clients": list(rec["clients"]),
        "aggregation": {k: agg[k] for k in ("strategy", "fixed_point_bits", "modulus_bits")
                        if k in agg},
        "methods_run": methods_run, "methods_refused": refused,
        "n_tested": int(tested.sum()), "n_hits": int(hits.sum()),
        "k_by_gene": {r["ncbi_gene"]: r["k"] for r in rows},
        "rows": rows, "not_in_release": not_in_release, "notes": notes,
    }
    ids = [rec["release_id"], rec["job_id"], rec["log_id"], *rec["clients"],
           *(r["ncbi_gene"] for r in rows), *not_in_release, *refused.values(),
           *(n["note_id"] for n in notes)]
    for n in notes:  # names the fixed note texts mention (SpaceX-16, SpaceX-8)
        ids += [m.group(0) for m in SLUG_SHAPE_RE.finditer(n["text"])]
    return envelope(NAME, VERSION, call_id=call_id, data=data, ids=ids)


TOOL = Tool(
    name=NAME, version=VERSION, input_model=ResultsReadInput, fn=results_read,
    description=(
        "Read a released phase 3a result by release ID: pooled rows for one method (fe by "
        "default; also fe_het, v2, v3, reml_hksj when they were run), either the top n genes "
        "(by q, p or |estimate|) or named genes (NCBIGene CURIEs). Returns hit counts, which "
        "methods ran or were refused (with the refusing decision), and the caveats that apply, "
        "each with a note ID to cite. Never returns per-site values."),
)
