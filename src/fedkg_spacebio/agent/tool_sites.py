"""``sites.list`` (CONTRACTS §7): the site registry, as the agent sees it.

What it returns
---------------
One entry per FLARE client of the run's policy profile, from the client's manifest (§3): its
site, stratum, tissue term, animals per arm, studies, DE design and policy ID, the SHA-256 of its
manifest file, and the SHA-256 of the inclusion flags it released before any job
(``inclusion_flags_sha256``, §4). The agent uses these facts to decide which clients are
compatible with the question (the decision itself is a ``decision`` record of kind
``compatibility``, so the rule that decided is in the trace) and to fill a job spec's
provenance.

Filters: ``stratum`` (``iss`` or ``earth``) and ``tissue``: ``liver`` accepts the three liver
terms (liver, left lateral lobe of liver, and the lobe-of-liver term) because the census records
SpaceX-16 and SpaceX-21 as "left lateral lobe of liver"; a filter on only the main liver term
would silently drop them. Any other value must be one exact ``UBERON:`` term.

The inclusion-flag hash is **recomputed** from the released flags and compared with the value the
site recorded. A mismatch is an error, not a warning: the per-gene policy checks and the gene plan
are built from these flags, so a hash that doesn't describe them would make every later check
about different data than it claims. Only the gene IDs and the three ``tested_*`` flags are read
from the released file (the open-profile run's copy also holds per-site estimates, which this
tool never opens; they aren't site metadata).

A client whose flags were never released has ``inclusion_flags_sha256: null`` and
``flags_released: false``; it can't be in a per-gene plan.

``ids`` (what a claim may cite): client and site names, OSD accessions, UBERON terms and policy
IDs. Manifest hashes are not IDs.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator

from fedkg_spacebio.agent.envelope import ToolError, envelope
from fedkg_spacebio.agent.tools import Tool, ToolContext
from fedkg_spacebio.sites.inclusion import FLAGS, inclusion_flags_sha256

NAME = "sites.list"
VERSION = "0.1.0"
LIVER_TERMS = ("UBERON:0002107", "UBERON:0006727", "UBERON:0001115")
MANIFEST_FIELDS = ("client", "site", "stratum", "policy_id", "tissue_uberon", "n_flight",
                   "n_ground", "studies", "design", "counts_variant")
UBERON_RE = re.compile(r"UBERON:\d{7}")


class SitesListInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stratum: Literal["iss", "earth"] | None = None
    tissue: str | None = None

    @field_validator("tissue")
    @classmethod
    def _tissue(cls, v):
        if v is not None and v != "liver" and not UBERON_RE.fullmatch(v):
            raise ValueError("tissue is 'liver' or one UBERON term (UBERON:<7 digits>)")
        return v


def _manifests(ctx: ToolContext) -> list[tuple[dict, str]]:
    mdir = ctx.manifest_dir
    paths = sorted(mdir.glob("*.json")) if mdir.is_dir() else []
    if not paths:
        raise ToolError("no_manifests", f"no site manifests for profile {ctx.profile!r} "
                                        f"({ctx.counts_variant} counts)")
    out = []
    for p in paths:
        raw = p.read_bytes()
        m = json.loads(raw)
        missing = [f for f in MANIFEST_FIELDS if f not in m]
        if missing:
            raise ToolError("manifest_incomplete", f"{p.name} lacks {missing}")
        if m["client"] != p.stem:
            raise ToolError("manifest_incomplete", f"{p.name}: client {m['client']!r} doesn't "
                                                   "match the file name")
        out.append((m, hashlib.sha256(raw).hexdigest()))
    return out


def _released_flags_sha(ctx: ToolContext, client: str, stratum: str) -> str | None:
    sent = ctx.released_flags_dir(stratum)
    npz, meta = sent / f"{client}.npz", sent / f"{client}.meta.json"
    if not npz.is_file() or not meta.is_file():
        return None
    recorded = json.loads(meta.read_text()).get("inclusion_flags_sha256")
    with np.load(npz) as z:
        flags = pd.DataFrame({k: z[k] for k in ("ncbi_gene", *FLAGS)})
    actual = inclusion_flags_sha256(flags)
    if recorded != actual:
        raise ToolError("flags_mismatch", f"{client}: released flags hash to {actual[:12]}…, "
                                          f"but the site recorded {str(recorded)[:12]}…")
    return actual


def sites_list(inp: SitesListInput, ctx: ToolContext, call_id: str | None) -> dict:
    wanted = (set(LIVER_TERMS) if inp.tissue == "liver"
              else {inp.tissue} if inp.tissue else None)
    clients, ids = [], []
    for m, sha in _manifests(ctx):
        if inp.stratum and m["stratum"] != inp.stratum:
            continue
        if wanted is not None and m["tissue_uberon"] not in wanted:
            continue
        flags_sha = _released_flags_sha(ctx, m["client"], m["stratum"])
        clients.append({
            "client": m["client"], "site": m["site"], "stratum": m["stratum"],
            "tissue_uberon": m["tissue_uberon"], "n_flight": m["n_flight"],
            "n_ground": m["n_ground"], "studies": list(m["studies"]), "design": m["design"],
            "policy_id": m["policy_id"], "counts_variant": m["counts_variant"],
            "manifest_sha256": sha, "inclusion_flags_sha256": flags_sha,
            "flags_released": flags_sha is not None,
        })
        ids += [m["client"], m["site"], *m["studies"], m["tissue_uberon"], m["policy_id"]]
    data = {"profile": ctx.profile, "counts_variant": ctx.counts_variant,
            "filters": {"stratum": inp.stratum, "tissue": inp.tissue},
            "tissue_terms": list(LIVER_TERMS) if inp.tissue == "liver" else
                            ([inp.tissue] if inp.tissue else None),
            "clients": clients}
    if inp.tissue:
        ids += data["tissue_terms"]
    return envelope(NAME, VERSION, call_id=call_id, data=data, ids=ids)


TOOL = Tool(
    name=NAME, version=VERSION, input_model=SitesListInput, fn=sites_list,
    description=(
        "List the federation's clients (one per mission and stratum) for this run's policy "
        "profile: site, stratum (iss or earth), tissue term, animals per arm, studies, DE "
        "design, the site's policy ID, and the hash of the inclusion flags it released. "
        "Optional filters: stratum; tissue ('liver' accepts all liver terms, or one UBERON "
        "term)."),
)
