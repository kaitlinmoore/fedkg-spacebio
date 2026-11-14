"""``kg.query``: named, parameterized queries against the GeneLab KG (CONTRACTS §7; D-033, D-038).

What the tool is for
--------------------
The demo's explanation step: for a gene the federated analysis found, ask the KG (1) what its
human orthologs are and (2) in which other flight-vs-control comparisons the gene was reported
changed. Every answer comes back in the §7 envelope, with every citable ID listed in ``ids``.

Why named templates, not free Cypher (D-033)
--------------------------------------------
An agent that writes its own Cypher can ask anything, including queries whose meaning nobody
reviewed (a confounded comparison, a mirror-image contrast counted twice). A fixed allow-list
(``kg_templates.json``) means every query the agent can run was designed and checked by hand
(WS-14 part 1), and its exact text is pinned by ``cypher_sha256``. The agent chooses only the
template and its parameters, and the parameters are validated here.

Templates
---------
- ``gene_human_orthologs(gene)``: the mouse gene's human orthologs (``IS_ORTHOLOG_MGiG``). Every
  ortholog is returned; with more than one, ``one_to_many`` is set. The KG gives no basis for
  ranking them, so none is picked as "best".
- ``gene_de_comparisons(gene, tissue_group="any", limit=25)``: the **strict** flight-vs-control
  comparisons with a DE edge to the gene: flight vs ground, vivarium or basal control; the same
  tissue on both sides; every other factor identical. Non-strict edges are only counted
  (``n_nonstrict``). Without this scope one gene returns 38–128 rows, mostly mismatched pairs
  from our own input studies (WS-14 design note).

What a DE row means (D-029)
---------------------------
The KG stores a DE edge only when the comparison reported adj_p < 0.1, with log2fc and adj_p but
no standard error. A row says "this comparison reported the gene changed". It is not an effect
size to pool and not a replication. The tool's ``notes`` say so in every result.

Post-processing (deterministic; changes bump ``TOOL_VERSION``)
--------------------------------------------------------------
1. ``not_found`` (gene not in the KG) and ``wrong_taxon`` (not a mouse gene; a human ID would
   otherwise come back as its own ortholog) are errors. A mouse gene with no rows is ``ok``.
2. Rows are sorted by (``adj_p``, ``comparison``) in Python; Neo4j doesn't guarantee the order
   that ``collect`` keeps. It's a **display order only**: adj_p values from different designs,
   sample sizes and platforms don't rank strength of evidence.
3. Each row is labelled:
   - ``relation_to_inputs`` (first match wins): ``phase3a_input`` · ``shares_animals_with_input``
     (OSD-168) · ``same_mission_as_input`` (may share animals; the KG can't tell) ·
     ``other_mission`` · ``no_mission``. Only ``other_mission`` rows are outside our inputs, and
     even those are "reported changed", never corroboration (D-038). Rodent Research missions
     usually take many tissues from the same animals, so "not one of our input studies" is not
     "independent";
   - ``is_liver``: a tissue term or the material is in the explicit list ``LIVER_TERMS``.
     Alternative considered: reason over UBERON ``part_of`` live. Rejected: it needs another
     source, and an explicit list can be audited; WS-14 checked it covers every liver material
     in the KG;
   - ``repeat_of``: the earlier comparison with the same (study, factors, material, log2fc,
     adj_p). OSD-462 reports each result under three comparison IDs; counting it once avoids
     tripling one result.
4. ``tissue_group``, then ``limit``; ``truncated`` says rows were cut.
5. ``summary`` describes the gene across all strict rows, before the tissue filter and the cap.
   ``n_strict`` is the KG's count; the per-relation counts, ``n_liver`` and ``n_studies`` count
   each repeat group once.

The input lists (``PHASE3A_INPUT_STUDIES`` etc.) are written out rather than read from the site
manifests: they label provenance relative to one fixed analysis (phase 3a, D-016/D-022), and a
literal list is easy to audit against ``SITES.md``.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Iterable, Mapping
from functools import cache
from importlib import resources
from typing import Any

from fedkg_spacebio.agent.envelope import ToolError, envelope
from fedkg_spacebio.agent.kg_cache import (
    CacheEntry,
    KgCache,
    cache_key,
    canonical_json,
    offline_from_env,
    sha256_hex,
    utc_now,
)
from fedkg_spacebio.agent.kg_client import (
    KG_MCP_URL,
    KgClient,
    KgQueryError,
    KgTransportError,
)

TOOL = "kg.query"
TOOL_VERSION = "0.1.0"
KG_ASSUMED_VERSION = "SPOKE-GeneLab v0.3.1+20260311 (assumed)"
FINGERPRINT_TEMPLATE = "kg_fingerprint"
MOUSE_TAXON = "10090"

# Phase 3a inputs and their relatives (SITES.md; D-038).
PHASE3A_INPUT_STUDIES = frozenset({"OSD-47", "OSD-48", "OSD-137", "OSD-245", "OSD-379", "OSD-463"})
SHARES_ANIMALS_WITH_INPUT = frozenset({"OSD-168"})
INPUT_MISSIONS = frozenset({"SpaceX-4", "SpaceX-8", "SpaceX-13", "SpaceX-16", "SpaceX-21"})
RELATIONS = (
    "phase3a_input",
    "shares_animals_with_input",
    "same_mission_as_input",
    "other_mission",
    "no_mission",
)
# liver; left lateral lobe of liver; left lobe of liver (D-038). Explicit on purpose.
LIVER_TERMS = ("UBERON:0002107", "UBERON:0006727", "UBERON:0001115")
TISSUE_GROUPS = {"any": None, "liver": LIVER_TERMS}
LIMIT_DEFAULT, LIMIT_MAX = 25, 50

REL_ORTHOLOG = "IS_ORTHOLOG_MGiG"
REL_DE = "MEASURED_DIFFERENTIAL_EXPRESSION_ASmMG"
REL_PERFORMED = "PERFORMED_SpAS"
REL_INVESTIGATED = "INVESTIGATED_ASiA"
REL_CONDUCTED = "CONDUCTED_MIcS"

NOTE_DE_EDGE = (
    "A DE edge means the comparison reported the gene at adj_p < 0.1 (the KG's storage filter). "
    "It is not an effect size to pool and not a replication."
)
NOTE_SAME_MISSION = "`same_mission_as_input` studies may share animals with the inputs."
NOTE_ORDER = (
    "Rows are sorted by adj_p for display only; adj_p values from different designs, sample "
    "sizes and platforms do not rank strength of evidence."
)
NOTE_SYMBOLS = "Symbols are display labels, never evidence."
NOTE_ONE_TO_MANY = (
    "This gene has more than one human ortholog; a claim citing one should say it is one of "
    "n_orthologs."
)

GENE_CURIE_RE = re.compile(r"NCBIGene:([1-9][0-9]{0,9})")
_KG_GENE_RE = re.compile(r"[1-9][0-9]{0,9}")
EDGE_KEY_RE = re.compile(r"kgedge:([A-Za-z_]+)\|([^|\s]+)\|([^|\s]+)")


# -- IDs (CONTRACTS §1) ------------------------------------------------------------------------
def gene_to_kg(curie: str) -> str:
    """``"NCBIGene:13170"`` → ``"13170"`` (the KG's bare string). Anything else is refused.

    Ensembl IDs, symbols, ``FAKE:`` IDs and integers are all ``invalid_param``: the KG is
    NCBI-Gene-only (D-012), and an integer would silently match nothing.
    """
    if not isinstance(curie, str):
        raise ToolError("invalid_param", f"gene must be a string CURIE, got {type(curie).__name__}")
    m = GENE_CURIE_RE.fullmatch(curie)
    if not m:
        raise ToolError("invalid_param", f"gene must look like NCBIGene:<n>, got {curie!r}")
    return m.group(1)


def kg_gene_to_curie(bare: Any) -> str:
    """``"13170"`` → ``"NCBIGene:13170"``; refuses anything that isn't bare digits as a string."""
    if not isinstance(bare, str) or not _KG_GENE_RE.fullmatch(bare):
        raise ToolError("kg_error", f"KG gene identifier is not bare digits: {bare!r}")
    return f"NCBIGene:{bare}"


def edge_key(rel_type: str, start: str, end: str) -> str:
    """The composite key ``kgedge:<REL_TYPE>|<start>|<end>`` (§1), endpoints in our CURIE forms.

    Unique because the KG holds at most one edge of each type between two nodes (checked for every
    type used here on 2026-11-14), and only within one graph snapshot, i.e. with ``kg_version``.
    """
    for part in (rel_type, start, end):
        if not isinstance(part, str) or not part or "|" in part or any(c.isspace() for c in part):
            raise ValueError(f"edge key part must be a non-empty string without '|': {part!r}")
    key = f"kgedge:{rel_type}|{start}|{end}"
    if not EDGE_KEY_RE.fullmatch(key):
        raise ValueError(f"bad edge key {key!r}")
    return key


def parse_edge_key(key: str) -> tuple[str, str, str]:
    """Inverse of ``edge_key``: ``(rel_type, start, end)``."""
    m = EDGE_KEY_RE.fullmatch(key)
    if not m:
        raise ValueError(f"not an edge key: {key!r}")
    return m.group(1), m.group(2), m.group(3)


# -- the allow-list ----------------------------------------------------------------------------
@cache
def load_templates() -> dict[str, dict[str, Any]]:
    """The templates in ``kg_templates.json``, each checked against its recorded hash."""
    text = resources.files("fedkg_spacebio.agent").joinpath("kg_templates.json").read_text(
        encoding="utf-8"
    )
    raw = json.loads(text)
    out: dict[str, dict[str, Any]] = {}
    for name, spec in raw.items():
        if name.startswith("_"):
            continue
        if sha256_hex(spec["cypher"]) != spec["cypher_sha256"]:
            raise RuntimeError(f"template {name}: cypher_sha256 doesn't match its Cypher")
        out[name] = spec
    return out


def agent_templates() -> list[str]:
    return sorted(n for n, s in load_templates().items() if s.get("agent_callable", True))


# -- the store: cache + live client ------------------------------------------------------------
class KgStore:
    """Where ``kg_query`` gets raw KG rows: the disk cache, then (unless offline) the live KG.

    The fingerprint (version signal) is taken live once per store and recorded in the cache
    manifest; offline, the recorded one is used.
    """

    def __init__(
        self,
        cache: KgCache | None = None,
        *,
        offline: bool | None = None,
        client_factory: Callable[[], KgClient] = KgClient,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.cache = cache or KgCache()
        self.offline = offline_from_env() if offline is None else offline
        self._client_factory = client_factory
        self._client: KgClient | None = None
        self._fingerprint: dict[str, Any] | None = None
        self._clock = clock
        self.last_elapsed_ms: float | None = None

    def __enter__(self) -> KgStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    @property
    def client(self) -> KgClient:
        if self.offline:
            raise ToolError("cache_miss", "offline mode: no live KG calls")
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    def _live(self, cypher: str, kg_params: Mapping[str, Any]) -> tuple[int, list[Any]]:
        t0 = self._clock()
        result = self.client.query(cypher, kg_params)
        self.last_elapsed_ms = (self._clock() - t0) * 1000.0
        return result

    def fingerprint(self) -> dict[str, Any]:
        """The current fingerprint record (live once per store, else the manifest's)."""
        if self._fingerprint is None:
            if self.offline:
                self._fingerprint = self.cache.current_fingerprint()
            else:
                spec = load_templates()[FINGERPRINT_TEMPLATE]
                total, rows = self._live(spec["cypher"], {})
                if total != 1:
                    raise ToolError("kg_error", f"fingerprint query returned {total} rows")
                record = {
                    "template": FINGERPRINT_TEMPLATE,
                    "kg_params": {},
                    "cypher_sha256": spec["cypher_sha256"],
                    "fetched": utc_now(),
                    "source": _source(),
                    "total_rows": total,
                    "rows": rows,
                    "fingerprint_sha256": sha256_hex(canonical_json(rows[0])),
                    "elapsed_ms": round(self.last_elapsed_ms or 0.0, 1),
                }
                self.cache.write_fingerprint(record)
                self._fingerprint = record
        return self._fingerprint

    def kg_version(self) -> str:
        return f"{KG_ASSUMED_VERSION}; fp:{self.fingerprint()['fingerprint_sha256'][:12]}"

    def fetch(self, template: str, kg_params: Mapping[str, Any]) -> tuple[CacheEntry, bool]:
        """Raw rows for (template, KG params): ``(entry, cache_hit)``. Offline, a miss raises."""
        spec = load_templates()[template]
        fp = self.fingerprint()["fingerprint_sha256"]
        key = cache_key(template, kg_params, spec["cypher_sha256"], fp)
        entry = self.cache.get(key)
        if entry is not None:
            return entry, True
        if self.offline:
            raise ToolError(
                "cache_miss",
                f"offline mode and no cached {template} result for {dict(kg_params)} "
                f"(fingerprint {fp[:12]})",
            )
        total, rows = self._live(spec["cypher"], kg_params)
        entry = self.cache.put(key, {
            "template": template,
            "kg_params": dict(kg_params),
            "cypher_sha256": spec["cypher_sha256"],
            "fetched": utc_now(),
            "source": _source(),
            "kg_fingerprint_sha256": fp,
            "total_rows": total,
            "rows": rows,
            "elapsed_ms": round(self.last_elapsed_ms or 0.0, 1),
        })
        return entry, False


def _source() -> str:
    return f"mcp-genelab `query` tool, direct POST to {KG_MCP_URL} (fedkg kg_client)"


# -- parameter validation ----------------------------------------------------------------------
_PARAMS = {
    "gene_human_orthologs": {"gene"},
    "gene_de_comparisons": {"gene", "tissue_group", "limit"},
}


def validate(template: Any, params: Any) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Check the call; return ``(template, tool params with defaults, KG params)``."""
    if not isinstance(template, str) or template not in agent_templates():
        raise ToolError(
            "invalid_template",
            f"unknown template {template!r}; allowed: {', '.join(agent_templates())}",
        )
    if not isinstance(params, Mapping):
        raise ToolError("invalid_param", "params must be an object")
    extra = set(params) - _PARAMS[template]
    if extra:
        raise ToolError("invalid_param", f"unknown parameter(s) for {template}: {sorted(extra)}")
    if "gene" not in params:
        raise ToolError("invalid_param", "missing parameter: gene")
    bare = gene_to_kg(params["gene"])
    out: dict[str, Any] = {"gene": params["gene"]}
    if template == "gene_de_comparisons":
        tg = params.get("tissue_group", "any")
        if not isinstance(tg, str) or tg not in TISSUE_GROUPS:
            raise ToolError("invalid_param", f"tissue_group must be one of {sorted(TISSUE_GROUPS)}")
        limit = params.get("limit", LIMIT_DEFAULT)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= LIMIT_MAX:
            raise ToolError("invalid_param", f"limit must be an integer from 1 to {LIMIT_MAX}")
        out.update(tissue_group=tg, limit=limit)
    return template, out, {"gene": bare}  # always a string: an integer matches nothing


# -- post-processing ---------------------------------------------------------------------------
def _check_gene_row(rows: list[Any], bare: str) -> dict[str, Any]:
    if not rows:
        raise ToolError("not_found", f"NCBIGene:{bare} is not a gene node in the KG")
    first = rows[0]
    if any(r.get("mgene") != bare for r in rows):
        raise ToolError("kg_error", "KG rows are for a different gene than requested")
    if first.get("mgene_taxon") != MOUSE_TAXON:
        raise ToolError(
            "wrong_taxon",
            f"NCBIGene:{bare} is taxon {first.get('mgene_taxon')!r}, not mouse ({MOUSE_TAXON}); "
            "the templates start from a mouse gene",
        )
    return first


def orthologs_data(rows: list[Any], bare: str) -> tuple[dict[str, Any], list[str]]:
    """``data`` and ``ids`` for ``gene_human_orthologs``."""
    first = _check_gene_row(rows, bare)
    gene = kg_gene_to_curie(bare)
    orth = sorted((r for r in rows if r.get("gene") is not None), key=lambda r: int(r["gene"]))
    items = []
    for r in orth:
        human = kg_gene_to_curie(r["gene"])
        items.append({
            "gene": human,
            "gene_symbol": r.get("gene_symbol"),
            "edge": edge_key(REL_ORTHOLOG, gene, human),
        })
    if len({i["gene"] for i in items}) != len(items):
        raise ToolError("kg_error", "duplicate ortholog rows (composite edge key not unique)")
    notes = [NOTE_SYMBOLS] + ([NOTE_ONE_TO_MANY] if len(items) > 1 else [])
    data = {
        "template": "gene_human_orthologs",
        "gene": gene,
        "gene_symbol": first.get("mgene_symbol"),
        "orthologs": items,
        "n_orthologs": len(items),
        "one_to_many": len(items) > 1,
        "notes": notes,
    }
    ids = [gene] + [x for i in items for x in (i["gene"], i["edge"])]
    return data, ids


def relation_to_inputs(study: str | None, missions: Iterable[str]) -> str:
    """The five-level provenance label (first match wins; D-038)."""
    missions = set(missions)
    if study in PHASE3A_INPUT_STUDIES:
        return "phase3a_input"
    if study in SHARES_ANIMALS_WITH_INPUT:
        return "shares_animals_with_input"
    if missions & INPUT_MISSIONS:
        return "same_mission_as_input"
    if missions:
        return "other_mission"
    return "no_mission"


def is_liver(tissues: Iterable[str], material_id: str | None) -> bool:
    terms = set(tissues) | ({material_id} if material_id else set())
    return bool(terms & set(LIVER_TERMS))


def _sort_key(r: Mapping[str, Any]) -> tuple[float, str]:
    p = r.get("adj_p")
    return (math.inf if p is None else float(p), r["comparison"])


def comparisons_data(
    rows: list[Any], bare: str, tissue_group: str = "any", limit: int = LIMIT_DEFAULT
) -> tuple[dict[str, Any], list[str]]:
    """``data`` and ``ids`` for ``gene_de_comparisons``."""
    first = _check_gene_row(rows, bare)
    if len(rows) != 1:
        raise ToolError("kg_error", f"expected one KG row per gene, got {len(rows)}")
    gene = kg_gene_to_curie(bare)
    strict = list(first.get("rows") or [])
    if first.get("n_strict") != len(strict):
        raise ToolError("kg_error", "n_strict doesn't match the rows returned")
    if len({r["comparison"] for r in strict}) != len(strict):
        raise ToolError("kg_error", "a comparison appears twice (composite edge key not unique)")

    out: list[dict[str, Any]] = []
    seen: dict[tuple, str] = {}
    for r in sorted(strict, key=_sort_key):
        study = r.get("study")
        missions = sorted(r.get("missions") or [])
        tissues = sorted(r.get("tissues") or [])
        comp = r["comparison"]
        rep_key = (
            study, tuple(r.get("factors_1") or ()), tuple(r.get("factors_2") or ()),
            r.get("material_id"), r.get("log2fc"), r.get("adj_p"),
        )
        repeat_of = seen.get(rep_key)
        if repeat_of is None:
            seen[rep_key] = comp
        edges: dict[str, Any] = {
            "de": edge_key(REL_DE, comp, gene),
            "performed": edge_key(REL_PERFORMED, study, comp) if study else None,
            "investigated": [edge_key(REL_INVESTIGATED, comp, t) for t in tissues],
            "conducted": [edge_key(REL_CONDUCTED, m, study) for m in missions] if study else [],
        }
        out.append({
            "comparison": comp,
            "study": study,
            "missions": missions,
            "tissues": tissues,
            "material_id": r.get("material_id"),
            "material_name": r.get("material_name"),
            "technology": r.get("technology"),
            "control": r.get("control"),
            "factors_1": r.get("factors_1"),
            "factors_2": r.get("factors_2"),
            "log2fc": r.get("log2fc"),
            "adj_p": r.get("adj_p"),
            "relation_to_inputs": relation_to_inputs(study, missions),
            "is_liver": is_liver(tissues, r.get("material_id")),
            "repeat_of": repeat_of,
            "edges": edges,
        })

    unique = [r for r in out if r["repeat_of"] is None]
    by_relation = {k: 0 for k in RELATIONS}
    for r in unique:
        by_relation[r["relation_to_inputs"]] += 1
    terms = TISSUE_GROUPS[tissue_group]
    in_group = out if terms is None else [r for r in out if r["is_liver"]]
    returned = in_group[:limit]

    summary = {
        "n_strict": len(out),
        "n_nonstrict": first.get("n_nonstrict"),
        "n_repeats": len(out) - len(unique),
        "n_studies": len({r["study"] for r in out if r["study"]}),
        "n_strict_by_relation": by_relation,
        "n_liver": sum(r["is_liver"] for r in unique),
        "n_in_tissue_group": len(in_group),
        "n_returned": len(returned),
    }
    data = {
        "template": "gene_de_comparisons",
        "gene": gene,
        "gene_symbol": first.get("mgene_symbol"),
        "scope": "flight_vs_control_strict",
        "tissue_group": tissue_group,
        "tissue_terms": list(terms) if terms is not None else None,
        "limit": limit,
        "truncated": len(in_group) > limit,
        "summary": summary,
        "rows": returned,
        "notes": [NOTE_DE_EDGE, NOTE_SAME_MISSION, NOTE_ORDER, NOTE_SYMBOLS],
    }

    ids: list[str] = [gene]
    for r in returned:
        ids += [r["comparison"]] + ([r["study"]] if r["study"] else [])
        ids += r["tissues"] + ([r["material_id"]] if r["material_id"] else [])
        ids += r["missions"]
        e = r["edges"]
        ids += [e["de"]] + ([e["performed"]] if e["performed"] else [])
        ids += e["investigated"] + e["conducted"]
    # A repeat always follows its original in the sort, and shares its tissue, so the original
    # is returned too; check rather than assume, since a dangling ID would be uncitable.
    returned_ids = {r["comparison"] for r in returned}
    for r in returned:
        if r["repeat_of"] is not None and r["repeat_of"] not in returned_ids:
            raise ToolError("kg_error", f"repeat_of {r['repeat_of']} not among the returned rows")
    if terms is not None:
        ids += list(terms)  # the filter's terms are in data, so they're listed too
    return data, ids


_POSTPROCESS = {
    "gene_human_orthologs": lambda rows, bare, p: orthologs_data(rows, bare),
    "gene_de_comparisons": lambda rows, bare, p: comparisons_data(
        rows, bare, p["tissue_group"], p["limit"]
    ),
}


# -- the tool ----------------------------------------------------------------------------------
def kg_query(
    template: Any, params: Any, *, store: KgStore, call_id: str | None = None
) -> dict[str, Any]:
    """Run one named template and return the §7 envelope. Never raises for a bad call."""
    cache_info = None
    try:
        name, tool_params, kg_params = validate(template, params)
        entry, hit = store.fetch(name, kg_params)
        cache_info = {"hit": hit, "key": f"sha256:{entry.key}", "fetched": entry.fetched}
        data, ids = _POSTPROCESS[name](entry.rows, kg_params["gene"], tool_params)
        data["params"] = tool_params
        data["cypher_sha256"] = load_templates()[name]["cypher_sha256"]
        data["kg_version"] = store.kg_version()
        return envelope(TOOL, TOOL_VERSION, call_id=call_id, data=data, ids=ids, cache=cache_info)
    except ToolError as err:
        return envelope(TOOL, TOOL_VERSION, call_id=call_id, cache=cache_info, error=err)
    except KgTransportError as exc:
        err = ToolError("transport_error", str(exc))
        return envelope(TOOL, TOOL_VERSION, call_id=call_id, cache=cache_info, error=err)
    except KgQueryError as exc:
        err = ToolError("kg_error", str(exc))
        return envelope(TOOL, TOOL_VERSION, call_id=call_id, cache=cache_info, error=err)
    except ValueError as exc:  # an identifier the KG returned can't form an edge key
        err = ToolError("kg_error", f"unusable KG identifier: {exc}")
        return envelope(TOOL, TOOL_VERSION, call_id=call_id, cache=cache_info, error=err)
