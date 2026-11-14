"""Site manifests for phase 3a: which samples and files each FLARE client may use.

What a manifest is
------------------
One JSON file per FLARE client, i.e. per (site, stratum) row of D-020. It lists the study (or
studies) the client draws from, the count file to read, every sample it may use with its arm and
covariates, and the DE design. The client opens only ``<manifest_dir>/<client>.json`` and the
files named in it (``sites/loader.py``).

**Framing.** In a real federation each site would declare its own manifest. Here the sites are
simulated over public OSDR data, so the coordinator builds the manifests from the public census
(``$FEDKG_WORKDIR/census/``). That is metadata, not site data. The coordinator never opens a
count file.

How a client's samples are chosen
---------------------------------
1. ``CLIENTS`` names, for each client, the census rows it uses, by study and by the flight arm's
   non-spaceflight factor values (the census lists them after the arm label). A spec must match
   **exactly one** Ground-Control census row with ``location_aware_match = True`` at that site.
   Anything else raises. That way a renamed factor fails loudly instead of silently picking a
   different comparison.
2. The row's animals (normalized OSDR source names) are mapped to sample names through the
   census sample table.
3. If an animal has several samples (OSD-463's technical replicates), keep only the one in
   GeneLab's SampleTable (``techrep1``, D-022). Exactly one sample per animal must remain.
4. Across clients, no sample and no animal may appear twice (``check_disjoint``). OSD-245 and
   OSD-379 each feed one ISS client and one Earth client from the **same** count file. Their
   arms are different animals, and this check proves it.

Which policy governs the client (CONTRACTS §3, §5)
--------------------------------------------------
Each manifest names its site's policy, ``policy_id`` = ``pol:<site>:<version>`` (§1). A site's
policy covers all of its clients (SpaceX-16's ISS and Earth clients share one). The default is
version 1, the open-profile policy of every testbed site. The restricted demo profile passes
``policy_ids={"SpaceX-16": "pol:SpaceX-16:2"}``. Because the policy is part of what a site
declares, a profile change also changes the manifest's SHA-256, which job specs record
(§6 ``provenance.manifest_sha256``). The policies themselves are invented for the testbed
(§5.5); the manifest only points at one. Evaluating it is the policy layer's job, not this one.

Designs (D-022): SpaceX-4 uses ``~ study + condition`` (two RR-1 cohorts). SpaceX-16 uses
``~ age + condition`` because both age groups appear in both arms in both strata. All others use
``~ condition``. Each client uses only its own flight and ground samples, with no all-group fits
(Kaitlin, WS-07), so a mission's ISS and Earth clients stay independent.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from fedkg_spacebio.census.osdr_client import default_workdir


@dataclass(frozen=True)
class RowSpec:
    """One census row a client uses, plus the covariate values its samples carry."""

    osd_id: str
    flight_factors: tuple[str, ...]  # non-spaceflight factor values of the flight arm, as listed
    covariates: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ClientSpec:
    client: str
    site: str  # mission (D-014)
    stratum: str  # "iss" or "earth" (D-020)
    design: str
    rows: tuple[RowSpec, ...]


CLIENTS: dict[str, ClientSpec] = {c.client: c for c in [
    ClientSpec("SpaceX-4", "SpaceX-4", "iss", "~ study + condition", (
        RowSpec("OSD-47", (), {"study": "OSD-47"}),
        RowSpec("OSD-48", ("Carcass",), {"study": "OSD-48"}),
    )),
    ClientSpec("SpaceX-8", "SpaceX-8", "iss", "~ condition", (RowSpec("OSD-137", ()),)),
    ClientSpec("SpaceX-13-ISST", "SpaceX-13", "iss", "~ condition", (
        RowSpec("OSD-245", ("Carcass", "~60 day", "On ISS")),
    )),
    ClientSpec("SpaceX-16-ISST", "SpaceX-16", "iss", "~ age + condition", (
        RowSpec("OSD-379", ("10 to 12 week", "Carcass", "22 to 23 day", "On ISS"),
                {"age": "10 to 12 week"}),
        RowSpec("OSD-379", ("32 week", "Carcass", "22 to 23 day", "On ISS"), {"age": "32 week"}),
    )),
    ClientSpec("SpaceX-13-LAR", "SpaceX-13", "earth", "~ condition", (
        RowSpec("OSD-245", ("Upon euthanasia", "~30 day", "On Earth")),
    )),
    ClientSpec("SpaceX-16-LAR", "SpaceX-16", "earth", "~ age + condition", (
        RowSpec("OSD-379", ("10 to 12 week", "Upon euthanasia", "~40 day", "On Earth"),
                {"age": "10 to 12 week"}),
        RowSpec("OSD-379", ("32 week", "Upon euthanasia", "~40 day", "On Earth"),
                {"age": "32 week"}),
    )),
    ClientSpec("SpaceX-21", "SpaceX-21", "earth", "~ condition", (RowSpec("OSD-463", ()),)),
]}

STRATA = ("iss", "earth")
COUNTS_VARIANTS = {"standard": "counts_file", "rrnarm": "counts_file_rrnarm"}


def clients_in(stratum: str, exclude: tuple[str, ...] = (),
               clients: Mapping[str, ClientSpec] = CLIENTS) -> list[str]:
    if stratum not in STRATA:
        raise ValueError(f"unknown stratum {stratum!r}; expected one of {STRATA}")
    names = [c for c, spec in clients.items() if spec.stratum == stratum and c not in exclude]
    unknown = set(exclude) - set(clients)
    if unknown:
        raise ValueError(f"cannot exclude unknown clients: {sorted(unknown)}")
    return names


def default_policy_id(site: str) -> str:
    """The open-profile policy of a testbed site: version 1."""
    return f"pol:{site}:1"


def check_policy_id(policy_id: str, site: str) -> str:
    """Return ``policy_id`` if it is ``pol:<site>:<version>`` for this site, else raise.

    A pointer to another site's policy would let one site's rules silently govern another's
    data, so the site part has to match exactly. Versions are positive integers.
    """
    if not isinstance(policy_id, str) or not re.fullmatch(
            rf"pol:{re.escape(site)}:[1-9][0-9]*", policy_id):
        raise ValueError(f"policy_id {policy_id!r} is not pol:{site}:<version> for site {site!r}")
    return policy_id


def _as_list(value) -> list:
    return json.loads(value) if isinstance(value, str) else list(value)


def find_row(census: pd.DataFrame, site: str, spec: RowSpec) -> pd.Series:
    """The single Ground-Control, location-aware census row this spec names."""
    cand = census[(census["osd_id"] == spec.osd_id) & (census["site"] == site)
                  & (census["control_type"] == "Ground Control")
                  & census["location_aware_match"].astype(bool)]
    hits = [r for _, r in cand.iterrows()
            if tuple(_as_list(r["factors_flight"])[1:]) == spec.flight_factors]
    if len(hits) != 1:
        raise ValueError(f"{site}/{spec.osd_id} {spec.flight_factors}: expected 1 census row, "
                         f"found {len(hits)}")
    return hits[0]


def _sample_table_names(path: Path) -> set[str]:
    st = pd.read_csv(path)
    return set(st.iloc[:, 0].astype(str))


def build_manifest(spec: ClientSpec, census: pd.DataFrame, samples: pd.DataFrame,
                   data_root: Path, counts_variant: str = "standard",
                   policy_id: str | None = None) -> dict:
    """The manifest dict for one client. ``samples`` is the census sample table.

    ``policy_id`` defaults to ``default_policy_id(spec.site)``.
    """
    policy_id = check_policy_id(policy_id or default_policy_id(spec.site), spec.site)
    if counts_variant not in COUNTS_VARIANTS:
        raise ValueError(f"unknown counts variant {counts_variant!r}")
    sources: dict[str, dict] = {}
    rows_used = []
    for rs in spec.rows:
        row = find_row(census, spec.site, rs)
        osd = rs.osd_id
        counts_file = row[COUNTS_VARIANTS[counts_variant]]
        st_path = data_root / osd / sample_table_name(row["counts_file"])
        in_de = _sample_table_names(st_path)
        src = sources.setdefault(osd, {"osd_id": osd, "counts_file": f"{osd}/{counts_file}",
                                       "samples": []})
        if src["counts_file"] != f"{osd}/{counts_file}":
            raise ValueError(f"{spec.client}: two count files for {osd}")
        study_samples = samples[samples["osd_id"] == osd]
        for arm, animals_col, label in (("flight", "flight_animals", "Space Flight"),
                                        ("ground", "control_animals", "Ground Control")):
            for animal in _as_list(row[animals_col]):
                hits = study_samples[(study_samples["source_key"] == animal)
                                     & (study_samples["spaceflight_label"] == label)]
                names = sorted(set(hits["sample_name"]))
                if len(names) > 1:
                    names = [n for n in names if n in in_de]
                if len(names) != 1:
                    raise ValueError(f"{spec.client}: animal {animal} in {osd} maps to "
                                     f"{len(names)} samples after the SampleTable filter")
                src["samples"].append({"sample": names[0], "animal": f"{osd}:{animal}",
                                       "arm": arm, **dict(rs.covariates)})
        rows_used.append({"osd_id": osd, "comparison_id": row["comparison_id"],
                          "genelab_contrast": row["genelab_contrast"],
                          "factors_flight": _as_list(row["factors_flight"]),
                          "factors_control": _as_list(row["factors_control"]),
                          "strict_match": bool(row["strict_match"]),
                          "location_aware_match": bool(row["location_aware_match"]),
                          "euthanasia_location_flight": row["euthanasia_location_flight"],
                          "euthanasia_location_flight_source":
                              row["euthanasia_location_flight_source"],
                          "de_file": row["de_file"]})
    all_samples = [s for src in sources.values() for s in src["samples"]]
    n_flight = sum(s["arm"] == "flight" for s in all_samples)
    n_ground = sum(s["arm"] == "ground" for s in all_samples)
    return {
        "client": spec.client, "site": spec.site, "stratum": spec.stratum,
        "policy_id": policy_id, "design": spec.design,
        "contrast": ["condition", "flight", "ground"],
        "counts_variant": counts_variant, "studies": sorted(sources),
        "sources": list(sources.values()), "census_rows": rows_used,
        "n_flight": n_flight, "n_ground": n_ground,
    }


def sample_table_name(counts_file: str) -> str:
    """``GLDS-47_rna_seq_RSEM_Unnormalized_Counts_GLbulkRNAseq.csv`` → its SampleTable name."""
    prefix = counts_file.split("_rna_seq_")[0]
    return f"{prefix}_rna_seq_SampleTable_GLbulkRNAseq.csv"


def check_disjoint(manifests: list[dict]) -> None:
    """Raise if any sample or animal belongs to more than one client."""
    seen_sample: dict[tuple[str, str], str] = {}
    seen_animal: dict[str, str] = {}
    for m in manifests:
        for src in m["sources"]:
            for s in src["samples"]:
                key = (src["osd_id"], s["sample"])
                for table, k in ((seen_sample, key), (seen_animal, s["animal"])):
                    if k in table and table[k] != m["client"]:
                        raise ValueError(f"{k} is in both {table[k]} and {m['client']}")
                    table[k] = m["client"]


def write_manifests(out_dir: Path | None = None, workdir: Path | None = None,
                    counts_variant: str = "standard",
                    clients: Mapping[str, ClientSpec] = CLIENTS,
                    policy_ids: Mapping[str, str] | None = None) -> dict[str, Path]:
    """Build, check and write every client's manifest. Returns {client: path}.

    ``policy_ids`` maps **site** (not client) → policy ID, for sites that don't use the default
    version 1. Naming a site that has no client here is an error, so a typo can't silently
    leave a site on its default policy.
    """
    policy_ids = dict(policy_ids or {})
    unknown = set(policy_ids) - {spec.site for spec in clients.values()}
    if unknown:
        raise ValueError(f"policy_ids names sites with no client: {sorted(unknown)}")
    workdir = workdir or default_workdir()
    census = pd.read_csv(workdir / "census" / "liver_census.csv")
    samples = pd.read_csv(workdir / "census" / "liver_samples.csv")
    data_root = workdir / "data" / "osdr"
    manifests = [build_manifest(spec, census, samples, data_root, counts_variant,
                                policy_ids.get(spec.site))
                 for spec in clients.values()]
    check_disjoint(manifests)
    out_dir = out_dir or workdir / "phase3a" / "manifests" / counts_variant
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for m in manifests:
        p = out_dir / f"{m['client']}.json"
        p.write_text(json.dumps(m, indent=1))
        paths[m["client"]] = p
    return paths
