"""The federated job spec (CONTRACTS §6.1): parsing, canonical form and hash.

A job spec is what the agent submits and what the policy check reads. This module answers one
question: **is this a well-formed spec?** Required fields are present, types are right, IDs have
the right syntax, nothing unknown appears at any level, and the fields that depend on the task
agree with it (weighting, rounds, split).

What it deliberately does *not* decide is whether a well-formed spec makes sense against the
contract tables and the site registry: an unknown client or method, outputs that don't cover a
method, a hash that doesn't match a file. Those are the policy check's ``spec#…`` stage
(CONTRACTS §5.4). The split matters for what the user sees: a malformed spec can't be evaluated
at all (``JobSpecError``, §7 ``ok: false``), while a wrong one is *denied* with a named reason
and gets a decision record. Unknown method names and output kinds therefore parse here, so the
check can deny them by name.

Choices
-------
- **pydantic with ``extra="forbid"`` at every level.** A misspelt optional field (``wieghting``)
  would otherwise be dropped silently and the job would run on defaults. pydantic also reports
  every problem at once, which is what a person fixing a spec by hand wants. Alternative: JSON
  Schema (a second language to maintain next to the Python that uses the fields).
- **Integers are strict** (``rounds: "2"`` or ``2.0`` is an error, not coerced), because a
  silently coerced value is a value nobody wrote.
- **The spec hash** is the SHA-256 of canonical JSON: sorted keys, ``,`` and ``:`` separators,
  UTF-8, dates as ISO strings, **null fields left out**, and without ``policy_decision`` (the
  decision points at the spec, so the spec's hash can't include the decision). Leaving nulls
  out means a YAML file that writes ``split: null`` and one that omits ``split`` hash the same.
  Alternative: hashing the file bytes (would make comments, key order and YAML vs JSON matter).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Annotated, Literal, get_args

import yaml
from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from fedkg_spacebio.ids import CLIENT_RE, DECISION_ID_RE, JOB_ID_RE, is_sha256

SCHEMA = "job-spec/1"

Task = Literal["de_meta", "clock_plane", "fedavg_classifier"]
OutputKind = Literal["site_metadata", "masked_contribution", "per_site_summary", "model_update",
                     "per_sample"]
WeightingScheme = Literal["inverse_variance", "sample_count", "equal", "custom"]
Stratum = Literal["iss", "earth"]

TASKS: tuple[str, ...] = get_args(Task)
OUTPUT_KINDS: tuple[str, ...] = get_args(OutputKind)
WEIGHTING_SCHEMES: tuple[str, ...] = get_args(WeightingScheme)


@dataclass(frozen=True)
class MethodNeed:
    """CONTRACTS §5.2: the task a method belongs to, and the output kinds that serve it."""

    task: str
    outputs: frozenset[str]


# §5.2. ``per_sample`` carries everything any method could use, so it "covers" every method;
# the project policy then refuses it by name. ``per_site_summary`` covers ``fe`` too (FE is a
# function of each site's estimate and SE, so the server can compute it from the summaries).
# ``fe_het`` accepts ``masked_contribution`` since WS-13 (D-039): Cochran's Q and I² across all
# clients are functions of three totals, Σw·y², Σw·y and Σw, which the secure sum delivers
# without any site's estimate. Q over a subset that leaves a site out is still a new release
# checked by differencing. The random-effects methods stay per-site only (D-039's caution).
METHODS: dict[str, MethodNeed] = {
    "fe": MethodNeed("de_meta", frozenset({"masked_contribution", "per_site_summary",
                                           "per_sample"})),
    "fe_het": MethodNeed("de_meta", frozenset({"masked_contribution", "per_site_summary",
                                               "per_sample"})),
    **{m: MethodNeed("de_meta", frozenset({"per_site_summary", "per_sample"}))
       for m in ("reml_hksj", "v2", "v3")},
    "clock_plane": MethodNeed("clock_plane", frozenset({"per_site_summary", "per_sample"})),
    "fedavg": MethodNeed("fedavg_classifier", frozenset({"model_update", "per_sample"})),
}

# §5.4 spec#weighting: the schemes each task accepts.
TASK_WEIGHTING: dict[str, frozenset[str]] = {
    "de_meta": frozenset({"inverse_variance"}),
    "clock_plane": frozenset({"inverse_variance"}),
    "fedavg_classifier": frozenset({"sample_count", "equal", "custom"}),
}

# Tasks whose result is pooled gene by gene. The others are checked as a single item (§5.3).
PER_GENE_TASKS: frozenset[str] = frozenset({"de_meta"})


class JobSpecError(ValueError):
    """A spec that is malformed (not merely wrong). ``problems`` lists every one found."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("malformed job spec:\n  " + "\n  ".join(self.problems))


# ------------------------------------------------------------------ strict YAML


class _StrictLoader(yaml.SafeLoader):
    """``yaml.safe_load``, but a key repeated in one mapping is an error.

    PyYAML keeps the last value of a repeated key without a word, so ``min_sites: 3`` followed
    later by ``min_sites: 1`` would quietly mean 1. In a policy or a job spec that has to fail.
    """


def _construct_mapping(loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False):
    loader.flatten_mapping(node)
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark)
        seen.add(key)
    return loader.construct_mapping(node, deep=deep)


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def load_yaml_strict(text: str):
    """Parse YAML (or JSON) text with the safe loader, refusing repeated keys."""
    return yaml.load(text, Loader=_StrictLoader)  # noqa: S506 (a SafeLoader subclass)


# ------------------------------------------------------------------ field types


def _matches(regex, what: str):
    def check(value: str) -> str:
        if not regex.fullmatch(value):
            raise ValueError(f"not a valid {what}: {value!r}")
        return value
    return check


def _sha(value: str) -> str:
    if not is_sha256(value):
        raise ValueError(f"not a SHA-256 (64 lowercase hex, or FAKE-sha256-NNNN): {value!r}")
    return value


ClientId = Annotated[StrictStr, AfterValidator(_matches(CLIENT_RE, "client ID"))]
JobId = Annotated[StrictStr, AfterValidator(_matches(JOB_ID_RE, "job ID (§1)"))]
DecisionId = Annotated[StrictStr, AfterValidator(_matches(DECISION_ID_RE, "decision ID (§1)"))]
Sha256 = Annotated[StrictStr, AfterValidator(_sha)]
Text = Annotated[StrictStr, Field(min_length=1)]
PositiveInt = Annotated[StrictInt, Field(ge=1)]
Weight = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FileRef(_Model):
    """A file the spec depends on, by path (relative to the policy root) and SHA-256."""

    ref: Text
    sha256: Sha256


class GenesExclude(_Model):
    decision_id: DecisionId
    ref: Text
    sha256: Sha256


class Inclusion(_Model):
    rule: Literal["mean10_finite", "indep_filter", "finite_p"]
    min_sites_per_gene: Literal["all"] | PositiveInt


MAX_MODULUS_BITS = 64  # §6.3: the secure sum's uint64 arithmetic


class Aggregation(_Model):
    strategy: Literal["plain", "secure_sum"]
    fixed_point_bits: PositiveInt | None = None
    modulus_bits: PositiveInt | None = None

    @model_validator(mode="after")
    def _bits(self):
        bits = (self.fixed_point_bits, self.modulus_bits)
        if self.strategy == "secure_sum":
            if None in bits:
                raise ValueError("secure_sum needs fixed_point_bits and modulus_bits (§6.3)")
            if self.fixed_point_bits >= self.modulus_bits:
                raise ValueError("fixed_point_bits must be smaller than modulus_bits")
            # The secure sum works in uint64 arithmetic, i.e. modulo 2^64 (WS-13, D-042). A larger
            # modulus can't be run, so it's a malformed spec, not a run-time surprise.
            if self.modulus_bits > MAX_MODULUS_BITS:
                raise ValueError(f"modulus_bits must be at most {MAX_MODULUS_BITS} (the secure "
                                 f"sum runs modulo 2^64), got {self.modulus_bits}")
        elif bits != (None, None):
            raise ValueError("fixed_point_bits and modulus_bits apply to secure_sum only")
        return self


class Weighting(_Model):
    scheme: WeightingScheme
    weights: dict[ClientId, Weight] | None = None
    reason: StrictStr | None = None

    @model_validator(mode="after")
    def _weights(self):
        if self.scheme == "custom" and not self.weights:
            raise ValueError("custom weighting needs weights (client -> number)")
        if self.scheme != "custom" and self.weights is not None:
            raise ValueError("weights apply to custom weighting only")
        return self


class Split(_Model):
    scheme: Literal["loso"]
    folds_ref: Text
    sha256: Sha256


class Window(_Model):
    start: AwareDatetime
    end: AwareDatetime

    @model_validator(mode="after")
    def _order(self):
        if self.end <= self.start:
            raise ValueError("a contact window must end after it starts")
        return self


class ContactWindows(_Model):
    """Phase 3b placeholder: when each client can be reached, and the job's deadline."""

    deadline: AwareDatetime
    clients: dict[ClientId, list[Window]]


class Provenance(_Model):
    kg_version: Text
    census_access_date: date
    annotation_sha256: Sha256
    manifest_sha256: dict[ClientId, Sha256]
    inclusion_flags_sha256: dict[ClientId, Sha256] = Field(default_factory=dict)


class JobSpec(_Model):
    """A parsed job spec. Immutable: ``with_job_id`` returns a new one."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_: Literal["job-spec/1"] = Field(alias="schema")
    job_id: JobId | None = None
    task: Task
    question: StrictStr
    stratum: Stratum
    clients: list[ClientId] = Field(min_length=1)
    methods: list[Text] = Field(min_length=1)
    counts_variant: Literal["standard", "rrnarm"]
    inclusion: Inclusion
    gene_plan: FileRef
    genes_exclude: GenesExclude | None = None
    outputs: dict[ClientId, Text]
    aggregation: Aggregation
    weighting: Weighting | None = None
    rounds: PositiveInt
    split: Split | None = None
    contact_windows: ContactWindows | None = None
    provenance: Provenance
    policy_decision: DecisionId | None = None

    @field_validator("clients", "methods")
    @classmethod
    def _unique(cls, value: list[str]) -> list[str]:
        repeated = sorted({v for v in value if value.count(v) > 1})
        if repeated:
            raise ValueError(f"listed more than once: {repeated}")
        return value

    @model_validator(mode="after")
    def _task_fields(self):
        problems = []
        if self.weighting is None and self.task != "fedavg_classifier":
            problems.append(f"weighting is required for task {self.task} (only "
                            "fedavg_classifier may leave it out, for the policy to deny)")
        if self.task != "fedavg_classifier" and self.rounds != 1:
            problems.append(f"rounds must be 1 for task {self.task}")
        if self.task == "fedavg_classifier" and self.split is None:
            problems.append("fedavg_classifier needs a split (fixed shared folds)")
        if self.task != "fedavg_classifier" and self.split is not None:
            problems.append(f"split applies to fedavg_classifier only, not {self.task}")
        if problems:
            raise ValueError("; ".join(problems))
        return self


# ------------------------------------------------------------------ parse, load, hash


def _problems(exc: ValidationError) -> list[str]:
    out = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err["loc"]) or "spec"
        msg = err["msg"].removeprefix("Value error, ")
        out.append(f"{loc}: {msg}")
    return out


def parse_job_spec(data: Mapping) -> JobSpec:
    """Validate a spec given as a mapping (from YAML or JSON). Raises ``JobSpecError``."""
    if isinstance(data, JobSpec):
        return data
    if not isinstance(data, Mapping):
        raise JobSpecError([f"spec: expected a mapping, got {type(data).__name__}"])
    try:
        return JobSpec.model_validate(dict(data))
    except ValidationError as exc:
        raise JobSpecError(_problems(exc)) from None


def load_job_spec(path: str | Path) -> JobSpec:
    """Read a spec file: ``.json`` as JSON, anything else as YAML (repeated keys refused)."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text) if path.suffix == ".json" else load_yaml_strict(text)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise JobSpecError([f"{path}: {exc}"]) from None
    return parse_job_spec(data)


def spec_to_dict(spec: JobSpec) -> dict:
    """Every field, JSON-safe (dates as ISO strings), nulls included. Parses back to ``spec``."""
    return spec.model_dump(mode="json", by_alias=True)


def canonical_spec_json(spec: JobSpec, exclude: tuple[str, ...] = ("policy_decision",)) -> bytes:
    """The bytes that are hashed: sorted keys, compact, UTF-8, nulls and ``exclude`` left out."""
    d = spec.model_dump(mode="json", by_alias=True, exclude_none=True, exclude=set(exclude))
    return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def job_spec_sha256(spec: JobSpec) -> str:
    """CONTRACTS §6.1 spec hash. Covers ``job_id``; ignores ``policy_decision``."""
    return hashlib.sha256(canonical_spec_json(spec)).hexdigest()


def with_job_id(spec: JobSpec, job_id: str) -> JobSpec:
    """The same spec with ``job_id`` set (validated), e.g. after the check mints one."""
    return parse_job_spec({**spec_to_dict(spec), "job_id": job_id})
