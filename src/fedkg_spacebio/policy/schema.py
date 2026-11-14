"""Site policies as code: the ``site-policy/1`` format, its loader, and profiles (CONTRACTS §5.1).

A policy is one YAML file per site (plus one project policy) listing rules, each of a known
``kind`` with its parameters and a ``reason``. This module only *reads* policies and checks that
they are well formed. Deciding a job against them is ``policy.evaluate``.

Every demo policy is invented for the testbed (first line of each file). No real mission
declared these rules.

What "well formed" means here
-----------------------------
- ``schema: site-policy/1`` and ``default: deny``. A policy can't default to allow: anything no
  rule allows is refused, so a forgotten rule fails closed.
- ``policy_id`` is ``pol:<site>:<version>`` for the file's own ``site`` (the same check the
  manifest builder applies), or ``pol:project:<version>`` with ``site: null``.
- Rule keys are unique within the file (the rule ID is ``<policy_id>#<key>``, so a repeated key
  would make two rules indistinguishable in a decision record).
- Every rule has a known ``kind``, exactly that kind's parameters, and a non-empty ``reason``.
  Tasks, methods and output kinds must be names from the contract tables (§5.2, §6.1): a typo
  in an allow-list would otherwise silently allow or refuse nothing.
- ``differencing.min_sites`` ≥ 2 (1 would protect nothing: every site "is" a group of one).
- ``weighting`` rules only in the project policy (D-017 is a project-wide rule).
- The YAML may not repeat a key in one mapping (PyYAML would keep the last one silently).

A **profile** is a folder of policies (``policies/<profile>/``): ``project.yaml`` and one
``<site>.yaml`` per site, the file name equal to the policy's ``site``. The demo has two
profiles, ``open`` and ``restricted``, which differ only in SpaceX-16's file.

Alternative considered: a general policy language (OPA/Rego, Cedar). Those are built for exactly
this, but the rules here are six fixed kinds with numeric parameters, and the per-gene checks
(§5.3) need numpy and exact linear algebra that a policy engine would call out to anyway.
Small typed models keep each rule's meaning in one readable place.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from fedkg_spacebio.federation.jobspec import METHODS, TASKS, OutputKind, load_yaml_strict

SCHEMA = "site-policy/1"
RULE_KINDS = ("analyses", "release", "min_aggregation", "differencing", "dp", "weighting")
PROJECT_POLICY_RE = re.compile(r"pol:project:[1-9][0-9]*")
SITE_POLICY_RE = re.compile(r"pol:(?P<site>[A-Za-z0-9][A-Za-z0-9._-]*):[1-9][0-9]*")


class PolicyError(ValueError):
    """A policy file or profile that is malformed. The message lists every problem."""


Key = Annotated[StrictStr, Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")]
Reason = Annotated[StrictStr, Field(min_length=1)]
Count = Annotated[StrictInt, Field(ge=1)]


class _Rule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: Key
    reason: Reason


class AnalysesRule(_Rule):
    """Passes when the job's task and every method are listed (``methods: "*"`` = any)."""

    kind: Literal["analyses"]
    tasks: list[StrictStr] = Field(min_length=1)
    methods: Literal["*"] | list[StrictStr]

    @field_validator("tasks")
    @classmethod
    def _known_tasks(cls, v):
        unknown = [t for t in v if t not in TASKS]
        if unknown:
            raise ValueError(f"unknown tasks {unknown}; expected some of {list(TASKS)}")
        return v

    @field_validator("methods")
    @classmethod
    def _known_methods(cls, v):
        unknown = [m for m in v if m not in METHODS] if v != "*" else []
        if unknown:
            raise ValueError(f"unknown methods {unknown}; expected some of {list(METHODS)}")
        return v


class ReleaseRule(_Rule):
    """Passes when every output the job asks of this site's clients is listed."""

    kind: Literal["release"]
    outputs: list[OutputKind] = Field(min_length=1)


class MinAggregationRule(_Rule):
    """k_g ≥ ``min_sites`` for every gene the site contributes to; n per arm ≥ ``min_n_per_arm``."""

    kind: Literal["min_aggregation"]
    min_sites: Count
    min_n_per_arm: Annotated[StrictInt, Field(ge=0)]


class DifferencingRule(_Rule):
    """No set of releases isolates the site, or a group of fewer than ``min_sites`` with it."""

    kind: Literal["differencing"]
    min_sites: Annotated[StrictInt, Field(ge=2)]


class DpRule(_Rule):
    """``required: true`` denies every job until a DP mechanism exists."""

    kind: Literal["dp"]
    required: StrictBool


class WeightingRule(_Rule):
    """Project policy only: FedAvg jobs declare their weighting (D-017)."""

    kind: Literal["weighting"]


Rule = Annotated[AnalysesRule | ReleaseRule | MinAggregationRule | DifferencingRule | DpRule
                 | WeightingRule, Field(discriminator="kind")]


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_: Literal["site-policy/1"] = Field(alias="schema")
    policy_id: StrictStr
    site: StrictStr | None
    effective: date
    default: Literal["deny"]
    rules: list[Rule]

    @model_validator(mode="after")
    def _consistent(self):
        problems = []
        if self.site is None:
            if not PROJECT_POLICY_RE.fullmatch(self.policy_id):
                problems.append(f"policy_id {self.policy_id!r}: a policy with site null must "
                                "be pol:project:<version>")
        else:
            m = SITE_POLICY_RE.fullmatch(self.policy_id)
            if not m or m["site"] != self.site:
                problems.append(f"policy_id {self.policy_id!r} is not pol:{self.site}:<version> "
                                f"for site {self.site!r}")
            if self.site == "project":
                problems.append("'project' is reserved for the project policy")
            for r in self.rules:
                if r.kind == "weighting":
                    problems.append(f"rule {r.key!r}: weighting rules belong in the project "
                                    "policy only")
        keys = [r.key for r in self.rules]
        repeated = sorted({k for k in keys if keys.count(k) > 1})
        if repeated:
            problems.append(f"rule keys used more than once: {repeated}")
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def rule_id(self, key: str) -> str:
        """``<policy_id>#<key>`` for a rule in this policy."""
        self.rule(key)
        return f"{self.policy_id}#{key}"

    def rule(self, key: str):
        for r in self.rules:
            if r.key == key:
                return r
        raise KeyError(f"{self.policy_id} has no rule {key!r}")


@dataclass(frozen=True)
class LoadedPolicy:
    """A parsed policy with the SHA-256 of the exact file bytes it came from."""

    policy: Policy
    sha256: str
    path: Path


def load_policy(path: str | Path) -> LoadedPolicy:
    path = Path(path)
    data = path.read_bytes()
    try:
        raw = load_yaml_strict(data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise PolicyError(f"{path}: {exc}") from None
    if not isinstance(raw, dict):
        raise PolicyError(f"{path}: expected a mapping")
    try:
        policy = Policy.model_validate(raw)
    except ValidationError as exc:
        lines = [f"{'.'.join(str(x) for x in e['loc']) or 'policy'}: "
                 f"{e['msg'].removeprefix('Value error, ')}" for e in exc.errors()]
        raise PolicyError(f"{path}:\n  " + "\n  ".join(lines)) from None
    return LoadedPolicy(policy, hashlib.sha256(data).hexdigest(), path)


@dataclass(frozen=True)
class Profile:
    """A folder of policies: the project policy and one policy per site."""

    name: str
    project: LoadedPolicy
    sites: dict[str, LoadedPolicy]

    def site_policy_ids(self) -> dict[str, str]:
        """site → policy_id. What the manifest builder writes into each client's manifest."""
        return {s: lp.policy.policy_id for s, lp in self.sites.items()}


def load_profile(directory: str | Path) -> Profile:
    """Load ``project.yaml`` and every ``<site>.yaml`` in ``directory``."""
    directory = Path(directory)
    files = sorted(directory.glob("*.yaml"))
    if not files:
        raise PolicyError(f"{directory}: no policy files")
    project, sites, problems = None, {}, []
    for f in files:
        lp = load_policy(f)
        site = lp.policy.site
        if site is None:
            if f.stem != "project":
                problems.append(f"{f.name}: a project policy must be in project.yaml")
            project = lp
        elif f.stem != site:
            problems.append(f"{f.name}: file name must be {site}.yaml (the policy's site)")
        else:
            sites[site] = lp
    if project is None:
        problems.append("no project policy (project.yaml with site: null)")
    if problems:
        raise PolicyError(f"{directory}:\n  " + "\n  ".join(problems))
    return Profile(name=directory.name, project=project, sites=sites)
