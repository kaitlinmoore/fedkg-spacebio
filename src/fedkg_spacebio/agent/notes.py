"""The caveats that travel with a phase 3a result, as citable notes (``note:<key>``).

Why caveats are tool output
---------------------------
A statement of the phase 3a result is misleading without its caveats: what question the method
answers, how much one mission drives it, the time-of-day explanation that can't be ruled out, the
fact that the methods were chosen after seeing results. If the report writer (a person, a
deterministic planner or a language model) had to remember them, some would be dropped or
reworded until they no longer said the same thing.

So ``results.read`` returns the caveats that apply to the release it reads, each as a fixed
sentence with an ID (``note:time-of-day``). A caveat claim in the report cites that ID from that
call, and the grounding checker treats it like any other citation: the caveat is grounded if the
tool that read the release said it applies. Texts are fixed strings, versioned with the tool, so
a trace shows exactly which wording the run used.

When a note applies
-------------------
Each note has a rule over the release's metadata (``ReleaseInfo``). Three kinds of rule:
- **method notes** (what the estimand is): by the method read;
- **finding notes** (numbers from the phase 3a analysis of the same data): only for releases of
  that analysis (ISS, standard counts, the primary inclusion rule, all four ISS clients), and only
  where the release could have carried the evidence. The SpaceX-16 and SpaceX-8 findings need
  per-site estimates, so they apply to plain releases (per-site summaries) and not to masked ones.
  Under the restricted profile the federation can't see that dependence; what the report says
  instead is that the check was refused, with the refusing decision;
- **framing notes** (always, or by profile and aggregation): invented policies, the simulated
  secure sum, the open profile as a counterfactual.

The wording comes from the project's tracked docs (``docs/phase3a-liver-meta.md``,
``docs/methodology.md``); each note says where.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

ISS_PRIMARY_CLIENTS = frozenset({"SpaceX-4", "SpaceX-8", "SpaceX-13-ISST", "SpaceX-16-ISST"})


@dataclass(frozen=True)
class ReleaseInfo:
    profile: str
    stratum: str
    counts_variant: str
    inclusion_rule: str
    min_sites_per_gene: str | int
    clients: frozenset[str]
    aggregation: str          # "plain" | "secure_sum"
    method: str               # the method being read
    methods_run: frozenset[str]

    @property
    def iss_primary(self) -> bool:
        """The phase 3a ISS analysis the finding notes describe."""
        return (self.stratum == "iss" and self.counts_variant == "standard"
                and self.inclusion_rule == "mean10_finite" and self.min_sites_per_gene == "all"
                and self.clients == ISS_PRIMARY_CLIENTS)


@dataclass(frozen=True)
class Note:
    key: str
    text: str
    source: str                         # tracked doc the wording comes from
    applies: Callable[[ReleaseInfo], bool]

    @property
    def note_id(self) -> str:
        return f"note:{self.key}"


NOTES: tuple[Note, ...] = (
    # --- what the method answers
    Note("fe-these-missions",
         "Fixed effect answers \"what is the precision-weighted average change in these "
         "missions?\". It does not answer \"what would a new mission show?\", and a large, "
         "precise site can drive it.",
         "docs/phase3a-liver-meta.md",
         lambda r: r.method in ("fe", "v2")),
    Note("het-annotation",
         "Cochran's Q p and I² are annotations, not filters: a gene with a high Q is still a "
         "fixed-effect hit.",
         "docs/phase3a-liver-meta.md",
         lambda r: "fe_het" in r.methods_run),
    Note("v3-new-mission",
         "V3 (empirical-Bayes τ²) answers \"a new mission\": whether the mean effect across "
         "missions, the one a new mission would be drawn around, is not 0. With four sites "
         "(ISS) its power is low and with three (Earth) essentially none, so few or no hits is a "
         "result, not a failure.",
         "docs/phase3a-method-comparison.md",
         lambda r: r.method == "v3"),
    # --- findings of the phase 3a analysis of the same data
    Note("time-of-day",
         "These results are consistent with a time-of-day mismatch of a few hours or a clock "
         "change; not separable.",
         "docs/phase3a-liver-meta.md; docs/phase3a-time-of-day.md",
         lambda r: True),
    Note("spacex16-dominance",
         "In the phase 3a analysis of the same data, every ISS V2 and V3 hit vanishes without "
         "SpaceX-16.",
         "docs/phase3a-liver-meta.md",
         lambda r: r.iss_primary and r.aggregation == "plain"),
    Note("rhythmic-share-fe",
         "In the phase 3a analysis of the same data, 1,097 of the 1,926 ISS fixed-effect hits "
         "remain when liver-rhythmic genes are removed.",
         "docs/phase3a-liver-meta.md",
         lambda r: r.iss_primary and r.method == "fe"),
    Note("rhythmic-share-v3",
         "In the phase 3a analysis of the same data, 61% of the ISS V3 hits are liver-rhythmic "
         "genes; 4 remain without them.",
         "docs/phase3a-liver-meta.md",
         lambda r: r.iss_primary and r.method == "v3"),
    Note("spacex8-stays-in",
         "SpaceX-8 disagrees with the other ISS missions and stays in: dropping a site because "
         "it disagrees, after seeing results, would be choosing the answer.",
         "docs/phase3a-liver-meta.md",
         lambda r: r.iss_primary and r.aggregation == "plain"),
    Note("methods-chosen-after-results",
         "The headline methods were chosen after seeing results, so hit counts should be read "
         "as exploratory.",
         "docs/phase3a-liver-meta.md",
         lambda r: True),
    # --- framing
    Note("policies-invented",
         "The site policies are invented for the testbed: no real mission declared them, and "
         "the sites are simulated over public data.",
         "docs/methodology.md §7",
         lambda r: True),
    Note("secure-sum-simulation",
         "The secure sum is a simulation of the arithmetic, not a security or privacy claim; "
         "the data are public.",
         "docs/methodology.md §7.7",
         lambda r: r.aggregation == "secure_sum"),
    Note("open-counterfactual",
         "This release was made under the open profile, in its own release log: it shows what "
         "the analysis gives if every site shares per-site summaries. It is a counterfactual, "
         "not a release made alongside the restricted one.",
         "docs/methodology.md §7",
         lambda r: r.profile == "open"),
)

_KEYS = [n.key for n in NOTES]
assert len(_KEYS) == len(set(_KEYS)), "note keys must be unique"


def notes_for(info: ReleaseInfo) -> list[Note]:
    """The notes that apply to a release read with ``info.method``, in catalog order."""
    return [n for n in NOTES if n.applies(info)]


def by_id(note_id: str) -> Note:
    for n in NOTES:
        if n.note_id == note_id:
            return n
    raise KeyError(note_id)
