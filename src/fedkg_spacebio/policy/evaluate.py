"""The policy check: one job spec against the sites' policies and the release log (CONTRACTS §5).

``check_job`` takes a parsed spec, a loaded profile (the policies), the site registry and the
active release log, and returns the spec with its job ID minted and a decision record. It
decides; it never runs anything. Releasing a result under an allow is ``ReleaseLog.append``.

Order of evaluation (§5.4)
--------------------------
1. **Spec checks** (``spec#…``): is the spec coherent with the contract tables and the registry?
   Unknown clients, methods of another task, outputs that don't meet a method's need, the
   aggregation not matching the outputs, the wrong weighting for the task, gene-plan and
   exclusion files that don't match their hashes, manifests that don't match. If any fails, the
   job is denied and **no policy is evaluated**: a policy's answer about a spec that doesn't hold
   together would mean nothing.
2. **The project policy**, rules in file order.
3. **Each site in the job**, in the order its first client appears in ``clients``, rules in file
   order.
4. **Every other site with a client in the stratum** (registry order), **differencing rules
   only**. A site outside the job has nothing at stake but what the release reveals about it,
   and a release over other sites can still reveal it by subtraction (example B, J4).

Combining: deny overrides allow; anything no rule allows is denied (every policy is
``default: deny``). The deciding rule is the first failure in that order, and every failure is
listed, so one decision tells the agent everything it would have to change.

What each rule kind checks (§5.1)
---------------------------------
- ``analyses``: the job's task and every method are listed.
- ``release``: every output the job asks of the site's clients is listed. The reason names the
  methods that have no allowed output at the refusing client (so the agent can see what to drop
  or re-plan, e.g. use ``fe`` with ``masked_contribution``).
- ``min_aggregation``: each of the site's clients in the job declares at least ``min_n_per_arm``
  animals per arm (an undeclared count fails), and every gene the site contributes to pools at
  least ``min_sites`` sites, counted in the gene plan whatever the output kind (§5.3).
- ``differencing``: for each gene in the release, the logged releases of the same (task,
  stratum) plus this one must not expose the site (``policy.differencing``).
- ``dp``: ``required: true`` fails: no DP mechanism exists, so no job can meet it.
- ``weighting`` (project): not applicable unless the task is ``fedavg_classifier``; then the
  job must declare a weighting: ``sample_count`` with every client's n_k usable by FLARE
  (``fedavg_client.check_n_k``, the same check the client helper applies), or ``equal``/``custom``
  with a reason (D-017).

Rows a release adds, per gene (§5.3): one row of the clients contributing to the gene, plus a
single-client row for each client whose output is ``per_site_summary`` or ``model_update``
(the receiver reads that client's value directly). A masked contribution adds no row of its own.
Tasks without per-gene pooling (``clock_plane``, ``fedavg_classifier``) are one item whose
contributors are the job's clients; their checks have scope ``job``.

Failing genes are listed in a gene list (canonical format, §6.1) next to the decision record and
referenced with its SHA-256, so the agent can resubmit with exactly those genes excluded.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from fedkg_spacebio.federation.fedavg_client import check_n_k
from fedkg_spacebio.federation.gene_plan import (
    GenePlanError,
    canonical_gene_list_csv,
    exclude_genes,
    read_gene_list,
    read_gene_plan,
)
from fedkg_spacebio.federation.jobspec import (
    METHODS,
    OUTPUT_KINDS,
    PER_GENE_TASKS,
    TASK_WEIGHTING,
    JobSpec,
    job_spec_sha256,
    with_job_id,
)
from fedkg_spacebio.ids import DECISION_ID_RE, mint_job_id
from fedkg_spacebio.policy.decision import (
    Decision,
    Evaluation,
    genes_failed_ref,
    next_decision_id,
    write_decision,
)
from fedkg_spacebio.policy.differencing import failing_genes, isolating_set
from fedkg_spacebio.policy.registry import ClientRecord, check_against_profile
from fedkg_spacebio.policy.release_log import ReleaseLog
from fedkg_spacebio.policy.schema import LoadedPolicy, Profile

# Output kinds whose receiver reads one client's own value: each adds a {client} row (§5.3).
OWN_VALUE_OUTPUTS = frozenset({"per_site_summary", "model_update"})


@dataclass
class _Result:
    """A rule's outcome before it becomes an Evaluation. ``explain`` goes into the reason."""

    result: str
    detail: str
    explain: str | None = None
    genes: list[int] | None = None  # gene scope: the failing genes (empty when none)


@dataclass
class _Context:
    spec: JobSpec
    registry: dict[str, ClientRecord]
    log: ReleaseLog
    plan: pd.DataFrame  # the released genes (plan minus exclusions), bool, clients as columns
    per_gene: bool
    _masks: tuple | None = field(default=None, repr=False)

    def site_clients(self, site: str | None) -> list[str]:
        """The job's clients governed by ``site`` (all of them for the project policy)."""
        return [c for c in self.spec.clients
                if site is None or self.registry[c].site == site]

    def sites_per_gene(self) -> pd.Series:
        """k_g: distinct sites contributing to each released gene."""
        site_of = {c: self.registry[c].site for c in self.plan.columns}
        by_site = self.plan.T.groupby(lambda c: site_of[c]).any().T
        return by_site.sum(axis=1)

    def masks(self) -> tuple[np.ndarray, list[str], list[str]]:
        """Release rows per item, as bitmasks over the stratum's clients (built once).

        Returns (masks: items × rows, column clients, column sites).
        """
        if self._masks is None:
            self._masks = _build_masks(self)
        return self._masks


# ------------------------------------------------------------------ the entry point


def check_job(spec: JobSpec, profile: Profile, registry: dict[str, ClientRecord],
              log: ReleaseLog, *, decision_id: str | None = None, now: datetime | None = None,
              write: bool = True) -> tuple[JobSpec, Decision]:
    """Decide ``spec``. Returns (the spec with its job ID, the decision record).

    Raises ``PolicyMismatchError`` if a manifest names a policy the profile doesn't load.
    With ``write`` (default) the decision record and failing-gene lists are written under the
    log's root; with ``write=False`` nothing is written (gene lists keep count and hash only).
    """
    check_against_profile(registry, profile)
    if spec.job_id is None:
        spec = with_job_id(spec, mint_job_id(now))
    if decision_id is None:
        decision_id = next_decision_id(spec.job_id, log.root)
    elif not DECISION_ID_RE.fullmatch(decision_id):
        raise ValueError(f"not a decision ID: {decision_id!r}")
    head = log.head()

    evaluations: list[Evaluation] = []
    first_fail: tuple[str, str] | None = None  # (rule_id, reason)
    policies: list[LoadedPolicy] = []

    spec_results, ctx = _spec_checks(spec, registry, log)
    for check, res in spec_results:
        rule_id = f"spec#{check}"
        evaluations.append(Evaluation(rule_id, None, "job", res.result, res.detail))
        if res.result == "fail" and first_fail is None:
            first_fail = (rule_id, f"{res.detail}. The spec doesn't hold together, so the "
                                   "policies were not evaluated.")

    if first_fail is None:
        for loaded, site, kinds in _policy_order(spec, profile, registry):
            policies.append(loaded)
            for rule in loaded.policy.rules:
                if kinds is not None and rule.kind not in kinds:
                    continue
                res = RULE_CHECKS[rule.kind](rule, site, ctx)
                ev = _evaluation(loaded, rule, site, ctx, res, decision_id, log, write)
                evaluations.append(ev)
                if res.result == "fail" and first_fail is None:
                    first_fail = (ev.rule_id, f"{rule.reason} ({res.explain or res.detail})")

    n_rules = len(evaluations)
    decision = Decision(
        decision_id=decision_id, job_id=spec.job_id, job_spec_sha256=job_spec_sha256(spec),
        release_log_id=log.log_id,
        release_log_head=head[0] if head else None,
        release_log_head_sha256=head[1] if head else None,
        outcome="deny" if first_fail else "allow",
        deciding_rule=first_fail[0] if first_fail else None,
        reason=first_fail[1] if first_fail else f"Allowed: all {n_rules} checks passed or "
                                                "didn't apply.",
        evaluations=evaluations,
        policies=[{"policy_id": lp.policy.policy_id, "sha256": lp.sha256} for lp in policies])
    if write:
        write_decision(decision, log.root)
    return spec, decision


def _policy_order(spec: JobSpec, profile: Profile, registry: dict[str, ClientRecord]):
    """(policy, site, rule kinds to evaluate or None for all), in §5.4 order."""
    yield profile.project, None, None
    job_sites = list(dict.fromkeys(registry[c].site for c in spec.clients))
    for site in job_sites:
        yield profile.sites[site], site, None
    others = dict.fromkeys(r.site for r in registry.values()
                           if r.stratum == spec.stratum and r.site not in job_sites)
    for site in others:
        loaded = profile.sites[site]
        if any(r.kind == "differencing" for r in loaded.policy.rules):
            yield loaded, site, {"differencing"}


def _evaluation(loaded: LoadedPolicy, rule, site: str | None, ctx: _Context, res: _Result,
                decision_id: str, log: ReleaseLog, write: bool) -> Evaluation:
    scope = "gene" if res.genes is not None else "job"
    ev = Evaluation(loaded.policy.rule_id(rule.key), site, scope, res.result, res.detail)
    if res.genes is not None:
        ev.genes_failed = len(res.genes)
        if res.genes:
            data = canonical_gene_list_csv(res.genes)
            ev.sha256 = hashlib.sha256(data).hexdigest()
            if write:
                ref = genes_failed_ref(decision_id, site, rule.key)
                path = log.root / ref
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as f:
                    f.write(data)
                ev.genes_failed_ref = ref
    return ev


# ------------------------------------------------------------------ spec checks (§5.4)


def _spec_checks(spec: JobSpec, registry: dict[str, ClientRecord], log: ReleaseLog):
    """All eight checks, in order, and the evaluation context (None if the plan can't load)."""
    out = []

    def add(name: str, problems: list[str], na: str | None = None):
        if na is not None:
            out.append((name, _Result("not_applicable", na)))
        elif problems:
            out.append((name, _Result("fail", "; ".join(problems))))
        else:
            out.append((name, _Result("pass", "ok")))

    # clients-known
    problems = []
    for c in spec.clients:
        if c not in registry:
            problems.append(f"{c} isn't in the site registry")
        elif registry[c].stratum != spec.stratum:
            problems.append(f"{c} is in stratum {registry[c].stratum}, not {spec.stratum}")
    add("clients-known", problems)
    known = [c for c in spec.clients if c in registry]

    # methods-known
    problems = []
    for m in spec.methods:
        if m not in METHODS:
            problems.append(f"{m} isn't a known method (§5.2: {', '.join(METHODS)})")
        elif METHODS[m].task != spec.task:
            problems.append(f"{m} belongs to task {METHODS[m].task}, not {spec.task}")
    add("methods-known", problems)

    # outputs-cover
    problems = []
    missing = [c for c in spec.clients if c not in spec.outputs]
    extra = [c for c in spec.outputs if c not in spec.clients]
    if missing:
        problems.append(f"no output given for {', '.join(missing)}")
    if extra:
        problems.append(f"outputs name clients not in the job: {', '.join(extra)}")
    for c, kind in spec.outputs.items():
        if kind not in OUTPUT_KINDS:
            problems.append(f"{c}: {kind} isn't an output kind (§5.1)")
            continue
        if c not in spec.clients:
            continue
        unmet = [m for m in spec.methods if m in METHODS and METHODS[m].task == spec.task
                 and kind not in METHODS[m].outputs]
        for m in unmet:
            need = " or ".join(sorted(METHODS[m].outputs - {"per_sample"}))
            problems.append(f"{c}'s {kind} doesn't serve {m} (needs {need})")
    add("outputs-cover", problems)

    # aggregation
    problems = []
    kinds = {c: k for c, k in spec.outputs.items() if c in spec.clients}
    if spec.aggregation.strategy == "secure_sum":
        unmasked = [c for c, k in kinds.items() if k != "masked_contribution"]
        if unmasked:
            problems.append("secure_sum needs every output to be masked_contribution; not: "
                            + ", ".join(unmasked))
    else:
        masked = [c for c, k in kinds.items() if k == "masked_contribution"]
        if masked:
            problems.append("masked_contribution needs secure_sum (only the sum is readable); "
                            f"under plain: {', '.join(masked)}")
    add("aggregation", problems)

    # weighting
    if spec.weighting is None:
        add("weighting", [], na="no weighting declared; the project policy decides (D-017)")
    else:
        allowed = TASK_WEIGHTING[spec.task]
        add("weighting", [] if spec.weighting.scheme in allowed else [
            f"{spec.task} takes {' or '.join(sorted(allowed))}, not {spec.weighting.scheme}"])

    # gene-plan
    plan = None
    try:
        plan = read_gene_plan(log.root / spec.gene_plan.ref, spec.gene_plan.sha256)
        if sorted(plan.columns) != sorted(spec.clients):
            add("gene-plan", [f"plan columns {sorted(plan.columns)} aren't the job's clients"])
            plan = None
        else:
            add("gene-plan", [])
    except GenePlanError as exc:
        add("gene-plan", [str(exc)])

    # genes-exclude
    if spec.genes_exclude is None:
        add("genes-exclude", [], na="no exclusions")
    else:
        try:
            excluded = read_gene_list(log.root / spec.genes_exclude.ref,
                                      spec.genes_exclude.sha256)
            if plan is not None:
                plan = exclude_genes(plan, excluded)
            add("genes-exclude", [])
        except GenePlanError as exc:
            add("genes-exclude", [str(exc)])
            plan = None

    # provenance
    problems = []
    for c in known:
        given = spec.provenance.manifest_sha256.get(c)
        if given is None:
            problems.append(f"no manifest_sha256 for {c}")
        elif given != registry[c].manifest_sha256:
            problems.append(f"{c}: manifest_sha256 doesn't match its manifest")
        if spec.task == "de_meta" and c not in spec.provenance.inclusion_flags_sha256:
            problems.append(f"no inclusion_flags_sha256 for {c} (needed for de_meta)")
    stray = sorted(set(spec.provenance.manifest_sha256) - set(spec.clients))
    if stray:
        problems.append(f"manifest_sha256 names clients not in the job: {', '.join(stray)}")
    add("provenance", problems)

    ctx = None
    if plan is not None and all(r.result != "fail" for _, r in out):
        ctx = _Context(spec, registry, log, plan, spec.task in PER_GENE_TASKS)
    return out, ctx


# ------------------------------------------------------------------ rule kinds (§5.1)


def _analyses(rule, site, ctx: _Context) -> _Result:
    problems = []
    if ctx.spec.task not in rule.tasks:
        problems.append(f"task {ctx.spec.task} isn't allowed (allowed: {', '.join(rule.tasks)})")
    if rule.methods != "*":
        refused = [m for m in ctx.spec.methods if m not in rule.methods]
        if refused:
            problems.append(f"methods not allowed: {', '.join(refused)}")
    return _Result("fail" if problems else "pass", "; ".join(problems) or "allowed")


def _release(rule, site, ctx: _Context) -> _Result:
    allowed = set(rule.outputs)
    refused = [(c, ctx.spec.outputs[c]) for c in ctx.site_clients(site)
               if ctx.spec.outputs[c] not in allowed]
    if not refused:
        return _Result("pass", "every requested output is allowed")
    detail = "; ".join(f"{kind} requested from {c}" for c, kind in refused)
    # Which methods have no output this rule allows? (They can't be served here at all.)
    stuck = [m for m in ctx.spec.methods if m in METHODS and not METHODS[m].outputs & allowed]
    explain = detail
    if stuck:
        explain += (f"; {', '.join(stuck)} need{'s' if len(stuck) == 1 else ''} an output "
                    f"{'this site' if site else 'the project'} doesn't allow")
    return _Result("fail", detail, explain)


def _min_aggregation(rule, site, ctx: _Context) -> _Result:
    problems = []
    for c in ctx.site_clients(site):
        rec = ctx.registry[c]
        for arm in ("n_flight", "n_ground"):
            n = getattr(rec, arm)
            if n is None:
                problems.append(f"{c} doesn't declare {arm}, so it can't show at least "
                                f"{rule.min_n_per_arm} per arm")
            elif n < rule.min_n_per_arm:
                problems.append(f"{c} has {arm} = {n} < {rule.min_n_per_arm}")
    if not ctx.per_gene:
        k = len({ctx.registry[c].site for c in ctx.spec.clients})
        if k < rule.min_sites:
            problems.append(f"the job pools {k} site{'s' if k != 1 else ''} < {rule.min_sites}")
        return _Result("fail" if problems else "pass", "; ".join(problems) or
                       f"{k} sites pooled; per-arm counts met")
    mine = ctx.plan[ctx.site_clients(site)].any(axis=1)
    k = ctx.sites_per_gene()
    genes = [int(g) for g in ctx.plan.index[mine & (k < rule.min_sites)]]
    if genes:
        problems.append(f"{len(genes)} gene{'s' if len(genes) != 1 else ''} this "
                        f"{'site' if site else 'job'} contributes to pool fewer than "
                        f"{rule.min_sites} sites")
    return _Result("fail" if problems else "pass", "; ".join(problems) or
                   f"every gene pools at least {rule.min_sites} sites; per-arm counts met",
                   genes=genes)


def _differencing(rule, site, ctx: _Context) -> _Result:
    masks, columns, col_sites = ctx.masks()
    protected = [j for j, c in enumerate(columns)
                 if site is None or col_sites[j] == site]
    failing = np.zeros(masks.shape[0], dtype=bool)
    for j in protected:
        failing |= failing_genes(masks, j, col_sites, rule.min_sites)
    genes = ([int(g) for g, f in zip(ctx.plan.index, failing, strict=True) if f]
             if ctx.per_gene else [])
    n_logged = len(ctx.log.history(ctx.spec.task, ctx.spec.stratum))
    if not failing.any():
        res = _Result("pass", f"no isolation against {n_logged} logged release"
                              f"{'s' if n_logged != 1 else ''} of {ctx.spec.task} in "
                              f"{ctx.spec.stratum}")
    else:
        # one concrete witness, to make the failure readable
        i = int(np.flatnonzero(failing)[0])
        group = None
        for j in protected:
            group = isolating_set(masks[i].tolist(), j, col_sites, rule.min_sites)
            if group:
                break
        names = ", ".join(columns[j] for j in sorted(group))
        what = (f"{len(genes)} gene{'s' if len(genes) != 1 else ''}" if ctx.per_gene
                else "the job")
        example = f", e.g. gene {genes[0]}" if ctx.per_gene else ""
        res = _Result("fail", f"{what}: this release with {n_logged} logged release"
                              f"{'s' if n_logged != 1 else ''} would reveal the sum over "
                              f"{{{names}}}, fewer than {rule.min_sites} sites{example}")
    if ctx.per_gene:
        res.genes = genes
    return res


def _dp(rule, site, ctx: _Context) -> _Result:
    if rule.required:
        return _Result("fail", "DP required, but no DP mechanism exists yet, so no job can "
                               "meet it")
    return _Result("pass", "not required")


def _weighting(rule, site, ctx: _Context) -> _Result:
    spec = ctx.spec
    if spec.task != "fedavg_classifier":
        return _Result("not_applicable", f"task {spec.task}")
    w = spec.weighting
    if w is None:
        return _Result("fail", "no weighting declared; FLARE would silently weight every "
                               "client equally (D-017)")
    problems = []
    if w.scheme == "sample_count":
        for c in spec.clients:
            try:
                check_n_k(ctx.registry[c].n_k)
            except ValueError as exc:
                problems.append(f"{c}: manifest n_k {ctx.registry[c].n_k!r} unusable: {exc}")
    elif w.scheme in ("equal", "custom"):
        if not (w.reason or "").strip():
            problems.append(f"{w.scheme} weighting needs a reason")
        if w.scheme == "custom":
            weights = w.weights or {}
            missing = [c for c in spec.clients if c not in weights]
            extra = [c for c in weights if c not in spec.clients]
            bad = [c for c, v in weights.items() if not v > 0]
            if missing:
                problems.append(f"custom weights miss {', '.join(missing)}")
            if extra:
                problems.append(f"custom weights name clients not in the job: "
                                f"{', '.join(extra)}")
            if bad:
                problems.append(f"custom weights must be positive: {', '.join(bad)}")
    else:  # spec#weighting already refuses other schemes; kept for completeness
        problems.append(f"{w.scheme} isn't a FedAvg weighting")
    return _Result("fail" if problems else "pass", "; ".join(problems) or
                   f"{w.scheme} weighting declared")


RULE_CHECKS: dict[str, Callable[..., _Result]] = {
    "analyses": _analyses,
    "release": _release,
    "min_aggregation": _min_aggregation,
    "differencing": _differencing,
    "dp": _dp,
    "weighting": _weighting,
}


# ------------------------------------------------------------------ release rows


def _rows_of(clients_df: pd.DataFrame, outputs: dict, bit: dict, items) -> list[np.ndarray]:
    """One release's rows over ``items``: the pooled row plus own-value rows (§5.3)."""
    frame = clients_df.reindex(index=items, fill_value=False).astype(bool)
    pooled = np.zeros(len(items), dtype=np.uint64)
    singles = []
    for c in frame.columns:
        col = frame[c].to_numpy() * np.uint64(bit[c])
        pooled |= col
        if outputs.get(c) in OWN_VALUE_OUTPUTS:
            singles.append(col)
    return [pooled, *singles]


def _build_masks(ctx: _Context) -> tuple[np.ndarray, list[str], list[str]]:
    spec = ctx.spec
    history = ctx.log.history(spec.task, spec.stratum)
    columns = [c for c, r in ctx.registry.items() if r.stratum == spec.stratum]
    for entry, _ in history:  # a client logged earlier but no longer registered still counts
        columns += [c for c in entry["clients"] if c not in columns]
    col_sites = [ctx.registry[c].site if c in ctx.registry else c for c in columns]
    bit = {c: 1 << j for j, c in enumerate(columns)}

    if ctx.per_gene:
        items = ctx.plan.index
        releases = [(contrib, entry["outputs"]) for entry, contrib in history]
        releases.append((ctx.plan, spec.outputs))
    else:  # one item; every job client contributes to it
        items = pd.Index([0])
        releases = [(pd.DataFrame({c: [True] for c in entry["clients"]}), entry["outputs"])
                    for entry, _ in history]
        releases.append((pd.DataFrame({c: [True] for c in spec.clients}), spec.outputs))
    rows = [r for contrib, outputs in releases for r in _rows_of(contrib, outputs, bit, items)]
    masks = np.column_stack(rows) if rows else np.zeros((len(items), 0), dtype=np.uint64)
    return masks, columns, col_sites
