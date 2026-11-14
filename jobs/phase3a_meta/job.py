"""Phase 3a FLARE job: federated DE meta-analysis for one stratum, in the FL simulator.

Run from WSL after ``fedkg``:

    uv run python jobs/phase3a_meta/job.py --stratum iss
    uv run python jobs/phase3a_meta/job.py --stratum earth
    uv run python jobs/phase3a_meta/job.py --stratum iss --counts rrnarm      # sensitivity
    uv run python jobs/phase3a_meta/central.py --stratum iss                  # Check 1
    uv run python jobs/phase3a_meta/job.py --stratum iss --profile restricted # D-036 profile

One FLARE client per (site, stratum) row of D-020 (``fedkg_spacebio.sites.manifest.CLIENTS``).
The server runs ``MetaAnalysisController``: one task, wait for every client, validate, pool.
In the ISS stratum the server also pools a ``primary_without_SpaceX-8`` variant from the same
summaries, because SpaceX-8's ISS euthanasia is inferred (D-020 caveat).

Outputs: ``$FEDKG_WORKDIR/results/phase3a/<stratum>/<counts>/flare/``.

From a job spec (CONTRACTS §6; WS-13), which is what ``job.submit`` (WS-15) calls:

    uv run python jobs/phase3a_meta/job.py --spec <spec.yaml> [--profile restricted]

``run_spec`` checks the spec against its files (``federation/spec_run.py``: plan and exclusion
hashes, manifest hashes, outputs vs aggregation, m ≤ 64) and runs it with
``SecureSumController`` (all outputs ``masked_contribution``) or ``SpecPlainController`` (all
``per_site_summary``). Outputs: ``…/<stratum>/<counts>/jobs/<job slug>/`` with ``server/`` (what
the server received and released) and ``sites/<client>/{sent,local}`` (what each client sent,
and what it kept). It uses the manifests as they are (their hashes are in the spec), never
rebuilds them. The mask seed is drawn fresh per run unless ``mask_seed`` is given.

``--profile`` (D-036) picks the policy profile (``policies/<profile>/``). It sets each
manifest's ``policy_id`` from the profile's policy files, so the manifests, and their hashes,
say which policies the run was made under. Profiles other than ``open`` write manifests and
results under ``profiles/<profile>/`` (see ``config.py``). The profile doesn't gate the run:
this script runs whatever it is told. Checking a job against the policies before it runs is the
policy check's job (``python -m fedkg_spacebio.policy.check``), and refusing to submit without
an allow is ``job.submit``'s (CONTRACTS §7).
"""

import argparse
import json
import secrets
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402
from nvflare.recipe import SimEnv  # noqa: E402

from fedkg_spacebio.federation.jobspec import load_job_spec, spec_to_dict  # noqa: E402
from fedkg_spacebio.federation.meta_flare import (  # noqa: E402
    MetaAnalysisRecipe,
    SecureSumController,
    SpecPlainController,
)
from fedkg_spacebio.federation.spec_run import policy_root_default, prepare  # noqa: E402
from fedkg_spacebio.sites.manifest import clients_in, write_manifests  # noqa: E402

DROP_VARIANTS = {"iss": ["SpaceX-8"], "earth": []}


def run(stratum: str, counts: str, root: Path | None = None, clients: list[str] | None = None,
        drop_variants: list[str] | None = None, n_cpus: int = config.N_CPUS_PER_CLIENT,
        verify_annotation: bool = True, build_manifests: bool = True,
        profile: str = "open") -> Path:
    manifests = config.manifest_dir(counts, root, profile)
    if build_manifests:
        write_manifests(out_dir=manifests, workdir=root or config.workdir(),
                        counts_variant=counts, policy_ids=config.policy_ids(profile))
    clients = clients or clients_in(stratum)
    out = config.results_dir(stratum, counts, "flare", root, profile)
    if out.exists():
        shutil.rmtree(out)  # never mix outputs from two runs
    args = (f"--manifest_dir {manifests} "
            f"--data_root {config.data_root(root)} --annotation {config.annotation_path(root)} "
            f"--sent_dir {out / 'sent'} --n_cpus {n_cpus}"
            + ("" if verify_annotation else " --no_verify_annotation"))
    recipe = MetaAnalysisRecipe(
        name=config.job_name(stratum, counts), clients=clients,
        client_script=str(HERE / "client.py"), client_args=args, results_dir=str(out),
        drop_variants=DROP_VARIANTS.get(stratum, []) if drop_variants is None else drop_variants)
    env = SimEnv(clients=clients, workspace_root=str(config.sim_workspace_root()))
    result = recipe.execute(env)
    status = str(result.get_status())
    print("Result workspace:", result.get_result())
    print("Job status:      ", status)
    if "COMPLETED" not in status or not (out / "pooling_run.json").exists():
        raise SystemExit(f"FLARE job did not complete cleanly ({status})")
    print("Outputs:         ", out)
    return out


def run_spec(spec_path: Path, profile: str = "open", root: Path | None = None,
             policy_root: Path | None = None, n_cpus: int = config.N_CPUS_PER_CLIENT,
             verify_annotation: bool = True, mask_seed: bytes | None = None) -> Path:
    """Run one §6 job spec (de_meta) in the simulator. Not gated: see the module docstring."""
    policy_root = Path(policy_root) if policy_root is not None else policy_root_default()
    spec_path = Path(spec_path)
    spec = load_job_spec(spec_path)
    manifests = config.manifest_dir(spec.counts_variant, root, profile)
    sr = prepare(spec, manifest_dir=manifests, policy_root=policy_root)
    sr.spec_path = spec_path
    out = config.spec_results_dir(spec.stratum, spec.counts_variant, sr.slug, root, profile)
    if out.exists():
        shutil.rmtree(out)  # never mix outputs from two runs
    out.mkdir(parents=True)
    (out / "job_spec.json").write_text(json.dumps(
        {"job_spec_sha256": sr.spec_sha256, "source": str(spec_path),
         "spec": spec_to_dict(spec)}, indent=1))
    sites = out / "sites"
    args = (f"--manifest_dir {manifests} --data_root {config.data_root(root)} "
            f"--annotation {config.annotation_path(root)} --n_cpus {n_cpus} "
            f"--spec {spec_path.resolve()} --policy_root {policy_root} --sites_dir {sites}"
            + ("" if verify_annotation else " --no_verify_annotation"))
    common = dict(
        results_dir=str(out), expected_clients=sr.clients, job_id=sr.job_id,
        job_spec_sha256=sr.spec_sha256, methods=list(spec.methods), flag=sr.flag,
        plan_path=str(sr.plan_path), plan_sha256=spec.gene_plan.sha256,
        exclude_path=str(sr.exclude_path) if sr.exclude_path else None,
        exclude_sha256=spec.genes_exclude.sha256 if spec.genes_exclude else None,
        inclusion_flags_sha256=dict(spec.provenance.inclusion_flags_sha256),
        provenance={"profile": profile, "counts_variant": spec.counts_variant,
                    "stratum": spec.stratum, "policy_decision": spec.policy_decision,
                    "annotation_sha256": spec.provenance.annotation_sha256,
                    "manifest_sha256": dict(spec.provenance.manifest_sha256),
                    "policy_ids": {c: m["policy_id"] for c, m in sr.site_meta.items()},
                    "gated": "no: job.py runs what it is given; the policy check and job.submit "
                             "decide whether a spec may run"},
        annotation_path=str(config.annotation_path(root)), verify_annotation=verify_annotation)
    if sr.mode == "secure_sum":
        seed = mask_seed if mask_seed is not None else secrets.token_bytes(32)
        sim = sites / "_simulation"
        sim.mkdir(parents=True)
        (sim / "mask_seed.json").write_text(json.dumps({
            "root_seed_hex": seed.hex(),
            "note": "Simulation only (CONTRACTS §6.3): the pairwise mask seeds derive from this "
                    "root seed, passed to the clients through the job config. Kept under sites/, "
                    "never under server/."}, indent=1))
        args += f" --mask_seed {seed.hex()}"
        controller = SecureSumController(
            fixed_point_bits=sr.params.fixed_point_bits, modulus_bits=sr.params.modulus_bits,
            quantities=list(sr.quantities), **common)
    else:
        controller = SpecPlainController(**common)
    recipe = MetaAnalysisRecipe(
        name=f"phase3a_{sr.slug.replace('-', '_')}", clients=sr.clients,
        client_script=str(HERE / "client.py"), client_args=args, results_dir=str(out),
        controller=controller)
    env = SimEnv(clients=sr.clients, workspace_root=str(config.sim_workspace_root()))
    result = recipe.execute(env)
    status = str(result.get_status())
    print("Result workspace:", result.get_result())
    print("Job status:      ", status)
    if "COMPLETED" not in status or not (out / "server" / "run.json").exists():
        raise SystemExit(f"FLARE spec job did not complete cleanly ({status})")
    print("Outputs:         ", out)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stratum", choices=["iss", "earth"])
    parser.add_argument("--counts", choices=["standard", "rrnarm"], default="standard")
    parser.add_argument("--n_cpus", type=int, default=config.N_CPUS_PER_CLIENT)
    parser.add_argument("--profile", choices=config.PROFILES, default="open",
                        help="policy profile: sets the manifests' policy_id (D-036)")
    parser.add_argument("--spec", type=Path, help="run a §6 job spec instead of the flags")
    args = parser.parse_args()
    if args.spec is not None:
        if args.stratum is not None:
            parser.error("--spec carries its own stratum and counts; don't combine with --stratum")
        run_spec(args.spec, profile=args.profile, n_cpus=args.n_cpus)
        return
    if args.stratum is None:
        parser.error("--stratum is required (or give --spec)")
    run(args.stratum, args.counts, n_cpus=args.n_cpus, profile=args.profile)


if __name__ == "__main__":
    main()
