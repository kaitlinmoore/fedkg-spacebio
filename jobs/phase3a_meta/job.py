"""Phase 3a FLARE job: federated DE meta-analysis for one stratum, in the FL simulator.

Run from WSL after ``fedkg``:

    uv run python jobs/phase3a_meta/job.py --stratum iss
    uv run python jobs/phase3a_meta/job.py --stratum earth
    uv run python jobs/phase3a_meta/job.py --stratum iss --counts rrnarm      # sensitivity
    uv run python jobs/phase3a_meta/central.py --stratum iss                  # Check 1

One FLARE client per (site, stratum) row of D-020 (``fedkg_spacebio.sites.manifest.CLIENTS``).
The server runs ``MetaAnalysisController``: one task, wait for every client, validate, pool.
In the ISS stratum the server also pools a ``primary_without_SpaceX-8`` variant from the same
summaries, because SpaceX-8's ISS euthanasia is inferred (D-020 caveat).

Outputs: ``$FEDKG_WORKDIR/results/phase3a/<stratum>/<counts>/flare/``.
"""

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402
from nvflare.recipe import SimEnv  # noqa: E402

from fedkg_spacebio.federation.meta_flare import MetaAnalysisRecipe  # noqa: E402
from fedkg_spacebio.sites.manifest import clients_in, write_manifests  # noqa: E402

DROP_VARIANTS = {"iss": ["SpaceX-8"], "earth": []}


def run(stratum: str, counts: str, root: Path | None = None, clients: list[str] | None = None,
        drop_variants: list[str] | None = None, n_cpus: int = config.N_CPUS_PER_CLIENT,
        verify_annotation: bool = True, build_manifests: bool = True) -> Path:
    if build_manifests:
        write_manifests(out_dir=config.manifest_dir(counts, root), workdir=root or config.workdir(),
                        counts_variant=counts)
    clients = clients or clients_in(stratum)
    out = config.results_dir(stratum, counts, "flare", root)
    if out.exists():
        shutil.rmtree(out)  # never mix outputs from two runs
    args = (f"--manifest_dir {config.manifest_dir(counts, root)} "
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stratum", choices=["iss", "earth"], required=True)
    parser.add_argument("--counts", choices=["standard", "rrnarm"], default="standard")
    parser.add_argument("--n_cpus", type=int, default=config.N_CPUS_PER_CLIENT)
    args = parser.parse_args()
    run(args.stratum, args.counts, n_cpus=args.n_cpus)


if __name__ == "__main__":
    main()
