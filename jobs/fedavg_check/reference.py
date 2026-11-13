"""In-process side of the FedAvg check, and the comparison report.

Runs the protocol from job.py (same data, init, rounds, local steps
and learning rate) with **no FLARE**: our own ``fedavg()`` does the aggregation. It then loads
FLARE's final global model and compares.

    uv run python jobs/fedavg_check/reference.py              # after job.py (both weightings)
    uv run python jobs/fedavg_check/reference.py --no-flare   # reference and baselines only

Rows in the report:
- FLARE (n_k): FLARE's FedAvg, with clients sending their sample count.
- FLARE (equal): FLARE's FedAvg, with clients omitting it, so FLARE silently weights equally.
- reference (n_k) / (equal): our fedavg() with the same two weightings.
- centralized GD: gradient descent on the pooled data with the same step budget
  (NUM_ROUNDS × LOCAL_STEPS steps), as a sanity baseline.
- centralized optimum: unpenalized logistic regression on the pooled data (sklearn), i.e. the
  best this model class can do on this training data.

The headline check is FLARE (n_k) == reference (n_k) within (RTOL, ATOL) from config.py. The exit
code is 1 if any FLARE result that exists fails to match its reference, or if the n_k result is
missing (unless --no-flare). These are training-set numbers: an equivalence check, not a
generalization estimate.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402

from fedkg_spacebio.federation.logreg import (  # noqa: E402
    accuracy,
    centralized_gd,
    log_loss,
    run_fedavg_inprocess,
)
from fedkg_spacebio.federation.synthetic import N_FEATURES, make_all_sites, pool  # noqa: E402


def flare_model_path(weighting: str) -> Path:
    return (
        config.sim_workspace_root()
        / config.job_name(weighting)
        / "server/simulate_job/models/server.npy"
    )


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-flare", action="store_true", help="skip loading FLARE results")
    parser.add_argument(
        "--out",
        type=Path,
        default=config.workdir() / "results" / "fedavg_check.json",
        help="where to write the JSON report (default: $FEDKG_WORKDIR/results/)",
    )
    args = parser.parse_args()

    sites = make_all_sites(config.SEED)
    X, y = pool(sites)
    theta0 = np.zeros(N_FEATURES + 1)
    protocol = dict(num_rounds=config.NUM_ROUNDS, local_steps=config.LOCAL_STEPS, lr=config.LR)

    models: dict[str, np.ndarray] = {}
    models["reference (n_k)"] = run_fedavg_inprocess(sites, theta0, weighting="samples", **protocol)
    models["reference (equal)"] = run_fedavg_inprocess(sites, theta0, weighting="equal", **protocol)
    models["centralized GD"] = centralized_gd(
        X, y, theta0, steps=config.NUM_ROUNDS * config.LOCAL_STEPS, lr=config.LR
    )
    # C=inf means no penalty (sklearn ≥1.8 deprecates penalty=None in favor of this).
    sk = LogisticRegression(C=np.inf, tol=1e-12, max_iter=100_000).fit(X, y)
    models["centralized optimum"] = np.append(sk.coef_[0], sk.intercept_[0])

    # Special case from methodology §5: with ONE full-batch local step per round and n_k weights,
    # FedAvg is exactly centralized gradient descent, because the pooled mean gradient is the
    # n_k-weighted mean of the site mean gradients.
    one_step_fed = run_fedavg_inprocess(
        sites, theta0, num_rounds=config.NUM_ROUNDS, local_steps=1, lr=config.LR
    )
    one_step_central = centralized_gd(X, y, theta0, steps=config.NUM_ROUNDS, lr=config.LR)

    checks: dict[str, dict] = {}
    failed = False
    if not args.no_flare:
        for weighting, label in (("samples", "FLARE (n_k)"), ("equal", "FLARE (equal)")):
            path = flare_model_path(weighting)
            if not path.exists():
                print(f"[missing] {label}: {path}")
                print(f"  run: uv run python jobs/fedavg_check/job.py --weighting {weighting}")
                failed |= weighting == "samples"
                continue
            # FLARE saves server.npy after every round, so a file can exist while a run is still
            # going (or after one crashed). Only trust it if the server log shows the run finished.
            server_log = path.parents[2] / "log.txt"
            if not server_log.exists() or "Finished FedAvg" not in server_log.read_text():
                print(f"[incomplete] {label}: no 'Finished FedAvg' in {server_log}")
                failed = True
                continue
            flare_theta = np.load(path)
            models[label] = flare_theta
            ref = models["reference (n_k)" if weighting == "samples" else "reference (equal)"]
            ok = bool(np.allclose(flare_theta, ref, rtol=config.RTOL, atol=config.ATOL))
            checks[f"{label} vs matching reference"] = {
                "max_abs_diff": max_abs_diff(flare_theta, ref),
                "allclose": ok,
                "flare_dtype": str(flare_theta.dtype),
                "path": str(path),
            }
            failed |= not ok

    # Does the check have teeth? n_k vs equal weighting should differ by far more than RTOL.
    checks["reference (n_k) vs reference (equal)"] = {
        "max_abs_diff": max_abs_diff(models["reference (n_k)"], models["reference (equal)"]),
    }
    checks["1-step FedAvg (n_k) vs centralized GD"] = {
        "max_abs_diff": max_abs_diff(one_step_fed, one_step_central),
        "allclose": bool(np.allclose(one_step_fed, one_step_central, rtol=1e-12, atol=1e-14)),
    }
    optimum = models["centralized optimum"]

    # ---- report ----
    sizes = "/".join(str(s.n_samples) for s in sites)
    print()
    print(
        f"FedAvg check: 3 synthetic sites (n = {sizes}), {config.NUM_ROUNDS} rounds x "
        f"{config.LOCAL_STEPS} local steps, lr {config.LR}, seed {config.SEED}"
    )
    print(f"tolerance: rtol={config.RTOL:g}, atol={config.ATOL:g} (see config.py)\n")
    header = (
        f"{'model':<22} {'theta = [w1, w2, w3, w4, b]':<48} "
        f"{'loss':>7} {'acc':>6} {'|Δ| to opt':>10}"
    )
    print(header)
    print("-" * len(header))
    order = ["FLARE (n_k)", "reference (n_k)", "FLARE (equal)", "reference (equal)",
             "centralized GD", "centralized optimum"]
    rows = {}
    for name in order:
        if name not in models:
            continue
        th = np.asarray(models[name], np.float64)
        rows[name] = {
            "theta": th.tolist(),
            "pooled_log_loss": log_loss(th, X, y),
            "pooled_accuracy": accuracy(th, X, y),
            "max_abs_diff_to_optimum": max_abs_diff(th, optimum),
        }
        r = rows[name]
        theta_s = "[" + ", ".join(f"{v:+.4f}" for v in th) + "]"
        print(
            f"{name:<22} {theta_s:<48} {r['pooled_log_loss']:7.4f} "
            f"{r['pooled_accuracy']:6.3f} {r['max_abs_diff_to_optimum']:10.4f}"
        )
    print()
    for name, c in checks.items():
        extra = f"  allclose={c['allclose']}" if "allclose" in c else ""
        print(f"{name:<44} max|Δθ| = {c['max_abs_diff']:.3e}{extra}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": {
            "seed": config.SEED, "num_rounds": config.NUM_ROUNDS,
            "local_steps": config.LOCAL_STEPS, "lr": config.LR,
            "site_sizes": [s.n_samples for s in sites],
            "rtol": config.RTOL, "atol": config.ATOL,
        },
        "models": rows,
        "checks": checks,
        "passed": not failed,
    }
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nreport: {args.out}")
    print("PASS" if not failed else "FAIL")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
