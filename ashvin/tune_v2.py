"""Optuna v2: tune multistart pipeline on small tests.

Tunes GD hyperparams + WL-priority legalization params.
Uses multithreaded trials on tests 1-5 (fastest, highest WL penalty).
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from ashvin.solver import solve, solve_multistart
from placement import calculate_normalized_metrics, generate_placement_input

TEST_CASES = {
    1: (2, 20, 1001),
    2: (3, 25, 1002),
    3: (2, 30, 1003),
    4: (3, 50, 1004),
    5: (4, 75, 1005),
    6: (5, 100, 1006),
    7: (5, 150, 1007),
    8: (7, 150, 1008),
    9: (8, 200, 1009),
}


def evaluate_config(config, test_ids, use_multistart=False):
    """Run solver with config on specified tests, return avg WL."""
    total_wl = 0
    total_ov = 0
    for test_id in test_ids:
        nm, ns, seed = TEST_CASES[test_id]
        torch.manual_seed(seed)
        cf, pf, el = generate_placement_input(nm, ns)
        N = cf.shape[0]
        area = cf[:, 0].sum().item()
        sr = (area ** 0.5) * 0.6
        a = torch.rand(N) * 2 * 3.14159
        r = torch.rand(N) * sr
        cf[:, 2] = r * torch.cos(a)
        cf[:, 3] = r * torch.sin(a)

        if use_multistart:
            result = solve_multistart(cf, pf, el, config=config)
        else:
            result = solve(cf, pf, el, config=config)
        m = calculate_normalized_metrics(result["final_cell_features"], pf, el)
        total_wl += m["normalized_wl"]
        total_ov += m["overlap_ratio"]

    n = len(test_ids)
    return total_ov / n, total_wl / n


def objective(trial):
    """Optuna objective for full pipeline tuning."""
    config = {
        # GD params
        "epochs": trial.suggest_int("epochs", 300, 1000, step=100),
        "lr": trial.suggest_float("lr", 0.001, 0.02, log=True),
        "lambda_wl": trial.suggest_float("lambda_wl", 1.0, 8.0),
        "lambda_overlap_start": trial.suggest_float("lambda_overlap_start", 0.5, 10.0),
        "lambda_overlap_end": trial.suggest_float("lambda_overlap_end", 30.0, 200.0),
        "lambda_density": trial.suggest_float("lambda_density", 0.0, 5.0),
        "beta_start": trial.suggest_float("beta_start", 0.05, 0.5),
        "beta_end": trial.suggest_float("beta_end", 1.0, 6.0),
        "warmup_epochs": trial.suggest_int("warmup_epochs", 50, 300, step=50),
        "lr_schedule": trial.suggest_categorical("lr_schedule", ["warmup", "warmup_cosine"]),
        # Pipeline params
        "pipeline_passes": trial.suggest_int("pipeline_passes", 1, 5),
        "repair_iterations": 200,
        # Skip expensive passes during tuning
        "_skip_scatter": True,   # scatter calls solve() recursively — 50% of runtime
        "_skip_global_swap": False,
        "gs_passes": trial.suggest_int("gs_passes", 3, 10),
    }

    # Evaluate
    avg_overlap, avg_wl = evaluate_config(config, objective.test_ids)

    # Primary: overlap = 0. Secondary: minimize WL.
    score = avg_wl + 100.0 * avg_overlap
    return score


def main():
    try:
        import optuna
    except ImportError:
        print("Install optuna: uv add optuna")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=80)
    parser.add_argument("--test-ids", type=str, default="1,2,3,4,5")
    parser.add_argument("--n-jobs", type=int, default=3, help="Parallel trials")
    parser.add_argument("--multistart", action="store_true", help="Use solve_multistart")
    parser.add_argument("--study-name", type=str, default="tune_v2")
    args = parser.parse_args()

    test_ids = [int(x) for x in args.test_ids.split(",")]
    objective.test_ids = test_ids

    print(f"Tuning on tests: {test_ids}")
    print(f"Trials: {args.n_trials}, parallel jobs: {args.n_jobs}")

    # Seed with previous best
    with open(Path(__file__).parent / "results" / "best_config.json") as f:
        prev_best = json.load(f)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Enqueue previous best as first trial
    study.enqueue_trial({
        "epochs": prev_best.get("epochs", 500),
        "lr": prev_best.get("lr", 0.003),
        "lambda_wl": prev_best.get("lambda_wl", 3.58),
        "lambda_overlap_start": prev_best.get("lambda_overlap_start", 1.23),
        "lambda_overlap_end": prev_best.get("lambda_overlap_end", 96.2),
        "lambda_density": prev_best.get("lambda_density", 1.64),
        "beta_start": prev_best.get("beta_start", 0.11),
        "beta_end": prev_best.get("beta_end", 2.03),
        "warmup_epochs": prev_best.get("warmup_epochs", 200),
        "lr_schedule": prev_best.get("lr_schedule", "warmup_cosine"),
        "pipeline_passes": 3,
        "gs_passes": 5,
    })

    study.optimize(objective, n_trials=args.n_trials,
                   n_jobs=args.n_jobs, show_progress_bar=True)

    print("\n" + "=" * 60)
    print("BEST TRIAL")
    print("=" * 60)
    print(f"Score: {study.best_trial.value:.4f}")
    print(f"Params: {study.best_trial.params}")

    # Evaluate best on all tests 1-9
    best_config = {
        **study.best_trial.params,
        "repair_iterations": 200,
        "_skip_global_swap": False,
    }

    print("\nEvaluating best config on all tests 1-9...")
    avg_overlap, avg_wl = evaluate_config(best_config, list(range(1, 10)))
    print(f"All tests avg: overlap={avg_overlap:.4f}, wl={avg_wl:.4f}")

    # Save
    config_path = Path(__file__).parent / "results" / "best_config_v2.json"
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"Best config saved: {config_path}")


if __name__ == "__main__":
    main()
