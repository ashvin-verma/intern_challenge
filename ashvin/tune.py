"""Optuna hyperparameter tuning for the placement solver.

Usage:
    uv run python ashvin/tune.py --n-trials 50 --test-ids 1,4,8
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from ashvin.solver import solve
from placement import calculate_normalized_metrics, generate_placement_input

# Test cases (same as test.py)
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
    10: (10, 2000, 1010),
}


def evaluate_config(config, test_ids, timeout=120):
    """Run solver with config on specified tests, return avg metrics."""
    results = []
    for test_id in test_ids:
        nm, ns, seed = TEST_CASES[test_id]
        torch.manual_seed(seed)
        cf, pf, el = generate_placement_input(nm, ns)
        N = cf.shape[0]
        area = cf[:, 0].sum().item()
        sr = (area**0.5) * 0.6
        a = torch.rand(N) * 2 * 3.14159
        r = torch.rand(N) * sr
        cf[:, 2] = r * torch.cos(a)
        cf[:, 3] = r * torch.sin(a)

        result = solve(cf, pf, el, config=config)
        m = calculate_normalized_metrics(result["final_cell_features"], pf, el)
        results.append(m)

    avg_overlap = sum(r["overlap_ratio"] for r in results) / len(results)
    avg_wl = sum(r["normalized_wl"] for r in results) / len(results)
    return avg_overlap, avg_wl


def objective(trial):
    """Optuna objective: minimize overlap first, then WL."""
    config = {
        "epochs": trial.suggest_int("epochs", 500, 2500, step=500),
        "lr": trial.suggest_float("lr", 0.003, 0.05, log=True),
        "lambda_wl": trial.suggest_float("lambda_wl", 0.5, 5.0),
        "lambda_overlap_start": trial.suggest_float("lambda_overlap_start", 1.0, 20.0),
        "lambda_overlap_end": trial.suggest_float("lambda_overlap_end", 50.0, 300.0),
        "lambda_density": trial.suggest_float("lambda_density", 0.0, 5.0),
        "beta_start": trial.suggest_float("beta_start", 0.05, 1.0),
        "beta_end": trial.suggest_float("beta_end", 2.0, 10.0),
        "warmup_epochs": trial.suggest_int("warmup_epochs", 20, 200, step=20),
        "lr_schedule": trial.suggest_categorical("lr_schedule", ["warmup", "warmup_cosine", "constant"]),
        "repair_iterations": 100,
        "_skip_wl_polish": True,  # skip slow post-processing during tuning
    }

    # Evaluate on a subset of tests for speed
    avg_overlap, avg_wl = evaluate_config(config, objective.test_ids)

    # Primary: overlap must be 0. Secondary: minimize WL.
    # Penalize any non-zero overlap heavily.
    score = avg_wl + 100.0 * avg_overlap
    return score


def main():
    try:
        import optuna
    except ImportError:
        print("Install optuna: uv add optuna")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Optuna tuning for placement solver")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials")
    parser.add_argument(
        "--test-ids", type=str, default="1,4,7,8",
        help="Comma-separated test IDs for tuning (default: 1,4,7,8)",
    )
    parser.add_argument("--study-name", type=str, default="placement_tune")
    args = parser.parse_args()

    test_ids = [int(x) for x in args.test_ids.split(",")]
    objective.test_ids = test_ids

    print(f"Tuning on tests: {test_ids}")
    print(f"Trials: {args.n_trials}")

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    print("\n" + "=" * 60)
    print("BEST TRIAL")
    print("=" * 60)
    print(f"Score: {study.best_trial.value:.4f}")
    print(f"Params: {study.best_trial.params}")

    # Evaluate best config on all tests 1-10
    best_config = {
        **study.best_trial.params,
        "repair_iterations": 200,
    }
    print("\nEvaluating best config on all tests 1-10...")
    avg_overlap, avg_wl = evaluate_config(best_config, list(range(1, 11)))
    print(f"All tests avg: overlap={avg_overlap:.4f}, wl={avg_wl:.4f}")

    # Save best config
    import json
    config_path = Path(__file__).parent / "results" / "best_config.json"
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"Best config saved: {config_path}")


if __name__ == "__main__":
    main()
