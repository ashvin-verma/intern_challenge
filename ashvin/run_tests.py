"""Instrumented test runner with per-phase timing and CSV output."""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure repo root is on sys.path for placement imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from ashvin.device_utils import move_runtime_tensors
from ashvin.instrumented_train import instrumented_train_placement, two_stage_train_placement
from ashvin.solver import solve as annealed_solve, solve_multistart, solve_scatter
from placement import calculate_normalized_metrics, generate_placement_input

# Same test cases as test.py
TEST_CASES = [
    (1, 2, 20, 1001),
    (2, 3, 25, 1002),
    (3, 2, 30, 1003),
    (4, 3, 50, 1004),
    (5, 4, 75, 1005),
    (6, 5, 100, 1006),
    (7, 5, 150, 1007),
    (8, 7, 150, 1008),
    (9, 8, 200, 1009),
    (10, 10, 2000, 1010),
    (11, 10, 10000, 1011),
    (12, 10, 100000, 1012),
]

RESULTS_DIR = Path(__file__).resolve().parent / "results"

CSV_COLUMNS = [
    "timestamp",
    "test_id",
    "num_macros",
    "num_std_cells",
    "total_cells",
    "num_nets",
    "seed",
    "overlap_ratio",
    "num_cells_with_overlaps",
    "normalized_wl",
    "elapsed_time",
    "train_time",
    "wl_loss_time",
    "overlap_loss_time",
    "density_loss_time",
    "backward_time",
    "optimizer_time",
    "eval_time",
    "skipped_eval",
    "tag",
]


def run_single_test(test_id, num_macros, num_std_cells, seed, max_cells_for_eval=200000, lambda_density=0.0, two_stage=False, config=None, solver_type=None):
    """Run one test case with instrumented training."""
    torch.manual_seed(seed)
    runtime_config = dict(config) if config else {}

    cell_features, pin_features, edge_list = generate_placement_input(
        num_macros, num_std_cells
    )

    # Position init — must match test.py:83-91 exactly
    total_cells = cell_features.shape[0]
    total_area = cell_features[:, 0].sum().item()
    spread_radius = (total_area**0.5) * 0.6

    angles = torch.rand(total_cells) * 2 * 3.14159
    radii = torch.rand(total_cells) * spread_radius
    cell_features[:, 2] = radii * torch.cos(angles)
    cell_features[:, 3] = radii * torch.sin(angles)
    cell_features, pin_features, edge_list, _runtime_device, _runtime_reason = move_runtime_tensors(
        cell_features, pin_features, edge_list, config=runtime_config, verbose=True
    )

    # Instrumented training
    start_time = time.perf_counter()
    if solver_type == "scatter":
        result = solve_scatter(
            cell_features, pin_features, edge_list,
            config=runtime_config, verbose=True,
        )
    elif solver_type == "multistart":
        result = solve_multistart(
            cell_features, pin_features, edge_list,
            config=runtime_config, verbose=True,
        )
    elif solver_type == "annealed":
        result = annealed_solve(
            cell_features, pin_features, edge_list,
            config=runtime_config,
        )
    elif two_stage or config is not None:
        result = two_stage_train_placement(
            cell_features, pin_features, edge_list,
            config=runtime_config,
        )
    else:
        result = instrumented_train_placement(
            cell_features, pin_features, edge_list, verbose=False,
            lambda_density=lambda_density,
        )
    train_end = time.perf_counter()

    timing = result["timing"]
    final_cell_features = result["final_cell_features"]

    # Evaluation
    skipped_eval = total_cells > max_cells_for_eval
    if skipped_eval:
        overlap_ratio = None
        num_cells_with_overlaps = None
        normalized_wl = None
        eval_time = 0.0
    else:
        eval_start = time.perf_counter()
        metrics = calculate_normalized_metrics(
            final_cell_features, pin_features, edge_list
        )
        eval_time = time.perf_counter() - eval_start
        overlap_ratio = metrics["overlap_ratio"]
        num_cells_with_overlaps = metrics["num_cells_with_overlaps"]
        normalized_wl = metrics["normalized_wl"]

    elapsed_time = time.perf_counter() - start_time

    return {
        "test_id": test_id,
        "num_macros": num_macros,
        "num_std_cells": num_std_cells,
        "total_cells": total_cells,
        "num_nets": edge_list.shape[0],
        "seed": seed,
        "overlap_ratio": overlap_ratio,
        "num_cells_with_overlaps": num_cells_with_overlaps,
        "normalized_wl": normalized_wl,
        "elapsed_time": elapsed_time,
        "train_time": timing["total_train_time"],
        "wl_loss_time": timing["wl_loss_time"],
        "overlap_loss_time": timing["overlap_loss_time"],
        "density_loss_time": timing.get("density_loss_time", 0.0),
        "backward_time": timing["backward_time"],
        "optimizer_time": timing["optimizer_time"],
        "eval_time": eval_time,
        "skipped_eval": skipped_eval,
    }


def run_all_tests(test_ids=None, max_cells_for_eval=200000, lambda_density=0.0, two_stage=False, config=None, solver_type=None):
    """Run specified tests (or all) and return results."""
    cases = TEST_CASES
    if test_ids:
        cases = [c for c in TEST_CASES if c[0] in test_ids]

    print("=" * 70)
    print("INSTRUMENTED PLACEMENT TEST SUITE")
    print("=" * 70)
    print(f"\nRunning {len(cases)} test cases...")
    print()

    results = []
    for idx, (test_id, num_macros, num_std_cells, seed) in enumerate(cases, 1):
        size = (
            "Small"
            if num_std_cells <= 30
            else "Medium"
            if num_std_cells <= 100
            else "Large"
        )
        print(
            f"Test {idx}/{len(cases)}: {size} "
            f"({num_macros} macros, {num_std_cells} std cells, seed={seed})"
        )

        result = run_single_test(
            test_id, num_macros, num_std_cells, seed, max_cells_for_eval,
            lambda_density=lambda_density, two_stage=two_stage, config=config,
            solver_type=solver_type,
        )
        results.append(result)

        # Print per-test summary
        if result["skipped_eval"]:
            print(f"  Overlap: SKIPPED (>{max_cells_for_eval} cells)")
        else:
            status = (
                "PASS" if result["num_cells_with_overlaps"] == 0 else "FAIL"
            )
            print(
                f"  Overlap: {result['overlap_ratio']:.4f} "
                f"({result['num_cells_with_overlaps']}/{result['total_cells']}) "
                f"[{status}]"
            )
            print(f"  Norm WL: {result['normalized_wl']:.4f}")

        print(
            f"  Time: {result['elapsed_time']:.2f}s "
            f"(train={result['train_time']:.2f}s, eval={result['eval_time']:.2f}s)"
        )
        print(
            f"  Breakdown: wl={result['wl_loss_time']:.2f}s "
            f"overlap={result['overlap_loss_time']:.2f}s "
            f"backward={result['backward_time']:.2f}s "
            f"optim={result['optimizer_time']:.2f}s"
        )
        print()

    return results


def print_summary(results):
    """Print aggregate summary."""
    evaluated = [r for r in results if not r["skipped_eval"]]

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if evaluated:
        avg_overlap = sum(r["overlap_ratio"] for r in evaluated) / len(evaluated)
        avg_wl = sum(r["normalized_wl"] for r in evaluated) / len(evaluated)
        print(f"Avg Overlap (evaluated): {avg_overlap:.4f}")
        print(f"Avg Norm WL (evaluated): {avg_wl:.4f}")

    total_time = sum(r["elapsed_time"] for r in results)
    total_train = sum(r["train_time"] for r in results)
    total_wl = sum(r["wl_loss_time"] for r in results)
    total_overlap = sum(r["overlap_loss_time"] for r in results)
    total_backward = sum(r["backward_time"] for r in results)
    total_optim = sum(r["optimizer_time"] for r in results)
    total_eval = sum(r["eval_time"] for r in results)

    print(f"Total time: {total_time:.2f}s")
    print(f"  Training: {total_train:.2f}s")
    print(f"    WL loss:      {total_wl:.2f}s")
    print(f"    Overlap loss: {total_overlap:.2f}s")
    print(f"    Backward:     {total_backward:.2f}s")
    print(f"    Optimizer:    {total_optim:.2f}s")
    print(f"  Evaluation: {total_eval:.2f}s")

    skipped = len(results) - len(evaluated)
    if skipped:
        print(f"  Skipped eval: {skipped} tests")


def save_results_csv(results, tag=""):
    """Save results to CSV in ashvin/results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    path = RESULTS_DIR / f"{ts}{suffix}.csv"

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in results:
            row = dict(r)
            row["timestamp"] = ts
            row["tag"] = tag
            writer.writerow(row)

    print(f"\nCSV saved: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Instrumented placement test runner")
    parser.add_argument(
        "--tests",
        type=str,
        default=None,
        help="Comma-separated test IDs to run (default: all)",
    )
    parser.add_argument("--tag", type=str, default="", help="Tag for CSV filename")
    parser.add_argument(
        "--max-cells",
        type=int,
        default=200000,
        help="Skip eval above this cell count (default: 200000)",
    )
    parser.add_argument(
        "--lambda-density",
        type=float,
        default=0.0,
        help="Weight for density loss (default: 0.0, disabled)",
    )
    parser.add_argument(
        "--two-stage",
        action="store_true",
        help="Use two-stage training (macros first, then std cells)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config preset name (default, aggressive, balanced, annealed) or JSON file path",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default=None,
        choices=["annealed", "multistart", "scatter"],
        help="Solver type (annealed = single-stage competitor-inspired)",
    )
    parser.add_argument(
        "--no-wl-polish",
        action="store_true",
        help="Skip WL polish + cell swap (faster)",
    )
    args = parser.parse_args()

    test_ids = None
    if args.tests:
        test_ids = [int(x) for x in args.tests.split(",")]

    # Load config
    solver_config = {}
    if args.no_wl_polish:
        solver_config["_skip_wl_polish"] = True
    if args.config:
        from ashvin.config import PRESETS
        if args.config in PRESETS:
            solver_config = PRESETS[args.config]
        else:
            import json
            with open(args.config) as f:
                solver_config = json.load(f)

    results = run_all_tests(
        test_ids=test_ids, max_cells_for_eval=args.max_cells,
        lambda_density=args.lambda_density, two_stage=args.two_stage,
        config=solver_config or None, solver_type=args.solver,
    )
    print_summary(results)
    save_results_csv(results, tag=args.tag)


if __name__ == "__main__":
    main()
