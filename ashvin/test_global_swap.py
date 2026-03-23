"""Quick test: measure global swap impact per-test with before/after plots."""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from placement import calculate_normalized_metrics, generate_placement_input
from ashvin.solver import solve as annealed_solve
from ashvin.global_swap import global_swap, edge_targeted_swap

PLOTS_DIR = Path(__file__).resolve().parent / "plots" / "global_swap"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

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
]

CONFIG_PATH = Path(__file__).resolve().parent / "results" / "best_config.json"


def plot_placement(cell_features, pin_features, edge_list, title, filepath,
                   highlight_edges=None):
    """Plot placement with optional edge highlighting."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    positions = cell_features[:, 2:4].detach()
    widths = cell_features[:, 4].detach()
    heights = cell_features[:, 5].detach()
    N = cell_features.shape[0]
    num_macros = (cell_features[:, 5] > 1.5).sum().item()

    # Draw edges (light gray for normal, red for worst)
    pin_to_cell = pin_features[:, 0].long()
    if highlight_edges is not None:
        highlight_set = set(highlight_edges)
    else:
        highlight_set = set()

    for e in range(min(edge_list.shape[0], 5000)):
        sp, tp = edge_list[e, 0].item(), edge_list[e, 1].item()
        sc, tc = pin_to_cell[sp].item(), pin_to_cell[tp].item()
        x1 = positions[sc, 0].item() + pin_features[sp, 1].item()
        y1 = positions[sc, 1].item() + pin_features[sp, 2].item()
        x2 = positions[tc, 0].item() + pin_features[tp, 1].item()
        y2 = positions[tc, 1].item() + pin_features[tp, 2].item()
        color = "red" if e in highlight_set else "#cccccc"
        alpha = 0.8 if e in highlight_set else 0.15
        lw = 1.5 if e in highlight_set else 0.3
        ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=lw, zorder=1)

    # Draw cells
    for i in range(N):
        x, y = positions[i, 0].item(), positions[i, 1].item()
        w, h = widths[i].item(), heights[i].item()
        color = "#4488cc" if i >= num_macros else "#cc4444"
        alpha = 0.6 if i >= num_macros else 0.8
        rect = plt.Rectangle((x - w/2, y - h/2), w, h,
                              facecolor=color, edgecolor="black",
                              alpha=alpha, linewidth=0.3, zorder=2)
        ax.add_patch(rect)

    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(filepath, dpi=120)
    plt.close()


def find_worst_edges(cell_features, pin_features, edge_list, top_k=50):
    """Return indices of top_k worst edges by Manhattan WL."""
    positions = cell_features[:, 2:4].detach()
    pin_to_cell = pin_features[:, 0].long()
    E = edge_list.shape[0]

    edge_wls = []
    for e in range(E):
        sp, tp = edge_list[e, 0].item(), edge_list[e, 1].item()
        sc, tc = pin_to_cell[sp].item(), pin_to_cell[tp].item()
        dx = abs(positions[sc, 0].item() + pin_features[sp, 1].item()
                 - positions[tc, 0].item() - pin_features[tp, 1].item())
        dy = abs(positions[sc, 1].item() + pin_features[sp, 2].item()
                 - positions[tc, 1].item() - pin_features[tp, 2].item())
        edge_wls.append((dx + dy, e))

    edge_wls.sort(reverse=True)
    return [e for _, e in edge_wls[:top_k]]


def run_test(test_id, num_macros, num_std_cells, seed, config):
    """Run solver WITHOUT global swap, then apply global swap passes manually."""
    torch.manual_seed(seed)
    cell_features, pin_features, edge_list = generate_placement_input(num_macros, num_std_cells)

    N = cell_features.shape[0]
    total_area = cell_features[:, 0].sum().item()
    spread_radius = (total_area ** 0.5) * 0.6
    angles = torch.rand(N) * 2 * 3.14159
    radii = torch.rand(N) * spread_radius
    cell_features[:, 2] = radii * torch.cos(angles)
    cell_features[:, 3] = radii * torch.sin(angles)

    # Run solver WITHOUT global swap
    no_gs_config = dict(config)
    no_gs_config["_skip_global_swap"] = True
    result = annealed_solve(cell_features, pin_features, edge_list,
                            config=no_gs_config, verbose=False)
    cell_features = result["final_cell_features"]

    # Measure baseline
    m0 = calculate_normalized_metrics(cell_features, pin_features, edge_list)
    wl_baseline = m0["normalized_wl"]
    overlap_baseline = m0["overlap_ratio"]

    print(f"  Baseline: overlap={overlap_baseline:.4f} wl={wl_baseline:.4f}")

    # Plot baseline with worst edges highlighted
    worst_edges = find_worst_edges(cell_features, pin_features, edge_list, top_k=30)
    plot_placement(cell_features, pin_features, edge_list,
                   f"Test {test_id} — Before global swap (WL={wl_baseline:.4f})",
                   PLOTS_DIR / f"test{test_id}_0_before.png",
                   highlight_edges=worst_edges)

    # Apply edge-targeted swap
    cf_backup = cell_features.clone()
    t0 = time.perf_counter()
    et_stats = edge_targeted_swap(cell_features, pin_features, edge_list,
                                  num_passes=5, top_edge_frac=0.2, verbose=True)
    t_et = time.perf_counter() - t0
    m1 = calculate_normalized_metrics(cell_features, pin_features, edge_list)

    if m1["overlap_ratio"] > 0:
        cell_features[:] = cf_backup
        m1 = m0
        print(f"  Edge-targeted: REVERTED (overlap)")
    else:
        print(f"  Edge-targeted: {et_stats['swaps']} swaps, "
              f"wl={m1['normalized_wl']:.4f} ({t_et:.1f}s)")

    worst_edges_1 = find_worst_edges(cell_features, pin_features, edge_list, top_k=30)
    plot_placement(cell_features, pin_features, edge_list,
                   f"Test {test_id} — After edge-targeted swap "
                   f"({et_stats['swaps']} swaps, WL={m1['normalized_wl']:.4f})",
                   PLOTS_DIR / f"test{test_id}_1_edge_swap.png",
                   highlight_edges=worst_edges_1)

    # Apply global (barycentric) swap
    cf_backup2 = cell_features.clone()
    t0 = time.perf_counter()
    gs_stats = global_swap(cell_features, pin_features, edge_list,
                           num_passes=5, top_frac=0.5, search_radius=3, verbose=True)
    t_gs = time.perf_counter() - t0
    m2 = calculate_normalized_metrics(cell_features, pin_features, edge_list)

    if m2["overlap_ratio"] > 0:
        cell_features[:] = cf_backup2
        m2 = m1
        print(f"  Global swap: REVERTED (overlap)")
    else:
        print(f"  Global swap: {gs_stats['swaps']} swaps, "
              f"wl={m2['normalized_wl']:.4f} ({t_gs:.1f}s)")

    worst_edges_2 = find_worst_edges(cell_features, pin_features, edge_list, top_k=30)
    plot_placement(cell_features, pin_features, edge_list,
                   f"Test {test_id} — After global swap "
                   f"({gs_stats['swaps']} swaps, WL={m2['normalized_wl']:.4f})",
                   PLOTS_DIR / f"test{test_id}_2_global_swap.png",
                   highlight_edges=worst_edges_2)

    return {
        "test_id": test_id,
        "N": N,
        "wl_baseline": wl_baseline,
        "wl_after_edge": m1["normalized_wl"],
        "wl_after_global": m2["normalized_wl"],
        "edge_swaps": et_stats["swaps"],
        "global_swaps": gs_stats["swaps"],
        "edge_time": t_et,
        "global_time": t_gs,
        "overlap_baseline": overlap_baseline,
        "overlap_final": m2["overlap_ratio"],
    }


def _run_test_wrapper(args):
    """Wrapper for multiprocessing."""
    test_id, nm, nsc, seed, config = args
    return run_test(test_id, nm, nsc, seed, config)


def main():
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    test_ids = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(1, 11))
    cases = [c for c in TEST_CASES if c[0] in test_ids]
    parallel = "--parallel" in sys.argv or "-p" in sys.argv

    print(f"Running {len(cases)} tests with global swap analysis" +
          (" (parallel)" if parallel else ""))
    print("=" * 70)

    all_results = []
    if parallel and len(cases) > 1:
        # Split: run small tests (1-9) in parallel, test 10 separately
        small_cases = [c for c in cases if c[2] <= 200]   # std_cells <= 200
        large_cases = [c for c in cases if c[2] > 200]

        workers = min(len(small_cases), 6)
        if small_cases:
            print(f"Running {len(small_cases)} small tests with {workers} workers...")
            args_list = [(tid, nm, nsc, seed, config) for tid, nm, nsc, seed in small_cases]
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_run_test_wrapper, a): a[0] for a in args_list}
                for future in as_completed(futures):
                    tid = futures[future]
                    r = future.result()
                    all_results.append(r)
                    print(f"  Test {tid} done: {r['wl_baseline']:.4f} -> {r['wl_after_global']:.4f}")

        for test_id, nm, nsc, seed in large_cases:
            print(f"\nTest {test_id} ({nm} macros, {nsc} std cells) — running sequentially")
            r = run_test(test_id, nm, nsc, seed, config)
            all_results.append(r)

        all_results.sort(key=lambda r: r["test_id"])
    else:
        for test_id, nm, nsc, seed in cases:
            print(f"\nTest {test_id} ({nm} macros, {nsc} std cells)")
            r = run_test(test_id, nm, nsc, seed, config)
            all_results.append(r)

    # Summary
    print("\n" + "=" * 70)
    print("GLOBAL SWAP SUMMARY")
    print("=" * 70)
    print(f"{'Test':>4} {'N':>5} {'Baseline':>10} {'Edge':>10} {'Global':>10} "
          f"{'Improve':>10} {'Swaps':>8} {'Time':>8}")
    print("-" * 70)

    total_improve = 0
    for r in all_results:
        improve = r["wl_baseline"] - r["wl_after_global"]
        total_improve += improve
        print(f"{r['test_id']:>4} {r['N']:>5} {r['wl_baseline']:>10.4f} "
              f"{r['wl_after_edge']:>10.4f} {r['wl_after_global']:>10.4f} "
              f"{improve:>+10.4f} {r['edge_swaps']+r['global_swaps']:>8} "
              f"{r['edge_time']+r['global_time']:>7.1f}s")

    avg_baseline = sum(r["wl_baseline"] for r in all_results) / len(all_results)
    avg_final = sum(r["wl_after_global"] for r in all_results) / len(all_results)
    print("-" * 70)
    print(f"{'AVG':>4} {'':>5} {avg_baseline:>10.4f} {'':>10} {avg_final:>10.4f} "
          f"{avg_baseline-avg_final:>+10.4f}")

    # Create summary plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    tests = [r["test_id"] for r in all_results]
    wl_base = [r["wl_baseline"] for r in all_results]
    wl_edge = [r["wl_after_edge"] for r in all_results]
    wl_global = [r["wl_after_global"] for r in all_results]

    x = range(len(tests))
    w = 0.25
    axes[0].bar([i - w for i in x], wl_base, w, label="Baseline", color="#cc4444")
    axes[0].bar(list(x), wl_edge, w, label="+ Edge swap", color="#44cc44")
    axes[0].bar([i + w for i in x], wl_global, w, label="+ Global swap", color="#4444cc")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels([f"T{t}" for t in tests])
    axes[0].set_ylabel("Normalized WL")
    axes[0].set_title("WL by test: before vs after global swap")
    axes[0].legend()
    axes[0].axhline(y=0.131, color="gold", linestyle="--", alpha=0.7, label="#1 target")

    improvements = [r["wl_baseline"] - r["wl_after_global"] for r in all_results]
    colors = ["#44cc44" if imp > 0 else "#cc4444" for imp in improvements]
    axes[1].bar(list(x), improvements, color=colors)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels([f"T{t}" for t in tests])
    axes[1].set_ylabel("WL improvement")
    axes[1].set_title("Per-test WL improvement from global swap")
    axes[1].axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "summary.png", dpi=120)
    plt.close()
    print(f"\nPlots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
