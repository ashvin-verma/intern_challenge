"""Visualize placement results for specific test cases."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from ashvin.instrumented_train import instrumented_train_placement
from placement import (
    calculate_overlap_metrics,
    generate_placement_input,
)

# Same test cases as test.py
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
    11: (10, 10000, 1011),
    12: (10, 100000, 1012),
}

OUTPUT_DIR = Path(__file__).resolve().parent / "plots"


def plot_test(test_id, initial_features, final_features, num_macros, pin_features, edge_list, version=""):
    """Plot initial vs final placement with macro/std cell distinction and overlap highlighting."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    for ax, cell_features, title in [
        (ax1, initial_features, "Initial"),
        (ax2, final_features, "Final"),
    ]:
        N = cell_features.shape[0]
        positions = cell_features[:, 2:4].detach().numpy()
        widths = cell_features[:, 4].detach().numpy()
        heights = cell_features[:, 5].detach().numpy()

        # Find which cells have overlaps
        overlap_cells = set()
        for i in range(min(N, 3000)):  # cap for performance
            for j in range(i + 1, min(N, 3000)):
                dx = abs(positions[i, 0] - positions[j, 0])
                dy = abs(positions[i, 1] - positions[j, 1])
                if dx < (widths[i] + widths[j]) / 2 and dy < (heights[i] + heights[j]) / 2:
                    overlap_cells.add(i)
                    overlap_cells.add(j)

        # Draw cells: macros first (behind), then std cells
        for i in range(N):
            x = positions[i, 0] - widths[i] / 2
            y = positions[i, 1] - heights[i] / 2
            is_macro = i < num_macros
            has_overlap = i in overlap_cells

            if is_macro:
                facecolor = "#ff6b6b" if has_overlap else "#74b9ff"
                edgecolor = "#c0392b" if has_overlap else "#2980b9"
                lw = 1.5
                alpha = 0.5
                zorder = 1
            else:
                facecolor = "#ff8787" if has_overlap else "#a8e6cf"
                edgecolor = "#e74c3c" if has_overlap else "#27ae60"
                lw = 0.5
                alpha = 0.6
                zorder = 2

            rect = Rectangle(
                (x, y), widths[i], heights[i],
                fill=True, facecolor=facecolor, edgecolor=edgecolor,
                linewidth=lw, alpha=alpha, zorder=zorder,
            )
            ax.add_patch(rect)

        metrics = calculate_overlap_metrics(cell_features) if N <= 3000 else {"overlap_count": "?", "total_overlap_area": "?"}

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        overlap_str = f"{metrics['overlap_count']}" if isinstance(metrics['overlap_count'], int) else "?"
        area_str = f"{metrics['total_overlap_area']:.0f}" if isinstance(metrics.get('total_overlap_area', '?'), float) else "?"
        ax.set_title(f"{title}\nOverlap pairs: {overlap_str}, Area: {area_str}", fontsize=12)

        all_x = positions[:, 0]
        all_y = positions[:, 1]
        max_dim = max(widths.max(), heights.max())
        margin = max_dim * 0.5 + 5
        ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax.set_ylim(all_y.min() - margin, all_y.max() + margin)

    num_std = N - num_macros
    fig.suptitle(
        f"Test {test_id}: {num_macros} macros + {num_std} std cells (seed {TEST_CASES[test_id][2]})",
        fontsize=14, fontweight="bold",
    )

    # Legend
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor="#74b9ff", edgecolor="#2980b9", label="Macro (no overlap)"),
        Patch(facecolor="#ff6b6b", edgecolor="#c0392b", label="Macro (overlap)"),
        Patch(facecolor="#a8e6cf", edgecolor="#27ae60", label="Std cell (no overlap)"),
        Patch(facecolor="#ff8787", edgecolor="#e74c3c", label="Std cell (overlap)"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=4, fontsize=10)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    output_dir = OUTPUT_DIR / version if version else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"test_{test_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Visualize placement test cases")
    parser.add_argument(
        "--tests", type=str, default="1",
        help="Comma-separated test IDs (default: 1)",
    )
    parser.add_argument(
        "--version", type=str, default="",
        help="Version subfolder for plots (e.g., 'run2_scalable')",
    )
    parser.add_argument(
        "--lambda-density", type=float, default=0.0,
        help="Density loss weight (default: 0.0)",
    )
    parser.add_argument(
        "--two-stage", action="store_true",
        help="Use two-stage training (macros first)",
    )
    args = parser.parse_args()

    test_ids = [int(x) for x in args.tests.split(",")]

    for test_id in test_ids:
        if test_id not in TEST_CASES:
            print(f"Unknown test {test_id}, skipping")
            continue

        num_macros, num_std_cells, seed = TEST_CASES[test_id]
        total_cells = num_macros + num_std_cells

        if total_cells > 3000:
            print(f"Test {test_id} ({total_cells} cells) too large to visualize usefully, skipping")
            continue

        print(f"Test {test_id}: {num_macros} macros + {num_std_cells} std cells...")
        torch.manual_seed(seed)

        cell_features, pin_features, edge_list = generate_placement_input(
            num_macros, num_std_cells
        )

        # Same init as test.py
        total_area = cell_features[:, 0].sum().item()
        spread_radius = (total_area ** 0.5) * 0.6
        angles = torch.rand(total_cells) * 2 * 3.14159
        radii = torch.rand(total_cells) * spread_radius
        cell_features[:, 2] = radii * torch.cos(angles)
        cell_features[:, 3] = radii * torch.sin(angles)

        initial_features = cell_features.clone()

        if args.two_stage:
            from ashvin.instrumented_train import two_stage_train_placement
            result = two_stage_train_placement(
                cell_features, pin_features, edge_list,
            )
        else:
            result = instrumented_train_placement(
                cell_features, pin_features, edge_list, verbose=False,
                lambda_density=args.lambda_density,
            )

        plot_test(
            test_id, initial_features, result["final_cell_features"],
            num_macros, pin_features, edge_list, version=args.version,
        )


if __name__ == "__main__":
    main()
