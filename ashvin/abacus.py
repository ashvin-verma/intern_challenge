"""Abacus legalization: minimize total displacement from GD positions.

Industrial-standard legalizer (Spindler et al., 2008). Instead of greedy
left-to-right packing, uses a cluster-merging DP within each row to find
positions that minimize sum of squared displacements from GD targets.

Key insight: GD already placed cells in approximately the right neighborhoods.
Abacus preserves those neighborhoods. Shelf-pack ignores them.

Algorithm per row:
1. Sort cells by GD x-position
2. Place each cell at its ideal (GD) position
3. If it overlaps the previous cluster, merge clusters
4. Merged cluster's position = weighted optimal that minimizes total displacement
5. Repeat until no overlaps remain
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def abacus_legalize(cell_features, num_macros=None, pin_features=None, edge_list=None):
    """Abacus legalization minimizing displacement from GD positions.

    Modifies cell_features[:, 2:4] in-place.
    Returns dict with stats.
    """
    start_time = time.perf_counter()
    N = cell_features.shape[0]
    if N <= 1:
        return {"time": 0.0, "cells_moved": 0, "max_displacement": 0.0}

    if num_macros is None:
        num_macros = (cell_features[:, 5] > 1.5).sum().item()

    positions = cell_features[:, 2:4].detach()
    widths = cell_features[:, 4].detach()
    heights = cell_features[:, 5].detach()
    original_positions = positions.clone()

    # Save GD target positions
    gd_x = positions[:, 0].clone()
    gd_y = positions[:, 1].clone()

    # --- Step 1: Legalize macros (same iterative push as before) ---
    if num_macros > 1:
        for _pass in range(200):
            any_ov = False
            for i in range(num_macros):
                for j in range(i + 1, num_macros):
                    xi, yi = positions[i, 0].item(), positions[i, 1].item()
                    xj, yj = positions[j, 0].item(), positions[j, 1].item()
                    wi, hi_v = widths[i].item(), heights[i].item()
                    wj, hj = widths[j].item(), heights[j].item()
                    dx, dy = xi - xj, yi - yj
                    ov_x = (wi + wj) / 2 - abs(dx)
                    ov_y = (hi_v + hj) / 2 - abs(dy)
                    if ov_x > 0 and ov_y > 0:
                        any_ov = True
                        if ov_x <= ov_y:
                            s = ov_x / 2 + 0.1
                            sign = 1.0 if dx >= 0 else -1.0
                            positions[i, 0] += sign * s
                            positions[j, 0] -= sign * s
                        else:
                            s = ov_y / 2 + 0.1
                            sign = 1.0 if dy >= 0 else -1.0
                            positions[i, 1] += sign * s
                            positions[j, 1] -= sign * s
            if not any_ov:
                break

    if num_macros >= N:
        cell_features[:, 2:4] = positions
        return {"time": time.perf_counter() - start_time, "cells_moved": 0, "max_displacement": 0.0}

    # --- Step 2: Collect macro obstacles ---
    obstacles = []
    for i in range(num_macros):
        ox, oy = positions[i, 0].item(), positions[i, 1].item()
        ow, oh = widths[i].item(), heights[i].item()
        obstacles.append((ox - ow / 2, oy - oh / 2, ox + ow / 2, oy + oh / 2))

    # --- Step 3: Assign std cells to rows ---
    std_indices = list(range(num_macros, N))
    row_height = 1.0
    all_y = gd_y[std_indices]
    y_min = all_y.min().item() - 10

    row_assignments = {}
    for idx in std_indices:
        y = gd_y[idx].item()
        row_idx = round((y - y_min) / row_height)
        if row_idx not in row_assignments:
            row_assignments[row_idx] = []
        row_assignments[row_idx].append(idx)

    # --- Step 4: Abacus DP per row ---
    for row_idx, cells_in_row in row_assignments.items():
        row_y = y_min + row_idx * row_height

        if not cells_in_row:
            continue

        # Sort by GD x-position
        cells_in_row.sort(key=lambda i: gd_x[i].item())

        # Build obstacle intervals for this row
        row_obstacles = []
        for ox_min, oy_min, ox_max, oy_max in obstacles:
            if oy_max > row_y - row_height / 2 and oy_min < row_y + row_height / 2:
                row_obstacles.append((ox_min, ox_max))
        row_obstacles.sort()

        # Abacus cluster-merging DP
        # Each cluster: list of (cell_idx, width), start_x, total_width
        clusters = []

        for ci in cells_in_row:
            wi = widths[ci].item()
            ideal_x = gd_x[ci].item()
            # Ideal left edge of this cell
            ideal_left = ideal_x - wi / 2

            # Create singleton cluster at ideal position
            new_cluster = {
                "cells": [ci],
                "widths": [wi],
                "total_w": wi,
                "left": ideal_left,  # left edge of cluster
                "ideal_sum": ideal_left,  # sum of ideal left edges (for weighted opt)
                "count": 1,
            }
            clusters.append(new_cluster)

            # Merge with previous clusters while overlapping
            while len(clusters) >= 2:
                prev = clusters[-2]
                cur = clusters[-1]
                prev_right = prev["left"] + prev["total_w"]

                if prev_right <= cur["left"] + 1e-6:
                    break  # no overlap

                # Merge: find optimal left-edge that minimizes displacement
                merged_cells = prev["cells"] + cur["cells"]
                merged_widths = prev["widths"] + cur["widths"]
                merged_total_w = prev["total_w"] + cur["total_w"]

                # Optimal cluster left = weighted average of ideal positions
                # Each cell i wants: cluster_left = ideal_x_i - cumulative_width_before_i - w_i/2
                # We minimize sum of (actual_x_i - ideal_x_i)^2
                # With cells packed left-to-right, actual_x_i = cluster_left + cumulative + w_i/2
                # So we want cluster_left = mean(ideal_x_i - cumulative_i - w_i/2)
                ideal_lefts_sum = 0.0
                cumulative = 0.0
                for ci2, wi2 in zip(merged_cells, merged_widths):
                    ideal_lefts_sum += gd_x[ci2].item() - wi2 / 2 - cumulative
                    cumulative += wi2

                opt_left = ideal_lefts_sum / len(merged_cells)

                # Don't let the cluster go left of the previous cluster's constrained position
                # (this prevents cascading leftward shifts)
                if len(clusters) >= 3:
                    prev_prev = clusters[-3]
                    min_left = prev_prev["left"] + prev_prev["total_w"]
                    opt_left = max(opt_left, min_left)

                merged = {
                    "cells": merged_cells,
                    "widths": merged_widths,
                    "total_w": merged_total_w,
                    "left": opt_left,
                    "ideal_sum": ideal_lefts_sum,
                    "count": len(merged_cells),
                }
                clusters.pop()
                clusters.pop()
                clusters.append(merged)

        # Assign positions from clusters
        for cluster in clusters:
            cur_left = cluster["left"]

            # Handle macro obstacles: shift cluster right if it overlaps
            for ox_min, ox_max in row_obstacles:
                cluster_right = cur_left + cluster["total_w"]
                if cluster_right > ox_min and cur_left < ox_max:
                    cur_left = ox_max + 0.1

            for ci2, wi2 in zip(cluster["cells"], cluster["widths"]):
                positions[ci2, 0] = cur_left + wi2 / 2
                positions[ci2, 1] = row_y
                cur_left += wi2

    # Write back
    cell_features[:, 2:4] = positions

    displacement = (positions - original_positions).abs()
    max_displacement = displacement.max().item()
    cells_moved = (displacement.sum(dim=1) > 0.01).sum().item()

    return {
        "time": time.perf_counter() - start_time,
        "cells_moved": cells_moved,
        "max_displacement": max_displacement,
    }
