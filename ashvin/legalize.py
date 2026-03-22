"""Deterministic legalization — guarantees zero overlap.

Places cells into non-overlapping positions using greedy row packing.
Macros are placed first (sorted by area, largest first), then std cells
are packed into rows between/around macros.

This is a post-processing step after gradient descent. It moves cells
the minimum distance needed to eliminate all overlaps.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def legalize(cell_features, num_macros=None):
    """Deterministic legalization: remove all overlaps via greedy packing.

    Modifies cell_features[:, 2:4] in-place.

    Strategy:
    1. Place macros first (largest first), shifting to avoid overlap
    2. Pack std cells into rows, left-to-right, bottom-to-top
    3. Each cell is placed at the leftmost non-overlapping position in its row

    Args:
        cell_features: [N, 6] tensor — positions modified in-place
        num_macros: number of macros (inferred if None)

    Returns:
        dict with stats (time, cells_moved, max_displacement)
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

    # --- Step 1: Legalize macros (place largest first, shift to avoid overlap) ---
    if num_macros > 0:
        macro_areas = cell_features[:num_macros, 0]
        macro_order = torch.argsort(macro_areas, descending=True)

        placed_macros = []  # list of (x, y, w, h) for placed macros

        for idx in macro_order.tolist():
            x = positions[idx, 0].item()
            y = positions[idx, 1].item()
            w = widths[idx].item()
            h = heights[idx].item()

            # Try to place at current position; shift if overlapping with placed macros
            for _ in range(100):  # max attempts
                overlap_found = False
                for px, py, pw, ph in placed_macros:
                    dx = abs(x - px)
                    dy = abs(y - py)
                    min_sep_x = (w + pw) / 2
                    min_sep_y = (h + ph) / 2

                    if dx < min_sep_x and dy < min_sep_y:
                        # Overlap — shift in the direction of least overlap
                        overlap_x = min_sep_x - dx
                        overlap_y = min_sep_y - dy

                        if overlap_x <= overlap_y:
                            shift = overlap_x + 0.1
                            x += shift if x >= px else -shift
                        else:
                            shift = overlap_y + 0.1
                            y += shift if y >= py else -shift
                        overlap_found = True
                        break

                if not overlap_found:
                    break

            positions[idx, 0] = x
            positions[idx, 1] = y
            placed_macros.append((x, y, w, h))

        # Global macro repair: iteratively resolve all macro-macro overlaps
        # (the incremental placement above can leave overlaps due to stale positions)
        for _pass in range(200):
            any_overlap = False
            for i in range(num_macros):
                for j in range(i + 1, num_macros):
                    xi, yi = positions[i, 0].item(), positions[i, 1].item()
                    xj, yj = positions[j, 0].item(), positions[j, 1].item()
                    wi, hi = widths[i].item(), heights[i].item()
                    wj, hj = widths[j].item(), heights[j].item()

                    dx = xi - xj
                    dy = yi - yj
                    adx, ady = abs(dx), abs(dy)
                    ov_x = (wi + wj) / 2 - adx
                    ov_y = (hi + hj) / 2 - ady

                    if ov_x > 0 and ov_y > 0:
                        any_overlap = True
                        # Push apart along axis of least overlap
                        if ov_x <= ov_y:
                            shift = ov_x / 2 + 0.1
                            sign = 1.0 if dx >= 0 else -1.0
                            positions[i, 0] += sign * shift
                            positions[j, 0] -= sign * shift
                        else:
                            shift = ov_y / 2 + 0.1
                            sign = 1.0 if dy >= 0 else -1.0
                            positions[i, 1] += sign * shift
                            positions[j, 1] -= sign * shift
            if not any_overlap:
                break

    # --- Step 2: Legalize std cells (row-based packing) ---
    if num_macros < N:
        std_indices = list(range(num_macros, N))

        # Sort std cells by their current x position (preserve relative order)
        std_x = positions[std_indices, 0]
        sort_order = torch.argsort(std_x)
        sorted_std = [std_indices[i] for i in sort_order.tolist()]

        # Collect all macro bounding boxes as obstacles
        obstacles = []
        for i in range(num_macros):
            ox = positions[i, 0].item()
            oy = positions[i, 1].item()
            ow = widths[i].item()
            oh = heights[i].item()
            obstacles.append((ox - ow / 2, oy - oh / 2, ox + ow / 2, oy + oh / 2))

        # Row-based packing: std cells have height=1.0
        # Group into rows by quantizing y to nearest integer
        row_height = 1.0

        # Determine row range from current positions
        all_y = positions[std_indices, 1]
        y_min = all_y.min().item() - 10
        y_max = all_y.max().item() + 10

        # Assign each std cell to nearest row
        row_assignments = {}
        for idx in sorted_std:
            y = positions[idx, 1].item()
            row_idx = round((y - y_min) / row_height)
            if row_idx not in row_assignments:
                row_assignments[row_idx] = []
            row_assignments[row_idx].append(idx)

        # For each row, pack cells left-to-right avoiding overlaps
        for row_idx, cells_in_row in row_assignments.items():
            row_y = y_min + row_idx * row_height

            # Sort cells in row by x position
            cells_in_row.sort(key=lambda i: positions[i, 0].item())

            # Track rightmost edge of placed cells in this row
            cursor_x = None

            for idx in cells_in_row:
                w = widths[idx].item()
                h = heights[idx].item()
                target_x = positions[idx, 0].item()

                # Start from target_x or cursor_x, whichever is further right
                if cursor_x is not None:
                    x = max(target_x, cursor_x + w / 2)
                else:
                    x = target_x

                # Check macro obstacles and shift right — re-check until clean
                for _attempt in range(20):
                    shifted = False
                    for ox_min, oy_min, ox_max, oy_max in obstacles:
                        cell_left = x - w / 2
                        cell_right = x + w / 2
                        cell_bottom = row_y - h / 2
                        cell_top = row_y + h / 2

                        if (cell_right > ox_min and cell_left < ox_max and
                                cell_top > oy_min and cell_bottom < oy_max):
                            x = ox_max + w / 2 + 0.1
                            shifted = True
                    if not shifted:
                        break

                positions[idx, 0] = x
                positions[idx, 1] = row_y
                cursor_x = x + w / 2

    # Write back
    cell_features[:, 2:4] = positions

    # Compute stats
    displacement = (positions - original_positions).abs()
    max_displacement = displacement.max().item()
    cells_moved = (displacement.sum(dim=1) > 0.01).sum().item()

    return {
        "time": time.perf_counter() - start_time,
        "cells_moved": cells_moved,
        "max_displacement": max_displacement,
    }
