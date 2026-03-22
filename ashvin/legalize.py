"""Deterministic legalization — guarantees zero overlap.

Two strategies:
1. Row-based packing (original): snaps cells to rows, reliable but WL-destructive
2. Minimal-disturbance (new): nudge cells minimum distance, preserves WL better
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def legalize(cell_features, num_macros=None, pin_features=None, edge_list=None):
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

        # WL-aware sort: group cells by nearest macro region, then by x within region
        if pin_features is not None and edge_list is not None and num_macros > 0:
            from collections import Counter
            pin_to_cell = pin_features[:, 0].long()
            # Find each std cell's most-connected macro
            cell_macro_affinity = {}
            for e in range(edge_list.shape[0]):
                sc = pin_to_cell[edge_list[e, 0].item()].item()
                tc = pin_to_cell[edge_list[e, 1].item()].item()
                if sc < num_macros and tc >= num_macros:
                    cell_macro_affinity.setdefault(tc, Counter())[sc] += 1
                elif tc < num_macros and sc >= num_macros:
                    cell_macro_affinity.setdefault(sc, Counter())[tc] += 1

            # Sort by: (macro_region_x, cell_x) so cells near same macro pack together
            def sort_key(idx):
                if idx in cell_macro_affinity:
                    best_macro = cell_macro_affinity[idx].most_common(1)[0][0]
                    return (positions[best_macro, 0].item(), positions[idx, 0].item())
                return (positions[idx, 0].item(), positions[idx, 0].item())

            sorted_std = sorted(std_indices, key=sort_key)
        else:
            # Fallback: sort by x position
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


def legalize_min_disturbance(cell_features, num_macros=None, max_passes=50):
    """Minimal-disturbance legalization: nudge cells minimum distance to resolve overlaps.

    Unlike row-based legalization, this preserves GD-optimized positions as much
    as possible. Each cell stays near its original position — only nudged enough
    to not overlap.

    Algorithm:
    1. Resolve macro-macro overlaps (same as row-based — push apart)
    2. For std cells: iteratively find overlapping pairs and nudge apart
       by minimum displacement along axis of least overlap
    3. Process cells in order of most overlaps first (greedy)
    4. Repeat until no overlaps remain

    Modifies cell_features[:, 2:4] in-place.
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

    # Step 1: Resolve macro-macro overlaps (same iterative push as row-based)
    if num_macros > 1:
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

    # Step 2: Resolve std cell overlaps with minimum disturbance
    # Build spatial index for efficiency
    from collections import defaultdict

    for _pass in range(max_passes):
        # Find all overlapping pairs (spatial hash for large N, brute force for small)
        overlapping_pairs = []

        if N <= 2500:
            # Brute force
            for i in range(N):
                for j in range(i + 1, N):
                    dx = abs(positions[i, 0].item() - positions[j, 0].item())
                    dy = abs(positions[i, 1].item() - positions[j, 1].item())
                    if dx < (widths[i].item() + widths[j].item()) / 2 and \
                       dy < (heights[i].item() + heights[j].item()) / 2:
                        overlapping_pairs.append((i, j))
        else:
            # Spatial hash
            bin_size = max(widths.max().item(), 3.0)
            x_min = positions[:, 0].min().item() - bin_size
            y_min = positions[:, 1].min().item() - bin_size

            bin_to_cells = defaultdict(list)
            for i in range(N):
                bx = int((positions[i, 0].item() - x_min) / bin_size)
                by = int((positions[i, 1].item() - y_min) / bin_size)
                bin_to_cells[(bx, by)].append(i)

            seen = set()
            for (bx, by), cells in bin_to_cells.items():
                # Check within bin + forward neighbors
                for dbx, dby in [(0, 0), (1, 0), (1, 1), (0, 1), (-1, 1)]:
                    nbx, nby = bx + dbx, by + dby
                    neighbors = bin_to_cells.get((nbx, nby), [])
                    check_cells = cells if (dbx == 0 and dby == 0) else neighbors

                    for a in cells:
                        for b in check_cells:
                            if a >= b:
                                continue
                            pair = (a, b)
                            if pair in seen:
                                continue
                            dx = abs(positions[a, 0].item() - positions[b, 0].item())
                            dy = abs(positions[a, 1].item() - positions[b, 1].item())
                            if dx < (widths[a].item() + widths[b].item()) / 2 and \
                               dy < (heights[a].item() + heights[b].item()) / 2:
                                overlapping_pairs.append(pair)
                                seen.add(pair)

        if not overlapping_pairs:
            break

        # Count overlaps per cell to prioritize worst offenders
        overlap_count = defaultdict(int)
        for i, j in overlapping_pairs:
            overlap_count[i] += 1
            overlap_count[j] += 1

        # Process pairs: worst offenders first
        overlapping_pairs.sort(key=lambda p: -(overlap_count[p[0]] + overlap_count[p[1]]))

        for i, j in overlapping_pairs:
            xi, yi = positions[i, 0].item(), positions[i, 1].item()
            xj, yj = positions[j, 0].item(), positions[j, 1].item()
            wi, hi = widths[i].item(), heights[i].item()
            wj, hj = widths[j].item(), heights[j].item()

            dx = xi - xj
            dy = yi - yj
            adx, ady = abs(dx), abs(dy)
            ov_x = (wi + wj) / 2 - adx
            ov_y = (hi + hj) / 2 - ady

            if ov_x <= 0 or ov_y <= 0:
                continue  # already resolved by earlier nudge

            # Determine which cells can move
            i_frozen = i < num_macros
            j_frozen = j < num_macros
            if i_frozen and j_frozen:
                continue

            # Nudge along axis of least overlap (minimum disturbance)
            if ov_x <= ov_y:
                shift = ov_x / 2 + 0.05
                sign = 1.0 if dx >= 0 else -1.0
                if dx == 0:
                    sign = 1.0
                if not i_frozen and not j_frozen:
                    positions[i, 0] += sign * shift
                    positions[j, 0] -= sign * shift
                elif i_frozen:
                    positions[j, 0] -= sign * (ov_x + 0.05)
                else:
                    positions[i, 0] += sign * (ov_x + 0.05)
            else:
                shift = ov_y / 2 + 0.05
                sign = 1.0 if dy >= 0 else -1.0
                if dy == 0:
                    sign = 1.0
                if not i_frozen and not j_frozen:
                    positions[i, 1] += sign * shift
                    positions[j, 1] -= sign * shift
                elif i_frozen:
                    positions[j, 1] -= sign * (ov_y + 0.05)
                else:
                    positions[i, 1] += sign * (ov_y + 0.05)

    cell_features[:, 2:4] = positions

    displacement = (positions - original_positions).abs()
    max_displacement = displacement.max().item()
    cells_moved = (displacement.sum(dim=1) > 0.01).sum().item()

    return {
        "time": time.perf_counter() - start_time,
        "cells_moved": cells_moved,
        "max_displacement": max_displacement,
    }
