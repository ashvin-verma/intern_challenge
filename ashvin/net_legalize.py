"""Net-aware legalization: minimize displacement + WL delta.

Instead of blind row-packing, assigns each cell to the best legal slot
considering both how far it moves AND how much wirelength changes.

Key insight: because the graph is sparse pairwise edges (not hyperedges),
the WL delta for moving a single cell is cheap to compute — just sum
the incident edge length changes.

Algorithm:
1. Resolve macro overlaps (same as before)
2. Form virtual rows from current y-positions
3. For each cell (sorted by WL-cost, worst first):
   a. Generate candidate x-slots in its row (gaps between placed cells/macros)
   b. Score each slot: alpha * displacement + beta * WL_delta
   c. Assign to best slot
4. Greedy with priority: cells with most WL to lose go first
"""

import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def _compute_cell_wl(cell_idx, positions, pin_features, edge_list, pin_to_cell, cell_edges):
    """Total WL of all edges incident to cell_idx."""
    total = 0.0
    for e_idx in cell_edges.get(cell_idx, []):
        sp = edge_list[e_idx, 0].item()
        tp = edge_list[e_idx, 1].item()
        sc = pin_to_cell[sp]
        tc = pin_to_cell[tp]
        dx = abs(positions[sc, 0].item() + pin_features[sp, 1].item()
                 - positions[tc, 0].item() - pin_features[tp, 1].item())
        dy = abs(positions[sc, 1].item() + pin_features[sp, 2].item()
                 - positions[tc, 1].item() - pin_features[tp, 2].item())
        total += dx + dy
    return total


def net_aware_legalize(cell_features, pin_features, edge_list, num_macros=None,
                       alpha=1.0, beta=2.0):
    """Net-aware legalization minimizing displacement + WL delta.

    Args:
        cell_features: [N, 6] — modified in-place
        pin_features: [P, 7]
        edge_list: [E, 2]
        num_macros: inferred if None
        alpha: weight for displacement cost
        beta: weight for WL delta cost

    Returns:
        dict with stats
    """
    start_time = time.perf_counter()

    N = cell_features.shape[0]
    if N <= 1:
        return {"time": 0.0, "cells_moved": 0}

    if num_macros is None:
        num_macros = (cell_features[:, 5] > 1.5).sum().item()

    positions = cell_features[:, 2:4].detach()
    widths = cell_features[:, 4].detach()
    heights = cell_features[:, 5].detach()
    original_positions = positions.clone()

    pin_to_cell = pin_features[:, 0].long().tolist()

    # Build cell -> edge index mapping
    cell_edges = defaultdict(list)
    for e_idx in range(edge_list.shape[0]):
        sc = pin_to_cell[edge_list[e_idx, 0].item()]
        tc = pin_to_cell[edge_list[e_idx, 1].item()]
        cell_edges[sc].append(e_idx)
        if tc != sc:
            cell_edges[tc].append(e_idx)

    # --- Step 1: Resolve macro overlaps (same iterative push) ---
    if num_macros > 1:
        for _pass in range(200):
            any_ov = False
            for i in range(num_macros):
                for j in range(i + 1, num_macros):
                    xi, yi = positions[i, 0].item(), positions[i, 1].item()
                    xj, yj = positions[j, 0].item(), positions[j, 1].item()
                    wi, hi = widths[i].item(), heights[i].item()
                    wj, hj = widths[j].item(), heights[j].item()
                    dx, dy = xi - xj, yi - yj
                    ov_x = (wi + wj) / 2 - abs(dx)
                    ov_y = (hi + hj) / 2 - abs(dy)
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

    # --- Step 2: Collect macro obstacles ---
    obstacles = []
    for i in range(num_macros):
        ox, oy = positions[i, 0].item(), positions[i, 1].item()
        ow, oh = widths[i].item(), heights[i].item()
        obstacles.append((ox - ow / 2, oy - oh / 2, ox + ow / 2, oy + oh / 2))

    # --- Step 3: Form rows and assign std cells ---
    if num_macros >= N:
        cell_features[:, 2:4] = positions
        return {"time": time.perf_counter() - start_time, "cells_moved": 0}

    std_indices = list(range(num_macros, N))
    row_height = 1.0

    # Assign cells to rows by quantizing y
    y_min = positions[std_indices, 1].min().item() - 10
    row_assignments = defaultdict(list)
    for idx in std_indices:
        row_idx = round((positions[idx, 1].item() - y_min) / row_height)
        row_assignments[row_idx].append(idx)

    # --- Step 4: For each row, assign cells to slots using net-aware cost ---
    for row_idx, cells_in_row in row_assignments.items():
        row_y = y_min + row_idx * row_height

        if not cells_in_row:
            continue

        # Sort by WL contribution (worst first — they get priority for good slots)
        cell_wl = []
        for idx in cells_in_row:
            wl = _compute_cell_wl(idx, positions, pin_features, edge_list, pin_to_cell, cell_edges)
            cell_wl.append((wl, idx))
        cell_wl.sort(reverse=True)  # worst WL first

        # Track occupied x-ranges in this row (from macros + already-placed cells)
        occupied = []
        for ox_min, oy_min, ox_max, oy_max in obstacles:
            if oy_max > row_y - row_height / 2 and oy_min < row_y + row_height / 2:
                occupied.append((ox_min, ox_max))

        placed_ranges = []  # (x_min, x_max) of placed std cells

        for _wl_score, idx in cell_wl:
            w = widths[idx].item()
            h = heights[idx].item()
            orig_x = positions[idx, 0].item()

            # Generate candidate slots:
            # 1. Original position
            # 2. Barycentric center of connected cells (best WL position)
            # 3. Gaps between obstacles/placed cells
            candidates = [orig_x]

            # Barycentric target — where WL wants this cell to be
            nbrs = []
            for e_idx in cell_edges.get(idx, []):
                sp = edge_list[e_idx, 0].item()
                tp = edge_list[e_idx, 1].item()
                sc, tc = pin_to_cell[sp], pin_to_cell[tp]
                other = tc if sc == idx else sc
                nbrs.append(other)
            if nbrs:
                bary_x = sum(positions[n, 0].item() for n in nbrs) / len(nbrs)
                candidates.append(bary_x)
                # Also try positions slightly left/right of barycentric
                candidates.append(bary_x - w)
                candidates.append(bary_x + w)

            # Add gap positions
            all_occupied = sorted(occupied + placed_ranges, key=lambda r: r[0])
            if all_occupied:
                # Before first obstacle
                candidates.append(all_occupied[0][0] - w / 2 - 0.1)
                # After last obstacle
                candidates.append(all_occupied[-1][1] + w / 2 + 0.1)
                # Gaps between obstacles
                for k in range(len(all_occupied) - 1):
                    gap_center = (all_occupied[k][1] + all_occupied[k + 1][0]) / 2
                    gap_width = all_occupied[k + 1][0] - all_occupied[k][1]
                    if gap_width >= w + 0.1:
                        candidates.append(gap_center)

            # Score each candidate: alpha * displacement + beta * WL_delta
            best_score = float("inf")
            best_x = orig_x

            for cand_x in candidates:
                # Check if legal (no overlap with occupied ranges)
                cell_left = cand_x - w / 2
                cell_right = cand_x + w / 2
                legal = True
                for occ_left, occ_right in all_occupied:
                    if cell_right > occ_left + 0.01 and cell_left < occ_right - 0.01:
                        legal = False
                        break
                if not legal:
                    continue

                # Displacement cost
                disp = abs(cand_x - orig_x)

                # WL delta: compute WL at candidate position
                old_x_val = positions[idx, 0].item()
                positions[idx, 0] = cand_x
                positions[idx, 1] = row_y
                wl_new = _compute_cell_wl(idx, positions, pin_features, edge_list, pin_to_cell, cell_edges)
                positions[idx, 0] = old_x_val  # restore

                # Compute WL at original position for reference
                wl_orig = _compute_cell_wl(idx, positions, pin_features, edge_list, pin_to_cell, cell_edges)

                wl_delta = wl_new - wl_orig

                score = alpha * disp + beta * wl_delta

                if score < best_score:
                    best_score = score
                    best_x = cand_x

            # Place cell at best slot
            positions[idx, 0] = best_x
            positions[idx, 1] = row_y
            placed_ranges.append((best_x - w / 2, best_x + w / 2))

    cell_features[:, 2:4] = positions

    displacement = (positions - original_positions).abs()
    cells_moved = (displacement.sum(dim=1) > 0.01).sum().item()

    return {
        "time": time.perf_counter() - start_time,
        "cells_moved": cells_moved,
        "max_displacement": displacement.max().item(),
    }
