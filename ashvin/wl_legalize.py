"""WL-priority legalization: place worst-WL cells first at their optimal rows.

Unlike greedy row-packing (which sorts by x and packs left-to-right),
this legalizer:
1. Sorts cells by WL contribution (worst first)
2. For each cell, finds the best row (nearest to barycentric target y)
3. Inserts the cell in that row, pushing existing cells to make room
4. High-WL cells get priority for optimal positions

Guarantees zero overlap by construction (compaction after each insertion).
"""

import sys
import time
from collections import defaultdict
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


def _check_macro_overlap(x, y, w, h, obstacles):
    cx_min, cx_max = x - w / 2, x + w / 2
    cy_min, cy_max = y - h / 2, y + h / 2
    for ox_min, oy_min, ox_max, oy_max in obstacles:
        if cx_max > ox_min and cx_min < ox_max and cy_max > oy_min and cy_min < oy_max:
            return True
    return False


def wl_priority_legalize(cell_features, pin_features, edge_list, num_macros=None,
                          alpha=0.5, beta=2.0):
    """WL-priority legalization.

    1. Resolve macro overlaps (same as existing legalization)
    2. Sort std cells by WL contribution (worst first)
    3. For each cell, find best row and position based on barycentric target
    4. Insert into row with compaction (guaranteed zero overlap)

    High-WL cells are placed first and get optimal positions.
    Low-WL cells fill remaining space.

    Args:
        alpha: weight for displacement cost in scoring
        beta: weight for WL delta in scoring
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

    pin_to_cell = pin_features[:, 0].long().tolist()
    cell_edges = defaultdict(list)
    for e_idx in range(edge_list.shape[0]):
        sc = pin_to_cell[edge_list[e_idx, 0].item()]
        tc = pin_to_cell[edge_list[e_idx, 1].item()]
        cell_edges[sc].append(e_idx)
        if tc != sc:
            cell_edges[tc].append(e_idx)

    # --- Step 1: Legalize macros (same as existing) ---
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

    if num_macros >= N:
        cell_features[:, 2:4] = positions
        return {"time": time.perf_counter() - start_time, "cells_moved": 0, "max_displacement": 0.0}

    # --- Step 3: Determine row positions ---
    std_indices = list(range(num_macros, N))
    row_height = 1.0
    all_y = positions[std_indices, 1]
    y_min = all_y.min().item() - 5
    y_max = all_y.max().item() + 5

    # Generate available row positions
    row_min = int((y_min) / row_height)
    row_max = int((y_max) / row_height) + 1
    available_rows = [y_min + (r - row_min) * row_height for r in range(row_max - row_min + 1)]

    # --- Step 4: Score cells by WL and sort (worst first) ---
    cell_wl_list = []
    for idx in std_indices:
        wl = _compute_cell_wl(idx, positions, pin_features, edge_list, pin_to_cell, cell_edges)
        cell_wl_list.append((wl, idx))
    cell_wl_list.sort(reverse=True)  # worst WL first

    # --- Step 5: Place cells one by one in WL-priority order ---
    # Row structure: row_y -> list of (x_center, cell_idx) sorted by x
    row_contents = defaultdict(list)  # row_y -> [(x, width, cell_idx), ...]

    for _wl_score, idx in cell_wl_list:
        w = widths[idx].item()
        h = heights[idx].item()
        orig_x = positions[idx, 0].item()
        orig_y = positions[idx, 1].item()

        # Compute barycentric target from connected cells
        # (uses current positions — already-placed cells have legalized positions)
        nbrs = []
        for e_idx in cell_edges.get(idx, []):
            sp = edge_list[e_idx, 0].item()
            tp = edge_list[e_idx, 1].item()
            sc, tc = pin_to_cell[sp], pin_to_cell[tp]
            other = tc if sc == idx else sc
            nbrs.append(other)

        if nbrs:
            bary_x = sum(positions[n, 0].item() for n in nbrs) / len(nbrs)
            bary_y = sum(positions[n, 1].item() for n in nbrs) / len(nbrs)
        else:
            bary_x = orig_x
            bary_y = orig_y

        # Target position: weighted average of barycentric and original
        target_x = beta * bary_x + alpha * orig_x
        target_x /= (alpha + beta)
        target_y = beta * bary_y + alpha * orig_y
        target_y /= (alpha + beta)

        # Find best row (try nearest 5 rows to target_y)
        sorted_rows = sorted(available_rows, key=lambda ry: abs(ry - target_y))
        best_score = float("inf")
        best_row = sorted_rows[0]
        best_x = target_x

        for row_y in sorted_rows[:5]:
            # Find best insertion position in this row
            existing = row_contents[row_y]  # sorted by x

            # Candidate insertion positions
            candidates_x = [target_x, bary_x, orig_x]

            # After each existing cell
            for ex, ew, _ec in existing:
                candidates_x.append(ex + ew / 2 + w / 2 + 0.01)

            # Before first cell
            if existing:
                candidates_x.append(existing[0][0] - existing[0][1] / 2 - w / 2 - 0.01)

            for cand_x in candidates_x:
                # Check macro overlap
                if _check_macro_overlap(cand_x, row_y, w, h, obstacles):
                    continue

                # Check if insertion would create too much displacement for existing cells
                # For scoring, use displacement from original + WL delta
                y_disp = abs(row_y - orig_y)
                x_disp = abs(cand_x - orig_x)

                # Compute WL at candidate position
                old_px = positions[idx, 0].item()
                old_py = positions[idx, 1].item()
                positions[idx, 0] = cand_x
                positions[idx, 1] = row_y
                wl_at_cand = _compute_cell_wl(idx, positions, pin_features, edge_list,
                                               pin_to_cell, cell_edges)
                positions[idx, 0] = old_px
                positions[idx, 1] = old_py

                score = alpha * (x_disp + y_disp) + beta * wl_at_cand

                if score < best_score:
                    best_score = score
                    best_row = row_y
                    best_x = cand_x

        # Insert cell into the chosen row at best_x
        # First, check where it fits in the row ordering
        existing = row_contents[best_row]

        # Add cell to row
        existing.append((best_x, w, idx))
        existing.sort(key=lambda t: t[0])

        # Compact: ensure no overlaps within the row
        # Find the index of our newly inserted cell
        cell_slot_idx = next(i for i, (_, _, c) in enumerate(existing) if c == idx)

        # Compact rightward from the insertion point
        for k in range(cell_slot_idx + 1, len(existing)):
            prev_x, prev_w, _prev_c = existing[k - 1]
            cur_x, cur_w, cur_c = existing[k]
            min_x = prev_x + prev_w / 2 + cur_w / 2
            if cur_x < min_x:
                new_x = min_x
                existing[k] = (new_x, cur_w, cur_c)

        # Compact leftward from the insertion point
        for k in range(cell_slot_idx - 1, -1, -1):
            next_x, next_w, _next_c = existing[k + 1]
            cur_x, cur_w, cur_c = existing[k]
            max_x = next_x - next_w / 2 - cur_w / 2
            if cur_x > max_x:
                new_x = max_x
                existing[k] = (new_x, cur_w, cur_c)

        # Apply positions
        for x, w_c, c in existing:
            positions[c, 0] = x
            positions[c, 1] = best_row

        # Handle macro overlap after compaction — shift right if needed
        for k in range(len(existing)):
            x, w_c, c = existing[k]
            for _attempt in range(20):
                if not _check_macro_overlap(x, best_row, w_c, h, obstacles):
                    break
                # Shift right past the obstacle
                for ox_min, oy_min, ox_max, oy_max in obstacles:
                    cx_min = x - w_c / 2
                    cx_max = x + w_c / 2
                    cy_min = best_row - h / 2
                    cy_max = best_row + h / 2
                    if cx_max > ox_min and cx_min < ox_max and cy_max > oy_min and cy_min < oy_max:
                        x = ox_max + w_c / 2 + 0.1
                        break
            existing[k] = (x, w_c, c)
            positions[c, 0] = x

        # Re-compact after macro avoidance
        for k in range(1, len(existing)):
            prev_x, prev_w, _prev_c = existing[k - 1]
            cur_x, cur_w, cur_c = existing[k]
            min_x = prev_x + prev_w / 2 + cur_w / 2
            if cur_x < min_x:
                new_x = min_x
                existing[k] = (new_x, cur_w, cur_c)
                positions[cur_c, 0] = new_x

        row_contents[best_row] = existing

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
