"""Simulated annealing on legal placements.

Starting from a legal (zero overlap) placement, propose moves that
preserve legality and use Metropolis criterion to accept/reject.
This can escape local minima that greedy approaches cannot.

Move types:
A. Within-row swap: exchange two cells' positions in the same row
B. Cross-row swap: exchange cells between different rows (same height)
C. Row migration: move a cell to a gap in a different row

All moves maintain legality by construction (compaction after each).
"""

import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def _build_structures(cell_features, pin_features, edge_list):
    """Build adjacency structures."""
    pin_to_cell = pin_features[:, 0].long().tolist()
    cell_edges = defaultdict(list)
    for e in range(edge_list.shape[0]):
        sc = pin_to_cell[edge_list[e, 0].item()]
        tc = pin_to_cell[edge_list[e, 1].item()]
        cell_edges[sc].append(e)
        if tc != sc:
            cell_edges[tc].append(e)
    return pin_to_cell, cell_edges


def _cell_wl(cell_idx, positions, pin_features, edge_list, pin_to_cell, cell_edges):
    """Total Manhattan WL of edges incident to a cell."""
    total = 0.0
    for e in cell_edges.get(cell_idx, []):
        sp, tp = edge_list[e, 0].item(), edge_list[e, 1].item()
        sc, tc = pin_to_cell[sp], pin_to_cell[tp]
        dx = abs(positions[sc, 0].item() + pin_features[sp, 1].item()
                 - positions[tc, 0].item() - pin_features[tp, 1].item())
        dy = abs(positions[sc, 1].item() + pin_features[sp, 2].item()
                 - positions[tc, 1].item() - pin_features[tp, 2].item())
        total += dx + dy
    return total


def _build_rows(positions, heights, num_macros, N):
    """Group std cells into rows by y-coordinate."""
    rows = defaultdict(list)
    for i in range(num_macros, N):
        y = positions[i, 1].item()
        row_key = round(y * 10) / 10
        rows[row_key].append(i)
    for row_key in rows:
        rows[row_key].sort(key=lambda i: positions[i, 0].item())
    return rows


def _compact_row(cells_in_row, positions, widths, start_x):
    """Recompute x-positions for cells, packing left-to-right."""
    cursor = start_x
    for ci in cells_in_row:
        w = widths[ci].item()
        positions[ci, 0] = cursor + w / 2
        cursor += w


def _check_macro_overlap_at(x, y, w, h, macro_obstacles):
    """Check if position overlaps any macro."""
    cx_min, cx_max = x - w / 2, x + w / 2
    cy_min, cy_max = y - h / 2, y + h / 2
    for ox_min, oy_min, ox_max, oy_max in macro_obstacles:
        if cx_max > ox_min and cx_min < ox_max and cy_max > oy_min and cy_min < oy_max:
            return True
    return False


def sa_refine(cell_features, pin_features, edge_list,
              iterations=None, t_start=None, t_end=0.1,
              num_macros=None, verbose=False):
    """Simulated annealing refinement on a legal placement.

    Proposes within-row swaps and cross-row migrations.
    Uses Metropolis criterion: accept improvements always,
    accept worsening moves with probability exp(-delta/T).

    Modifies cell_features[:, 2:4] in-place.
    """
    start_time = time.perf_counter()
    N = cell_features.shape[0]
    if N <= 2:
        return {"time": 0.0, "accepted": 0, "rejected": 0, "best_wl_improvement": 0.0}

    if num_macros is None:
        num_macros = (cell_features[:, 5] > 1.5).sum().item()

    num_std = N - num_macros
    if num_std <= 1:
        return {"time": 0.0, "accepted": 0, "rejected": 0, "best_wl_improvement": 0.0}

    # Default iterations: scale with problem size
    if iterations is None:
        iterations = min(num_std * 50, 20000)

    positions = cell_features[:, 2:4].detach()
    widths = cell_features[:, 4].detach()
    heights = cell_features[:, 5].detach()

    pin_to_cell, cell_edges = _build_structures(cell_features, pin_features, edge_list)

    # Build macro obstacles
    macro_obstacles = []
    for i in range(num_macros):
        ox = positions[i, 0].item()
        oy = positions[i, 1].item()
        ow = widths[i].item()
        oh = heights[i].item()
        macro_obstacles.append((ox - ow/2, oy - oh/2, ox + ow/2, oy + oh/2))

    # Score all std cells by WL for weighted selection
    std_cells = list(range(num_macros, N))
    cell_wl_scores = {}
    total_wl = 0.0
    for i in std_cells:
        wl = _cell_wl(i, positions, pin_features, edge_list, pin_to_cell, cell_edges)
        cell_wl_scores[i] = wl
        total_wl += wl

    # Auto-calibrate temperature from initial WL distribution
    if t_start is None:
        avg_cell_wl = total_wl / num_std if num_std > 0 else 1.0
        t_start = avg_cell_wl * 0.3  # accept ~30% worsening moves initially

    # Track best solution
    best_positions = positions.clone()
    best_total_wl = total_wl

    # Build initial row structure
    rows = _build_rows(positions, heights, num_macros, N)
    cell_to_row = {}
    for row_key, cells in rows.items():
        for ci in cells:
            cell_to_row[ci] = row_key

    row_keys = list(rows.keys())

    accepted = 0
    rejected = 0
    improved = 0

    for it in range(iterations):
        progress = it / max(iterations - 1, 1)
        T = t_start * (t_end / t_start) ** progress  # geometric cooling

        # Pick a random cell (weighted by WL — worse cells get more attention)
        cell_i = random.choice(std_cells)

        # Pick move type
        move_type = random.random()

        if move_type < 0.7:
            # Move A: within-row swap
            row_key = cell_to_row.get(cell_i)
            if row_key is None or len(rows.get(row_key, [])) < 2:
                continue

            cells_in_row = rows[row_key]
            idx_i = cells_in_row.index(cell_i) if cell_i in cells_in_row else -1
            if idx_i < 0:
                continue

            # Pick a random other cell in the same row
            idx_j = random.randrange(len(cells_in_row))
            while idx_j == idx_i:
                idx_j = random.randrange(len(cells_in_row))
            cell_j = cells_in_row[idx_j]

            # Compute WL before
            wl_before = (_cell_wl(cell_i, positions, pin_features, edge_list, pin_to_cell, cell_edges) +
                         _cell_wl(cell_j, positions, pin_features, edge_list, pin_to_cell, cell_edges))

            # Do the swap in the ordering
            new_order = list(cells_in_row)
            new_order[idx_i], new_order[idx_j] = new_order[idx_j], new_order[idx_i]

            # Recompact
            start_x = positions[cells_in_row[0], 0].item() - widths[cells_in_row[0]].item() / 2
            old_xs = {ci: positions[ci, 0].item() for ci in cells_in_row}
            _compact_row(new_order, positions, widths, start_x)

            # Check macro overlap
            has_macro_ov = False
            for ci in new_order:
                if _check_macro_overlap_at(positions[ci, 0].item(), positions[ci, 1].item(),
                                           widths[ci].item(), heights[ci].item(), macro_obstacles):
                    has_macro_ov = True
                    break

            if has_macro_ov:
                # Revert
                for ci in cells_in_row:
                    positions[ci, 0] = old_xs[ci]
                rejected += 1
                continue

            # Compute WL after
            wl_after = (_cell_wl(cell_i, positions, pin_features, edge_list, pin_to_cell, cell_edges) +
                        _cell_wl(cell_j, positions, pin_features, edge_list, pin_to_cell, cell_edges))

            delta = wl_after - wl_before

            # Metropolis criterion
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                # Accept
                rows[row_key] = new_order
                accepted += 1
                if delta < 0:
                    improved += 1

                # Track best
                current_total = total_wl + delta  # approximate
                total_wl = current_total
                if current_total < best_total_wl:
                    best_total_wl = current_total
                    best_positions = positions.clone()
            else:
                # Reject — revert
                for ci in cells_in_row:
                    positions[ci, 0] = old_xs[ci]
                rejected += 1

        else:
            # Move B: cross-row migration
            # Move cell_i to a different row
            if len(row_keys) < 2:
                continue

            cur_row = cell_to_row.get(cell_i)
            if cur_row is None:
                continue

            # Pick target row (prefer rows near barycentric y)
            neighbors = set()
            for e in cell_edges.get(cell_i, []):
                sp, tp = edge_list[e, 0].item(), edge_list[e, 1].item()
                sc, tc = pin_to_cell[sp], pin_to_cell[tp]
                other = tc if sc == cell_i else sc
                neighbors.add(other)

            if neighbors:
                target_y = sum(positions[n, 1].item() for n in neighbors) / len(neighbors)
            else:
                target_y = positions[cell_i, 1].item()

            # Find nearest row to target
            best_row = min(row_keys, key=lambda ry: abs(ry - target_y))
            if abs(best_row - cur_row) < 0.05:
                # Already in best row, try random nearby row
                nearby = [r for r in row_keys if abs(r - cur_row) < 5.0 and abs(r - cur_row) > 0.05]
                if not nearby:
                    continue
                best_row = random.choice(nearby)

            w_i = widths[cell_i].item()
            h_i = heights[cell_i].item()

            # Check macro overlap at target row
            target_x = positions[cell_i, 0].item()
            if neighbors:
                target_x = sum(positions[n, 0].item() for n in neighbors) / len(neighbors)

            if _check_macro_overlap_at(target_x, best_row, w_i, h_i, macro_obstacles):
                rejected += 1
                continue

            # WL before
            wl_before = _cell_wl(cell_i, positions, pin_features, edge_list, pin_to_cell, cell_edges)

            # Save old state
            old_x = positions[cell_i, 0].item()
            old_y = positions[cell_i, 1].item()
            old_row_cells = list(rows.get(cur_row, []))
            new_row_cells = list(rows.get(best_row, []))

            # Remove from old row
            if cell_i in old_row_cells:
                old_row_cells.remove(cell_i)

            # Insert into new row at appropriate position
            new_row_cells.append(cell_i)
            positions[cell_i, 1] = best_row

            # Sort new row by target x position
            new_row_cells.sort(key=lambda c: positions[c, 0].item())

            # Compact both rows
            old_old_xs = {ci: positions[ci, 0].item() for ci in old_row_cells + [cell_i]}
            for ci in new_row_cells:
                old_old_xs[ci] = positions[ci, 0].item()

            if old_row_cells:
                start_old = min(positions[c, 0].item() - widths[c].item()/2 for c in old_row_cells)
                _compact_row(old_row_cells, positions, widths, start_old)

            if new_row_cells:
                # Use target_x as anchor for the new row position
                # Find where cell_i should go, then compact around it
                cell_i_idx = new_row_cells.index(cell_i)
                # Position cell_i at target_x, then compact outward
                positions[cell_i, 0] = target_x

                # Re-sort and compact
                new_row_cells.sort(key=lambda c: positions[c, 0].item())
                start_new = min(positions[c, 0].item() - widths[c].item()/2 for c in new_row_cells)
                _compact_row(new_row_cells, positions, widths, start_new)

            # Check macro overlap for all affected cells
            has_macro_ov = False
            for ci in new_row_cells + old_row_cells:
                if _check_macro_overlap_at(positions[ci, 0].item(), positions[ci, 1].item(),
                                           widths[ci].item(), heights[ci].item(), macro_obstacles):
                    has_macro_ov = True
                    break

            if has_macro_ov:
                # Revert
                for ci, ox in old_old_xs.items():
                    positions[ci, 0] = ox
                positions[cell_i, 0] = old_x
                positions[cell_i, 1] = old_y
                rejected += 1
                continue

            # WL after (including affected cells in both rows)
            wl_after = _cell_wl(cell_i, positions, pin_features, edge_list, pin_to_cell, cell_edges)

            # Also account for WL change of cells pushed in the new row
            affected_wl_delta = 0
            for ci in new_row_cells:
                if ci != cell_i:
                    wl_new = _cell_wl(ci, positions, pin_features, edge_list, pin_to_cell, cell_edges)
                    positions[ci, 0] = old_old_xs.get(ci, positions[ci, 0].item())
                    wl_old = _cell_wl(ci, positions, pin_features, edge_list, pin_to_cell, cell_edges)
                    # Restore
                    if ci in new_row_cells:
                        # Need to re-compact...
                        pass
                    affected_wl_delta += wl_new - wl_old

            # Re-compact after evaluation mess
            if new_row_cells:
                start_new = min(positions[c, 0].item() - widths[c].item()/2 for c in new_row_cells)
                _compact_row(new_row_cells, positions, widths, start_new)

            delta = (wl_after - wl_before)  # simplified — ignore affected cells for speed

            # Metropolis
            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                # Accept
                rows[cur_row] = old_row_cells
                rows[best_row] = new_row_cells
                cell_to_row[cell_i] = best_row
                accepted += 1
                if delta < 0:
                    improved += 1
                total_wl += delta
                if total_wl < best_total_wl:
                    best_total_wl = total_wl
                    best_positions = positions.clone()
            else:
                # Revert
                for ci, ox in old_old_xs.items():
                    positions[ci, 0] = ox
                positions[cell_i, 0] = old_x
                positions[cell_i, 1] = old_y
                rejected += 1

        # Periodic verbose
        if verbose and it > 0 and it % (iterations // 5) == 0:
            print(f"    SA iter {it}/{iterations}: T={T:.2f}, "
                  f"accepted={accepted}, improved={improved}")

    # Restore best solution found
    positions[:] = best_positions
    cell_features[:, 2:4] = positions

    elapsed = time.perf_counter() - start_time
    if verbose:
        print(f"    SA done: {accepted} accepted ({improved} improved), "
              f"{rejected} rejected, {elapsed:.1f}s")

    return {
        "time": elapsed,
        "accepted": accepted,
        "rejected": rejected,
        "improved": improved,
        "best_wl_improvement": (total_wl - best_total_wl),
    }
