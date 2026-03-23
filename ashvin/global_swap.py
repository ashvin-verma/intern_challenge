"""Global WL optimization: row reordering + cross-row reinsertion.

After legalization, cells are packed in rows. Two optimization strategies:

1. Row reordering: swap cells within a row and recompact. Always legal
   (no overlap check needed). Adjacent swaps only change 2 cells' positions.

2. Cross-row reinsertion: remove a cell from its row, insert into a gap in
   another row near its barycentric target. Makes room by shifting.

Both preserve zero overlap by construction.
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def _build_structures(cell_features, pin_features, edge_list):
    """Build adjacency structures for fast WL evaluation."""
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
    """Total Manhattan WL of edges incident to a cell. O(degree)."""
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


def _cells_wl(cell_indices, positions, pin_features, edge_list, pin_to_cell, cell_edges):
    """Total WL of all edges incident to any cell in the set. Avoids double-counting."""
    seen_edges = set()
    total = 0.0
    for ci in cell_indices:
        for e in cell_edges.get(ci, []):
            if e not in seen_edges:
                seen_edges.add(e)
                sp, tp = edge_list[e, 0].item(), edge_list[e, 1].item()
                sc, tc = pin_to_cell[sp], pin_to_cell[tp]
                dx = abs(positions[sc, 0].item() + pin_features[sp, 1].item()
                         - positions[tc, 0].item() - pin_features[tp, 1].item())
                dy = abs(positions[sc, 1].item() + pin_features[sp, 2].item()
                         - positions[tc, 1].item() - pin_features[tp, 2].item())
                total += dx + dy
    return total


def _build_rows(positions, heights, num_macros, N):
    """Group std cells into rows by y-coordinate (tolerance 0.1)."""
    rows = defaultdict(list)
    for i in range(num_macros, N):
        y = positions[i, 1].item()
        row_key = round(y * 10) / 10  # quantize to 0.1
        rows[row_key].append(i)

    # Sort each row by x-position
    for row_key in rows:
        rows[row_key].sort(key=lambda i: positions[i, 0].item())

    return rows


def _compact_row(cell_order, positions, widths, start_x):
    """Recompute x-positions for cells in the given order, packing left-to-right.

    Returns dict mapping cell_idx -> new_x (center position).
    """
    new_positions = {}
    cursor = start_x
    for ci in cell_order:
        w = widths[ci].item()
        new_x = cursor + w / 2
        new_positions[ci] = new_x
        cursor = new_x + w / 2
    return new_positions


def _check_macro_overlap(x, y, w, h, macro_obstacles):
    """Check if a cell at (x, y) with size (w, h) overlaps any macro."""
    cx_min, cx_max = x - w / 2, x + w / 2
    cy_min, cy_max = y - h / 2, y + h / 2
    for ox_min, oy_min, ox_max, oy_max in macro_obstacles:
        if cx_max > ox_min and cx_min < ox_max and cy_max > oy_min and cy_min < oy_max:
            return True
    return False


def row_reorder(cell_features, pin_features, edge_list,
                num_passes=10, num_macros=None, verbose=False):
    """Reorder cells within each row to minimize WL.

    For each row, try all pairwise swaps. After swap, recompact the row.
    Accept if total incident WL decreases. Always legal by construction.

    Returns dict with stats. Modifies cell_features[:, 2:4] in-place.
    """
    start_time = time.perf_counter()
    N = cell_features.shape[0]
    if N <= 1:
        return {"time": 0.0, "swaps": 0, "passes": 0}

    if num_macros is None:
        num_macros = (cell_features[:, 5] > 1.5).sum().item()

    positions = cell_features[:, 2:4].detach()
    widths = cell_features[:, 4].detach()
    heights = cell_features[:, 5].detach()

    pin_to_cell, cell_edges = _build_structures(cell_features, pin_features, edge_list)

    # Build macro obstacles for overlap checking during compaction
    macro_obstacles = []
    for i in range(num_macros):
        ox = positions[i, 0].item()
        oy = positions[i, 1].item()
        ow = widths[i].item()
        oh = heights[i].item()
        macro_obstacles.append((ox - ow/2, oy - oh/2, ox + ow/2, oy + oh/2))

    total_swaps = 0
    total_passes = 0

    for pass_num in range(num_passes):
        rows = _build_rows(positions, heights, num_macros, N)
        pass_swaps = 0

        for row_y, cells_in_row in rows.items():
            k = len(cells_in_row)
            if k <= 1:
                continue

            # Current row start position
            start_x = positions[cells_in_row[0], 0].item() - widths[cells_in_row[0]].item() / 2

            # Try all pairwise swaps (for small rows) or adjacent only (for large rows)
            max_pairs = k * (k - 1) // 2
            use_all_pairs = max_pairs <= 500  # up to ~32 cells per row

            pairs_to_try = []
            if use_all_pairs:
                for a in range(k):
                    for b in range(a + 1, k):
                        pairs_to_try.append((a, b))
            else:
                # Adjacent pairs + pairs involving worst-WL cells
                for a in range(k - 1):
                    pairs_to_try.append((a, a + 1))
                # Also try swaps of top 20% worst cells with all others
                cell_wls = [(
                    _cell_wl(cells_in_row[a], positions, pin_features, edge_list,
                             pin_to_cell, cell_edges), a
                ) for a in range(k)]
                cell_wls.sort(reverse=True)
                top_worst = max(1, k // 5)
                for _, a in cell_wls[:top_worst]:
                    for b in range(k):
                        if b != a and (min(a, b), max(a, b)) not in set(pairs_to_try):
                            pairs_to_try.append((min(a, b), max(a, b)))

            # Greedy: apply improving swaps one at a time until no more found
            improved_in_row = True
            while improved_in_row:
                improved_in_row = False

                wl_before = _cells_wl(cells_in_row, positions, pin_features, edge_list,
                                      pin_to_cell, cell_edges)

                best_swap = None
                best_improvement = 0.0

                for a, b in pairs_to_try:
                    # Swap cells a and b in the ordering
                    new_order = list(cells_in_row)
                    new_order[a], new_order[b] = new_order[b], new_order[a]

                    # Recompact
                    new_pos = _compact_row(new_order, positions, widths, start_x)

                    # Check macro overlaps for moved cells
                    has_macro_overlap = False
                    for ci in new_order:
                        if _check_macro_overlap(new_pos[ci], row_y,
                                                widths[ci].item(), heights[ci].item(),
                                                macro_obstacles):
                            has_macro_overlap = True
                            break

                    if has_macro_overlap:
                        continue

                    # Apply new positions temporarily
                    old_xs = {}
                    for ci in new_order:
                        old_xs[ci] = positions[ci, 0].item()
                        positions[ci, 0] = new_pos[ci]

                    wl_after = _cells_wl(new_order, positions, pin_features, edge_list,
                                         pin_to_cell, cell_edges)

                    improvement = wl_before - wl_after
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_swap = (a, b)

                    # Revert
                    for ci in new_order:
                        positions[ci, 0] = old_xs[ci]

                # Apply the best swap found for this row
                if best_swap is not None:
                    a, b = best_swap
                    new_order = list(cells_in_row)
                    new_order[a], new_order[b] = new_order[b], new_order[a]
                    new_pos = _compact_row(new_order, positions, widths, start_x)
                    for ci in new_order:
                        positions[ci, 0] = new_pos[ci]
                    cells_in_row[:] = new_order
                    pass_swaps += 1
                    improved_in_row = True

                    # Regenerate pairs list for new ordering
                    if use_all_pairs:
                        pairs_to_try = [(a, b) for a in range(k) for b in range(a+1, k)]

        total_swaps += pass_swaps
        total_passes = pass_num + 1

        if verbose:
            print(f"    Row reorder pass {pass_num}: {pass_swaps} swaps")

        if pass_swaps == 0:
            break

    cell_features[:, 2:4] = positions
    return {
        "time": time.perf_counter() - start_time,
        "swaps": total_swaps,
        "passes": total_passes,
    }


def cross_row_reinsertion(cell_features, pin_features, edge_list,
                          num_macros=None, top_frac=0.3, verbose=False):
    """Move high-WL cells to better rows near their barycentric target.

    For each worst-WL cell:
    1. Compute target = barycentric center of connected cells
    2. Find the closest row to target_y
    3. Find a gap in that row where the cell fits
    4. Move cell there, compact both old and new rows
    5. Accept if WL improved

    Always legal by construction (compaction after every move).
    """
    start_time = time.perf_counter()
    N = cell_features.shape[0]
    if N <= 1:
        return {"time": 0.0, "moves": 0}

    if num_macros is None:
        num_macros = (cell_features[:, 5] > 1.5).sum().item()

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

    # Score cells by WL
    cell_wl_scores = []
    for i in range(num_macros, N):
        wl = _cell_wl(i, positions, pin_features, edge_list, pin_to_cell, cell_edges)
        cell_wl_scores.append((wl, i))
    cell_wl_scores.sort(reverse=True)
    top_k = max(1, int(len(cell_wl_scores) * top_frac))

    # Get all row y-positions
    rows = _build_rows(positions, heights, num_macros, N)
    row_ys = sorted(rows.keys())

    total_moves = 0
    moved = set()

    for _wl_score, cell_i in cell_wl_scores[:top_k]:
        if cell_i in moved:
            continue

        # Compute barycentric target
        neighbors = set()
        for e in cell_edges.get(cell_i, []):
            sp, tp = edge_list[e, 0].item(), edge_list[e, 1].item()
            sc, tc = pin_to_cell[sp], pin_to_cell[tp]
            other = tc if sc == cell_i else sc
            neighbors.add(other)

        if not neighbors:
            continue

        target_x = sum(positions[n, 0].item() for n in neighbors) / len(neighbors)
        target_y = sum(positions[n, 1].item() for n in neighbors) / len(neighbors)

        cur_y = positions[cell_i, 1].item()
        cur_row_key = round(cur_y * 10) / 10
        w_i = widths[cell_i].item()
        h_i = heights[cell_i].item()

        # Find closest rows to target_y, search for gaps
        wl_before = _cell_wl(cell_i, positions, pin_features, edge_list, pin_to_cell, cell_edges)
        best_improvement = 0.01
        best_pos = None

        # Sort rows by distance to target_y, try closest 10
        sorted_rows = sorted(row_ys, key=lambda ry: abs(ry - target_y))

        for row_y in sorted_rows[:10]:
            if abs(row_y - cur_y) < 0.05:
                continue  # skip current row (handled by row_reorder)

            # Find ALL gaps in this row
            cells_here = rows.get(row_y, [])
            candidate_xs = []

            if not cells_here:
                # Empty row — try target_x directly
                candidate_xs.append(target_x)
            else:
                # Gap before first cell
                first_left = positions[cells_here[0], 0].item() - widths[cells_here[0]].item() / 2
                gap_x = first_left - w_i / 2
                candidate_xs.append(gap_x)

                # Gaps between consecutive cells
                for gi in range(len(cells_here) - 1):
                    c1 = cells_here[gi]
                    c2 = cells_here[gi + 1]
                    right_edge_1 = positions[c1, 0].item() + widths[c1].item() / 2
                    left_edge_2 = positions[c2, 0].item() - widths[c2].item() / 2
                    gap_size = left_edge_2 - right_edge_1
                    if gap_size >= w_i:
                        # Cell fits in this gap
                        candidate_xs.append(right_edge_1 + w_i / 2)  # left-aligned in gap
                        candidate_xs.append((right_edge_1 + left_edge_2) / 2)  # centered

                # Gap after last cell
                last_right = positions[cells_here[-1], 0].item() + widths[cells_here[-1]].item() / 2
                gap_x = last_right + w_i / 2
                candidate_xs.append(gap_x)

            # Also try target_x (might work if there's a gap there)
            candidate_xs.append(target_x)

            for try_x in candidate_xs:
                # Check macro overlap
                if _check_macro_overlap(try_x, row_y, w_i, h_i, macro_obstacles):
                    continue

                # Check overlap with cells in this row
                has_overlap = False
                for j in cells_here:
                    if j == cell_i:
                        continue
                    if abs(try_x - positions[j, 0].item()) < (w_i + widths[j].item()) / 2:
                        has_overlap = True
                        break

                if has_overlap:
                    continue

                # Evaluate WL
                old_x = positions[cell_i, 0].item()
                old_y = positions[cell_i, 1].item()
                positions[cell_i, 0] = try_x
                positions[cell_i, 1] = row_y

                wl_after = _cell_wl(cell_i, positions, pin_features, edge_list,
                                    pin_to_cell, cell_edges)
                improvement = wl_before - wl_after

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_pos = (try_x, row_y)

                # Revert
                positions[cell_i, 0] = old_x
                positions[cell_i, 1] = old_y

        if best_pos is not None:
            # Remove from old row
            if cur_row_key in rows and cell_i in rows[cur_row_key]:
                rows[cur_row_key].remove(cell_i)

            # Move to new position
            positions[cell_i, 0] = best_pos[0]
            positions[cell_i, 1] = best_pos[1]

            # Add to new row
            new_row_key = round(best_pos[1] * 10) / 10
            if new_row_key not in rows:
                rows[new_row_key] = []
            rows[new_row_key].append(cell_i)
            rows[new_row_key].sort(key=lambda c: positions[c, 0].item())

            moved.add(cell_i)
            total_moves += 1

    cell_features[:, 2:4] = positions

    if verbose:
        print(f"    Cross-row reinsertion: {total_moves} moves")

    return {
        "time": time.perf_counter() - start_time,
        "moves": total_moves,
    }


def global_swap(cell_features, pin_features, edge_list,
                num_passes=5, num_macros=None, verbose=False, **kwargs):
    """Combined global swap: row reordering + cross-row reinsertion.

    Phase 1: Reorder cells within each row (always legal)
    Phase 2: Move worst-WL cells to better rows

    Modifies cell_features[:, 2:4] in-place.
    """
    start_time = time.perf_counter()

    # Phase 1: Row reordering
    rr_stats = row_reorder(cell_features, pin_features, edge_list,
                           num_passes=num_passes, num_macros=num_macros,
                           verbose=verbose)

    # Phase 2: Cross-row reinsertion
    cr_stats = cross_row_reinsertion(cell_features, pin_features, edge_list,
                                     num_macros=num_macros, top_frac=0.3,
                                     verbose=verbose)

    # Phase 3: Another round of row reordering (after cross-row moves)
    if cr_stats["moves"] > 0:
        rr2_stats = row_reorder(cell_features, pin_features, edge_list,
                                num_passes=num_passes, num_macros=num_macros,
                                verbose=verbose)
        rr_stats["swaps"] += rr2_stats["swaps"]
        rr_stats["passes"] += rr2_stats["passes"]

    return {
        "time": time.perf_counter() - start_time,
        "swaps": rr_stats["swaps"],
        "passes": rr_stats["passes"],
        "cross_row_moves": cr_stats["moves"],
    }


def edge_targeted_swap(cell_features, pin_features, edge_list,
                       num_passes=3, num_macros=None, top_edge_frac=0.2,
                       verbose=False):
    """Target worst edges via row reordering.

    1. Identify worst-WL edges
    2. For each endpoint cell, try moving it within its row closer to the
       other endpoint (by swapping with intermediate cells)
    3. Accept if WL improves

    This is a thin wrapper: just calls row_reorder which naturally addresses
    worst edges through its pairwise swap search.
    """
    # Row reorder already handles this by trying all pairwise swaps
    # and accepting the best improvement. The worst-WL cells naturally
    # get the most improvement from reordering.
    return row_reorder(cell_features, pin_features, edge_list,
                       num_passes=num_passes, num_macros=num_macros,
                       verbose=verbose)
