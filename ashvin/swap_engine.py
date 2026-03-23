"""Fast iterative cell-swap engine — the core WL optimizer.

After legalization, this engine runs hundreds of targeted moves to recover
WL destroyed by legalization. Each move is O(degree) to evaluate.

Two move types:
A. Within-row swap: exchange two cells' ordering in the same row, recompact.
   Always legal. O(degree_i + degree_j) to evaluate.
B. Cross-row reinsertion: remove cell from its row, insert into another row
   near its barycentric target. Compact both rows. Always legal.
   O(degree_i + cells_in_target_row) to evaluate.

Key design: operate on ROW STRUCTURE not positions. Rows are ordered lists
of cell indices. Compaction converts a row ordering into x-positions.
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


# ── Data structures ──────────────────────────────────────────────────

def build_adjacency(pin_features, edge_list):
    """Build cell→edges and edge→cells mappings."""
    pin_to_cell = pin_features[:, 0].long().tolist()
    cell_edges = defaultdict(list)
    E = edge_list.shape[0]
    for e in range(E):
        sc = pin_to_cell[edge_list[e, 0].item()]
        tc = pin_to_cell[edge_list[e, 1].item()]
        cell_edges[sc].append(e)
        if tc != sc:
            cell_edges[tc].append(e)
    return pin_to_cell, cell_edges


def build_rows(positions, heights, num_macros, N):
    """Build row structure: row_y → [cell indices sorted by x]."""
    rows = {}
    cell_row = {}
    for i in range(num_macros, N):
        ry = round(positions[i, 1].item() * 10) / 10
        if ry not in rows:
            rows[ry] = []
        rows[ry].append(i)
        cell_row[i] = ry
    for ry in rows:
        rows[ry].sort(key=lambda c: positions[c, 0].item())
    return rows, cell_row


def compact_row(row_cells, widths, start_x):
    """Given ordered cells, compute x-positions by left-to-right packing.
    Returns list of (cell_idx, new_x) pairs."""
    result = []
    cursor = start_x
    for ci in row_cells:
        w = widths[ci].item()
        x = cursor + w / 2
        result.append((ci, x))
        cursor = x + w / 2
    return result


def get_row_start(row_cells, positions, widths):
    """Get the leftmost edge of a row's current extent."""
    if not row_cells:
        return 0.0
    first = row_cells[0]
    return positions[first, 0].item() - widths[first].item() / 2


# ── WL evaluation ───────────────────────────────────────────────────

def cell_wl(ci, positions, pin_features, edge_list, pin_to_cell, cell_edges):
    """Total Manhattan WL of edges incident to cell ci."""
    total = 0.0
    for e in cell_edges.get(ci, []):
        sp = edge_list[e, 0].item()
        tp = edge_list[e, 1].item()
        sc, tc = pin_to_cell[sp], pin_to_cell[tp]
        dx = abs(positions[sc, 0].item() + pin_features[sp, 1].item()
                 - positions[tc, 0].item() - pin_features[tp, 1].item())
        dy = abs(positions[sc, 1].item() + pin_features[sp, 2].item()
                 - positions[tc, 1].item() - pin_features[tp, 2].item())
        total += dx + dy
    return total


def barycentric_target(ci, positions, pin_features, edge_list, pin_to_cell, cell_edges):
    """Compute barycentric center of cell's connected neighbors."""
    sx, sy, cnt = 0.0, 0.0, 0
    for e in cell_edges.get(ci, []):
        sp = edge_list[e, 0].item()
        tp = edge_list[e, 1].item()
        sc, tc = pin_to_cell[sp], pin_to_cell[tp]
        other = tc if sc == ci else sc
        sx += positions[other, 0].item()
        sy += positions[other, 1].item()
        cnt += 1
    if cnt == 0:
        return positions[ci, 0].item(), positions[ci, 1].item()
    return sx / cnt, sy / cnt


# ── Macro obstacle checking ────────────────────────────────────────

def build_macro_obstacles(positions, widths, heights, num_macros):
    obs = []
    for i in range(num_macros):
        x, y = positions[i, 0].item(), positions[i, 1].item()
        w, h = widths[i].item(), heights[i].item()
        obs.append((x - w/2, y - h/2, x + w/2, y + h/2))
    return obs


def check_macro_overlap(x, y, w, h, obstacles):
    l, r, b, t = x - w/2, x + w/2, y - h/2, y + h/2
    for ol, ob, or_, ot in obstacles:
        if r > ol and l < or_ and t > ob and b < ot:
            return True
    return False


# ── Move operations ─────────────────────────────────────────────────

def try_within_row_swap(ci, cj, row_cells, positions, widths, heights,
                        pin_features, edge_list, pin_to_cell, cell_edges,
                        obstacles, row_y):
    """Try swapping ci and cj within their row. Returns WL delta (negative = better)."""
    idx_i = row_cells.index(ci)
    idx_j = row_cells.index(cj)

    # Swap in ordering
    new_order = list(row_cells)
    new_order[idx_i], new_order[idx_j] = new_order[idx_j], new_order[idx_i]

    # Recompact
    start_x = get_row_start(row_cells, positions, widths)
    new_pos = compact_row(new_order, widths, start_x)

    # Check macro overlaps
    for c, nx in new_pos:
        if check_macro_overlap(nx, row_y, widths[c].item(), heights[c].item(), obstacles):
            return 0.0, None  # blocked

    # WL before for ALL cells in row (not just swapped pair)
    # Only edges incident to cells in this row are affected
    affected = set()
    lo, hi = min(idx_i, idx_j), max(idx_i, idx_j)
    for k in range(lo, hi + 1):
        affected.add(row_cells[k])

    wl_before = 0.0
    seen_edges = set()
    for c in affected:
        for e in cell_edges.get(c, []):
            if e not in seen_edges:
                seen_edges.add(e)
                sp, tp = edge_list[e, 0].item(), edge_list[e, 1].item()
                sc, tc = pin_to_cell[sp], pin_to_cell[tp]
                dx = abs(positions[sc, 0].item() + pin_features[sp, 1].item()
                         - positions[tc, 0].item() - pin_features[tp, 1].item())
                dy = abs(positions[sc, 1].item() + pin_features[sp, 2].item()
                         - positions[tc, 1].item() - pin_features[tp, 2].item())
                wl_before += dx + dy

    # Apply temporarily
    old_xs = {}
    for c, nx in new_pos:
        old_xs[c] = positions[c, 0].item()
        positions[c, 0] = nx

    wl_after = 0.0
    for e in seen_edges:
        sp, tp = edge_list[e, 0].item(), edge_list[e, 1].item()
        sc, tc = pin_to_cell[sp], pin_to_cell[tp]
        dx = abs(positions[sc, 0].item() + pin_features[sp, 1].item()
                 - positions[tc, 0].item() - pin_features[tp, 1].item())
        dy = abs(positions[sc, 1].item() + pin_features[sp, 2].item()
                 - positions[tc, 1].item() - pin_features[tp, 2].item())
        wl_after += dx + dy

    # Revert
    for c, _ in new_pos:
        positions[c, 0] = old_xs[c]

    delta = wl_after - wl_before  # negative = improvement
    return delta, new_order


def try_cross_row_move(ci, src_row_cells, dst_row_cells, dst_row_y,
                       insert_x, positions, widths, heights,
                       pin_features, edge_list, pin_to_cell, cell_edges,
                       obstacles):
    """Try moving ci from src_row to dst_row at insert_x. Returns WL delta."""
    # WL before (cell i + cells that will be displaced)
    wl_before_i = cell_wl(ci, positions, pin_features, edge_list, pin_to_cell, cell_edges)

    # Save old state
    old_x = positions[ci, 0].item()
    old_y = positions[ci, 1].item()

    # New source row (without ci)
    new_src = [c for c in src_row_cells if c != ci]

    # New dest row (with ci inserted at correct position)
    new_dst = list(dst_row_cells)
    new_dst.append(ci)
    # Temporarily set ci's position for sorting
    positions[ci, 0] = insert_x
    positions[ci, 1] = dst_row_y
    new_dst.sort(key=lambda c: positions[c, 0].item())

    # Compact dest row
    if new_dst:
        # Anchor compaction near the GD centroid of the row
        centroid_x = sum(positions[c, 0].item() for c in new_dst) / len(new_dst)
        total_w = sum(widths[c].item() for c in new_dst)
        start_x = centroid_x - total_w / 2
        dst_packed = compact_row(new_dst, widths, start_x)
    else:
        dst_packed = []

    # Check macro overlaps for dest
    for c, nx in dst_packed:
        if check_macro_overlap(nx, dst_row_y, widths[c].item(), heights[c].item(), obstacles):
            positions[ci, 0] = old_x
            positions[ci, 1] = old_y
            return 0.0, None, None

    # Apply dest positions temporarily
    old_positions = {}
    for c, nx in dst_packed:
        old_positions[c] = (positions[c, 0].item(), positions[c, 1].item())
        positions[c, 0] = nx
        positions[c, 1] = dst_row_y

    # Compact source row
    if new_src:
        src_start = get_row_start(src_row_cells, positions, widths)
        # But ci was removed, so start from first remaining
        src_centroid = sum(positions[c, 0].item() for c in new_src) / len(new_src)
        src_total_w = sum(widths[c].item() for c in new_src)
        src_start = src_centroid - src_total_w / 2
        src_packed = compact_row(new_src, widths, src_start)
        for c, nx in src_packed:
            if c not in old_positions:
                old_positions[c] = (positions[c, 0].item(), positions[c, 1].item())
            positions[c, 0] = nx
    else:
        src_packed = []

    # WL after
    wl_after_i = cell_wl(ci, positions, pin_features, edge_list, pin_to_cell, cell_edges)

    delta = wl_after_i - wl_before_i

    # Revert all
    for c, (ox, oy) in old_positions.items():
        positions[c, 0] = ox
        positions[c, 1] = oy
    positions[ci, 0] = old_x
    positions[ci, 1] = old_y

    return delta, new_src, new_dst


# ── Main engine ─────────────────────────────────────────────────────

def swap_engine(cell_features, pin_features, edge_list,
                max_iterations=20, num_macros=None, verbose=False):
    """Fast iterative cell-swap engine.

    Runs many rounds of within-row swaps and cross-row reinsertions
    until convergence. Modifies cell_features[:, 2:4] in-place.
    """
    start_time = time.perf_counter()
    N = cell_features.shape[0]
    if N <= 1:
        return {"time": 0.0, "swaps": 0, "moves": 0, "iterations": 0}

    if num_macros is None:
        num_macros = (cell_features[:, 5] > 1.5).sum().item()

    positions = cell_features[:, 2:4].detach()
    widths = cell_features[:, 4].detach()
    heights = cell_features[:, 5].detach()

    pin_to_cell, cell_edges = build_adjacency(pin_features, edge_list)
    obstacles = build_macro_obstacles(positions, widths, heights, num_macros)

    total_swaps = 0
    total_moves = 0

    for iteration in range(max_iterations):
        rows, cell_row = build_rows(positions, heights, num_macros, N)
        row_keys = sorted(rows.keys())

        iter_swaps = 0
        iter_moves = 0

        # Score all std cells by WL contribution
        cell_scores = []
        for i in range(num_macros, N):
            wl = cell_wl(i, positions, pin_features, edge_list, pin_to_cell, cell_edges)
            cell_scores.append((wl, i))
        cell_scores.sort(reverse=True)

        moved_cells = set()

        for _score, ci in cell_scores:
            if ci in moved_cells:
                continue

            cur_row_y = cell_row.get(ci)
            if cur_row_y is None:
                continue

            cur_row = rows.get(cur_row_y, [])
            if ci not in cur_row:
                continue

            # Compute barycentric target
            tx, ty = barycentric_target(ci, positions, pin_features, edge_list,
                                        pin_to_cell, cell_edges)

            # ── Move type A: within-row swaps ──
            best_delta = -0.01  # minimum improvement threshold
            best_swap = None
            best_order = None

            for cj in cur_row:
                if cj == ci or cj in moved_cells:
                    continue
                if abs(heights[ci].item() - heights[cj].item()) > 0.01:
                    continue

                delta, new_order = try_within_row_swap(
                    ci, cj, cur_row, positions, widths, heights,
                    pin_features, edge_list, pin_to_cell, cell_edges,
                    obstacles, cur_row_y)

                if delta < best_delta:
                    best_delta = delta
                    best_swap = cj
                    best_order = new_order

            if best_order is not None:
                # Apply the swap
                start_x = get_row_start(cur_row, positions, widths)
                packed = compact_row(best_order, widths, start_x)
                for c, nx in packed:
                    positions[c, 0] = nx
                rows[cur_row_y] = best_order
                moved_cells.add(ci)
                moved_cells.add(best_swap)
                iter_swaps += 1
                continue

            # ── Move type B: cross-row reinsertion ──
            # Try rows near barycentric target
            best_delta = -0.01  # low threshold — accept any improvement
            best_move = None

            # Sort candidate rows by distance to target y
            sorted_dst_rows = sorted(row_keys, key=lambda ry: abs(ry - ty))

            for dst_ry in sorted_dst_rows[:8]:  # try up to 8 nearest rows
                if abs(dst_ry - cur_row_y) < 0.05:
                    continue  # skip same row

                dst_row = rows.get(dst_ry, [])

                delta, new_src, new_dst = try_cross_row_move(
                    ci, cur_row, dst_row, dst_ry, tx,
                    positions, widths, heights,
                    pin_features, edge_list, pin_to_cell, cell_edges,
                    obstacles)

                if delta < best_delta:
                    best_delta = delta
                    best_move = (dst_ry, new_src, new_dst)

            if best_move is not None:
                dst_ry, new_src, new_dst = best_move

                # Apply: compact source row using its original start
                if new_src:
                    src_start = get_row_start(cur_row, positions, widths)
                    for c, nx in compact_row(new_src, widths, src_start):
                        positions[c, 0] = nx

                # Apply: position ci and compact dest row
                positions[ci, 0] = tx
                positions[ci, 1] = dst_ry
                new_dst.sort(key=lambda c: positions[c, 0].item())
                if new_dst:
                    dst_start = get_row_start(dst_row, positions, widths) if dst_row else tx - widths[ci].item() / 2
                    for c, nx in compact_row(new_dst, widths, dst_start):
                        positions[c, 0] = nx
                        positions[c, 1] = dst_ry

                rows[cur_row_y] = new_src
                rows[dst_ry] = new_dst
                cell_row[ci] = dst_ry
                moved_cells.add(ci)
                iter_moves += 1

        total_swaps += iter_swaps
        total_moves += iter_moves

        if verbose:
            print(f"    Swap engine iter {iteration}: {iter_swaps} swaps, {iter_moves} moves")

        if iter_swaps == 0 and iter_moves == 0:
            break

    cell_features[:, 2:4] = positions

    elapsed = time.perf_counter() - start_time
    if verbose:
        print(f"    Swap engine done: {total_swaps} swaps, {total_moves} moves, "
              f"{iteration+1} iters, {elapsed:.1f}s")

    return {
        "time": elapsed,
        "swaps": total_swaps,
        "moves": total_moves,
        "iterations": iteration + 1,
    }
