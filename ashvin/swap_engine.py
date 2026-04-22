"""
Fast row-structured local search after legalization.

Two move types:
1. Within-row swap: exchange two cells in the same row and recompact.
2. Cross-row reinsertion: remove a cell from one row and insert it into
   another row near a connectivity-driven target.
"""

import sys
import time
from bisect import bisect_left
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from ashvin.connectivity import (
    build_connectivity_context,
    collect_incident_edges,
    compute_cell_wl_scores,
    compute_neighbor_centroids,
    edge_wl_sum,
    get_cell_neighbors,
)


def build_rows(positions, num_macros, num_cells):
    """Build row structure: row_y -> [cell indices sorted by x]."""
    rows = {}
    cell_row = {}
    row_index = {}

    for cell_idx in range(num_macros, num_cells):
        row_y = round(positions[cell_idx, 1].item() * 10.0) / 10.0
        rows.setdefault(row_y, []).append(cell_idx)
        cell_row[cell_idx] = row_y

    for row_y, row_cells in rows.items():
        row_cells.sort(key=lambda c: positions[c, 0].item())
        row_index[row_y] = {cell_idx: idx for idx, cell_idx in enumerate(row_cells)}

    return rows, cell_row, row_index


def compact_row(row_cells, widths, start_x):
    """Pack a row from left to right and return (cell_idx, new_x) pairs."""
    packed = []
    cursor = start_x
    for cell_idx in row_cells:
        width = widths[cell_idx].item()
        new_x = cursor + width / 2.0
        packed.append((cell_idx, new_x))
        cursor = new_x + width / 2.0
    return packed


def get_row_start(row_cells, positions, widths):
    """Get the left edge of a row's current extent."""
    if not row_cells:
        return 0.0
    first = row_cells[0]
    return positions[first, 0].item() - widths[first].item() / 2.0


def build_macro_obstacles(positions, widths, heights, num_macros):
    obstacles = []
    for macro_idx in range(num_macros):
        x = positions[macro_idx, 0].item()
        y = positions[macro_idx, 1].item()
        w = widths[macro_idx].item()
        h = heights[macro_idx].item()
        obstacles.append((x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0))
    return obstacles


def check_macro_overlap(x, y, w, h, obstacles):
    left = x - w / 2.0
    right = x + w / 2.0
    bottom = y - h / 2.0
    top = y + h / 2.0
    for obs_left, obs_bottom, obs_right, obs_top in obstacles:
        if right > obs_left and left < obs_right and top > obs_bottom and bottom < obs_top:
            return True
    return False


def try_within_row_swap(
    ci,
    cj,
    row_cells,
    idx_i,
    idx_j,
    positions,
    widths,
    heights,
    wl_ctx,
    obstacles,
    row_y,
):
    """Try swapping ci and cj in the same row. Returns (delta, plan)."""
    new_order = list(row_cells)
    new_order[idx_i], new_order[idx_j] = new_order[idx_j], new_order[idx_i]
    start_x = get_row_start(row_cells, positions, widths)
    packed = compact_row(new_order, widths, start_x)

    for cell_idx, new_x in packed:
        if check_macro_overlap(new_x, row_y, widths[cell_idx].item(), heights[cell_idx].item(), obstacles):
            return 0.0, None

    lo = min(idx_i, idx_j)
    hi = max(idx_i, idx_j)
    affected = row_cells[lo:hi + 1]
    incident_edges = collect_incident_edges(affected, wl_ctx)
    wl_before = edge_wl_sum(incident_edges, positions, wl_ctx)

    old_x = {}
    for cell_idx, new_x in packed:
        old_x[cell_idx] = positions[cell_idx, 0].item()
        positions[cell_idx, 0] = new_x

    wl_after = edge_wl_sum(incident_edges, positions, wl_ctx)

    for cell_idx, _ in packed:
        positions[cell_idx, 0] = old_x[cell_idx]

    return wl_after - wl_before, {
        "new_order": new_order,
        "packed_positions": packed,
        "swap_pair": (ci, cj),
    }


def try_cross_row_move(
    ci,
    src_row_cells,
    dst_row_cells,
    dst_row_y,
    insert_x,
    positions,
    widths,
    heights,
    wl_ctx,
    obstacles,
):
    """Try moving ci to dst_row. Returns (delta, plan)."""
    new_src = [cell_idx for cell_idx in src_row_cells if cell_idx != ci]
    dst_x = [positions[cell_idx, 0].item() for cell_idx in dst_row_cells]
    insert_idx = bisect_left(dst_x, insert_x)
    new_dst = list(dst_row_cells)
    new_dst.insert(insert_idx, ci)

    src_start = get_row_start(src_row_cells, positions, widths) if new_src else None
    dst_start = (
        get_row_start(dst_row_cells, positions, widths)
        if dst_row_cells
        else insert_x - widths[ci].item() / 2.0
    )

    src_packed = compact_row(new_src, widths, src_start) if new_src else []
    dst_packed = compact_row(new_dst, widths, dst_start) if new_dst else []

    for cell_idx, new_x in src_packed:
        cur_y = positions[cell_idx, 1].item()
        if check_macro_overlap(new_x, cur_y, widths[cell_idx].item(), heights[cell_idx].item(), obstacles):
            return 0.0, None
    for cell_idx, new_x in dst_packed:
        if check_macro_overlap(new_x, dst_row_y, widths[cell_idx].item(), heights[cell_idx].item(), obstacles):
            return 0.0, None

    affected = new_src + new_dst
    incident_edges = collect_incident_edges(affected, wl_ctx)
    wl_before = edge_wl_sum(incident_edges, positions, wl_ctx)

    old_positions = {}
    for cell_idx in set(affected):
        old_positions[cell_idx] = (
            positions[cell_idx, 0].item(),
            positions[cell_idx, 1].item(),
        )

    for cell_idx, new_x in src_packed:
        positions[cell_idx, 0] = new_x
    for cell_idx, new_x in dst_packed:
        positions[cell_idx, 0] = new_x
        positions[cell_idx, 1] = dst_row_y

    wl_after = edge_wl_sum(incident_edges, positions, wl_ctx)

    for cell_idx, (old_x, old_y) in old_positions.items():
        positions[cell_idx, 0] = old_x
        positions[cell_idx, 1] = old_y

    return wl_after - wl_before, {
        "new_src": new_src,
        "new_dst": new_dst,
        "src_packed": src_packed,
        "dst_packed": dst_packed,
    }


def _build_row_lookup(rows, row_keys, num_cells, device):
    cell_row_ids = torch.full((num_cells,), -1, dtype=torch.long, device=device)
    for row_id, row_y in enumerate(row_keys):
        row_cells = rows[row_y]
        if row_cells:
            row_tensor = torch.as_tensor(row_cells, dtype=torch.long, device=device)
            cell_row_ids[row_tensor] = row_id
    return cell_row_ids


def _rank_destination_rows(
    ci,
    cur_row_y,
    target_x,
    target_y,
    row_keys,
    cell_row_ids,
    positions,
    wl_ctx,
    cross_row_limit,
):
    ranked = []
    seen_rows = set()
    neighbors = get_cell_neighbors(ci, wl_ctx)

    if neighbors.numel() > 0 and row_keys:
        neighbor_row_ids = cell_row_ids[neighbors]
        valid_mask = neighbor_row_ids >= 0
        valid_neighbors = neighbors[valid_mask]
        valid_row_ids = neighbor_row_ids[valid_mask]

        if valid_row_ids.numel() > 0:
            unique_row_ids, counts = torch.unique(valid_row_ids, return_counts=True)
            preferred = []
            for row_id, count in zip(unique_row_ids.tolist(), counts.tolist()):
                row_y = row_keys[row_id]
                if abs(row_y - cur_row_y) < 0.05:
                    continue
                row_neighbors = valid_neighbors[valid_row_ids == row_id]
                row_target_x = positions[row_neighbors, 0].mean().item()
                preferred.append((-count, abs(row_y - target_y), row_y, row_target_x))

            preferred.sort()
            for _neg_count, _dist, row_y, row_target_x in preferred:
                ranked.append((row_y, row_target_x))
                seen_rows.add(row_y)
                if len(ranked) >= cross_row_limit:
                    return ranked

    fallback_rows = sorted(row_keys, key=lambda row_y: abs(row_y - target_y))
    for row_y in fallback_rows:
        if abs(row_y - cur_row_y) < 0.05 or row_y in seen_rows:
            continue
        ranked.append((row_y, target_x))
        if len(ranked) >= cross_row_limit:
            break

    return ranked


def swap_engine(
    cell_features,
    pin_features,
    edge_list,
    max_iterations=20,
    num_macros=None,
    enable_within_row_swaps=False,
    within_row_window=3,
    cross_row_limit=None,
    verbose=False,
):
    """Fast iterative cell swap engine."""
    start_time = time.perf_counter()
    num_cells = cell_features.shape[0]
    if num_cells <= 1:
        return {"time": 0.0, "swaps": 0, "moves": 0, "iterations": 0}

    if num_macros is None:
        num_macros = (cell_features[:, 5] > 1.5).sum().item()

    positions = cell_features[:, 2:4].detach()
    widths = cell_features[:, 4].detach()
    heights = cell_features[:, 5].detach()
    wl_ctx = build_connectivity_context(pin_features, edge_list, num_cells=num_cells)
    obstacles = build_macro_obstacles(positions, widths, heights, num_macros)

    total_swaps = 0
    total_moves = 0
    if cross_row_limit is None:
        cross_row_limit = 6 if num_cells <= 300 else 4

    executed_iterations = 0
    for iteration in range(max_iterations):
        rows, cell_row, row_index = build_rows(positions, num_macros, num_cells)
        row_keys = sorted(rows.keys())
        if not row_keys:
            break
        executed_iterations = iteration + 1
        row_id_by_y = {row_y: row_id for row_id, row_y in enumerate(row_keys)}

        cell_scores = compute_cell_wl_scores(positions, wl_ctx, num_cells)
        target_x, target_y, _degree = compute_neighbor_centroids(positions, wl_ctx, num_cells)
        cell_row_ids = _build_row_lookup(rows, row_keys, num_cells, positions.device)

        ordered_cells = torch.argsort(cell_scores[num_macros:], descending=True) + num_macros
        iter_swaps = 0
        iter_moves = 0
        moved_cells = set()

        for ci in ordered_cells.tolist():
            if ci in moved_cells:
                continue

            cur_row_y = cell_row.get(ci)
            if cur_row_y is None:
                continue

            cur_row = rows.get(cur_row_y, [])
            idx_ci = row_index[cur_row_y].get(ci)
            if idx_ci is None:
                continue

            neighbors = get_cell_neighbors(ci, wl_ctx)
            neighbor_set = set(neighbors.tolist()) if neighbors.numel() > 0 else set()

            best_swap_delta = -0.01
            best_swap_plan = None
            if enable_within_row_swaps and len(cur_row) > 1:
                lo = max(0, idx_ci - within_row_window)
                hi = min(len(cur_row), idx_ci + within_row_window + 1)
                candidate_indices = list(range(idx_ci - 1, lo - 1, -1))
                candidate_indices.extend(range(idx_ci + 1, hi))
                candidate_indices.sort(
                    key=lambda idx_j: (
                        0 if cur_row[idx_j] in neighbor_set else 1,
                        abs(idx_j - idx_ci),
                    )
                )

                for idx_j in candidate_indices:
                    cj = cur_row[idx_j]
                    if cj in moved_cells:
                        continue

                    delta, swap_plan = try_within_row_swap(
                        ci,
                        cj,
                        cur_row,
                        idx_ci,
                        idx_j,
                        positions,
                        widths,
                        heights,
                        wl_ctx,
                        obstacles,
                        cur_row_y,
                    )

                    if delta < best_swap_delta:
                        best_swap_delta = delta
                        best_swap_plan = swap_plan

            if best_swap_plan is not None:
                for cell_idx, new_x in best_swap_plan["packed_positions"]:
                    positions[cell_idx, 0] = new_x
                rows[cur_row_y] = best_swap_plan["new_order"]
                row_index[cur_row_y] = {
                    cell_idx: idx for idx, cell_idx in enumerate(best_swap_plan["new_order"])
                }
                moved_cells.add(ci)
                iter_swaps += 1
                continue

            candidate_rows = _rank_destination_rows(
                ci,
                cur_row_y,
                target_x[ci].item(),
                target_y[ci].item(),
                row_keys,
                cell_row_ids,
                positions,
                wl_ctx,
                cross_row_limit,
            )

            best_delta = -0.01
            best_move = None
            for dst_row_y, insert_x in candidate_rows:
                dst_row = rows.get(dst_row_y, [])
                delta, move_plan = try_cross_row_move(
                    ci,
                    cur_row,
                    dst_row,
                    dst_row_y,
                    insert_x,
                    positions,
                    widths,
                    heights,
                    wl_ctx,
                    obstacles,
                )
                if delta < best_delta:
                    best_delta = delta
                    best_move = (dst_row_y, move_plan)

            if best_move is not None:
                dst_row_y, move_plan = best_move
                for cell_idx, new_x in move_plan["src_packed"]:
                    positions[cell_idx, 0] = new_x
                for cell_idx, new_x in move_plan["dst_packed"]:
                    positions[cell_idx, 0] = new_x
                    positions[cell_idx, 1] = dst_row_y

                rows[cur_row_y] = move_plan["new_src"]
                rows[dst_row_y] = move_plan["new_dst"]
                row_index[cur_row_y] = {
                    cell_idx: idx for idx, cell_idx in enumerate(move_plan["new_src"])
                }
                row_index[dst_row_y] = {
                    cell_idx: idx for idx, cell_idx in enumerate(move_plan["new_dst"])
                }
                cell_row[ci] = dst_row_y
                cell_row_ids[ci] = row_id_by_y[dst_row_y]
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
        print(
            f"    Swap engine done: {total_swaps} swaps, {total_moves} moves, "
            f"{executed_iterations} iters, {elapsed:.1f}s"
        )

    return {
        "time": elapsed,
        "swaps": total_swaps,
        "moves": total_moves,
        "iterations": executed_iterations,
    }
