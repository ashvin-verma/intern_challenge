"""Bounded mid-size row refinement.

This is a legal-to-legal detailed-placement pass for cases that are too large
for pairwise detailed placement but small enough for row-level WL evaluation.
It avoids all-pairs swaps: each row gets a few deterministic connectivity-driven
order/shift candidates, and a candidate is accepted only if affected-edge WL
improves.
"""

from __future__ import annotations

import time

import torch

from ashvin.connectivity import (
    build_connectivity_context,
    collect_incident_edges,
    compute_edge_wl,
    compute_neighbor_centroids,
    edge_wl_sum,
)
from ashvin.swap_engine import build_macro_obstacles, build_rows, check_macro_overlap, get_row_start


def _packed_positions(order, width_vals, start_x, gap):
    packed = []
    cursor = start_x
    for cell_idx in order:
        width = width_vals[cell_idx]
        new_x = cursor + width / 2.0
        packed.append((cell_idx, new_x))
        cursor = new_x + width / 2.0 + gap
    return packed


def _target_start(order, width_vals, target_x_vals, current_start, gap):
    if not order:
        return current_start
    offsets = []
    cursor = 0.0
    for cell_idx in order:
        width = width_vals[cell_idx]
        offsets.append(cursor + width / 2.0)
        cursor += width + gap
    desired = sorted(target_x_vals[cell_idx] - offset for cell_idx, offset in zip(order, offsets))
    mid = len(desired) // 2
    if len(desired) % 2:
        return desired[mid]
    return 0.5 * (desired[mid - 1] + desired[mid])


def _row_has_macro_overlap(packed, row_y, width_vals, height_vals, obstacles):
    for cell_idx, new_x in packed:
        if check_macro_overlap(
            new_x,
            row_y,
            width_vals[cell_idx],
            height_vals[cell_idx],
            obstacles,
        ):
            return True
    return False


def _apply_packed(positions, packed, row_y=None):
    old = {}
    for cell_idx, new_x in packed:
        old[cell_idx] = (positions[cell_idx, 0].item(), positions[cell_idx, 1].item())
        positions[cell_idx, 0] = new_x
        if row_y is not None:
            positions[cell_idx, 1] = row_y
    return old


def _restore_positions(positions, old):
    for cell_idx, (old_x, old_y) in old.items():
        positions[cell_idx, 0] = old_x
        positions[cell_idx, 1] = old_y


def _unique_orders(row_cells, target_x_vals, current_x_vals, max_window):
    current = list(row_cells)
    orders = [current]

    by_target = sorted(current, key=lambda c: (target_x_vals[c], current_x_vals[c]))
    orders.append(by_target)

    blended = sorted(
        current,
        key=lambda c: (0.7 * target_x_vals[c] + 0.3 * current_x_vals[c]),
    )
    orders.append(blended)

    if max_window and len(current) > max_window:
        windowed = current[:]
        for start in range(0, len(windowed), max_window):
            stop = min(len(windowed), start + max_window)
            windowed[start:stop] = sorted(
                windowed[start:stop],
                key=lambda c: (target_x_vals[c], current_x_vals[c]),
            )
        orders.append(windowed)

    deduped = []
    seen = set()
    for order in orders:
        key = tuple(order)
        if key not in seen:
            seen.add(key)
            deduped.append(order)
    return deduped


def _try_row_candidate(
    row_y,
    candidate_order,
    candidate_start,
    positions,
    width_vals,
    height_vals,
    obstacles,
    wl_ctx,
    incident_edges,
    wl_before,
    gap,
):
    packed = _packed_positions(candidate_order, width_vals, candidate_start, gap)
    if _row_has_macro_overlap(packed, row_y, width_vals, height_vals, obstacles):
        return 0.0, None

    old = _apply_packed(positions, packed)
    wl_after = edge_wl_sum(incident_edges, positions, wl_ctx)
    _restore_positions(positions, old)

    return wl_after - wl_before, packed


def _refine_rows_once(
    positions,
    widths,
    heights,
    num_macros,
    wl_ctx,
    target_x,
    obstacles,
    min_row_cells,
    max_window,
    gap,
):
    num_cells = positions.shape[0]
    rows, _cell_row, _row_index = build_rows(positions, num_macros, num_cells)
    row_items = sorted(rows.items(), key=lambda kv: (len(kv[1]), kv[0]), reverse=True)
    width_vals = widths.tolist()
    height_vals = heights.tolist()
    target_x_vals = target_x.tolist()
    current_x_vals = positions[:, 0].tolist()
    accepted = 0
    total_delta = 0.0

    for row_y, row_cells in row_items:
        if len(row_cells) < min_row_cells:
            continue

        current_start = get_row_start(row_cells, positions, widths)
        incident_edges = collect_incident_edges(row_cells, wl_ctx)
        wl_before = edge_wl_sum(incident_edges, positions, wl_ctx)
        best_delta = -1e-4
        best_packed = None

        for order in _unique_orders(row_cells, target_x_vals, current_x_vals, max_window):
            starts = [current_start]
            target_start = _target_start(order, width_vals, target_x_vals, current_start, gap)
            if abs(target_start - current_start) > 1e-4:
                starts.append(target_start)
            for start_x in starts:
                delta, packed = _try_row_candidate(
                    row_y,
                    order,
                    start_x,
                    positions,
                    width_vals,
                    height_vals,
                    obstacles,
                    wl_ctx,
                    incident_edges,
                    wl_before,
                    gap,
                )
                if delta < best_delta:
                    best_delta = delta
                    best_packed = packed

        if best_packed is not None:
            _apply_packed(positions, best_packed)
            accepted += 1
            total_delta += best_delta

    return accepted, total_delta


def _try_global_row_remap(
    positions,
    widths,
    heights,
    num_macros,
    wl_ctx,
    target_x,
    target_y,
    obstacles,
    gap,
):
    num_cells = positions.shape[0]
    rows, _cell_row, _row_index = build_rows(positions, num_macros, num_cells)
    if len(rows) <= 1:
        return False, 0.0

    row_specs = []
    width_vals = widths.tolist()
    height_vals = heights.tolist()
    target_x_vals = target_x.tolist()
    current_x_vals = positions[:, 0].tolist()
    for row_y, row_cells in sorted(rows.items()):
        if not row_cells:
            continue
        row_specs.append((row_y, get_row_start(row_cells, positions, widths), len(row_cells)))

    movable = list(range(num_macros, num_cells))
    target_y_vals = target_y.tolist()
    movable.sort(key=lambda c: (target_y_vals[c], target_x_vals[c]))

    assignments = []
    cursor = 0
    for row_y, start_x, count in row_specs:
        assigned = movable[cursor:cursor + count]
        cursor += count
        assigned.sort(key=lambda c: (target_x_vals[c], current_x_vals[c]))
        target_start = _target_start(assigned, width_vals, target_x_vals, start_x, gap)
        packed = _packed_positions(assigned, width_vals, target_start, gap)
        if _row_has_macro_overlap(packed, row_y, width_vals, height_vals, obstacles):
            return False, 0.0
        assignments.append((row_y, packed))

    wl_before = compute_edge_wl(positions, wl_ctx).sum().item()
    old = {}
    for row_y, packed in assignments:
        old.update(_apply_packed(positions, packed, row_y=row_y))
    wl_after = compute_edge_wl(positions, wl_ctx).sum().item()

    if wl_after < wl_before - 1e-4:
        return True, wl_after - wl_before

    _restore_positions(positions, old)
    return False, 0.0


def mid_size_row_refine(
    cell_features,
    pin_features,
    edge_list,
    num_passes=2,
    num_macros=None,
    min_row_cells=4,
    max_window=16,
    try_row_remap=True,
    gap=1e-3,
    verbose=False,
):
    """Run bounded row-order/shift refinement in-place."""
    start_time = time.perf_counter()
    num_cells = cell_features.shape[0]
    if num_cells <= 1:
        return {"time": 0.0, "rows_changed": 0, "remaps": 0, "passes": 0}

    if num_macros is None:
        num_macros = (cell_features[:, 5] > 1.5).sum().item()

    positions = cell_features[:, 2:4].detach()
    widths = cell_features[:, 4].detach()
    heights = cell_features[:, 5].detach()
    wl_ctx = build_connectivity_context(pin_features, edge_list, num_cells=num_cells)
    obstacles = build_macro_obstacles(positions, widths, heights, num_macros)

    rows_changed = 0
    remaps = 0
    executed_passes = 0

    for pass_idx in range(num_passes):
        target_x, target_y, _degree = compute_neighbor_centroids(positions, wl_ctx, num_cells)

        if try_row_remap and pass_idx == 0:
            accepted, delta = _try_global_row_remap(
                positions,
                widths,
                heights,
                num_macros,
                wl_ctx,
                target_x,
                target_y,
                obstacles,
                gap,
            )
            if accepted:
                remaps += 1
                if verbose:
                    print(f"    Mid-row remap accepted: delta={delta:.2f}")

        changed, delta = _refine_rows_once(
            positions,
            widths,
            heights,
            num_macros,
            wl_ctx,
            target_x,
            obstacles,
            min_row_cells,
            max_window,
            gap,
        )
        rows_changed += changed
        executed_passes = pass_idx + 1
        if verbose:
            print(f"    Mid-row pass {pass_idx}: rows={changed} delta={delta:.2f}")
        if changed == 0:
            break

    cell_features[:, 2:4] = positions
    return {
        "time": time.perf_counter() - start_time,
        "rows_changed": rows_changed,
        "remaps": remaps,
        "passes": executed_passes,
    }
