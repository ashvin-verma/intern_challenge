"""WL-aware shelf legalizer candidate.

This is intentionally conservative: it builds a fresh legal shelf placement from
the current positions and lets the solver accept it only if exact metrics improve.
"""

from __future__ import annotations

import time

import torch

from ashvin.connectivity import (
    build_connectivity_context,
    compute_cell_wl_scores,
    compute_neighbor_centroids,
)


def _macro_overlaps(x, y, width, height, obstacles):
    left = x - width / 2.0
    right = x + width / 2.0
    bottom = y - height / 2.0
    top = y + height / 2.0
    for ox0, oy0, ox1, oy1 in obstacles:
        if right > ox0 and left < ox1 and top > oy0 and bottom < oy1:
            return True
    return False


def _push_past_macros(x, y, width, height, obstacles, gap):
    for _ in range(20):
        shifted = False
        left = x - width / 2.0
        right = x + width / 2.0
        bottom = y - height / 2.0
        top = y + height / 2.0
        for ox0, oy0, ox1, oy1 in obstacles:
            if right > ox0 and left < ox1 and top > oy0 and bottom < oy1:
                x = ox1 + width / 2.0 + gap
                shifted = True
                break
        if not shifted:
            break
    return x


def _legalize_macros(positions, widths, heights, num_macros):
    if num_macros <= 1:
        return
    for _ in range(200):
        any_overlap = False
        for i in range(num_macros):
            for j in range(i + 1, num_macros):
                dx = positions[i, 0].item() - positions[j, 0].item()
                dy = positions[i, 1].item() - positions[j, 1].item()
                ov_x = (widths[i].item() + widths[j].item()) / 2.0 - abs(dx)
                ov_y = (heights[i].item() + heights[j].item()) / 2.0 - abs(dy)
                if ov_x > 0 and ov_y > 0:
                    any_overlap = True
                    if ov_x <= ov_y:
                        shift = ov_x / 2.0 + 0.1
                        sign = 1.0 if dx >= 0 else -1.0
                        positions[i, 0] += sign * shift
                        positions[j, 0] -= sign * shift
                    else:
                        shift = ov_y / 2.0 + 0.1
                        sign = 1.0 if dy >= 0 else -1.0
                        positions[i, 1] += sign * shift
                        positions[j, 1] -= sign * shift
        if not any_overlap:
            break


def _build_macro_obstacles(positions, widths, heights, num_macros):
    obstacles = []
    for idx in range(num_macros):
        x = positions[idx, 0].item()
        y = positions[idx, 1].item()
        w = widths[idx].item()
        h = heights[idx].item()
        obstacles.append((x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0))
    return obstacles


def _row_candidates(target_y, orig_y, row_values, row_limit):
    ranked = sorted(row_values, key=lambda ry: (abs(ry - target_y), abs(ry - orig_y), ry))
    return ranked[: max(1, row_limit)]


def _candidate_insert_positions(row_items, target_x, orig_x, width, gap):
    candidates = [target_x, orig_x]
    if not row_items:
        return candidates
    for item in row_items:
        cx = item[0]
        cw = item[1]
        candidates.append(cx + (cw + width) / 2.0 + gap)
        candidates.append(cx - (cw + width) / 2.0 - gap)
    return candidates


def _compact_items(items, row_y, obstacles, gap):
    if not items:
        return []
    items = sorted(items, key=lambda item: item[0])
    packed = []
    cursor_right = None
    for target_x, width, height, cell_idx in items:
        x = target_x if cursor_right is None else max(target_x, cursor_right + width / 2.0 + gap)
        x = _push_past_macros(x, row_y, width, height, obstacles, gap)
        packed.append((x, width, height, cell_idx))
        cursor_right = x + width / 2.0
    return packed


def shelf_legalize_v2(
    cell_features,
    pin_features,
    edge_list,
    num_macros=None,
    row_limit=5,
    max_cells=3000,
    gap=1e-3,
):
    """Build a WL-aware shelf placement in-place.

    Returns a stats dict. The caller should verify overlap/WL before accepting.
    """
    start_time = time.perf_counter()
    num_cells = cell_features.shape[0]
    if num_cells <= 1:
        return {"time": 0.0, "cells_moved": 0, "max_displacement": 0.0, "rows": 0}
    if num_cells > max_cells:
        return {"time": 0.0, "cells_moved": 0, "max_displacement": 0.0, "rows": 0}
    if num_macros is None:
        num_macros = int((cell_features[:, 5] > 1.5).sum().item())

    positions = cell_features[:, 2:4].detach()
    widths = cell_features[:, 4].detach()
    heights = cell_features[:, 5].detach()
    original = positions.clone()

    _legalize_macros(positions, widths, heights, num_macros)
    obstacles = _build_macro_obstacles(positions, widths, heights, num_macros)

    if num_macros >= num_cells:
        return {
            "time": time.perf_counter() - start_time,
            "cells_moved": 0,
            "max_displacement": 0.0,
            "rows": 0,
        }

    ctx = build_connectivity_context(pin_features, edge_list, num_cells=num_cells)
    target_x, target_y, degree = compute_neighbor_centroids(positions, ctx, num_cells)
    wl_scores = compute_cell_wl_scores(positions, ctx, num_cells)

    std_cells = list(range(num_macros, num_cells))
    std_y = positions[num_macros:, 1]
    row_height = max(1.0, heights[num_macros:].max().item())
    y_min = (torch.floor(std_y.min() / row_height).item() - 4.0) * row_height
    y_max = (torch.ceil(std_y.max() / row_height).item() + 4.0) * row_height
    row_count = max(1, int(round((y_max - y_min) / row_height)) + 1)
    row_values = [y_min + idx * row_height for idx in range(row_count)]
    rows = {row_y: [] for row_y in row_values}

    std_cells.sort(
        key=lambda ci: (
            -float(wl_scores[ci].item()),
            -float(degree[ci].item()),
            float(target_y[ci].item()),
            float(target_x[ci].item()),
        )
    )

    for cell_idx in std_cells:
        width = widths[cell_idx].item()
        height = heights[cell_idx].item()
        tx = target_x[cell_idx].item()
        ty = target_y[cell_idx].item()
        ox = positions[cell_idx, 0].item()
        oy = positions[cell_idx, 1].item()
        best = None
        best_score = float("inf")

        for row_y in _row_candidates(ty, oy, row_values, row_limit):
            row_items = rows[row_y]
            for cand_x in _candidate_insert_positions(row_items, tx, ox, width, gap):
                if _macro_overlaps(cand_x, row_y, width, height, obstacles):
                    cand_x = _push_past_macros(cand_x, row_y, width, height, obstacles, gap)
                trial_items = row_items + [(cand_x, width, height, cell_idx)]
                packed = _compact_items(trial_items, row_y, obstacles, gap)
                if len(packed) != len(trial_items):
                    continue
                placed_x = next(x for x, _w, _h, ci in packed if ci == cell_idx)
                score = abs(placed_x - tx) + 1.25 * abs(row_y - ty) + 0.05 * abs(placed_x - ox)
                score += 0.02 * len(row_items)
                if score < best_score:
                    best_score = score
                    best = (row_y, packed)

        if best is None:
            row_y = min(row_values, key=lambda ry: abs(ry - oy))
            packed = _compact_items(rows[row_y] + [(ox, width, height, cell_idx)], row_y, obstacles, gap)
            best = (row_y, packed)

        row_y, packed = best
        rows[row_y] = packed
        for x, _width, _height, ci in packed:
            positions[ci, 0] = x
            positions[ci, 1] = row_y

    cell_features[:, 2:4] = positions
    displacement = (positions - original).abs()
    return {
        "time": time.perf_counter() - start_time,
        "cells_moved": int((displacement.sum(dim=1) > 0.01).sum().item()),
        "max_displacement": displacement.max().item() if displacement.numel() else 0.0,
        "rows": sum(1 for items in rows.values() if items),
    }
