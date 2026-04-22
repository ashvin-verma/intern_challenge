"""Lightweight legality projection for projected GD.

The projection is intentionally simple: keep macros fixed, snap standard cells
to horizontal rows, then compact each row in x while skipping macro obstacles.
It is meant to run inside an optimizer loop without resetting optimizer state.
"""

from __future__ import annotations

import time

import torch


def _build_macro_obstacles(pos, widths, heights, num_macros):
    obstacles = []
    for idx in range(num_macros):
        x = pos[idx, 0].item()
        y = pos[idx, 1].item()
        w = widths[idx].item()
        h = heights[idx].item()
        obstacles.append((x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0))
    return obstacles


def _push_past_obstacles(x, row_y, width, height, obstacles, gap):
    for _ in range(20):
        shifted = False
        cell_left = x - width / 2.0
        cell_right = x + width / 2.0
        cell_bottom = row_y - height / 2.0
        cell_top = row_y + height / 2.0
        for ox_min, oy_min, ox_max, oy_max in obstacles:
            if (
                cell_right > ox_min
                and cell_left < ox_max
                and cell_top > oy_min
                and cell_bottom < oy_max
            ):
                x = ox_max + width / 2.0 + gap
                shifted = True
                break
        if not shifted:
            break
    return x


def project_to_legal_rows(pos, widths, heights, num_macros, gap=1e-3, row_height=1.0):
    """Project standard cells to compacted rows in-place.

    Args:
        pos: [N, 2] position tensor. Mutated in-place.
        widths: [N] cell widths matching the current GD sizes.
        heights: [N] cell heights matching the current GD sizes.
        num_macros: prefix count of macro cells to keep fixed.
        gap: horizontal spacing inserted during compaction.
        row_height: row pitch for snapped standard-cell y values.

    Returns:
        dict with lightweight stats: time, rows, cells_projected, max_displacement.
    """
    start = time.perf_counter()
    num_cells = pos.shape[0]
    if num_cells <= num_macros:
        return {
            "time": 0.0,
            "rows": 0,
            "cells_projected": 0,
            "max_displacement": 0.0,
        }

    row_height = max(float(row_height), 1e-6)
    original = pos[num_macros:].detach().clone()
    obstacles = _build_macro_obstacles(pos, widths, heights, num_macros)
    rows = {}

    with torch.no_grad():
        for cell_idx in range(num_macros, num_cells):
            y = pos[cell_idx, 1].item()
            row_y = round(y / row_height) * row_height
            rows.setdefault(row_y, []).append(cell_idx)

        for row_y, row_cells in rows.items():
            row_cells.sort(key=lambda idx: pos[idx, 0].item())
            cursor_right = None
            for cell_idx in row_cells:
                width = widths[cell_idx].item()
                height = heights[cell_idx].item()
                target_x = pos[cell_idx, 0].item()
                if cursor_right is None:
                    x = target_x
                else:
                    x = max(target_x, cursor_right + width / 2.0 + gap)
                x = _push_past_obstacles(x, row_y, width, height, obstacles, gap)
                pos[cell_idx, 0] = x
                pos[cell_idx, 1] = row_y
                cursor_right = x + width / 2.0

    displacement = (pos[num_macros:] - original).abs()
    max_displacement = displacement.max().item() if displacement.numel() else 0.0
    return {
        "time": time.perf_counter() - start,
        "rows": len(rows),
        "cells_projected": num_cells - num_macros,
        "max_displacement": max_displacement,
    }
