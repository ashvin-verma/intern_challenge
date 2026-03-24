"""WL-aware Abacus legalization.

Same cluster-merging DP structure as Abacus, but optimizes
wirelength instead of displacement. For each cluster position,
evaluates the WL of all incident edges and picks the position
that minimizes total WL.

Key insight: displacement-minimizing Abacus preserves GD neighborhoods,
but GD neighborhoods for overlapping cells are meaningless. WL-aware
Abacus places cells where their EDGES want them, not where GD put them.
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def abacus_legalize(cell_features, num_macros=None, pin_features=None, edge_list=None):
    """WL-aware Abacus legalization.

    For each row:
    1. Sort cells by GD x (preserve topology)
    2. Pack left-to-right resolving overlaps via cluster merge
    3. For each merged cluster, try a few candidate positions and pick
       the one with lowest total incident WL

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

    gd_x = positions[:, 0].clone()
    gd_y = positions[:, 1].clone()

    # Build WL evaluation structures if we have connectivity info
    pin_to_cell = None
    cell_edges = None
    if pin_features is not None and edge_list is not None:
        pin_to_cell = pin_features[:, 0].long().tolist()
        cell_edges = defaultdict(list)
        for e in range(edge_list.shape[0]):
            sc = pin_to_cell[edge_list[e, 0].item()]
            tc = pin_to_cell[edge_list[e, 1].item()]
            cell_edges[sc].append(e)
            if tc != sc:
                cell_edges[tc].append(e)

    def cluster_wl(cells, lefts, row_y):
        """Compute total WL of edges incident to cells at given positions."""
        if pin_to_cell is None:
            return 0.0
        # Temporarily set positions
        old = {}
        cursor = lefts
        for ci, wi in zip(cells, [widths[c].item() for c in cells]):
            old[ci] = (positions[ci, 0].item(), positions[ci, 1].item())
            positions[ci, 0] = cursor + wi / 2
            positions[ci, 1] = row_y
            cursor += wi

        total = 0.0
        seen = set()
        for ci in cells:
            for e in cell_edges.get(ci, []):
                if e in seen:
                    continue
                seen.add(e)
                sp, tp = edge_list[e, 0].item(), edge_list[e, 1].item()
                sc, tc = pin_to_cell[sp], pin_to_cell[tp]
                dx = abs(positions[sc, 0].item() + pin_features[sp, 1].item()
                         - positions[tc, 0].item() - pin_features[tp, 1].item())
                dy = abs(positions[sc, 1].item() + pin_features[sp, 2].item()
                         - positions[tc, 1].item() - pin_features[tp, 2].item())
                total += dx + dy

        # Restore
        for ci, (ox, oy) in old.items():
            positions[ci, 0] = ox
            positions[ci, 1] = oy
        return total

    # --- Legalize macros ---
    if num_macros > 1:
        for _pass in range(200):
            any_ov = False
            for i in range(num_macros):
                for j in range(i + 1, num_macros):
                    xi, yi = positions[i, 0].item(), positions[i, 1].item()
                    xj, yj = positions[j, 0].item(), positions[j, 1].item()
                    wi, hi = widths[i].item(), heights[i].item()
                    wj, hj = widths[j].item(), heights[j].item()
                    ov_x = (wi + wj) / 2 - abs(xi - xj)
                    ov_y = (hi + hj) / 2 - abs(yi - yj)
                    if ov_x > 0 and ov_y > 0:
                        any_ov = True
                        if ov_x <= ov_y:
                            s = ov_x / 2 + 0.1
                            sign = 1.0 if xi >= xj else -1.0
                            positions[i, 0] += sign * s
                            positions[j, 0] -= sign * s
                        else:
                            s = ov_y / 2 + 0.1
                            sign = 1.0 if yi >= yj else -1.0
                            positions[i, 1] += sign * s
                            positions[j, 1] -= sign * s
            if not any_ov:
                break

    if num_macros >= N:
        cell_features[:, 2:4] = positions
        return {"time": time.perf_counter() - start_time, "cells_moved": 0, "max_displacement": 0.0}

    # --- Macro obstacles ---
    obstacles = []
    for i in range(num_macros):
        ox, oy = positions[i, 0].item(), positions[i, 1].item()
        ow, oh = widths[i].item(), heights[i].item()
        obstacles.append((ox - ow / 2, oy - oh / 2, ox + ow / 2, oy + oh / 2))

    # --- Row assignment ---
    std_indices = list(range(num_macros, N))
    row_height = 1.0
    y_min = gd_y[std_indices].min().item() - 10

    row_assignments = {}
    for idx in std_indices:
        row_idx = round((gd_y[idx].item() - y_min) / row_height)
        row_assignments.setdefault(row_idx, []).append(idx)

    # --- WL-aware cluster DP per row ---
    for row_idx, cells_in_row in row_assignments.items():
        row_y = y_min + row_idx * row_height
        if not cells_in_row:
            continue

        cells_in_row.sort(key=lambda i: gd_x[i].item())

        # Obstacle intervals for this row
        row_obs = sorted([
            (ox_min, ox_max) for ox_min, oy_min, ox_max, oy_max in obstacles
            if oy_max > row_y - row_height / 2 and oy_min < row_y + row_height / 2
        ])

        def push_past_obstacles(left, width):
            for _a in range(20):
                moved = False
                for omin, omax in row_obs:
                    if left + width > omin and left < omax:
                        left = omax + 0.1
                        moved = True
                if not moved:
                    break
            return left

        # Build clusters with WL-aware positioning
        clusters = []

        for ci in cells_in_row:
            wi = widths[ci].item()

            # Candidate positions for this singleton:
            # 1. GD position (displacement = 0)
            # 2. Barycentric x of neighbors (WL-optimal for this cell)
            candidates = [gd_x[ci].item() - wi / 2]

            if cell_edges is not None:
                nbr_xs = []
                for e in cell_edges.get(ci, []):
                    sp, tp = edge_list[e, 0].item(), edge_list[e, 1].item()
                    sc, tc = pin_to_cell[sp], pin_to_cell[tp]
                    other = tc if sc == ci else sc
                    nbr_xs.append(positions[other, 0].item())
                if nbr_xs:
                    bary_x = sum(nbr_xs) / len(nbr_xs)
                    candidates.append(bary_x - wi / 2)

            # Constraint: right of previous cluster, past obstacles
            min_left = 0.0
            if clusters:
                min_left = clusters[-1]["left"] + clusters[-1]["total_w"]
            min_left = push_past_obstacles(min_left, wi)

            # Pick best candidate
            best_left = max(candidates[0], min_left)
            best_wl = float("inf")

            for cand in candidates:
                cl = max(cand, min_left)
                cl = push_past_obstacles(cl, wi)
                wl = cluster_wl([ci], cl, row_y)
                if wl < best_wl:
                    best_wl = wl
                    best_left = cl

            clusters.append({
                "cells": [ci],
                "widths": [wi],
                "total_w": wi,
                "left": best_left,
            })

            # Merge overlapping clusters
            while len(clusters) >= 2:
                prev = clusters[-2]
                cur = clusters[-1]
                if prev["left"] + prev["total_w"] <= cur["left"] + 1e-6:
                    break

                merged_cells = prev["cells"] + cur["cells"]
                merged_widths = prev["widths"] + cur["widths"]
                merged_total_w = prev["total_w"] + cur["total_w"]

                # Try candidate positions for merged cluster:
                # 1. Displacement-optimal (classic Abacus)
                cum = 0.0
                disp_sum = 0.0
                for c2, w2 in zip(merged_cells, merged_widths):
                    disp_sum += gd_x[c2].item() - w2 / 2 - cum
                    cum += w2
                disp_opt = disp_sum / len(merged_cells)

                # 2. Previous cluster's left (compact)
                min_left = 0.0
                if len(clusters) >= 3:
                    min_left = clusters[-3]["left"] + clusters[-3]["total_w"]
                min_left = push_past_obstacles(min_left, merged_total_w)

                # Evaluate candidates
                cands = [disp_opt, prev["left"]]
                best_left = max(disp_opt, min_left)
                best_wl = float("inf")

                for cand in cands:
                    cl = max(cand, min_left)
                    cl = push_past_obstacles(cl, merged_total_w)
                    wl = cluster_wl(merged_cells, cl, row_y)
                    if wl < best_wl:
                        best_wl = wl
                        best_left = cl

                clusters.pop()
                clusters.pop()
                clusters.append({
                    "cells": merged_cells,
                    "widths": merged_widths,
                    "total_w": merged_total_w,
                    "left": best_left,
                })

        # Assign final positions
        for cluster in clusters:
            cur = cluster["left"]
            for c2, w2 in zip(cluster["cells"], cluster["widths"]):
                positions[c2, 0] = cur + w2 / 2
                positions[c2, 1] = row_y
                cur += w2

    cell_features[:, 2:4] = positions
    displacement = (positions - original_positions).abs()
    return {
        "time": time.perf_counter() - start_time,
        "cells_moved": (displacement.sum(dim=1) > 0.01).sum().item(),
        "max_displacement": displacement.max().item(),
    }
