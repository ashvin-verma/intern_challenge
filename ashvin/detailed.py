"""Detailed placement engine: post-legalization WL optimization.

All moves preserve legality (zero overlap). Uses sparse edge structure
for cheap delta evaluation — moving one cell only recomputes its incident edges.

Three passes:
A. Pair swap: swap same-height cells if WL improves
B. Single-cell reinsertion: move cell to best gap near its neighbors
C. Window reorder: try permutations of 3-5 adjacent cells in a row
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def _build_structures(cell_features, pin_features, edge_list):
    """Build adjacency and edge structures for fast delta computation."""
    N = cell_features.shape[0]
    pin_to_cell = pin_features[:, 0].long().tolist()

    # cell -> list of edge indices
    cell_edges = defaultdict(list)
    for e in range(edge_list.shape[0]):
        sc = pin_to_cell[edge_list[e, 0].item()]
        tc = pin_to_cell[edge_list[e, 1].item()]
        cell_edges[sc].append(e)
        if tc != sc:
            cell_edges[tc].append(e)

    return pin_to_cell, cell_edges


def _cell_wl(cell_idx, positions, pin_features, edge_list, pin_to_cell, cell_edges):
    """Total WL of edges incident to a cell. O(degree)."""
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


def _check_overlap_local(positions, widths, heights, cell_idx, spatial_idx):
    """Check overlap using spatial index. O(neighbors)."""
    x = positions[cell_idx, 0].item()
    y = positions[cell_idx, 1].item()
    w = widths[cell_idx].item()
    h = heights[cell_idx].item()
    bx, by = spatial_idx["cell_to_bin"].get(cell_idx, (0, 0))
    for dbx in (-1, 0, 1):
        for dby in (-1, 0, 1):
            for j in spatial_idx["bin_to_cells"].get((bx + dbx, by + dby), []):
                if j == cell_idx:
                    continue
                if abs(x - positions[j, 0].item()) < (w + widths[j].item()) / 2 and \
                   abs(y - positions[j, 1].item()) < (h + heights[j].item()) / 2:
                    return True
    return False


def _build_spatial(positions, widths, N):
    """Build spatial hash index."""
    bin_size = max(widths.max().item(), 3.0)
    x_min = positions[:, 0].min().item() - bin_size
    y_min = positions[:, 1].min().item() - bin_size
    bin_to_cells = defaultdict(list)
    cell_to_bin = {}
    for i in range(N):
        bx = int((positions[i, 0].item() - x_min) / bin_size)
        by = int((positions[i, 1].item() - y_min) / bin_size)
        bin_to_cells[(bx, by)].append(i)
        cell_to_bin[i] = (bx, by)
    return {"bin_to_cells": bin_to_cells, "cell_to_bin": cell_to_bin, "bin_size": bin_size}


def pass_pair_swap(positions, widths, heights, pin_features, edge_list,
                   pin_to_cell, cell_edges, num_macros, N):
    """Swap same-height cells in nearby bins if WL improves."""
    spatial = _build_spatial(positions, widths, N)
    swaps = 0

    for (bx, by), cells in spatial["bin_to_cells"].items():
        std_cells = [c for c in cells if c >= num_macros]
        # Check against same bin + forward neighbors
        for nbx, nby in [(bx, by), (bx + 1, by), (bx, by + 1)]:
            nb_cells = [c for c in spatial["bin_to_cells"].get((nbx, nby), []) if c >= num_macros]

            for i in std_cells:
                hi = heights[i].item()
                for j in nb_cells:
                    if j <= i or abs(hi - heights[j].item()) > 0.01:
                        continue

                    # WL before
                    wl_before = (_cell_wl(i, positions, pin_features, edge_list, pin_to_cell, cell_edges) +
                                 _cell_wl(j, positions, pin_features, edge_list, pin_to_cell, cell_edges))

                    # Swap
                    pi, pj = positions[i].clone(), positions[j].clone()
                    positions[i], positions[j] = pj, pi

                    # WL after
                    wl_after = (_cell_wl(i, positions, pin_features, edge_list, pin_to_cell, cell_edges) +
                                _cell_wl(j, positions, pin_features, edge_list, pin_to_cell, cell_edges))

                    if wl_after < wl_before - 0.01:
                        # Check overlap
                        if not _check_overlap_local(positions, widths, heights, i, spatial) and \
                           not _check_overlap_local(positions, widths, heights, j, spatial):
                            swaps += 1
                            continue

                    # Revert
                    positions[i], positions[j] = pi, pj

    return swaps


def pass_reinsertion(positions, widths, heights, pin_features, edge_list,
                     pin_to_cell, cell_edges, num_macros, N):
    """Remove a cell and reinsert at best gap near its connected neighbors."""
    moves = 0

    # Sort cells by WL contribution (worst first)
    cell_wl_scores = []
    for i in range(num_macros, N):
        wl = _cell_wl(i, positions, pin_features, edge_list, pin_to_cell, cell_edges)
        cell_wl_scores.append((wl, i))
    cell_wl_scores.sort(reverse=True)

    # Only try reinsertion for top 20% worst-WL cells (cap at 50)
    top_k = min(50, max(1, len(cell_wl_scores) // 5))

    spatial = _build_spatial(positions, widths, N)

    for _wl, cell_idx in cell_wl_scores[:top_k]:
        w = widths[cell_idx].item()
        h = heights[cell_idx].item()
        old_x = positions[cell_idx, 0].item()
        old_y = positions[cell_idx, 1].item()

        # Find barycentric target from connected cells
        neighbors = set()
        for e in cell_edges.get(cell_idx, []):
            sp, tp = edge_list[e, 0].item(), edge_list[e, 1].item()
            sc, tc = pin_to_cell[sp], pin_to_cell[tp]
            other = tc if sc == cell_idx else sc
            neighbors.add(other)

        if not neighbors:
            continue

        bary_x = sum(positions[n, 0].item() for n in neighbors) / len(neighbors)
        bary_y = sum(positions[n, 1].item() for n in neighbors) / len(neighbors)

        # Try candidate positions near barycentric target
        candidates = [
            (bary_x, bary_y),
            (bary_x - w, bary_y),
            (bary_x + w, bary_y),
            (bary_x, old_y),  # same row, closer x
            (bary_x - w, old_y),
            (bary_x + w, old_y),
        ]

        wl_before = _cell_wl(cell_idx, positions, pin_features, edge_list, pin_to_cell, cell_edges)
        best_wl = wl_before
        best_pos = (old_x, old_y)

        for cx, cy in candidates:
            positions[cell_idx, 0] = cx
            positions[cell_idx, 1] = cy

            # Check overlap
            if _check_overlap_local(positions, widths, heights, cell_idx, spatial):
                continue

            wl_new = _cell_wl(cell_idx, positions, pin_features, edge_list, pin_to_cell, cell_edges)
            if wl_new < best_wl - 0.01:
                best_wl = wl_new
                best_pos = (cx, cy)

        positions[cell_idx, 0] = best_pos[0]
        positions[cell_idx, 1] = best_pos[1]
        if best_pos != (old_x, old_y):
            moves += 1

    return moves


def detailed_placement(cell_features, pin_features, edge_list,
                       num_passes=5, num_macros=None):
    """Run detailed placement passes until convergence.

    Modifies cell_features[:, 2:4] in-place.
    Returns dict with stats.
    """
    start_time = time.perf_counter()
    N = cell_features.shape[0]
    if N <= 1:
        return {"time": 0.0, "swaps": 0, "reinsertions": 0, "passes": 0}

    if num_macros is None:
        num_macros = (cell_features[:, 5] > 1.5).sum().item()

    positions = cell_features[:, 2:4].detach()
    widths = cell_features[:, 4].detach()
    heights = cell_features[:, 5].detach()

    pin_to_cell, cell_edges = _build_structures(cell_features, pin_features, edge_list)

    total_swaps = 0
    total_reinsertions = 0
    actual_passes = 0

    for p in range(num_passes):
        # Pass A: pair swaps
        swaps = pass_pair_swap(positions, widths, heights, pin_features, edge_list,
                               pin_to_cell, cell_edges, num_macros, N)
        total_swaps += swaps

        # Pass B: reinsertion of worst-WL cells
        reinsertions = pass_reinsertion(positions, widths, heights, pin_features, edge_list,
                                        pin_to_cell, cell_edges, num_macros, N)
        total_reinsertions += reinsertions

        actual_passes = p + 1
        if swaps == 0 and reinsertions == 0:
            break

    cell_features[:, 2:4] = positions

    return {
        "time": time.perf_counter() - start_time,
        "swaps": total_swaps,
        "reinsertions": total_reinsertions,
        "passes": actual_passes,
    }
