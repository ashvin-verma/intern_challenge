"""Post-legalization wirelength optimization.

1. Gradient WL polish: GD on wirelength → re-legalize cycles
2. Cell swap: swap nearby same-size cells if WL improves, O(N) per pass
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.optim as optim

from placement import wirelength_attraction_loss


def _compute_edge_wl(positions, pin_features, edge_list):
    """Compute per-edge Manhattan distance. Returns [E] tensor."""
    pin_to_cell = pin_features[:, 0].long()
    pin_abs_x = positions[pin_to_cell, 0] + pin_features[:, 1]
    pin_abs_y = positions[pin_to_cell, 1] + pin_features[:, 2]

    src = edge_list[:, 0].long()
    tgt = edge_list[:, 1].long()
    dx = torch.abs(pin_abs_x[src] - pin_abs_x[tgt])
    dy = torch.abs(pin_abs_y[src] - pin_abs_y[tgt])
    return dx + dy


def _cell_wl_contribution(cell_idx, positions, pin_features, edge_list, pin_to_cell, cell_to_edges):
    """Total WL of all edges touching a cell."""
    total = 0.0
    for e_idx in cell_to_edges[cell_idx]:
        src_pin = edge_list[e_idx, 0].item()
        tgt_pin = edge_list[e_idx, 1].item()
        src_cell = pin_to_cell[src_pin].item()
        tgt_cell = pin_to_cell[tgt_pin].item()
        dx = abs(positions[src_cell, 0].item() + pin_features[src_pin, 1].item()
                 - positions[tgt_cell, 0].item() - pin_features[tgt_pin, 1].item())
        dy = abs(positions[src_cell, 1].item() + pin_features[src_pin, 2].item()
                 - positions[tgt_cell, 1].item() - pin_features[tgt_pin, 2].item())
        total += dx + dy
    return total


def _build_cell_to_edges(pin_features, edge_list, N):
    """Map each cell to its edge indices."""
    pin_to_cell = pin_features[:, 0].long()
    cell_to_edges = defaultdict(list)
    for e_idx in range(edge_list.shape[0]):
        src_cell = pin_to_cell[edge_list[e_idx, 0].item()].item()
        tgt_cell = pin_to_cell[edge_list[e_idx, 1].item()].item()
        cell_to_edges[src_cell].append(e_idx)
        if tgt_cell != src_cell:
            cell_to_edges[tgt_cell].append(e_idx)
    return cell_to_edges, pin_to_cell


def _check_overlap_pair(pos_i, w_i, h_i, pos_j, w_j, h_j):
    """Check if two cells overlap."""
    dx = abs(pos_i[0] - pos_j[0])
    dy = abs(pos_i[1] - pos_j[1])
    return dx < (w_i + w_j) / 2 and dy < (h_i + h_j) / 2


def cell_swap_optimization(
    cell_features, pin_features, edge_list,
    num_passes=5,
    num_macros=None,
):
    """Swap nearby same-height cells if it improves WL without creating overlap.

    Uses spatial hash for fast overlap checking (O(1) per swap instead of O(N)).

    Returns dict with stats.
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

    cell_to_edges, pin_to_cell = _build_cell_to_edges(pin_features, edge_list, N)

    total_swaps = 0

    for pass_num in range(num_passes):
        # Build spatial index for overlap checking
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

        def get_nearby(cell_idx):
            bx, by = cell_to_bin[cell_idx]
            nearby = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nearby.extend(bin_to_cells.get((bx + dx, by + dy), []))
            return nearby

        def check_overlap_fast(cell_idx):
            """Check overlap using spatial hash — O(neighbors) not O(N)."""
            x = positions[cell_idx, 0].item()
            y = positions[cell_idx, 1].item()
            w = widths[cell_idx].item()
            h = heights[cell_idx].item()
            for j in get_nearby(cell_idx):
                if j == cell_idx:
                    continue
                if abs(x - positions[j, 0].item()) < (w + widths[j].item()) / 2 and \
                   abs(y - positions[j, 1].item()) < (h + heights[j].item()) / 2:
                    return True
            return False

        swaps_this_pass = 0

        # Only try swaps between std cells in same/adjacent bins
        for (bx, by), cells in bin_to_cells.items():
            std_cells = [c for c in cells if c >= num_macros]
            # Neighbor bins (forward only to avoid double-checking)
            for nbx, nby in [(bx, by), (bx + 1, by), (bx, by + 1)]:
                nb_std = [c for c in bin_to_cells.get((nbx, nby), []) if c >= num_macros]
                if nbx == bx and nby == by:
                    nb_std = std_cells  # same bin

                for i in std_cells:
                    hi = heights[i].item()
                    for j in nb_std:
                        if j <= i:
                            continue
                        if abs(hi - heights[j].item()) > 0.01:
                            continue

                        # Compute WL before
                        wl_before = (_cell_wl_contribution(i, positions, pin_features, edge_list, pin_to_cell, cell_to_edges) +
                                     _cell_wl_contribution(j, positions, pin_features, edge_list, pin_to_cell, cell_to_edges))

                        # Swap
                        pos_i = positions[i].clone()
                        pos_j = positions[j].clone()
                        positions[i] = pos_j
                        positions[j] = pos_i

                        wl_after = (_cell_wl_contribution(i, positions, pin_features, edge_list, pin_to_cell, cell_to_edges) +
                                    _cell_wl_contribution(j, positions, pin_features, edge_list, pin_to_cell, cell_to_edges))

                        if wl_after < wl_before * 0.99:
                            # Fast overlap check using spatial hash
                            if not check_overlap_fast(i) and not check_overlap_fast(j):
                                swaps_this_pass += 1
                            else:
                                positions[i] = pos_i
                                positions[j] = pos_j
                        else:
                            positions[i] = pos_i
                            positions[j] = pos_j

        total_swaps += swaps_this_pass
        if swaps_this_pass == 0:
            break

    cell_features[:, 2:4] = positions
    return {
        "time": time.perf_counter() - start_time,
        "swaps": total_swaps,
        "passes": pass_num + 1,
    }


def gradient_wl_polish(
    cell_features, pin_features, edge_list,
    epochs=200, lr=0.005,
):
    """Run gradient descent on wirelength only, then re-legalize.

    Returns dict with stats.
    """
    from ashvin.legalize import legalize
    from ashvin.repair import repair_overlaps

    start_time = time.perf_counter()

    num_macros = (cell_features[:, 5] > 1.5).sum().item()
    N = cell_features.shape[0]

    if N > 10000:
        epochs = 50
    elif N > 2000:
        epochs = 100

    initial_wl = wirelength_attraction_loss(cell_features, pin_features, edge_list).item()

    for cycle in range(3):
        pos = cell_features[:, 2:4].clone().detach()
        std_pos = pos[num_macros:].clone().detach()
        std_pos.requires_grad_(True)
        macro_pos = pos[:num_macros].detach()

        optimizer = optim.Adam([std_pos], lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            full_pos = torch.cat([macro_pos, std_pos], dim=0)
            cf_cur = cell_features.clone()
            cf_cur[:, 2:4] = full_pos
            wl_loss = wirelength_attraction_loss(cf_cur, pin_features, edge_list)
            wl_loss.backward()
            torch.nn.utils.clip_grad_norm_([std_pos], max_norm=2.0)
            optimizer.step()

        cell_features[:, 2:4] = torch.cat([macro_pos, std_pos.detach()], dim=0)
        legalize(cell_features, num_macros=num_macros)
        repair_overlaps(cell_features, num_macros=num_macros, max_iterations=50)

        lr *= 0.5

    final_wl = wirelength_attraction_loss(cell_features, pin_features, edge_list).item()

    return {
        "time": time.perf_counter() - start_time,
        "wl_before": initial_wl,
        "wl_after": final_wl,
        "improvement": (initial_wl - final_wl) / initial_wl if initial_wl > 0 else 0,
    }
