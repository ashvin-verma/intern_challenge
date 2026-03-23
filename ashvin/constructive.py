"""Constructive placement: island clustering → coarse placement.

Build placement bottom-up:
1. Form islands: cluster connected cells into small legal blocks (3-8 cells)
2. Promote islands to macro-like units (bounding box = island size)
3. Coarse placement: GD on island-macros with WL + overlap loss

No legalization needed — islands are internally legal by construction.
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.optim as optim

from placement import wirelength_attraction_loss
from ashvin.overlap import scalable_overlap_loss, _pair_cache
from ashvin.density import density_loss


def _build_adjacency(pin_features, edge_list):
    """cell → set of connected cells."""
    pin_to_cell = pin_features[:, 0].long().tolist()
    neighbors = defaultdict(set)
    for e in range(edge_list.shape[0]):
        sc = pin_to_cell[edge_list[e, 0].item()]
        tc = pin_to_cell[edge_list[e, 1].item()]
        if sc != tc:
            neighbors[sc].add(tc)
            neighbors[tc].add(sc)
    return neighbors


def form_islands(cell_features, pin_features, edge_list, num_macros,
                 max_island_size=6, max_island_width=8.0):
    """Cluster connected std cells into small legal islands.

    Greedy: pick highest-degree unassigned cell, grow island by adding
    its most-connected unassigned neighbor until size limit.

    Returns list of islands, each a list of cell indices.
    Macros are each their own island.
    """
    N = cell_features.shape[0]
    widths = cell_features[:, 4].detach()
    neighbors = _build_adjacency(pin_features, edge_list)

    # Macros are singleton islands
    islands = [[i] for i in range(num_macros)]
    assigned = set(range(num_macros))

    # Sort std cells by degree (most connected first — they anchor islands)
    std_cells = list(range(num_macros, N))
    std_cells.sort(key=lambda c: len(neighbors.get(c, set())), reverse=True)

    for seed_cell in std_cells:
        if seed_cell in assigned:
            continue

        island = [seed_cell]
        assigned.add(seed_cell)
        island_width = widths[seed_cell].item()

        # Grow island greedily
        while len(island) < max_island_size:
            # Find best unassigned neighbor of any island member
            best_neighbor = None
            best_connections = 0  # connections to island members
            for member in island:
                for nb in neighbors.get(member, set()):
                    if nb in assigned or nb < num_macros:
                        continue
                    nb_w = widths[nb].item()
                    if island_width + nb_w > max_island_width:
                        continue
                    # Count connections to island members
                    conn = sum(1 for m in island if nb in neighbors.get(m, set()))
                    if conn > best_connections:
                        best_connections = conn
                        best_neighbor = nb

            if best_neighbor is None:
                break

            island.append(best_neighbor)
            assigned.add(best_neighbor)
            island_width += widths[best_neighbor].item()

        islands.append(island)

    # Assign remaining singletons
    for c in range(num_macros, N):
        if c not in assigned:
            islands.append([c])

    return islands


def pack_island(cell_features, island_cells):
    """Pack cells within an island into a single-row legal block.

    Returns (island_width, island_height, cell_offsets).
    cell_offsets: list of (dx, dy) relative to island center.
    """
    widths = cell_features[:, 4].detach()
    heights = cell_features[:, 5].detach()

    if len(island_cells) == 1:
        c = island_cells[0]
        return widths[c].item(), heights[c].item(), [(0.0, 0.0)]

    # Pack left-to-right in a single row
    total_w = sum(widths[c].item() for c in island_cells)
    max_h = max(heights[c].item() for c in island_cells)

    offsets = []
    cursor = -total_w / 2
    for c in island_cells:
        w = widths[c].item()
        dx = cursor + w / 2
        dy = 0.0
        offsets.append((dx, dy))
        cursor += w

    return total_w, max_h, offsets


def build_island_features(cell_features, pin_features, edge_list, islands, island_packing):
    """Build cell_features and pin_features for the island-level problem.

    Each island becomes a single "cell" with:
    - Position = island centroid
    - Width/height = island bounding box
    - Pins remapped to island-level
    """
    N_islands = len(islands)
    pin_to_cell = pin_features[:, 0].long().tolist()
    positions = cell_features[:, 2:4].detach()

    # Cell-to-island mapping
    cell_to_island = {}
    for isl_idx, cells in enumerate(islands):
        for c in cells:
            cell_to_island[c] = isl_idx

    # Island features: [area, num_pins, x, y, width, height]
    island_cf = torch.zeros(N_islands, 6)
    for isl_idx, cells in enumerate(islands):
        isl_w, isl_h, _ = island_packing[isl_idx]
        # Centroid from member cells' current positions
        cx = sum(positions[c, 0].item() for c in cells) / len(cells)
        cy = sum(positions[c, 1].item() for c in cells) / len(cells)
        island_cf[isl_idx, 0] = isl_w * isl_h  # area
        island_cf[isl_idx, 1] = sum(cell_features[c, 1].item() for c in cells)  # pins
        island_cf[isl_idx, 2] = cx
        island_cf[isl_idx, 3] = cy
        island_cf[isl_idx, 4] = isl_w
        island_cf[isl_idx, 5] = isl_h

    # Remap pins: pin's cell_idx → island_idx, pin offset adjusted
    P = pin_features.shape[0]
    island_pf = pin_features.clone()
    for p in range(P):
        old_cell = pin_to_cell[p]
        isl_idx = cell_to_island[old_cell]
        island_pf[p, 0] = isl_idx

        # Find cell's offset within island
        cells = islands[isl_idx]
        _, _, offsets = island_packing[isl_idx]
        cell_pos_in_island = cells.index(old_cell) if old_cell in cells else 0
        if cell_pos_in_island < len(offsets):
            dx, dy = offsets[cell_pos_in_island]
        else:
            dx, dy = 0.0, 0.0

        # Pin position = island_center + cell_offset_in_island + pin_offset_in_cell
        island_pf[p, 1] = pin_features[p, 1].item() + dx
        island_pf[p, 2] = pin_features[p, 2].item() + dy

    return island_cf, island_pf


def coarse_place(island_cf, island_pf, edge_list,
                 epochs=800, lr=0.01, lambda_wl=3.0,
                 lambda_overlap_start=10.0, lambda_overlap_end=300.0,
                 lambda_density=3.0,
                 beta_start=0.1, beta_end=4.0, verbose=False):
    """GD placement on island-level features.

    Islands are big — treat them like macros. Need strong overlap penalty
    and many epochs to spread them apart properly.
    """
    N = island_cf.shape[0]

    # Inflate islands slightly so GD pushes them further apart
    island_cf_gd = island_cf.clone()
    island_cf_gd[:, 4] *= 1.1
    island_cf_gd[:, 5] *= 1.1

    pos = island_cf_gd[:, 2:4].clone().detach()
    pos.requires_grad_(True)

    optimizer = optim.Adam([pos], lr=lr)
    # Warmup then cosine
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=50)
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs - 50, 1))
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[50])

    _pair_cache["pairs"] = None
    _pair_cache["call_count"] = 0

    for epoch in range(epochs):
        optimizer.zero_grad()
        cf_cur = island_cf_gd.clone()
        cf_cur[:, 2:4] = pos

        progress = epoch / max(epochs - 1, 1)
        beta = beta_start + (beta_end - beta_start) * progress
        lam_ov = lambda_overlap_start + (lambda_overlap_end - lambda_overlap_start) * progress

        wl_loss = wirelength_attraction_loss(cf_cur, island_pf, edge_list)
        ov_loss = scalable_overlap_loss(cf_cur, beta=beta)
        d_loss = density_loss(cf_cur) if lambda_density > 0 else torch.tensor(0.0)

        total = lambda_wl * wl_loss + lam_ov * ov_loss + lambda_density * d_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_([pos], max_norm=5.0)
        optimizer.step()
        scheduler.step()

        if verbose and (epoch % 200 == 0 or epoch == epochs - 1):
            print(f"    Coarse epoch {epoch}/{epochs}: wl={wl_loss.item():.4f} "
                  f"ov={ov_loss.item():.4f} beta={beta:.2f} lam_ov={lam_ov:.0f}")

    island_cf[:, 2:4] = pos.detach()
    return island_cf


def uncluster(cell_features, islands, island_packing, island_cf):
    """Expand island positions back to individual cell positions."""
    positions = cell_features[:, 2:4].detach()
    for isl_idx, cells in enumerate(islands):
        cx = island_cf[isl_idx, 2].item()
        cy = island_cf[isl_idx, 3].item()
        _, _, offsets = island_packing[isl_idx]
        for k, c in enumerate(cells):
            if k < len(offsets):
                dx, dy = offsets[k]
            else:
                dx, dy = 0.0, 0.0
            positions[c, 0] = cx + dx
            positions[c, 1] = cy + dy
    cell_features[:, 2:4] = positions


def island_init(cell_features, pin_features, edge_list, config=None, verbose=False):
    """Create island-clustered initial positions for cell_features.

    Forms islands, packs internally, coarse-places islands, unclusters.
    Modifies cell_features[:, 2:4] in-place with connectivity-aware positions.
    """
    N = cell_features.shape[0]
    num_macros = (cell_features[:, 5] > 1.5).sum().item()

    max_island = config.get("max_island_size", 6) if config else 6
    islands = form_islands(cell_features, pin_features, edge_list,
                           num_macros, max_island_size=max_island)

    if verbose:
        sizes = [len(isl) for isl in islands]
        print(f"  Island init: {len(islands)} islands (sizes {min(sizes)}-{max(sizes)})")

    island_packing = [pack_island(cell_features, isl) for isl in islands]

    island_cf, island_pf = build_island_features(
        cell_features, pin_features, edge_list, islands, island_packing)

    coarse_epochs = config.get("coarse_epochs", 800) if config else 800
    island_cf = coarse_place(island_cf, island_pf, edge_list,
                             epochs=coarse_epochs, verbose=verbose)

    uncluster(cell_features, islands, island_packing, island_cf)
