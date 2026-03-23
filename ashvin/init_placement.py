"""Constructive initial placement.

Instead of random positions, place cells based on connectivity.
Cells connected by many edges should start near each other.
This gives gradient descent a much better starting point for WL.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def spectral_placement(cell_features, pin_features, edge_list):
    """Place cells using spectral (eigenvector) initial positions.

    Compute the graph Laplacian of cell connectivity,
    then use the 2nd and 3rd smallest eigenvectors as x,y coordinates.
    This minimizes squared wirelength and clusters connected cells.

    Modifies cell_features[:, 2:4] in-place.
    """
    N = cell_features.shape[0]
    if N <= 2:
        return

    pin_to_cell = pin_features[:, 0].long()

    # Build adjacency matrix (cell-level, weighted by edge count)
    adj = torch.zeros(N, N)
    for e in range(edge_list.shape[0]):
        src_cell = pin_to_cell[edge_list[e, 0].item()].item()
        tgt_cell = pin_to_cell[edge_list[e, 1].item()].item()
        if src_cell != tgt_cell:
            adj[src_cell, tgt_cell] += 1.0
            adj[tgt_cell, src_cell] += 1.0

    # Laplacian: L = D - A
    degree = adj.sum(dim=1)
    laplacian = torch.diag(degree) - adj

    # Compute eigenvectors (smallest eigenvalues after 0)
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
        # 2nd and 3rd eigenvectors give optimal 2D embedding
        x_coords = eigenvectors[:, 1]
        y_coords = eigenvectors[:, 2]
    except Exception:
        # Fallback: random placement
        return

    # Scale to appropriate spread
    total_area = cell_features[:, 0].sum().item()
    spread = (total_area ** 0.5) * 0.8

    # Normalize to [-spread, spread]
    x_range = x_coords.max() - x_coords.min()
    y_range = y_coords.max() - y_coords.min()
    if x_range > 0:
        x_coords = (x_coords - x_coords.mean()) / x_range * spread
    if y_range > 0:
        y_coords = (y_coords - y_coords.mean()) / y_range * spread

    cell_features[:, 2] = x_coords
    cell_features[:, 3] = y_coords


def force_directed_init(cell_features, pin_features, edge_list, iterations=30):
    """Place cells via iterative force-directed averaging.

    Start from random spread, then iteratively move each cell toward
    the centroid of its connected neighbors. No overlap consideration —
    just get the topology right. GD handles overlap later.

    Modifies cell_features[:, 2:4] in-place.
    """
    N = cell_features.shape[0]
    if N <= 2:
        return

    pin_to_cell = pin_features[:, 0].long()
    positions = cell_features[:, 2:4].detach()

    # Build cell adjacency: cell -> list of connected cells (with weights)
    from collections import defaultdict
    neighbors = defaultdict(lambda: defaultdict(float))
    for e in range(edge_list.shape[0]):
        sc = pin_to_cell[edge_list[e, 0].item()].item()
        tc = pin_to_cell[edge_list[e, 1].item()].item()
        if sc != tc:
            neighbors[sc][tc] += 1.0
            neighbors[tc][sc] += 1.0

    num_macros = (cell_features[:, 5] > 1.5).sum().item()

    for _it in range(iterations):
        new_pos = positions.clone()
        for i in range(num_macros, N):  # only move std cells
            nbrs = neighbors.get(i, {})
            if not nbrs:
                continue
            # Weighted centroid of neighbors
            wx, wy, total_w = 0.0, 0.0, 0.0
            for j, w in nbrs.items():
                wx += positions[j, 0].item() * w
                wy += positions[j, 1].item() * w
                total_w += w
            # Move 70% toward centroid, keep 30% current (damping)
            cx, cy = wx / total_w, wy / total_w
            new_pos[i, 0] = 0.3 * positions[i, 0].item() + 0.7 * cx
            new_pos[i, 1] = 0.3 * positions[i, 1].item() + 0.7 * cy
        positions[:] = new_pos

    cell_features[:, 2:4] = positions


def sequential_placement(cell_features, pin_features, edge_list):
    """Place cells one by one near the centroid of already-placed neighbors.

    Sort by degree (most connected first). Each cell placed at the centroid
    of its already-placed neighbors, or at a random position if no neighbors
    are placed yet.

    Modifies cell_features[:, 2:4] in-place.
    """
    N = cell_features.shape[0]
    if N <= 2:
        return

    pin_to_cell = pin_features[:, 0].long()
    positions = cell_features[:, 2:4].detach()
    total_area = cell_features[:, 0].sum().item()
    spread = (total_area ** 0.5) * 0.6

    # Build adjacency
    from collections import defaultdict
    neighbors = defaultdict(set)
    for e in range(edge_list.shape[0]):
        sc = pin_to_cell[edge_list[e, 0].item()].item()
        tc = pin_to_cell[edge_list[e, 1].item()].item()
        if sc != tc:
            neighbors[sc].add(tc)
            neighbors[tc].add(sc)

    num_macros = (cell_features[:, 5] > 1.5).sum().item()

    # Place macros first (keep their current positions)
    placed = set(range(num_macros))

    # Sort std cells by degree (most connected first — they anchor the placement)
    std_cells = list(range(num_macros, N))
    std_cells.sort(key=lambda c: len(neighbors.get(c, set())), reverse=True)

    for ci in std_cells:
        nbrs = neighbors.get(ci, set())
        placed_nbrs = nbrs & placed
        if placed_nbrs:
            # Place at centroid of placed neighbors + small jitter
            cx = sum(positions[n, 0].item() for n in placed_nbrs) / len(placed_nbrs)
            cy = sum(positions[n, 1].item() for n in placed_nbrs) / len(placed_nbrs)
            # Small jitter to avoid exact overlap
            import random
            positions[ci, 0] = cx + random.gauss(0, 0.5)
            positions[ci, 1] = cy + random.gauss(0, 0.5)
        else:
            # No placed neighbors — place randomly
            import random
            angle = random.uniform(0, 6.28)
            radius = random.uniform(0, spread)
            positions[ci, 0] = radius * __import__('math').cos(angle)
            positions[ci, 1] = radius * __import__('math').sin(angle)
        placed.add(ci)

    cell_features[:, 2:4] = positions
