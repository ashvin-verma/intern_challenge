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
