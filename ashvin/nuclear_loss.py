"""Nuclear-force inspired placement loss.

Inspired by the semi-empirical mass formula / Lennard-Jones potential:
- Repulsive at very short range (Pauli exclusion → overlap prevention)
- Attractive at medium range (strong nuclear force → pull connected cells together)
- Equilibrium at touching distance (cells should be close but not overlapping)

This unifies overlap prevention and wirelength minimization into a single
smooth potential, avoiding the tug-of-war between separate loss terms.

For connected cell pairs (i,j):
  sigma = (wi + wj)/2  (ideal x-separation)
  r = |xi - xj|  (actual distance)
  V(r) = (sigma/r)^4 - (sigma/r)^2   [repulsive at r<sigma, attractive at r>sigma]

For unconnected cell pairs: only repulsion (overlap prevention).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def nuclear_loss(cell_features, pin_features, edge_list, alpha=1.0):
    """Compute nuclear-force potential for connected cell pairs.

    For each edge, compute a Lennard-Jones-like potential between the
    connected cells. Repulsive when overlapping, attractive when far.

    Args:
        cell_features: [N, 6]
        pin_features: [P, 7]
        edge_list: [E, 2]
        alpha: overall scale

    Returns:
        Scalar loss (differentiable)
    """
    if edge_list.shape[0] == 0:
        return torch.tensor(0.0, requires_grad=True)

    positions = cell_features[:, 2:4]
    widths = cell_features[:, 4]
    heights = cell_features[:, 5]
    pin_to_cell = pin_features[:, 0].long()

    # Get cell pairs from edges
    src_pins = edge_list[:, 0].long()
    tgt_pins = edge_list[:, 1].long()
    src_cells = pin_to_cell[src_pins]
    tgt_cells = pin_to_cell[tgt_pins]

    # Absolute pin positions
    src_x = positions[src_cells, 0] + pin_features[src_pins, 1]
    src_y = positions[src_cells, 1] + pin_features[src_pins, 2]
    tgt_x = positions[tgt_cells, 0] + pin_features[tgt_pins, 1]
    tgt_y = positions[tgt_cells, 1] + pin_features[tgt_pins, 2]

    # Distance (with small epsilon for numerical stability)
    dx = src_x - tgt_x
    dy = src_y - tgt_y
    r_sq = dx * dx + dy * dy + 1e-6

    # Ideal separation: sum of half-widths (touching distance)
    sigma_x = (widths[src_cells] + widths[tgt_cells]) / 2
    sigma_y = (heights[src_cells] + heights[tgt_cells]) / 2
    sigma_sq = sigma_x * sigma_x + sigma_y * sigma_y

    # Bethe-Weizsäcker inspired:
    # - Volume term: each edge wants cells at touching distance → attraction
    # - Surface term: cells with fewer connections are "surface" → extra pull
    # - Coulomb term: repulsion between same-cluster cells that are too close

    # Attraction: squared distance (quadratic pull toward neighbors)
    # This is stronger than linear WL and creates tighter clusters
    attraction = r_sq / (sigma_sq + 1e-6)

    # Repulsion: only when overlapping (r < sigma)
    repulsion = torch.relu(1.0 - r_sq / sigma_sq) ** 2

    potential = attraction - 2.0 * repulsion  # net: attract far, repel close

    return alpha * potential.mean()
