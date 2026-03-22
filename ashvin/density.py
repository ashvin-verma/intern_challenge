"""Differentiable density penalty via bilinear interpolation.

Penalizes bins where accumulated cell area exceeds a uniform target.
Gradients push cells from dense bins toward sparse bins.

Cost: O(N) per epoch — each cell contributes to exactly 4 bins.
"""

import torch


def density_loss(cell_features, bin_size=10.0):
    """Compute differentiable density penalty.

    Args:
        cell_features: [N, 6] tensor with [area, num_pins, x, y, width, height]
        bin_size: grid bin size (larger = smoother density field)

    Returns:
        Scalar loss (differentiable w.r.t. positions via bilinear weights)
    """
    N = cell_features.shape[0]
    if N <= 1:
        return torch.tensor(0.0, requires_grad=True, device=cell_features.device)

    positions = cell_features[:, 2:4]  # [N, 2] — has grad
    areas = cell_features[:, 0]  # [N] — fixed

    # Grid bounds (detached — grid doesn't move with cells)
    x_min = positions[:, 0].detach().min() - bin_size
    y_min = positions[:, 1].detach().min() - bin_size
    x_max = positions[:, 0].detach().max() + bin_size
    y_max = positions[:, 1].detach().max() + bin_size

    nbx = int(((x_max - x_min) / bin_size).item()) + 2
    nby = int(((y_max - y_min) / bin_size).item()) + 2
    num_bins = nbx * nby

    # Fractional bin coordinates (DIFFERENTIABLE through positions)
    fx = (positions[:, 0] - x_min) / bin_size
    fy = (positions[:, 1] - y_min) / bin_size

    # Integer bin coords (detached — index only)
    ix = fx.detach().long().clamp(0, nbx - 2)
    iy = fy.detach().long().clamp(0, nby - 2)

    # Fractional parts (differentiable!)
    dx = fx - ix.float()
    dy = fy - iy.float()

    # Bilinear weights × area (differentiable through dx, dy)
    w00 = (1 - dx) * (1 - dy) * areas
    w10 = dx * (1 - dy) * areas
    w01 = (1 - dx) * dy * areas
    w11 = dx * dy * areas

    # Flatten bin indices
    idx00 = (ix * nby + iy).clamp(0, num_bins - 1)
    idx10 = ((ix + 1) * nby + iy).clamp(0, num_bins - 1)
    idx01 = (ix * nby + (iy + 1)).clamp(0, num_bins - 1)
    idx11 = ((ix + 1) * nby + (iy + 1)).clamp(0, num_bins - 1)

    # Accumulate density (scatter_add is differentiable through src values)
    density = torch.zeros(num_bins, device=positions.device)
    density = density.scatter_add(0, idx00, w00)
    density = density.scatter_add(0, idx10, w10)
    density = density.scatter_add(0, idx01, w01)
    density = density.scatter_add(0, idx11, w11)

    # Target: uniform distribution of total area across bins
    target = areas.sum() / num_bins

    # Penalty for exceeding target
    overflow = torch.relu(density - target)
    return overflow.sum() / N
