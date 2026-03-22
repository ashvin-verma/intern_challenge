"""Greedy overlap repair pass.

After gradient descent converges with residual overlaps, this pass
nudges overlapping pairs apart by the minimum amount needed.
Non-differentiable — operates on detached positions as post-processing.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from ashvin.overlap import compute_overlap_for_pairs, generate_candidate_pairs


def _brute_force_overlapping_pairs(positions, widths, heights, N):
    """O(N²) exact overlap check. Only use when N is small or conflicts are rare."""
    pairs = []
    for i in range(N):
        xi, yi = positions[i, 0].item(), positions[i, 1].item()
        wi, hi = widths[i].item(), heights[i].item()
        for j in range(i + 1, N):
            dx = abs(xi - positions[j, 0].item())
            dy = abs(yi - positions[j, 1].item())
            if dx < (wi + widths[j].item()) / 2 and dy < (hi + heights[j].item()) / 2:
                pairs.append((i, j))
    return pairs


def repair_overlaps(
    cell_features,
    num_macros=None,
    max_iterations=100,
    epsilon=0.01,
    freeze_macros=True,
    bin_size=3.0,
):
    """Greedy repair: nudge overlapping pairs apart.

    Modifies cell_features[:, 2:4] in-place.

    Args:
        cell_features: [N, 6] tensor
        num_macros: number of macros (inferred if None)
        max_iterations: max repair iterations
        epsilon: extra push beyond exact overlap resolution
        freeze_macros: if True, don't move macros
        bin_size: spatial hash bin size

    Returns:
        dict with repair stats
    """
    start_time = time.perf_counter()

    N = cell_features.shape[0]
    if N <= 1:
        return {"iterations": 0, "overlaps_before": 0, "overlaps_after": 0, "time": 0.0}

    if num_macros is None:
        num_macros = (cell_features[:, 5] > 1.5).sum().item()

    positions = cell_features[:, 2:4].detach()
    widths = cell_features[:, 4].detach()
    heights = cell_features[:, 5].detach()

    # Count initial overlaps
    pairs = generate_candidate_pairs(positions, widths, heights, num_macros, bin_size)
    if pairs.shape[0] > 0:
        areas = compute_overlap_for_pairs(positions, widths, heights, pairs)
        overlaps_before = (areas > 0).sum().item()
    else:
        overlaps_before = 0

    if overlaps_before == 0:
        return {
            "iterations": 0,
            "overlaps_before": 0,
            "overlaps_after": 0,
            "time": time.perf_counter() - start_time,
        }

    iteration = 0
    for iteration in range(max_iterations):
        # Find current overlapping pairs — use spatial hash first
        pairs = generate_candidate_pairs(positions, widths, heights, num_macros, bin_size)

        if pairs.shape[0] > 0:
            areas = compute_overlap_for_pairs(positions, widths, heights, pairs)
            overlap_mask = areas > 0
            num_overlaps = overlap_mask.sum().item()
        else:
            num_overlaps = 0

        # If spatial hash finds few/no overlaps, do brute-force exact check
        # to catch bin-boundary edge cases (affordable when conflicts are rare)
        if num_overlaps < max(N // 1000, 3) and N <= 2500:
            all_pairs = _brute_force_overlapping_pairs(positions, widths, heights, N)
            if len(all_pairs) > 0:
                pairs = torch.tensor(all_pairs, dtype=torch.long, device=positions.device)
                areas = compute_overlap_for_pairs(positions, widths, heights, pairs)
                overlap_mask = areas > 0
                num_overlaps = overlap_mask.sum().item()
            else:
                break  # truly zero overlaps

        if num_overlaps == 0:
            break

        overlapping_pairs = pairs[overlap_mask]
        made_progress = False

        for k in range(overlapping_pairs.shape[0]):
            i = overlapping_pairs[k, 0].item()
            j = overlapping_pairs[k, 1].item()

            # Read current positions (may have changed from earlier nudges this iteration)
            xi, yi = positions[i, 0].item(), positions[i, 1].item()
            xj, yj = positions[j, 0].item(), positions[j, 1].item()
            wi, hi = widths[i].item(), heights[i].item()
            wj, hj = widths[j].item(), heights[j].item()

            dx = xi - xj
            dy = yi - yj
            adx = abs(dx)
            ady = abs(dy)

            overlap_x = (wi + wj) / 2 - adx
            overlap_y = (hi + hj) / 2 - ady

            if overlap_x <= 0 or overlap_y <= 0:
                continue  # no longer overlapping

            # Determine which cells can move
            i_frozen = freeze_macros and i < num_macros
            j_frozen = freeze_macros and j < num_macros
            if i_frozen and j_frozen:
                continue  # both macros frozen, can't repair

            # Push apart along axis with less overlap (easier to resolve)
            if overlap_x <= overlap_y:
                shift = overlap_x / 2 + epsilon
                sign_d = 1.0 if dx >= 0 else -1.0
                if dx == 0:
                    sign_d = 1.0  # arbitrary direction
                if not i_frozen and not j_frozen:
                    positions[i, 0] += sign_d * shift
                    positions[j, 0] -= sign_d * shift
                elif i_frozen:
                    positions[j, 0] -= sign_d * (overlap_x + epsilon)
                else:
                    positions[i, 0] += sign_d * (overlap_x + epsilon)
            else:
                shift = overlap_y / 2 + epsilon
                sign_d = 1.0 if dy >= 0 else -1.0
                if dy == 0:
                    sign_d = 1.0
                if not i_frozen and not j_frozen:
                    positions[i, 1] += sign_d * shift
                    positions[j, 1] -= sign_d * shift
                elif i_frozen:
                    positions[j, 1] -= sign_d * (overlap_y + epsilon)
                else:
                    positions[i, 1] += sign_d * (overlap_y + epsilon)

            made_progress = True

        if not made_progress:
            break  # no pairs could be nudged — truly stuck

    # Count final overlaps — brute-force for accuracy
    final_pairs = _brute_force_overlapping_pairs(positions, widths, heights, N) if N <= 2500 else []
    if not final_pairs:
        # Fallback to spatial hash for very large N
        pairs = generate_candidate_pairs(positions, widths, heights, num_macros, bin_size)
        if pairs.shape[0] > 0:
            areas = compute_overlap_for_pairs(positions, widths, heights, pairs)
            overlaps_after = (areas > 0).sum().item()
        else:
            overlaps_after = 0
    else:
        overlaps_after = len(final_pairs)

    # Write back to cell_features
    cell_features[:, 2:4] = positions

    return {
        "iterations": iteration + 1,
        "overlaps_before": overlaps_before,
        "overlaps_after": overlaps_after,
        "time": time.perf_counter() - start_time,
    }
