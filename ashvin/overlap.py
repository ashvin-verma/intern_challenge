"""Scalable overlap engine using two-tier spatial hashing.

Tier 1: Macro pairs (exhaustive) — O(M*N) where M is small (~10)
Tier 2: StdCell-StdCell pairs (spatial hash) — O(N) average
"""

from collections import defaultdict

import torch


def _soft_positive(x, beta=None):
    """Smooth positive-part activation. ReLU when beta=None, softplus when beta>0."""
    if beta is None or beta <= 0:
        return torch.relu(x)
    return torch.nn.functional.softplus(x, beta=beta)


def compute_overlap_for_pairs(positions, widths, heights, pairs, beta=None):
    """Compute overlap area for candidate pairs. Differentiable.

    Args:
        positions: [N, 2] cell positions (must have grad if used in loss)
        widths: [N] cell widths
        heights: [N] cell heights
        pairs: [P, 2] int64 candidate pair indices
        beta: softplus beta (None = ReLU, >0 = softplus smoothing)

    Returns:
        [P] tensor of overlap areas (differentiable)
    """
    if pairs.shape[0] == 0:
        return torch.zeros(0, device=positions.device)

    i_idx = pairs[:, 0]
    j_idx = pairs[:, 1]

    dx = torch.abs(positions[i_idx, 0] - positions[j_idx, 0])
    dy = torch.abs(positions[i_idx, 1] - positions[j_idx, 1])

    min_sep_x = (widths[i_idx] + widths[j_idx]) / 2
    min_sep_y = (heights[i_idx] + heights[j_idx]) / 2

    overlap_x = _soft_positive(min_sep_x - dx, beta)
    overlap_y = _soft_positive(min_sep_y - dy, beta)

    return overlap_x * overlap_y


def _generate_macro_pairs(positions, widths, heights, num_macros):
    """Generate candidate pairs involving at least one macro.

    Macro-macro: all C(M,2) pairs.
    Macro-stdcell: vectorized distance filter per macro.

    Returns [P, 2] int64 tensor.
    """
    N = positions.shape[0]
    pair_list = []

    # Macro-macro: all pairs (at most C(10,2) = 45)
    for i in range(num_macros):
        for j in range(i + 1, num_macros):
            pair_list.append((i, j))

    # Macro-stdcell: vectorized filter per macro
    if num_macros < N:
        std_pos = positions[num_macros:].detach()
        std_w = widths[num_macros:]
        std_h = heights[num_macros:]

        for m in range(num_macros):
            mx = positions[m, 0].detach()
            my = positions[m, 1].detach()
            mw = widths[m]
            mh = heights[m]

            dx = torch.abs(std_pos[:, 0] - mx)
            dy = torch.abs(std_pos[:, 1] - my)
            max_dx = (mw + std_w) / 2
            max_dy = (mh + std_h) / 2

            mask = (dx < max_dx) & (dy < max_dy)
            std_indices = torch.where(mask)[0] + num_macros

            for s in std_indices.tolist():
                pair_list.append((m, s))

    if not pair_list:
        return torch.zeros((0, 2), dtype=torch.long, device=positions.device)

    return torch.tensor(pair_list, dtype=torch.long, device=positions.device)


def _generate_stdcell_pairs(positions, widths, heights, num_macros, bin_size):
    """Generate candidate pairs among std cells using GPU-friendly spatial hashing.

    Uses sort-based binning with torch ops — no Python loops over cells.

    Returns [P, 2] int64 tensor with global indices.
    """
    N = positions.shape[0]
    if num_macros >= N:
        return torch.zeros((0, 2), dtype=torch.long, device=positions.device)

    std_pos = positions[num_macros:].detach()
    num_std = std_pos.shape[0]

    if num_std <= 1:
        return torch.zeros((0, 2), dtype=torch.long, device=positions.device)

    dev = positions.device
    x = std_pos[:, 0]
    y = std_pos[:, 1]

    x_min = x.min() - bin_size
    y_min = y.min() - bin_size

    bx = ((x - x_min) / bin_size).long()
    by = ((y - y_min) / bin_size).long()

    # Encode bin as single int for sorting: bx * large_prime + by
    bx_range = (bx.max() - bx.min() + 3).item()
    bin_key = bx * max(bx_range, 1) + by

    # Sort cells by bin key
    sort_order = torch.argsort(bin_key)
    sorted_keys = bin_key[sort_order]
    sorted_global_idx = sort_order + num_macros  # global cell indices

    # Find bin boundaries using torch.unique
    unique_keys, counts = torch.unique_consecutive(sorted_keys, return_counts=True)
    offsets = torch.zeros(len(counts) + 1, dtype=torch.long, device=dev)
    torch.cumsum(counts, dim=0, out=offsets[1:])

    # For each bin, generate within-bin pairs (i < j)
    pair_chunks = []

    # Forward neighbor offsets (bin-key deltas)
    # (0,0) = 0, (1,0) = bx_range, (0,1) = 1, (1,1) = bx_range+1, (-1,1) = -bx_range+1
    bx_range_int = max(int(bx_range), 1)
    neighbor_deltas = [0, bx_range_int, 1, bx_range_int + 1, -bx_range_int + 1]

    # Build key-to-offset lookup
    key_to_offset = {}
    for b in range(len(unique_keys)):
        key_to_offset[unique_keys[b].item()] = (offsets[b].item(), offsets[b + 1].item())

    for b in range(len(unique_keys)):
        key_val = unique_keys[b].item()
        start_a, end_a = offsets[b].item(), offsets[b + 1].item()
        cells_a = sorted_global_idx[start_a:end_a]

        for delta in neighbor_deltas:
            nb_key = key_val + delta
            lookup = key_to_offset.get(nb_key)
            if lookup is None:
                continue
            start_b, end_b = lookup

            if delta == 0:
                # Same bin: generate i < j pairs
                n = end_a - start_a
                if n >= 2:
                    idx = torch.arange(n, device=dev)
                    ii, jj = torch.meshgrid(idx, idx, indexing="ij")
                    mask = ii < jj
                    pairs_local = torch.stack([cells_a[ii[mask]], cells_a[jj[mask]]], dim=1)
                    pair_chunks.append(pairs_local)
            else:
                # Cross-bin: all pairs
                cells_b = sorted_global_idx[start_b:end_b]
                na, nb = len(cells_a), len(cells_b)
                if na > 0 and nb > 0:
                    aa = cells_a.unsqueeze(1).expand(na, nb).reshape(-1)
                    bb = cells_b.unsqueeze(0).expand(na, nb).reshape(-1)
                    # Ensure i < j
                    lo = torch.min(aa, bb)
                    hi = torch.max(aa, bb)
                    pairs_local = torch.stack([lo, hi], dim=1)
                    pair_chunks.append(pairs_local)

    if not pair_chunks:
        return torch.zeros((0, 2), dtype=torch.long, device=dev)

    pairs = torch.cat(pair_chunks, dim=0)
    pairs = torch.unique(pairs, dim=0)
    return pairs


def generate_candidate_pairs(positions, widths, heights, num_macros, bin_size=3.0):
    """Generate all candidate overlapping pairs using two-tier approach.

    Returns [P, 2] int64 tensor of candidate pairs (i < j).
    """
    macro_pairs = _generate_macro_pairs(positions, widths, heights, num_macros)
    std_pairs = _generate_stdcell_pairs(positions, widths, heights, num_macros, bin_size)

    if macro_pairs.shape[0] == 0 and std_pairs.shape[0] == 0:
        return torch.zeros((0, 2), dtype=torch.long, device=positions.device)
    if macro_pairs.shape[0] == 0:
        return std_pairs
    if std_pairs.shape[0] == 0:
        return macro_pairs

    return torch.cat([macro_pairs, std_pairs], dim=0)


# Cache for pair lists — avoids rebuilding every epoch
_pair_cache = {"pairs": None, "call_count": 0, "N": 0}


def scalable_overlap_loss(
    cell_features, num_macros=None, bin_size=3.0, rebuild_interval=50, beta=None
):
    """Scalable overlap loss using spatial hashing. Differentiable.

    Candidate pairs are cached and rebuilt every `rebuild_interval` calls.

    Args:
        cell_features: [N, 6] tensor
        num_macros: number of macro cells (inferred if None)
        bin_size: spatial hash bin size for std cells
        rebuild_interval: rebuild pair list every N calls
        beta: softplus beta for annealed overlap (None = ReLU)

    Returns:
        Scalar loss tensor (differentiable)
    """
    N = cell_features.shape[0]
    if N <= 1:
        return torch.tensor(0.0, requires_grad=True, device=cell_features.device)

    if num_macros is None:
        num_macros = (cell_features[:, 5] > 1.5).sum().item()

    positions = cell_features[:, 2:4]
    widths = cell_features[:, 4]
    heights = cell_features[:, 5]

    # Rebuild pairs periodically or on first call / size change
    cache = _pair_cache
    need_rebuild = (
        cache["pairs"] is None
        or cache["N"] != N
        or cache["call_count"] % rebuild_interval == 0
    )

    if need_rebuild:
        pairs = generate_candidate_pairs(
            positions, widths, heights, num_macros, bin_size
        )
        cache["pairs"] = pairs
        cache["N"] = N

    cache["call_count"] += 1
    pairs = cache["pairs"]

    if pairs.shape[0] == 0:
        return torch.tensor(0.0, requires_grad=True, device=cell_features.device)

    overlap_areas = compute_overlap_for_pairs(positions, widths, heights, pairs, beta=beta)
    return overlap_areas.sum() / N


def scalable_cells_with_overlaps(cell_features, num_macros=None, bin_size=3.0):
    """Scalable evaluation: find cells involved in overlaps.

    Non-differentiable. Returns set of cell indices.
    """
    N = cell_features.shape[0]
    if N <= 1:
        return set()

    if num_macros is None:
        num_macros = (cell_features[:, 5] > 1.5).sum().item()

    positions = cell_features[:, 2:4].detach()
    widths = cell_features[:, 4].detach()
    heights = cell_features[:, 5].detach()

    pairs = generate_candidate_pairs(
        positions, widths, heights, num_macros, bin_size
    )

    if pairs.shape[0] == 0:
        return set()

    # Compute overlaps for candidate pairs
    overlap_areas = compute_overlap_for_pairs(positions, widths, heights, pairs)

    # Find pairs with actual overlap
    has_overlap = overlap_areas > 0
    overlapping_pairs = pairs[has_overlap]

    cells = set()
    for i, j in overlapping_pairs.tolist():
        cells.add(i)
        cells.add(j)

    return cells


def scalable_overlap_metrics(cell_features, num_macros=None, bin_size=3.0):
    """Scalable evaluation: overlap statistics.

    Non-differentiable. Returns same format as calculate_overlap_metrics().
    """
    N = cell_features.shape[0]
    if N <= 1:
        return {
            "overlap_count": 0,
            "total_overlap_area": 0.0,
            "max_overlap_area": 0.0,
            "overlap_percentage": 0.0,
        }

    if num_macros is None:
        num_macros = (cell_features[:, 5] > 1.5).sum().item()

    positions = cell_features[:, 2:4].detach()
    widths = cell_features[:, 4].detach()
    heights = cell_features[:, 5].detach()

    pairs = generate_candidate_pairs(
        positions, widths, heights, num_macros, bin_size
    )

    if pairs.shape[0] == 0:
        return {
            "overlap_count": 0,
            "total_overlap_area": 0.0,
            "max_overlap_area": 0.0,
            "overlap_percentage": 0.0,
        }

    overlap_areas = compute_overlap_for_pairs(positions, widths, heights, pairs)

    has_overlap = overlap_areas > 0
    overlap_count = has_overlap.sum().item()
    overlapping_areas = overlap_areas[has_overlap]

    total_overlap_area = overlapping_areas.sum().item() if overlap_count > 0 else 0.0
    max_overlap_area = overlapping_areas.max().item() if overlap_count > 0 else 0.0
    overlap_percentage = (overlap_count / N * 100) if N > 0 else 0.0

    return {
        "overlap_count": overlap_count,
        "total_overlap_area": total_overlap_area,
        "max_overlap_area": max_overlap_area,
        "overlap_percentage": overlap_percentage,
    }
