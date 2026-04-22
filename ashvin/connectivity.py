import torch


def build_connectivity_context(pin_features, edge_list, num_cells=None):
    """Build reusable tensor connectivity structures for WL/local-search code."""
    pin_to_cell = pin_features[:, 0].long()
    src_pin = edge_list[:, 0].long()
    tgt_pin = edge_list[:, 1].long()
    src_cell = pin_to_cell[src_pin]
    tgt_cell = pin_to_cell[tgt_pin]
    non_self = src_cell != tgt_cell

    if num_cells is None:
        if pin_to_cell.numel() == 0:
            num_cells = 0
        else:
            num_cells = int(pin_to_cell.max().item()) + 1

    edge_ids = torch.arange(edge_list.shape[0], device=edge_list.device, dtype=torch.long)
    flat_cells = torch.cat([src_cell, tgt_cell[non_self]])
    flat_edges = torch.cat([edge_ids, edge_ids[non_self]])

    if flat_cells.numel() > 0:
        edge_order = torch.argsort(flat_cells)
        sorted_cells = flat_cells[edge_order]
        cell_edge_ids = flat_edges[edge_order]
        edge_counts = torch.bincount(sorted_cells, minlength=num_cells)
    else:
        cell_edge_ids = torch.zeros(0, dtype=torch.long, device=edge_list.device)
        edge_counts = torch.zeros(num_cells, dtype=torch.long, device=edge_list.device)

    cell_edge_ptr = torch.zeros(num_cells + 1, dtype=torch.long, device=edge_list.device)
    if num_cells > 0:
        cell_edge_ptr[1:] = torch.cumsum(edge_counts, dim=0)

    if non_self.any():
        lo = torch.minimum(src_cell[non_self], tgt_cell[non_self])
        hi = torch.maximum(src_cell[non_self], tgt_cell[non_self])
        unique_pairs = torch.unique(torch.stack([lo, hi], dim=1), dim=0)
        adj_src = torch.cat([unique_pairs[:, 0], unique_pairs[:, 1]])
        adj_tgt = torch.cat([unique_pairs[:, 1], unique_pairs[:, 0]])
    else:
        adj_src = torch.zeros(0, dtype=torch.long, device=edge_list.device)
        adj_tgt = torch.zeros(0, dtype=torch.long, device=edge_list.device)

    if adj_src.numel() > 0:
        neigh_order = torch.argsort(adj_src)
        sorted_neigh_src = adj_src[neigh_order]
        cell_neighbor_ids = adj_tgt[neigh_order]
        neighbor_counts = torch.bincount(sorted_neigh_src, minlength=num_cells)
    else:
        cell_neighbor_ids = torch.zeros(0, dtype=torch.long, device=edge_list.device)
        neighbor_counts = torch.zeros(num_cells, dtype=torch.long, device=edge_list.device)

    cell_neighbor_ptr = torch.zeros(num_cells + 1, dtype=torch.long, device=edge_list.device)
    if num_cells > 0:
        cell_neighbor_ptr[1:] = torch.cumsum(neighbor_counts, dim=0)

    return {
        "pin_to_cell": pin_to_cell,
        "src_pin": src_pin,
        "tgt_pin": tgt_pin,
        "src_cell": src_cell,
        "tgt_cell": tgt_cell,
        "non_self": non_self,
        "pin_offset_x": pin_features[:, 1],
        "pin_offset_y": pin_features[:, 2],
        "cell_edge_ptr": cell_edge_ptr,
        "cell_edge_ids": cell_edge_ids,
        "cell_neighbor_ptr": cell_neighbor_ptr,
        "cell_neighbor_ids": cell_neighbor_ids,
        "adj_src": adj_src,
        "adj_tgt": adj_tgt,
    }


def get_cell_edges(cell_idx, ctx):
    start = int(ctx["cell_edge_ptr"][cell_idx].item())
    end = int(ctx["cell_edge_ptr"][cell_idx + 1].item())
    return ctx["cell_edge_ids"][start:end]


def get_cell_neighbors(cell_idx, ctx):
    start = int(ctx["cell_neighbor_ptr"][cell_idx].item())
    end = int(ctx["cell_neighbor_ptr"][cell_idx + 1].item())
    return ctx["cell_neighbor_ids"][start:end]


def collect_incident_edges(cells, ctx):
    """Deduplicate edges touching any cell in `cells`."""
    if torch.is_tensor(cells):
        cell_tensor = cells.long().flatten().unique()
    else:
        if not cells:
            return torch.zeros(0, dtype=torch.long, device=ctx["src_pin"].device)
        cell_tensor = torch.as_tensor(
            sorted(set(int(c) for c in cells)),
            dtype=torch.long,
            device=ctx["src_pin"].device,
        )

    spans = []
    ptr = ctx["cell_edge_ptr"]
    edge_ids = ctx["cell_edge_ids"]
    for cell_idx in cell_tensor.tolist():
        start = int(ptr[cell_idx].item())
        end = int(ptr[cell_idx + 1].item())
        if end > start:
            spans.append(edge_ids[start:end])

    if not spans:
        return torch.zeros(0, dtype=torch.long, device=ctx["src_pin"].device)
    if len(spans) == 1:
        return spans[0].unique()
    return torch.unique(torch.cat(spans))


def compute_edge_wl(positions, ctx):
    pin_abs_x = positions[ctx["pin_to_cell"], 0] + ctx["pin_offset_x"]
    pin_abs_y = positions[ctx["pin_to_cell"], 1] + ctx["pin_offset_y"]
    dx = torch.abs(pin_abs_x[ctx["src_pin"]] - pin_abs_x[ctx["tgt_pin"]])
    dy = torch.abs(pin_abs_y[ctx["src_pin"]] - pin_abs_y[ctx["tgt_pin"]])
    return dx + dy


def edge_wl_sum(edge_indices, positions, ctx):
    """Total Manhattan WL across a deduplicated edge set."""
    if edge_indices is None:
        return 0.0

    if torch.is_tensor(edge_indices):
        idx = edge_indices.long().flatten()
    else:
        edge_list = list(edge_indices)
        if not edge_list:
            return 0.0
        idx = torch.as_tensor(edge_list, dtype=torch.long, device=positions.device)

    if idx.numel() == 0:
        return 0.0

    src_pin = ctx["src_pin"][idx]
    tgt_pin = ctx["tgt_pin"][idx]
    src_cell = ctx["src_cell"][idx]
    tgt_cell = ctx["tgt_cell"][idx]
    dx = torch.abs(
        positions[src_cell, 0] + ctx["pin_offset_x"][src_pin]
        - positions[tgt_cell, 0] - ctx["pin_offset_x"][tgt_pin]
    )
    dy = torch.abs(
        positions[src_cell, 1] + ctx["pin_offset_y"][src_pin]
        - positions[tgt_cell, 1] - ctx["pin_offset_y"][tgt_pin]
    )
    return (dx + dy).sum().item()


def compute_cell_wl_scores(positions, ctx, num_cells):
    """Accumulate per-cell WL from vectorized per-edge distances."""
    edge_wl = compute_edge_wl(positions, ctx)
    scores = torch.zeros(num_cells, dtype=positions.dtype, device=positions.device)
    scores.index_add_(0, ctx["src_cell"], edge_wl)
    if ctx["non_self"].any():
        scores.index_add_(0, ctx["tgt_cell"][ctx["non_self"]], edge_wl[ctx["non_self"]])
    return scores


def compute_neighbor_centroids(positions, ctx, num_cells):
    """Compute connected-neighbor centroid for each cell."""
    target_x = positions[:, 0].clone()
    target_y = positions[:, 1].clone()
    degree = torch.zeros(num_cells, dtype=positions.dtype, device=positions.device)
    if ctx["adj_src"].numel() == 0:
        return target_x, target_y, degree

    sum_x = torch.zeros(num_cells, dtype=positions.dtype, device=positions.device)
    sum_y = torch.zeros(num_cells, dtype=positions.dtype, device=positions.device)
    ones = torch.ones(ctx["adj_src"].shape[0], dtype=positions.dtype, device=positions.device)
    sum_x.index_add_(0, ctx["adj_src"], positions[ctx["adj_tgt"], 0])
    sum_y.index_add_(0, ctx["adj_src"], positions[ctx["adj_tgt"], 1])
    degree.index_add_(0, ctx["adj_src"], ones)

    movable = degree > 0
    target_x[movable] = sum_x[movable] / degree[movable]
    target_y[movable] = sum_y[movable] / degree[movable]
    return target_x, target_y, degree
