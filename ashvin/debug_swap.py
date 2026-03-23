"""Debug: why does global swap find 0 swaps?"""
import json, sys, time
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from placement import calculate_normalized_metrics, generate_placement_input
from ashvin.solver import solve as annealed_solve
from ashvin.global_swap import (_build_structures, _cell_wl, _build_spatial,
                                 _find_cells_near, _check_overlap, _pos_to_bin)

CONFIG_PATH = Path(__file__).resolve().parent / "results" / "best_config.json"

with open(CONFIG_PATH) as f:
    config = json.load(f)
config["_skip_global_swap"] = True

torch.manual_seed(1001)
cell_features, pin_features, edge_list = generate_placement_input(2, 20)
N = cell_features.shape[0]
total_area = cell_features[:, 0].sum().item()
spread_radius = (total_area ** 0.5) * 0.6
angles = torch.rand(N) * 2 * 3.14159
radii = torch.rand(N) * spread_radius
cell_features[:, 2] = radii * torch.cos(angles)
cell_features[:, 3] = radii * torch.sin(angles)

result = annealed_solve(cell_features, pin_features, edge_list, config=config, verbose=False)
cell_features = result["final_cell_features"]

positions = cell_features[:, 2:4].detach()
widths = cell_features[:, 4].detach()
heights = cell_features[:, 5].detach()
num_macros = (cell_features[:, 5] > 1.5).sum().item()

pin_to_cell, cell_edges = _build_structures(cell_features, pin_features, edge_list)
spatial = _build_spatial(positions, widths, N)

print(f"N={N}, macros={num_macros}")
print(f"Width range (std cells): {widths[num_macros:].min():.3f} - {widths[num_macros:].max():.3f}")
print(f"Height range (std cells): {heights[num_macros:].min():.3f} - {heights[num_macros:].max():.3f}")
print()

# Check worst-WL cells
cell_wl_scores = []
for i in range(num_macros, N):
    wl = _cell_wl(i, positions, pin_features, edge_list, pin_to_cell, cell_edges)
    cell_wl_scores.append((wl, i))
cell_wl_scores.sort(reverse=True)

print("Top 5 worst-WL cells:")
for wl, i in cell_wl_scores[:5]:
    # Compute barycentric target
    neighbors = set()
    for e in cell_edges.get(i, []):
        sp, tp = edge_list[e, 0].item(), edge_list[e, 1].item()
        sc, tc = pin_to_cell[sp], pin_to_cell[tp]
        other = tc if sc == i else sc
        neighbors.add(other)

    if neighbors:
        target_x = sum(positions[n, 0].item() for n in neighbors) / len(neighbors)
        target_y = sum(positions[n, 1].item() for n in neighbors) / len(neighbors)
    else:
        target_x = target_y = 0

    cur_x = positions[i, 0].item()
    cur_y = positions[i, 1].item()
    dist = ((cur_x - target_x)**2 + (cur_y - target_y)**2)**0.5

    candidates = _find_cells_near(target_x, target_y, spatial, radius_bins=3)
    same_h = [j for j in candidates if j != i and j >= num_macros
              and abs(heights[j].item() - heights[i].item()) < 0.01]

    print(f"  Cell {i}: wl={wl:.2f}, pos=({cur_x:.1f},{cur_y:.1f}), "
          f"target=({target_x:.1f},{target_y:.1f}), dist={dist:.1f}, "
          f"w={widths[i].item():.2f}, neighbors={len(neighbors)}")
    print(f"    Candidates near target: {len(candidates)}, same-height: {len(same_h)}")

    # Try each candidate
    wl_before_i = _cell_wl(i, positions, pin_features, edge_list, pin_to_cell, cell_edges)
    tried = 0
    wl_improved = 0
    overlap_blocked = 0

    for j in same_h[:10]:
        tried += 1
        wl_before_j = _cell_wl(j, positions, pin_features, edge_list, pin_to_cell, cell_edges)
        wl_before = wl_before_i + wl_before_j

        # Swap
        pi = positions[i].clone()
        pj = positions[j].clone()
        positions[i] = pj
        positions[j] = pi

        # Update spatial hash for correct overlap check
        old_bin_i = spatial["cell_to_bin"][i]
        old_bin_j = spatial["cell_to_bin"][j]
        new_bin_i = _pos_to_bin(positions[i, 0].item(), positions[i, 1].item(), spatial)
        new_bin_j = _pos_to_bin(positions[j, 0].item(), positions[j, 1].item(), spatial)

        # Temporarily update spatial
        spatial["cell_to_bin"][i] = new_bin_i
        spatial["cell_to_bin"][j] = new_bin_j
        if old_bin_i != new_bin_i:
            if i in spatial["bin_to_cells"][old_bin_i]:
                spatial["bin_to_cells"][old_bin_i].remove(i)
            spatial["bin_to_cells"][new_bin_i].append(i)
        if old_bin_j != new_bin_j:
            if j in spatial["bin_to_cells"][old_bin_j]:
                spatial["bin_to_cells"][old_bin_j].remove(j)
            spatial["bin_to_cells"][new_bin_j].append(j)

        wl_after = (_cell_wl(i, positions, pin_features, edge_list, pin_to_cell, cell_edges) +
                    _cell_wl(j, positions, pin_features, edge_list, pin_to_cell, cell_edges))

        improvement = wl_before - wl_after
        if improvement > 0.01:
            wl_improved += 1
            has_overlap_i = _check_overlap(positions, widths, heights, i, spatial)
            has_overlap_j = _check_overlap(positions, widths, heights, j, spatial)
            if has_overlap_i or has_overlap_j:
                overlap_blocked += 1
                if tried <= 3:
                    print(f"      Swap {i}<->{j}: WL improved by {improvement:.3f} but OVERLAP "
                          f"(i_overlap={has_overlap_i}, j_overlap={has_overlap_j}, "
                          f"w_i={widths[i].item():.2f}, w_j={widths[j].item():.2f})")
            else:
                if tried <= 3:
                    print(f"      Swap {i}<->{j}: WL improved by {improvement:.3f} NO OVERLAP -- SHOULD ACCEPT!")

        # Revert
        positions[i] = pi
        positions[j] = pj
        spatial["cell_to_bin"][i] = old_bin_i
        spatial["cell_to_bin"][j] = old_bin_j
        if old_bin_i != new_bin_i:
            if i in spatial["bin_to_cells"][new_bin_i]:
                spatial["bin_to_cells"][new_bin_i].remove(i)
            spatial["bin_to_cells"][old_bin_i].append(i)
        if old_bin_j != new_bin_j:
            if j in spatial["bin_to_cells"][new_bin_j]:
                spatial["bin_to_cells"][new_bin_j].remove(j)
            spatial["bin_to_cells"][old_bin_j].append(j)

    print(f"    Tried: {tried}, WL-improved: {wl_improved}, overlap-blocked: {overlap_blocked}")
    print()
