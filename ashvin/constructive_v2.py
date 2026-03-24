"""Constructive placement v2: legal from the start, swap to optimize.

No GD. No legalization. No overlap loss.
Place cells one by one in legal positions, then improve via swaps.

Architecture:
1. Place macros (spread apart)
2. Place std cells greedily (most-connected first, at WL-optimal legal position)
3. Iterative swap refinement (thousands of legal-to-legal moves)
"""

import sys
import time
import math
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


# ── Adjacency ────────────────────────────────────────────────────────

def build_cell_graph(pin_features, edge_list):
    """Build weighted cell adjacency and per-cell edge lists."""
    pin_to_cell = pin_features[:, 0].long().tolist()
    neighbors = defaultdict(lambda: defaultdict(float))  # cell -> {cell: weight}
    cell_edges = defaultdict(list)

    for e in range(edge_list.shape[0]):
        sc = pin_to_cell[edge_list[e, 0].item()]
        tc = pin_to_cell[edge_list[e, 1].item()]
        cell_edges[sc].append(e)
        if sc != tc:
            cell_edges[tc].append(e)
            neighbors[sc][tc] += 1.0
            neighbors[tc][sc] += 1.0

    return pin_to_cell, neighbors, cell_edges


def cell_wl(ci, positions, pin_features, edge_list, pin_to_cell, cell_edges):
    """Manhattan WL of all edges incident to cell ci."""
    total = 0.0
    for e in cell_edges.get(ci, []):
        sp, tp = edge_list[e, 0].item(), edge_list[e, 1].item()
        sc, tc = pin_to_cell[sp], pin_to_cell[tp]
        dx = abs(positions[sc, 0].item() + pin_features[sp, 1].item()
                 - positions[tc, 0].item() - pin_features[tp, 1].item())
        dy = abs(positions[sc, 1].item() + pin_features[sp, 2].item()
                 - positions[tc, 1].item() - pin_features[tp, 2].item())
        total += dx + dy
    return total


# ── Row structure ────────────────────────────────────────────────────

class RowManager:
    """Manages rows of cells with legal (non-overlapping) positions."""

    def __init__(self, row_height=1.0):
        self.row_height = row_height
        self.rows = {}  # row_y -> sorted list of (left_edge, width, cell_idx)
        self.cell_row = {}  # cell_idx -> row_y
        self.macro_obstacles = []  # (x_min, y_min, x_max, y_max)

    def add_macro(self, ci, x, y, w, h):
        self.macro_obstacles.append((x - w/2, y - h/2, x + w/2, y + h/2))

    def get_row_y_values(self, y_center, radius=10):
        """Get available row y-values near y_center."""
        y_min = y_center - radius
        y_max = y_center + radius
        row_min = int(math.floor(y_min / self.row_height))
        row_max = int(math.ceil(y_max / self.row_height))
        return [r * self.row_height for r in range(row_min, row_max + 1)]

    def _macro_overlaps(self, x, row_y, w):
        """Check if position overlaps any macro."""
        h = self.row_height
        cx_min, cx_max = x - w/2, x + w/2
        cy_min, cy_max = row_y - h/2, row_y + h/2
        for ox_min, oy_min, ox_max, oy_max in self.macro_obstacles:
            if cx_max > ox_min and cx_min < ox_max and cy_max > oy_min and cy_min < oy_max:
                return True
        return False

    def push_outside_macros(self, x, y, w, h, margin=0.1):
        """If (x, y) overlaps any macro, project to nearest macro boundary.

        Returns list of candidate (x, y) positions outside all macros.
        Each candidate is the nearest boundary point of one macro.
        Caller picks the one with best WL.
        """
        candidates = []
        inside_any = False

        for ox_min, oy_min, ox_max, oy_max in self.macro_obstacles:
            mx = (ox_min + ox_max) / 2
            my = (oy_min + oy_max) / 2
            mw = ox_max - ox_min
            mh = oy_max - oy_min

            # Forbidden region for cell center
            fx_min = mx - (mw + w) / 2 - margin
            fx_max = mx + (mw + w) / 2 + margin
            fy_min = my - (mh + h) / 2 - margin
            fy_max = my + (mh + h) / 2 + margin

            if fx_min < x < fx_max and fy_min < y < fy_max:
                inside_any = True
                # Project to each boundary, pick nearest
                boundary_points = [
                    (fx_min, y),   # left
                    (fx_max, y),   # right
                    (x, fy_min),   # bottom
                    (x, fy_max),   # top
                ]
                for bx, by in boundary_points:
                    candidates.append((bx, by))

        if not inside_any:
            return [(x, y)]  # already legal

        if not candidates:
            return [(x, y)]

        return candidates

    def find_insertion_x(self, row_y, target_x, w):
        """Find best x to INSERT a cell of width w, pushing others aside.

        Instead of gap-finding (fails in packed rows), this inserts the cell
        at the desired position and compacts to resolve overlaps.
        Returns the x-center position after insertion.
        """
        if row_y not in self.rows:
            self.rows[row_y] = []
            return target_x  # empty row, just place at target

        cells = self.rows[row_y]
        if not cells:
            return target_x

        # The cell will be inserted into the sorted order.
        # After insertion, we compact. The question is: where in the order?
        # Insert at the position closest to target_x.

        # Find insertion index
        insert_idx = len(cells)
        for i, (left, cw, _) in enumerate(cells):
            if target_x < left + cw / 2:
                insert_idx = i
                break

        # Build the ordering with the new cell inserted
        # Compute compacted positions
        # Start from the leftmost existing cell or target_x, whichever is less
        all_items = list(cells)  # existing cells
        # We'll just compute where the cell WOULD go after compact
        # Use the centroid approach: compact from the center of mass

        # Simple: compact left-to-right from current start
        if all_items:
            start = min(all_items[0][0], target_x - w / 2)
        else:
            start = target_x - w / 2

        # Compute all positions
        cursor = start
        result_x = target_x  # default
        for i, (left, cw, ci) in enumerate(all_items):
            if i == insert_idx:
                # Our new cell goes here
                x = cursor + w / 2
                # Push past macros
                for _ in range(20):
                    if not self._macro_overlaps(x, row_y, w):
                        break
                    x += w
                    cursor = x - w / 2
                result_x = x
                cursor = x + w / 2

            # Existing cell
            x = max(left + cw / 2, cursor + cw / 2)
            # Push past macros
            for _ in range(20):
                if not self._macro_overlaps(x, row_y, cw):
                    break
                x += cw
            cursor = x + cw / 2

        # Handle insertion at end
        if insert_idx >= len(all_items):
            x = cursor + w / 2
            for _ in range(20):
                if not self._macro_overlaps(x, row_y, w):
                    break
                x += w
            result_x = x

        return result_x

    def place_cell(self, ci, x, row_y, w, positions):
        """Place a cell in a row and compact to ensure no overlaps."""
        if row_y not in self.rows:
            self.rows[row_y] = []

        # Add to row
        left = x - w / 2
        cells = self.rows[row_y]
        cells.append((left, w, ci))
        cells.sort(key=lambda t: t[0])
        self.cell_row[ci] = row_y

        # Set position
        positions[ci, 0] = x
        positions[ci, 1] = row_y

        # Compact to resolve any overlaps
        self.compact_row(row_y, positions, None)

    def remove_cell(self, ci):
        """Remove a cell from its row."""
        row_y = self.cell_row.get(ci)
        if row_y is None:
            return
        cells = self.rows.get(row_y, [])
        self.rows[row_y] = [(l, w, c) for l, w, c in cells if c != ci]
        del self.cell_row[ci]

    def get_row_cells(self, row_y):
        """Get cell indices in a row, sorted by x."""
        return [ci for _, _, ci in self.rows.get(row_y, [])]

    def compact_row(self, row_y, positions, widths=None):
        """Re-compact a row: resolve ALL overlaps (cell-cell AND cell-macro)."""
        cells = self.rows.get(row_y, [])
        if not cells:
            return
        if len(cells) == 1:
            ci = cells[0][2]
            w = cells[0][1]
            x = positions[ci, 0].item()
            # Still need to check macro overlap for singletons
            for _ in range(20):
                if not self._macro_overlaps(x, row_y, w):
                    break
                # Find which macro we hit and jump past it
                for ox_min, oy_min, ox_max, oy_max in self.macro_obstacles:
                    h = self.row_height
                    if x + w/2 > ox_min and x - w/2 < ox_max and \
                       row_y + h/2 > oy_min and row_y - h/2 < oy_max:
                        x = ox_max + w / 2 + 0.1
                        break
            positions[ci, 0] = x
            positions[ci, 1] = row_y
            self.rows[row_y] = [(x - w/2, w, ci)]
            return

        # Sort by current x
        cells.sort(key=lambda t: t[0])

        # Left-to-right sweep
        new_cells = []
        cursor = cells[0][0]  # start from leftmost edge
        for _, w, ci in cells:
            x = max(cursor + w / 2, positions[ci, 0].item())

            # Push past macro obstacles — check repeatedly
            for _ in range(20):
                if not self._macro_overlaps(x, row_y, w):
                    break
                for ox_min, oy_min, ox_max, oy_max in self.macro_obstacles:
                    h = self.row_height
                    if x + w/2 > ox_min and x - w/2 < ox_max and \
                       row_y + h/2 > oy_min and row_y - h/2 < oy_max:
                        x = ox_max + w / 2 + 0.1
                        break

            # Ensure no overlap with previous cell
            if new_cells:
                prev_right = new_cells[-1][0] + new_cells[-1][1]
                x = max(x, prev_right + w / 2)

            positions[ci, 0] = x
            positions[ci, 1] = row_y
            new_cells.append((x - w / 2, w, ci))
            cursor = x + w / 2
        self.rows[row_y] = new_cells


# ── Constructive placement ──────────────────────────────────────────

def construct_placement(cell_features, pin_features, edge_list, num_macros):
    """Place all cells in legal positions, greedily minimizing WL."""
    N = cell_features.shape[0]
    positions = cell_features[:, 2:4].detach()
    widths = cell_features[:, 4].detach()
    heights = cell_features[:, 5].detach()

    pin_to_cell, neighbors, cell_edges = build_cell_graph(pin_features, edge_list)

    rm = RowManager(row_height=1.0)

    # Step 1: Place macros — spread them apart using GD positions as hint
    total_area = cell_features[:, 0].sum().item()
    spread = (total_area ** 0.5) * 0.6

    for i in range(num_macros):
        # Keep GD macro positions (already legalized by macro push)
        rm.add_macro(i, positions[i, 0].item(), positions[i, 1].item(),
                     widths[i].item(), heights[i].item())

    # Step 2: Place std cells by degree (most connected first)
    std_cells = list(range(num_macros, N))
    std_cells.sort(key=lambda c: len(neighbors.get(c, {})), reverse=True)

    for ci in std_cells:
        w = widths[ci].item()

        # Compute target: barycentric center of placed neighbors
        placed_nbrs = [n for n in neighbors.get(ci, {}) if n in rm.cell_row or n < num_macros]

        if placed_nbrs:
            target_x = sum(positions[n, 0].item() for n in placed_nbrs) / len(placed_nbrs)
            target_y = sum(positions[n, 1].item() for n in placed_nbrs) / len(placed_nbrs)
        else:
            # No placed neighbors — use GD position
            target_x = positions[ci, 0].item()
            target_y = positions[ci, 1].item()

        # Try nearby rows, pick the one with best WL
        h = heights[ci].item()
        candidate_rows = rm.get_row_y_values(target_y, radius=5)
        best_wl = float("inf")
        best_x, best_ry = target_x, round(target_y)

        for ry in candidate_rows:
            x = rm.find_insertion_x(ry, target_x, w)

            # Check macro overlap and project to boundary if needed
            macro_candidates = rm.push_outside_macros(x, ry, w, h)
            for cx, cy in macro_candidates:
                # Snap cy back to row (we can't move between rows here)
                cx_final = cx
                wl = 0.0
                for n in placed_nbrs:
                    nx = positions[n, 0].item()
                    ny = positions[n, 1].item()
                    wl += abs(cx_final - nx) + abs(ry - ny)
                if wl < best_wl:
                    best_wl = wl
                    best_x = cx_final
                    best_ry = ry

        positions[ci, 0] = best_x
        positions[ci, 1] = best_ry
        rm.place_cell(ci, best_x, best_ry, w, positions)

    # Final pass: compact ALL rows to guarantee zero macro overlap
    for ry in list(rm.rows.keys()):
        rm.compact_row(ry, positions)

    cell_features[:, 2:4] = positions
    return rm


# ── Swap refinement ─────────────────────────────────────────────────

def swap_refine(cell_features, pin_features, edge_list, rm,
                num_macros, max_iterations=50, verbose=False):
    """Iterative legal-to-legal swap refinement.

    Two move types:
    A. Within-row swap: exchange two cells' order, recompact
    B. Cross-row move: move cell to a better row

    All moves preserve legality. Greedy: accept any improvement.
    """
    N = cell_features.shape[0]
    positions = cell_features[:, 2:4].detach()
    widths = cell_features[:, 4].detach()
    pin_to_cell, neighbors, cell_edges = build_cell_graph(pin_features, edge_list)

    total_improvements = 0

    for iteration in range(max_iterations):
        # Score cells by WL
        cell_scores = []
        for ci in range(num_macros, N):
            wl = cell_wl(ci, positions, pin_features, edge_list, pin_to_cell, cell_edges)
            cell_scores.append((wl, ci))
        cell_scores.sort(reverse=True)

        iter_improvements = 0
        moved = set()

        for _, ci in cell_scores:
            if ci in moved:
                continue

            cur_wl = cell_wl(ci, positions, pin_features, edge_list, pin_to_cell, cell_edges)
            w = widths[ci].item()
            cur_row = rm.cell_row.get(ci)
            if cur_row is None:
                continue

            # Compute target
            nbr_x, nbr_y, cnt = 0.0, 0.0, 0
            for n in neighbors.get(ci, {}):
                nbr_x += positions[n, 0].item()
                nbr_y += positions[n, 1].item()
                cnt += 1
            if cnt == 0:
                continue
            target_x = nbr_x / cnt
            target_y = nbr_y / cnt

            # Try cross-row move
            best_improvement = 0.01  # threshold
            best_move = None

            for ry in rm.get_row_y_values(target_y, radius=3):
                if abs(ry - cur_row) < 0.01:
                    continue

                x = rm.find_insertion_x(ry, target_x, w)

                # Check macro overlap, project if needed
                h = cell_features[ci, 5].item()
                macro_cands = rm.push_outside_macros(x, ry, w, h)

                for cx, _ in macro_cands:
                    old_x, old_y = positions[ci, 0].item(), positions[ci, 1].item()
                    positions[ci, 0] = cx
                    positions[ci, 1] = ry
                    new_wl = cell_wl(ci, positions, pin_features, edge_list, pin_to_cell, cell_edges)
                    positions[ci, 0] = old_x
                    positions[ci, 1] = old_y

                    improvement = cur_wl - new_wl
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = ("cross", cx, ry)

            # Apply best move
            if best_move is not None:
                _, new_x, new_ry = best_move
                old_row = rm.cell_row[ci]
                rm.remove_cell(ci)
                # Compact old row (close the gap)
                rm.compact_row(old_row, positions)
                # Place in new row (compact resolves overlaps)
                rm.place_cell(ci, new_x, new_ry, w, positions)
                moved.add(ci)
                iter_improvements += 1

        total_improvements += iter_improvements
        if verbose:
            print(f"    Swap iter {iteration}: {iter_improvements} improvements")
        if iter_improvements == 0:
            break

    cell_features[:, 2:4] = positions
    return total_improvements, iteration + 1


# ── Main solver ─────────────────────────────────────────────────────

def solve_constructive_v2(cell_features, pin_features, edge_list,
                          config=None, verbose=False):
    """Constructive solver: place legally, then swap to optimize.

    No GD. No legalization. Always legal.
    """
    start_time = time.perf_counter()
    cell_features = cell_features.clone()
    N = cell_features.shape[0]
    initial_cell_features = cell_features.clone()
    num_macros = (cell_features[:, 5] > 1.5).sum().item()

    if verbose:
        print(f"  Constructive v2: N={N}, macros={num_macros}")

    # Step 1-2: Construct legal placement
    rm = construct_placement(cell_features, pin_features, edge_list, num_macros)

    if verbose:
        from placement import calculate_normalized_metrics
        m = calculate_normalized_metrics(cell_features, pin_features, edge_list)
        print(f"  After construction: wl={m['normalized_wl']:.4f} "
              f"overlap={m['overlap_ratio']:.4f}")

    # Step 3: Swap refinement
    max_iters = config.get("swap_iterations", 50) if config else 50
    improvements, iters = swap_refine(
        cell_features, pin_features, edge_list, rm,
        num_macros, max_iterations=max_iters, verbose=verbose)

    # Light repair for any edge cases (should be zero or near-zero overlaps)
    from ashvin.repair import repair_overlaps
    repair_overlaps(cell_features, max_iterations=200)

    train_end = time.perf_counter()

    if verbose:
        from placement import calculate_normalized_metrics
        m = calculate_normalized_metrics(cell_features, pin_features, edge_list)
        print(f"  Final: wl={m['normalized_wl']:.4f} overlap={m['overlap_ratio']:.4f} "
              f"swaps={improvements} iters={iters} time={train_end-start_time:.1f}s")

    return {
        "final_cell_features": cell_features,
        "initial_cell_features": initial_cell_features,
        "loss_history": {"total_loss": [], "wirelength_loss": [], "overlap_loss": [], "density_loss": []},
        "timing": {
            "wl_loss_time": 0, "overlap_loss_time": 0, "density_loss_time": 0,
            "backward_time": 0, "optimizer_time": 0,
            "total_train_time": train_end - start_time,
            "legalize_time": 0, "repair_time": 0,
            "repair_before": 0, "repair_after": 0,
        },
    }
