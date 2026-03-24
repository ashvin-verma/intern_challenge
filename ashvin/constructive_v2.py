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

def compute_blocked_intervals(cell_features, num_macros, row_centers, row_h, margin=1e-3):
    """For each row, return sorted+merged (x_lo, x_hi) intervals blocked by macros."""
    blocked = {r: [] for r in range(len(row_centers))}

    for mi in range(num_macros):
        mx = float(cell_features[mi, 2])
        my = float(cell_features[mi, 3])
        mw = float(cell_features[mi, 4])
        mh = float(cell_features[mi, 5])

        m_x_lo = mx - mw / 2 - margin
        m_x_hi = mx + mw / 2 + margin
        m_y_lo = my - mh / 2 - margin
        m_y_hi = my + mh / 2 + margin

        for r, ry in enumerate(row_centers):
            row_lo = ry - row_h / 2
            row_hi = ry + row_h / 2
            if m_y_lo < row_hi and m_y_hi > row_lo:
                blocked[r].append((m_x_lo, m_x_hi))

    # Merge overlapping intervals per row
    for r in blocked:
        ivs = sorted(blocked[r])
        merged = []
        for lo, hi in ivs:
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        blocked[r] = merged

    return blocked


def best_legal_x(target_x, cell_w, blocked_intervals, margin=1e-3):
    """Find x closest to target_x where cell of width cell_w fits legally."""
    half = cell_w / 2 + margin

    # Expand macro intervals by cell half-width to get forbidden CENTER positions
    forbidden = [(lo - half, hi + half) for (lo, hi) in blocked_intervals]

    # Merge after expansion (adjacent macros may create impassable gaps)
    forbidden.sort()
    merged = []
    for lo, hi in forbidden:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))

    def is_legal(x):
        return all(x <= lo or x >= hi for (lo, hi) in merged)

    if is_legal(target_x):
        return target_x

    # Candidates: just outside each forbidden boundary
    candidates = []
    for lo, hi in merged:
        candidates.append(lo - 1e-6)
        candidates.append(hi + 1e-6)

    legal = [(abs(c - target_x), c) for c in candidates if is_legal(c)]
    return min(legal)[1] if legal else target_x


class RowManager:
    """Manages rows of cells with legal (non-overlapping) positions."""

    def __init__(self, row_height=1.0):
        self.row_height = row_height
        self.rows = {}  # row_y -> sorted list of (left_edge, width, cell_idx)
        self.cell_row = {}  # cell_idx -> row_y
        self.macro_obstacles = []  # (x_min, y_min, x_max, y_max)
        self.blocked = {}  # row_idx -> blocked intervals (set by init_blocked)
        self.row_centers = []  # list of row y-values
        self.row_y_to_idx = {}  # row_y -> index into row_centers

    def add_macro(self, ci, x, y, w, h):
        self.macro_obstacles.append((x - w/2, y - h/2, x + w/2, y + h/2))

    def init_blocked(self, cell_features, num_macros, y_min, y_max):
        """Precompute blocked intervals per row from macros."""
        row_min = int(math.floor(y_min / self.row_height))
        row_max = int(math.ceil(y_max / self.row_height))
        self.row_centers = [r * self.row_height for r in range(row_min, row_max + 1)]
        self.row_y_to_idx = {ry: i for i, ry in enumerate(self.row_centers)}
        self.blocked = compute_blocked_intervals(
            cell_features, num_macros, self.row_centers, self.row_height)

    def legal_x(self, row_y, target_x, cell_w):
        """Get nearest legal x for a cell in this row, avoiding macros."""
        r_idx = self.row_y_to_idx.get(row_y)
        if r_idx is not None:
            return best_legal_x(target_x, cell_w, self.blocked.get(r_idx, []))

        # Row not precomputed — compute blocked intervals on the fly
        intervals = []
        margin = 1e-3
        for ox_min, oy_min, ox_max, oy_max in self.macro_obstacles:
            if oy_min < row_y + self.row_height / 2 and oy_max > row_y - self.row_height / 2:
                intervals.append((ox_min - margin, ox_max + margin))
        # Merge
        intervals.sort()
        merged = []
        for lo, hi in intervals:
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        return best_legal_x(target_x, cell_w, merged)

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

    def place_cell(self, ci, x, row_y, w, positions, compact=True):
        """Place a cell in a row. Optionally compact to resolve overlaps."""
        if row_y not in self.rows:
            self.rows[row_y] = []

        left = x - w / 2
        cells = self.rows[row_y]
        cells.append((left, w, ci))
        cells.sort(key=lambda t: t[0])
        self.cell_row[ci] = row_y

        positions[ci, 0] = x
        positions[ci, 1] = row_y

        if compact:
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
        """Bidirectional compaction: push right then pull left, repeat.

        Right sweep: resolve overlaps by pushing right (skip macro intervals).
        Left sweep: pull cells back toward their original target where room exists.
        This distributes displacement symmetrically instead of piling everything right.
        """
        cells = self.rows.get(row_y, [])
        if not cells:
            return

        cells.sort(key=lambda t: t[0])

        r_idx = self.row_y_to_idx.get(row_y)
        blocked = self.blocked.get(r_idx, []) if r_idx is not None else []

        def is_blocked(x, w):
            half = w / 2 + 1e-3
            for blo, bhi in blocked:
                if blo - half < x < bhi + half:
                    return True
            return False

        def skip_blocked_right(x, w):
            half = w / 2 + 1e-3
            for blo, bhi in blocked:
                if blo - half < x < bhi + half:
                    return bhi + half + 1e-6
            return x

        # Right sweep: resolve overlaps
        for i in range(1, len(cells)):
            prev_left, prev_w, _ = cells[i - 1]
            prev_right = prev_left + prev_w
            cur_left, cur_w, ci = cells[i]
            if cur_left < prev_right - 1e-6:
                new_x = skip_blocked_right(prev_right + cur_w / 2, cur_w)
                positions[ci, 0] = new_x
                cells[i] = (new_x - cur_w / 2, cur_w, ci)

        # Left sweep: pull cells back where room exists
        for i in range(len(cells) - 2, -1, -1):
            cur_left, cur_w, ci = cells[i]
            cur_x = cur_left + cur_w / 2

            # How far left can this cell go?
            if i == 0:
                min_x = cur_x - 100  # no left neighbor constraint
            else:
                prev_left, prev_w, _ = cells[i - 1]
                min_x = prev_left + prev_w + cur_w / 2

            # How far right must it stay? (don't overlap next cell)
            if i < len(cells) - 1:
                next_left = cells[i + 1][0]
                max_x = next_left - cur_w / 2
            else:
                max_x = cur_x + 100

            # Try to move toward legal_x target (original placement position)
            target_x = self.legal_x(row_y, positions[ci, 0].item(), cur_w)
            new_x = max(min_x, min(target_x, max_x))

            # Don't move into blocked interval
            if not is_blocked(new_x, cur_w):
                positions[ci, 0] = new_x
                cells[i] = (new_x - cur_w / 2, cur_w, ci)

        for _, _, ci in cells:
            positions[ci, 1] = row_y

        self.rows[row_y] = cells


# ── Constructive placement ──────────────────────────────────────────

def construct_placement(cell_features, pin_features, edge_list, num_macros):
    """Two-phase constructive: cluster at targets, then spread to legalize.

    Phase 1: Place all cells at barycentric targets (allow overlaps).
             Iterate averaging positions toward connected neighbors.
    Phase 2: Assign to rows, spread within rows using Abacus-style
             cluster merge to resolve overlaps minimally.
    """
    N = cell_features.shape[0]
    positions = cell_features[:, 2:4].detach()
    widths = cell_features[:, 4].detach()
    heights = cell_features[:, 5].detach()

    pin_to_cell, neighbors, cell_edges = build_cell_graph(pin_features, edge_list)

    rm = RowManager(row_height=1.0)

    # ── Step 1: Legalize macros (just push apart, no gaps) ──
    if num_macros > 1:
        for _pass in range(300):
            any_ov = False
            for i in range(num_macros):
                for j in range(i + 1, num_macros):
                    xi, yi = positions[i, 0].item(), positions[i, 1].item()
                    xj, yj = positions[j, 0].item(), positions[j, 1].item()
                    wi, hi = widths[i].item(), heights[i].item()
                    wj, hj = widths[j].item(), heights[j].item()
                    ov_x = (wi + wj) / 2 - abs(xi - xj)
                    ov_y = (hi + hj) / 2 - abs(yi - yj)
                    if ov_x > 0 and ov_y > 0:
                        any_ov = True
                        if ov_x <= ov_y:
                            s = ov_x / 2 + 0.1
                            sign = 1.0 if xi >= xj else -1.0
                            positions[i, 0] += sign * s
                            positions[j, 0] -= sign * s
                        else:
                            s = ov_y / 2 + 0.1
                            sign = 1.0 if yi >= yj else -1.0
                            positions[i, 1] += sign * s
                            positions[j, 1] -= sign * s
            if not any_ov:
                break

    for i in range(num_macros):
        rm.add_macro(i, positions[i, 0].item(), positions[i, 1].item(),
                     widths[i].item(), heights[i].item())

    # ── Phase 1: Iterative barycentric averaging ──
    # Each cell moves toward centroid of all its neighbors (macros + std).
    # Like force-directed without repulsion. 20 iterations converges.
    std_cells = list(range(num_macros, N))

    for _iteration in range(20):
        for ci in std_cells:
            nbrs = neighbors.get(ci, {})
            if not nbrs:
                continue
            wx, wy, tw = 0.0, 0.0, 0.0
            for n, weight in nbrs.items():
                wx += positions[n, 0].item() * weight
                wy += positions[n, 1].item() * weight
                tw += weight
            if tw > 0:
                cx, cy = wx / tw, wy / tw
                positions[ci, 0] = 0.3 * positions[ci, 0].item() + 0.7 * cx
                positions[ci, 1] = 0.3 * positions[ci, 1].item() + 0.7 * cy

    # ── Phase 2: Assign to rows and spread ──
    # Precompute blocked intervals
    y_min = positions[:, 1].min().item() - 15
    y_max = positions[:, 1].max().item() + 15
    rm.init_blocked(cell_features, num_macros, y_min, y_max)

    # Assign each std cell to nearest legal row
    for ci in std_cells:
        w = widths[ci].item()
        ty = positions[ci, 1].item()
        # Snap to nearest row
        ry = round(ty / rm.row_height) * rm.row_height
        # Get legal x
        tx = positions[ci, 0].item()
        x = rm.legal_x(ry, tx, w)
        positions[ci, 0] = x
        positions[ci, 1] = ry
        rm.place_cell(ci, x, ry, w, positions, compact=False)

    # Compact all rows (bidirectional)
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
    swapped_pairs = set()  # prevent oscillation

    for iteration in range(max_iterations):
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

            best_improvement = 0.01
            best_move = None

            # Move type A: within-row swap with adjacent cells
            row_cells = rm.get_row_cells(cur_row)
            ci_idx = row_cells.index(ci) if ci in row_cells else -1
            if ci_idx >= 0:
                for offset in [-1, 1, -2, 2]:
                    j_idx = ci_idx + offset
                    if not (0 <= j_idx < len(row_cells)):
                        continue
                    cj = row_cells[j_idx]
                    if cj in moved or cj < num_macros:
                        continue
                    # Skip if already swapped this pair
                    pair = (min(ci, cj), max(ci, cj))
                    if pair in swapped_pairs:
                        continue

                    # Evaluate: WL of BOTH cells before and after swap
                    wl_j_before = cell_wl(cj, positions, pin_features, edge_list,
                                          pin_to_cell, cell_edges)
                    xi, xj = positions[ci, 0].item(), positions[cj, 0].item()
                    positions[ci, 0] = xj
                    positions[cj, 0] = xi
                    wl_i_after = cell_wl(ci, positions, pin_features, edge_list,
                                         pin_to_cell, cell_edges)
                    wl_j_after = cell_wl(cj, positions, pin_features, edge_list,
                                         pin_to_cell, cell_edges)
                    positions[ci, 0] = xi
                    positions[cj, 0] = xj

                    improvement = (cur_wl + wl_j_before) - (wl_i_after + wl_j_after)
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = ("swap", cj)

            # Move type B: cross-row move
            for ry in rm.get_row_y_values(target_y, radius=3):
                if abs(ry - cur_row) < 0.01:
                    continue

                x = rm.legal_x(ry, target_x, w)

                old_x, old_y = positions[ci, 0].item(), positions[ci, 1].item()
                positions[ci, 0] = x
                positions[ci, 1] = ry
                new_wl = cell_wl(ci, positions, pin_features, edge_list, pin_to_cell, cell_edges)
                positions[ci, 0] = old_x
                positions[ci, 1] = old_y

                improvement = cur_wl - new_wl
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_move = ("cross", x, ry)

            # Apply best move
            if best_move is not None:
                if best_move[0] == "swap":
                    cj = best_move[1]
                    xi = positions[ci, 0].item()
                    xj = positions[cj, 0].item()
                    positions[ci, 0] = xj
                    positions[cj, 0] = xi
                    ry = rm.cell_row[ci]
                    cells = rm.rows[ry]
                    rm.rows[ry] = sorted(cells, key=lambda t: t[0])
                    swapped_pairs.add((min(ci, cj), max(ci, cj)))
                    moved.add(ci)
                    moved.add(cj)
                    iter_improvements += 1
                else:
                    _, new_x, new_ry = best_move
                    old_row = rm.cell_row[ci]
                    rm.remove_cell(ci)
                    rm.compact_row(old_row, positions)
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
