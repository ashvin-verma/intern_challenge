"""Compare Phase 1 variants: averaging only, BFS only, BFS+averaging.
Plot the PHASE 1 output (before spreading) to see if they're different local minima."""
import sys, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque, defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from placement import calculate_normalized_metrics, generate_placement_input

PLOTS = Path(__file__).resolve().parent / "plots" / "phase1_compare"
PLOTS.mkdir(parents=True, exist_ok=True)


def build_cell_graph(pf, el):
    ptc = pf[:, 0].long().tolist()
    nbrs = defaultdict(lambda: defaultdict(float))
    for e in range(el.shape[0]):
        sc = ptc[el[e, 0].item()]
        tc = ptc[el[e, 1].item()]
        if sc != tc:
            nbrs[sc][tc] += 1.0
            nbrs[tc][sc] += 1.0
    return nbrs


def phase1_averaging(cf, pf, el, nm):
    pos = cf[:, 2:4].detach()
    N = cf.shape[0]
    nbrs = build_cell_graph(pf, el)
    for _ in range(20):
        for ci in range(nm, N):
            nb = nbrs.get(ci, {})
            if not nb:
                continue
            wx, wy, tw = 0., 0., 0.
            for n, w in nb.items():
                wx += pos[n, 0].item() * w
                wy += pos[n, 1].item() * w
                tw += w
            if tw > 0:
                pos[ci, 0] = 0.3 * pos[ci, 0].item() + 0.7 * wx / tw
                pos[ci, 1] = 0.3 * pos[ci, 1].item() + 0.7 * wy / tw


def phase1_bfs(cf, pf, el, nm):
    pos = cf[:, 2:4].detach()
    N = cf.shape[0]
    nbrs = build_cell_graph(pf, el)
    placed = set(range(nm))
    queue = deque()
    for mi in range(nm):
        for n in nbrs.get(mi, {}):
            if n >= nm:
                queue.append(n)
    visited = set(queue)
    order = []
    while queue:
        ci = queue.popleft()
        if ci in placed:
            continue
        placed.add(ci)
        order.append(ci)
        for n in nbrs.get(ci, {}):
            if n not in placed and n not in visited:
                queue.append(n)
                visited.add(n)
    for ci in range(nm, N):
        if ci not in placed:
            order.append(ci)
            placed.add(ci)
    for ci in order:
        pn = [n for n in nbrs.get(ci, {}) if n in placed or n < nm]
        if pn:
            pos[ci, 0] = sum(pos[n, 0].item() for n in pn) / len(pn)
            pos[ci, 1] = sum(pos[n, 1].item() for n in pn) / len(pn)


def phase1_bfs_then_avg(cf, pf, el, nm):
    phase1_bfs(cf, pf, el, nm)
    phase1_averaging(cf, pf, el, nm)


def plot_positions(cf, pf, el, title, filepath):
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = cf[:, 2:4].detach()
    w = cf[:, 4].detach()
    h = cf[:, 5].detach()
    N = cf.shape[0]
    nm = (cf[:, 5] > 1.5).sum().item()
    ptc = pf[:, 0].long()
    for e in range(min(el.shape[0], 2000)):
        sp, tp = el[e, 0].item(), el[e, 1].item()
        sc, tc = ptc[sp].item(), ptc[tp].item()
        x1 = pos[sc, 0].item() + pf[sp, 1].item()
        y1 = pos[sc, 1].item() + pf[sp, 2].item()
        x2 = pos[tc, 0].item() + pf[tp, 1].item()
        y2 = pos[tc, 1].item() + pf[tp, 2].item()
        ax.plot([x1, x2], [y1, y2], color="#999", alpha=0.15, linewidth=0.3)
    for i in range(N):
        x, y = pos[i, 0].item(), pos[i, 1].item()
        wi, hi = w[i].item(), h[i].item()
        color = "#cc4444" if i < nm else "#4488cc"
        alpha = 0.7 if i < nm else 0.5
        rect = plt.Rectangle((x - wi/2, y - hi/2), wi, hi, facecolor=color,
                              edgecolor="black", alpha=alpha, linewidth=0.3)
        ax.add_patch(rect)
    ax.set_aspect("equal")
    ax.autoscale()
    ax.grid(True, alpha=0.2)
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(filepath, dpi=120)
    plt.close()


for tid, nm_count, nsc, seed in [(4, 3, 50, 1004), (6, 5, 100, 1006)]:
    for name, fn in [("averaging", phase1_averaging),
                     ("bfs", phase1_bfs),
                     ("bfs+avg", phase1_bfs_then_avg)]:
        torch.manual_seed(seed)
        cf, pf, el = generate_placement_input(nm_count, nsc)
        N = cf.shape[0]
        ta = cf[:, 0].sum().item()
        sr = (ta ** 0.5) * 0.6
        ang = torch.rand(N) * 2 * 3.14159
        rad = torch.rand(N) * sr
        cf[:, 2] = rad * torch.cos(ang)
        cf[:, 3] = rad * torch.sin(ang)
        nm = (cf[:, 5] > 1.5).sum().item()

        # Legalize macros first
        pos = cf[:, 2:4].detach()
        widths = cf[:, 4].detach()
        heights = cf[:, 5].detach()
        if nm > 1:
            for _ in range(300):
                done = True
                for i in range(nm):
                    for j in range(i+1, nm):
                        ov_x = (widths[i].item()+widths[j].item())/2 - abs(pos[i,0].item()-pos[j,0].item())
                        ov_y = (heights[i].item()+heights[j].item())/2 - abs(pos[i,1].item()-pos[j,1].item())
                        if ov_x > 0 and ov_y > 0:
                            done = False
                            if ov_x <= ov_y:
                                s = ov_x/2+0.1
                                sign = 1.0 if pos[i,0].item() >= pos[j,0].item() else -1.0
                                pos[i,0] += sign*s; pos[j,0] -= sign*s
                            else:
                                s = ov_y/2+0.1
                                sign = 1.0 if pos[i,1].item() >= pos[j,1].item() else -1.0
                                pos[i,1] += sign*s; pos[j,1] -= sign*s
                if done:
                    break

        fn(cf, pf, el, nm)
        m = calculate_normalized_metrics(cf, pf, el)
        plot_positions(cf, pf, el,
            f"Test {tid} - {name} (WL={m['normalized_wl']:.4f})",
            PLOTS / f"t{tid}_{name}.png")
        print(f"T{tid} {name}: WL={m['normalized_wl']:.4f}")

print(f"Plots saved to {PLOTS}/")
