"""Plot constructive v2 results with overlaps highlighted."""
import sys, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from placement import calculate_normalized_metrics, generate_placement_input
from ashvin.constructive_v2 import construct_placement, swap_refine, solve_constructive_v2

PLOTS = Path(__file__).resolve().parent / "plots" / "constructive_v2"
PLOTS.mkdir(parents=True, exist_ok=True)


def plot_with_overlaps(cf, pf, el, title, filepath):
    fig, ax = plt.subplots(figsize=(12, 9))
    pos = cf[:, 2:4].detach()
    w = cf[:, 4].detach()
    h = cf[:, 5].detach()
    N = cf.shape[0]
    nm = (cf[:, 5] > 1.5).sum().item()
    ptc = pf[:, 0].long()

    # Find overlapping cells
    ov_cells = set()
    for i in range(N):
        for j in range(i + 1, N):
            dx = abs(pos[i, 0].item() - pos[j, 0].item())
            dy = abs(pos[i, 1].item() - pos[j, 1].item())
            if dx < (w[i].item() + w[j].item()) / 2 - 0.01 and \
               dy < (h[i].item() + h[j].item()) / 2 - 0.01:
                ov_cells.add(i)
                ov_cells.add(j)

    # Edges
    for e in range(min(el.shape[0], 2000)):
        sp, tp = el[e, 0].item(), el[e, 1].item()
        sc, tc = ptc[sp].item(), ptc[tp].item()
        x1 = pos[sc, 0].item() + pf[sp, 1].item()
        y1 = pos[sc, 1].item() + pf[sp, 2].item()
        x2 = pos[tc, 0].item() + pf[tp, 1].item()
        y2 = pos[tc, 1].item() + pf[tp, 2].item()
        ax.plot([x1, x2], [y1, y2], color="#999", alpha=0.1, linewidth=0.3)

    # Cells
    for i in range(N):
        x, y = pos[i, 0].item(), pos[i, 1].item()
        wi, hi = w[i].item(), h[i].item()
        if i < nm:
            color, ec, lw = "#cc4444", "black", 1.0
        elif i in ov_cells:
            color, ec, lw = "#ff6666", "red", 2.0
        else:
            color, ec, lw = "#4488cc", "black", 0.3
        rect = plt.Rectangle((x - wi/2, y - hi/2), wi, hi,
                              facecolor=color, edgecolor=ec, alpha=0.6, linewidth=lw)
        ax.add_patch(rect)

    ax.set_aspect("equal")
    ax.autoscale()
    ax.grid(True, alpha=0.2)
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(filepath, dpi=120)
    plt.close()


for tid, nm, nsc, seed in [(1, 2, 20, 1001), (4, 3, 50, 1004), (8, 7, 150, 1008)]:
    torch.manual_seed(seed)
    cf, pf, el = generate_placement_input(nm, nsc)
    N = cf.shape[0]
    ta = cf[:, 0].sum().item()
    sr = (ta ** 0.5) * 0.6
    ang = torch.rand(N) * 2 * 3.14159
    rad = torch.rand(N) * sr
    cf[:, 2] = rad * torch.cos(ang)
    cf[:, 3] = rad * torch.sin(ang)

    nmac = (cf[:, 5] > 1.5).sum().item()

    # After construction only (for intermediate plot)
    cf_construct = cf.clone()
    rm = construct_placement(cf_construct, pf, el, nmac)
    m1 = calculate_normalized_metrics(cf_construct, pf, el)
    plot_with_overlaps(cf_construct, pf, el,
        f"Test {tid} - After construction (WL={m1['normalized_wl']:.4f}, OV={m1['overlap_ratio']:.4f})",
        PLOTS / f"t{tid}_1_construct.png")

    # Full pipeline
    r = solve_constructive_v2(cf, pf, el, config={"swap_iterations": 50})
    cf = r["final_cell_features"]
    m2 = calculate_normalized_metrics(cf, pf, el)
    plot_with_overlaps(cf, pf, el,
        f"Test {tid} - After swaps (WL={m2['normalized_wl']:.4f}, OV={m2['overlap_ratio']:.4f})",
        PLOTS / f"t{tid}_2_swaps.png")

    # Overlap breakdown
    pos = cf[:, 2:4].detach()
    w = cf[:, 4].detach()
    h = cf[:, 5].detach()
    macro_ov = std_same = std_cross = 0
    for i in range(N):
        for j in range(i + 1, N):
            dx = abs(pos[i, 0].item() - pos[j, 0].item())
            dy = abs(pos[i, 1].item() - pos[j, 1].item())
            if dx < (w[i].item() + w[j].item()) / 2 - 0.01 and \
               dy < (h[i].item() + h[j].item()) / 2 - 0.01:
                if i < nmac or j < nmac:
                    macro_ov += 1
                elif abs(pos[i, 1].item() - pos[j, 1].item()) < 0.1:
                    std_same += 1
                else:
                    std_cross += 1
    print(f"T{tid}: WL={m2['normalized_wl']:.4f} OV={m2['overlap_ratio']:.4f} "
          f"macro_ov={macro_ov} same_row={std_same} cross_row={std_cross}")

print(f"Plots saved to {PLOTS}/")
