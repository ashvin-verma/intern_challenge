"""Debug: check how clustered cells are after averaging."""
import sys, torch
from collections import Counter, defaultdict
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from placement import generate_placement_input
from ashvin.constructive_v2 import build_cell_graph

torch.manual_seed(1004)
cf, pf, el = generate_placement_input(3, 50)
N = cf.shape[0]
ta = cf[:, 0].sum().item()
sr = (ta ** 0.5) * 0.6
ang = torch.rand(N) * 2 * 3.14159
rad = torch.rand(N) * sr
cf[:, 2] = rad * torch.cos(ang)
cf[:, 3] = rad * torch.sin(ang)
nm = (cf[:, 5] > 1.5).sum().item()
pos = cf[:, 2:4].detach()
_, nbrs, _ = build_cell_graph(pf, el)

# Run averaging
for _ in range(20):
    for ci in range(nm, N):
        nb = nbrs.get(ci, {})
        if not nb:
            continue
        wx, wy, tw = 0, 0, 0
        for n, w in nb.items():
            wx += pos[n, 0].item() * w
            wy += pos[n, 1].item() * w
            tw += w
        if tw > 0:
            pos[ci, 0] = 0.3 * pos[ci, 0].item() + 0.7 * wx / tw
            pos[ci, 1] = 0.3 * pos[ci, 1].item() + 0.7 * wy / tw

xs = [pos[i, 0].item() for i in range(nm, N)]
ys = [pos[i, 1].item() for i in range(nm, N)]
print("X range: %.1f to %.1f (span=%.1f)" % (min(xs), max(xs), max(xs)-min(xs)))
print("Y range: %.1f to %.1f (span=%.1f)" % (min(ys), max(ys), max(ys)-min(ys)))

rows = Counter(round(y) for y in ys)
print("Unique rows: %d for %d cells" % (len(rows), N - nm))
print("Row sizes (top 10):", sorted(rows.values(), reverse=True)[:10])

# Check pairwise distances
close_pairs = 0
for i in range(nm, N):
    for j in range(i+1, N):
        dx = abs(pos[i,0].item() - pos[j,0].item())
        dy = abs(pos[i,1].item() - pos[j,1].item())
        dist = (dx*dx + dy*dy) ** 0.5
        if dist < 5.0:
            close_pairs += 1
print("Pairs within 5.0 units: %d" % close_pairs)
print("Total std pairs: %d" % ((N-nm) * (N-nm-1) // 2))
