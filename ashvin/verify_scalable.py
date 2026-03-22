"""Verify scalable overlap engine matches naive implementation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from ashvin.overlap import scalable_cells_with_overlaps, scalable_overlap_metrics
from placement import calculate_cells_with_overlaps, calculate_overlap_metrics, generate_placement_input

TEST_CASES = [
    (1, 2, 20, 1001),
    (2, 3, 25, 1002),
    (3, 2, 30, 1003),
    (4, 3, 50, 1004),
    (5, 4, 75, 1005),
    (6, 5, 100, 1006),
    (7, 5, 150, 1007),
    (8, 7, 150, 1008),
    (9, 8, 200, 1009),
]

all_pass = True
for test_id, nm, ns, seed in TEST_CASES:
    torch.manual_seed(seed)
    cf, pf, el = generate_placement_input(nm, ns)
    N = cf.shape[0]
    area = cf[:, 0].sum().item()
    sr = (area**0.5) * 0.6
    a = torch.rand(N) * 2 * 3.14159
    r = torch.rand(N) * sr
    cf[:, 2] = r * torch.cos(a)
    cf[:, 3] = r * torch.sin(a)

    naive_cells = calculate_cells_with_overlaps(cf)
    scale_cells = scalable_cells_with_overlaps(cf)

    naive_m = calculate_overlap_metrics(cf)
    scale_m = scalable_overlap_metrics(cf)

    cells_match = naive_cells == scale_cells
    count_match = naive_m["overlap_count"] == scale_m["overlap_count"]
    area_close = abs(naive_m["total_overlap_area"] - scale_m["total_overlap_area"]) < 0.01

    status = "PASS" if (cells_match and count_match and area_close) else "FAIL"
    if status == "FAIL":
        all_pass = False

    print(
        f"Test {test_id:2d} (N={N:4d}): {status} | "
        f"cells: {len(naive_cells):3d} vs {len(scale_cells):3d} | "
        f"pairs: {naive_m['overlap_count']:4d} vs {scale_m['overlap_count']:4d} | "
        f"area: {naive_m['total_overlap_area']:.1f} vs {scale_m['total_overlap_area']:.1f}"
    )

print()
if all_pass:
    print("ALL TESTS PASSED — scalable matches naive exactly")
else:
    print("SOME TESTS FAILED — check spatial hash logic")
