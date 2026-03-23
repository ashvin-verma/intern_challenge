"""Quick A/B test: with vs without row snapping, on tests 1-5."""

import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from placement import calculate_normalized_metrics, generate_placement_input
from ashvin.solver import solve as annealed_solve

CONFIG_PATH = Path(__file__).resolve().parent / "results" / "best_config.json"

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


def run_one(test_id, nm, nsc, seed, config, label):
    torch.manual_seed(seed)
    cf, pf, el = generate_placement_input(nm, nsc)
    N = cf.shape[0]
    total_area = cf[:, 0].sum().item()
    spread_radius = (total_area ** 0.5) * 0.6
    angles = torch.rand(N) * 2 * 3.14159
    radii = torch.rand(N) * spread_radius
    cf[:, 2] = radii * torch.cos(angles)
    cf[:, 3] = radii * torch.sin(angles)

    t0 = time.perf_counter()
    result = annealed_solve(cf, pf, el, config=config, verbose=False)
    t1 = time.perf_counter()

    m = calculate_normalized_metrics(result["final_cell_features"], pf, el)
    return {
        "test_id": test_id, "N": N, "label": label,
        "wl": m["normalized_wl"], "overlap": m["overlap_ratio"],
        "time": t1 - t0,
    }


def main():
    with open(CONFIG_PATH) as f:
        base_config = json.load(f)

    test_ids = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(1, 10))
    cases = [c for c in TEST_CASES if c[0] in test_ids]

    configs = [
        ("baseline (no snap, no gs)", {**base_config, "_skip_row_snap": True, "_skip_global_swap": True}),
        ("+ row snap only", {**base_config, "_skip_global_swap": True}),
        ("+ global swap only", {**base_config, "_skip_row_snap": True}),
        ("+ both", dict(base_config)),
    ]

    print(f"{'Test':>4} {'N':>5}", end="")
    for label, _ in configs:
        print(f" {label[:20]:>22}", end="")
    print()
    print("-" * (10 + 22 * len(configs)))

    avg_wls = {label: [] for label, _ in configs}

    for test_id, nm, nsc, seed in cases:
        print(f"{test_id:>4} {nm+nsc:>5}", end="", flush=True)
        for label, cfg in configs:
            r = run_one(test_id, nm, nsc, seed, cfg, label)
            ov_flag = " OV!" if r["overlap"] > 0 else ""
            print(f" {r['wl']:>10.4f} ({r['time']:>5.1f}s){ov_flag}", end="", flush=True)
            avg_wls[label].append(r["wl"])
        print()

    print("-" * (10 + 22 * len(configs)))
    print(f"{'AVG':>4} {'':>5}", end="")
    for label, _ in configs:
        avg = sum(avg_wls[label]) / len(avg_wls[label])
        print(f" {avg:>10.4f}{'':>8}", end="")
    print()


if __name__ == "__main__":
    main()
