"""A/B test: WL-priority legalization vs greedy row-packing."""

import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from placement import calculate_normalized_metrics, generate_placement_input, wirelength_attraction_loss
from ashvin.solver import solve as annealed_solve
from ashvin.overlap import scalable_overlap_loss, _pair_cache
from ashvin.legalize import legalize as legalize_greedy
from ashvin.wl_legalize import wl_priority_legalize
from ashvin.repair import repair_overlaps

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


def run_test(test_id, nm, nsc, seed, config):
    """Run GD, then compare two legalization strategies."""
    torch.manual_seed(seed)
    cf, pf, el = generate_placement_input(nm, nsc)
    N = cf.shape[0]
    total_area = cf[:, 0].sum().item()
    sr = (total_area ** 0.5) * 0.6
    angles = torch.rand(N) * 2 * 3.14159
    radii = torch.rand(N) * sr
    cf[:, 2] = radii * torch.cos(angles)
    cf[:, 3] = radii * torch.sin(angles)

    # Run GD only (no legalization or post-processing)
    import torch.optim as optim
    pos = cf[:, 2:4].clone().detach()
    pos.requires_grad_(True)
    optimizer = optim.Adam([pos], lr=config.get("lr", 0.003))

    epochs = config.get("epochs", 500)
    _pair_cache["pairs"] = None
    _pair_cache["call_count"] = 0

    lambda_wl = config.get("lambda_wl", 3.58)
    lambda_overlap_start = config.get("lambda_overlap_start", 1.23)
    lambda_overlap_end = config.get("lambda_overlap_end", 96.2)
    beta_start = config.get("beta_start", 0.11)
    beta_end = config.get("beta_end", 2.03)
    lambda_density = config.get("lambda_density", 1.64)
    from ashvin.density import density_loss

    warmup_epochs = config.get("warmup_epochs", 200)
    schedulers = []
    schedulers.append(optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=max(warmup_epochs, 1)))
    schedulers.append(optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_epochs, 1)))
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers, milestones=[warmup_epochs])

    for epoch in range(epochs):
        optimizer.zero_grad()
        cf_cur = cf.clone()
        cf_cur[:, 2:4] = pos
        progress = epoch / max(epochs - 1, 1)
        beta = beta_start + (beta_end - beta_start) * progress
        lam_ov = lambda_overlap_start + (lambda_overlap_end - lambda_overlap_start) * progress
        wl_loss = wirelength_attraction_loss(cf_cur, pf, el)
        ov_loss = scalable_overlap_loss(cf_cur, beta=beta)
        d_loss = density_loss(cf_cur) if lambda_density > 0 else torch.tensor(0.0)
        total = lambda_wl * wl_loss + lam_ov * ov_loss + lambda_density * d_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_([pos], max_norm=5.0)
        optimizer.step()
        scheduler.step()

    cf[:, 2:4] = pos.detach()

    # Measure pre-legalization WL
    m_pre = calculate_normalized_metrics(cf, pf, el)
    wl_pre = m_pre["normalized_wl"]

    results = {"test_id": test_id, "N": N, "wl_pre_legalize": wl_pre}

    # Strategy A: Greedy row-packing (existing)
    cf_a = cf.clone()
    legalize_greedy(cf_a, pin_features=pf, edge_list=el)
    repair_overlaps(cf_a, max_iterations=200)
    m_a = calculate_normalized_metrics(cf_a, pf, el)
    results["wl_greedy"] = m_a["normalized_wl"]
    results["overlap_greedy"] = m_a["overlap_ratio"]

    # Strategy B: WL-priority legalization
    cf_b = cf.clone()
    wl_priority_legalize(cf_b, pf, el, alpha=0.5, beta=2.0)
    repair_overlaps(cf_b, max_iterations=200)
    m_b = calculate_normalized_metrics(cf_b, pf, el)
    results["wl_priority"] = m_b["normalized_wl"]
    results["overlap_priority"] = m_b["overlap_ratio"]

    # Strategy C: WL-priority with higher beta (more WL-focused)
    cf_c = cf.clone()
    wl_priority_legalize(cf_c, pf, el, alpha=0.1, beta=5.0)
    repair_overlaps(cf_c, max_iterations=200)
    m_c = calculate_normalized_metrics(cf_c, pf, el)
    results["wl_priority_b5"] = m_c["normalized_wl"]
    results["overlap_priority_b5"] = m_c["overlap_ratio"]

    return results


def main():
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    test_ids = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(1, 10))
    cases = [c for c in TEST_CASES if c[0] in test_ids]

    print(f"{'Test':>4} {'N':>5} {'Pre-legal':>10} {'Greedy':>10} {'WL-prio':>10} {'WL-prio-b5':>12} {'Best':>6}")
    print("-" * 65)

    for test_id, nm, nsc, seed in cases:
        r = run_test(test_id, nm, nsc, seed, config)
        wls = [r["wl_greedy"], r["wl_priority"], r["wl_priority_b5"]]
        best = min(wls)
        best_label = ["greedy", "wl-prio", "wl-b5"][wls.index(best)]

        ov_flags = []
        for k in ["overlap_greedy", "overlap_priority", "overlap_priority_b5"]:
            ov_flags.append("OV!" if r[k] > 0 else "   ")

        print(f"{r['test_id']:>4} {r['N']:>5} {r['wl_pre_legalize']:>10.4f} "
              f"{r['wl_greedy']:>8.4f}{ov_flags[0]} "
              f"{r['wl_priority']:>8.4f}{ov_flags[1]} "
              f"{r['wl_priority_b5']:>10.4f}{ov_flags[2]} "
              f"{best_label:>6}")


if __name__ == "__main__":
    main()
