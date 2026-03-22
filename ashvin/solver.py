"""Single-stage annealed solver — competitor-inspired approach.

Annealed softplus + lambda ramp + warmup LR + repair.
All cells optimized simultaneously (no macro/std split).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.optim as optim

from ashvin.density import density_loss
from ashvin.overlap import _pair_cache, scalable_overlap_loss
from ashvin.repair import repair_overlaps
from placement import wirelength_attraction_loss


def solve(
    cell_features, pin_features, edge_list,
    epochs=2000,
    lr=0.01,
    lambda_wl=1.0,
    lambda_overlap_start=5.0,
    lambda_overlap_end=100.0,
    lambda_density=1.0,
    beta_start=0.1,
    beta_end=6.0,
    warmup_epochs=100,
    lr_schedule="warmup",  # "warmup" (warmup only), "warmup_cosine", "constant"
    repair_iterations=200,
    config=None,
    verbose=False,
):
    """Single-stage annealed solver.

    Args:
        config: dict overriding all keyword args (for optuna)
    """
    if config is not None:
        epochs = config.get("epochs", epochs)
        lr = config.get("lr", lr)
        lambda_wl = config.get("lambda_wl", lambda_wl)
        lambda_overlap_start = config.get("lambda_overlap_start", lambda_overlap_start)
        lambda_overlap_end = config.get("lambda_overlap_end", lambda_overlap_end)
        lr_schedule = config.get("lr_schedule", lr_schedule)
        lambda_density = config.get("lambda_density", lambda_density)
        beta_start = config.get("beta_start", beta_start)
        beta_end = config.get("beta_end", beta_end)
        warmup_epochs = config.get("warmup_epochs", warmup_epochs)
        repair_iterations = config.get("repair_iterations", repair_iterations)

    cell_features = cell_features.clone()
    N = cell_features.shape[0]

    initial_cell_features = cell_features.clone()

    # Adaptive epoch scaling: fewer epochs for larger designs
    # (legalization handles remaining overlaps)
    if epochs == 2000:  # only auto-scale if using default
        if N > 10000:
            epochs = 200
            warmup_epochs = min(warmup_epochs, 20)
        elif N > 2000:
            epochs = 500
            warmup_epochs = min(warmup_epochs, 50)

    pos = cell_features[:, 2:4].clone().detach()
    pos.requires_grad_(True)

    optimizer = optim.Adam([pos], lr=lr)

    # LR schedule
    schedulers = []
    if lr_schedule in ("warmup", "warmup_cosine"):
        schedulers.append(optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=max(warmup_epochs, 1)
        ))
    if lr_schedule == "warmup_cosine":
        schedulers.append(optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs - warmup_epochs, 1)
        ))
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers, milestones=[warmup_epochs]
    ) if len(schedulers) == 2 else (schedulers[0] if schedulers else None)

    _pair_cache["pairs"] = None
    _pair_cache["call_count"] = 0

    wl_time = overlap_time = density_time = backward_time = optimizer_time = 0.0
    train_start = time.perf_counter()

    for epoch in range(epochs):
        optimizer.zero_grad()

        cell_features_current = cell_features.clone()
        cell_features_current[:, 2:4] = pos

        progress = epoch / max(epochs - 1, 1)

        # Annealed beta (softplus sharpness)
        beta = beta_start + (beta_end - beta_start) * progress

        # Ramped lambda_overlap
        lam_ov = lambda_overlap_start + (lambda_overlap_end - lambda_overlap_start) * progress

        t0 = time.perf_counter()
        wl_loss = wirelength_attraction_loss(cell_features_current, pin_features, edge_list)
        t1 = time.perf_counter()
        ov_loss = scalable_overlap_loss(cell_features_current, beta=beta)
        t2 = time.perf_counter()
        d_loss = density_loss(cell_features_current) if lambda_density > 0 else torch.tensor(0.0)
        t3 = time.perf_counter()

        total_loss = lambda_wl * wl_loss + lam_ov * ov_loss + lambda_density * d_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([pos], max_norm=5.0)
        t4 = time.perf_counter()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        t5 = time.perf_counter()

        wl_time += t1 - t0
        overlap_time += t2 - t1
        density_time += t3 - t2
        backward_time += t4 - t3
        optimizer_time += t5 - t4

        if verbose and (epoch % 200 == 0 or epoch == epochs - 1):
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch}/{epochs}: total={total_loss.item():.4f} "
                  f"wl={wl_loss.item():.4f} overlap={ov_loss.item():.4f} "
                  f"beta={beta:.2f} lam_ov={lam_ov:.1f} lr={lr_now:.5f}")

    cell_features[:, 2:4] = pos.detach()

    # Iterative legalization + repair until zero overlap
    from ashvin.legalize import legalize
    legalize_time = 0.0
    repair_time = 0.0
    repair_before = 0
    repair_after = 0

    for leg_pass in range(5):  # max 5 legalize-repair cycles
        leg_stats = legalize(cell_features)
        legalize_time += leg_stats["time"]

        rep_stats = repair_overlaps(
            cell_features, max_iterations=repair_iterations
        )
        repair_time += rep_stats["time"]

        if leg_pass == 0:
            repair_before = rep_stats["overlaps_before"]
        repair_after = rep_stats["overlaps_after"]

        if repair_after == 0:
            break

    # Barycentric WL refinement (fast, always on)
    from ashvin.wl_optimize import barycentric_refinement
    bary_stats = barycentric_refinement(cell_features, pin_features, edge_list)

    train_end = time.perf_counter()

    return {
        "final_cell_features": cell_features,
        "initial_cell_features": initial_cell_features,
        "loss_history": {"total_loss": [], "wirelength_loss": [], "overlap_loss": [], "density_loss": []},
        "timing": {
            "wl_loss_time": wl_time,
            "overlap_loss_time": overlap_time,
            "density_loss_time": density_time,
            "backward_time": backward_time,
            "optimizer_time": optimizer_time,
            "total_train_time": train_end - train_start,
            "legalize_time": legalize_time,
            "repair_time": repair_time,
            "repair_before": repair_before,
            "repair_after": repair_after,
        },
    }


def solve_scatter(cell_features, pin_features, edge_list, config=None, verbose=False):
    """Explosive scatter + reconverge: escape local minima.

    1. GD for 300 epochs (converge)
    2. Scatter positions outward from centroid
    3. GD for 200 more epochs (reconverge)
    4. Try 3 scatter magnitudes, keep best
    5. Legalize + repair + barycentric
    """
    from placement import calculate_normalized_metrics

    N = cell_features.shape[0]
    best_result = None
    best_wl = float("inf")

    scatter_factors = [1.0, 1.3, 1.5, 2.0]  # 1.0 = no scatter (baseline)

    for scatter in scatter_factors:
        cf = cell_features.clone()

        # Build config for this run
        run_config = dict(config) if config else {}
        run_config["_skip_wl_polish"] = True  # barycentric is in solve() already

        if scatter == 1.0:
            # Normal run (baseline)
            result = solve(cf, pin_features, edge_list, config=run_config, verbose=False)
        else:
            # Phase 1: short GD
            phase1_config = dict(run_config)
            phase1_config["epochs"] = 300
            result = solve(cf, pin_features, edge_list, config=phase1_config, verbose=False)

            # Scatter from centroid
            pos = result["final_cell_features"][:, 2:4]
            cx = pos[:, 0].mean()
            cy = pos[:, 1].mean()
            pos[:, 0] = cx + (pos[:, 0] - cx) * scatter
            pos[:, 1] = cy + (pos[:, 1] - cy) * scatter
            cf = result["final_cell_features"].clone()
            cf[:, 2:4] = pos

            # Phase 2: reconverge
            phase2_config = dict(run_config)
            phase2_config["epochs"] = 200
            result = solve(cf, pin_features, edge_list, config=phase2_config, verbose=False)

        m = calculate_normalized_metrics(result["final_cell_features"], pin_features, edge_list)
        if verbose:
            print(f"  scatter={scatter:.1f}: overlap={m['overlap_ratio']:.4f} wl={m['normalized_wl']:.4f}")

        if m["overlap_ratio"] == 0 and m["normalized_wl"] < best_wl:
            best_wl = m["normalized_wl"]
            best_result = result

    if best_result is None:
        best_result = solve(cell_features, pin_features, edge_list, config=config, verbose=verbose)

    return best_result


def solve_multistart(cell_features, pin_features, edge_list, config=None, verbose=False):
    """Run solver with multiple initial placements, pick best WL.

    Tries: original positions (from test.py init) + spectral placement.
    Returns the result with lowest WL (that has 0 overlap).
    """
    from placement import calculate_normalized_metrics

    N = cell_features.shape[0]
    best_result = None
    best_wl = float("inf")

    inits = [("original", cell_features.clone())]

    # Add spectral init for small/medium designs
    if N <= 5000:
        from ashvin.init_placement import spectral_placement
        spectral_cf = cell_features.clone()
        spectral_placement(spectral_cf, pin_features, edge_list)
        inits.append(("spectral", spectral_cf))

    for name, cf in inits:
        if verbose:
            print(f"  Multi-start: trying {name} init...")

        # Suppress WL polish config to keep it fast, re-enable for best
        fast_config = dict(config) if config else {}
        fast_config["_skip_wl_polish"] = True

        result = solve(cf, pin_features, edge_list, config=fast_config, verbose=False)
        m = calculate_normalized_metrics(result["final_cell_features"], pin_features, edge_list)

        if verbose:
            print(f"    {name}: overlap={m['overlap_ratio']:.4f} wl={m['normalized_wl']:.4f}")

        if m["overlap_ratio"] == 0 and m["normalized_wl"] < best_wl:
            best_wl = m["normalized_wl"]
            best_result = result

    # If no zero-overlap result, fall back to original
    if best_result is None:
        best_result = solve(cell_features, pin_features, edge_list, config=config, verbose=verbose)

    return best_result
