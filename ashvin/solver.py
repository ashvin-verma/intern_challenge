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

    # Cell inflation: inflate widths/heights during GD so cells spread further apart.
    # When deflated back before legalization, cells have natural gaps between them,
    # so legalization only needs minor adjustments instead of major reshuffling.
    inflate = config.get("inflate", 1.08) if config else 1.08
    if inflate > 1.0:
        cell_features[:, 4] *= inflate  # width
        cell_features[:, 5] *= inflate  # height

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
        beta = beta_start + (beta_end - beta_start) * progress
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
            print(f"  Epoch {epoch}/{epochs}: wl={wl_loss.item():.4f} "
                  f"overlap={ov_loss.item():.4f} beta={beta:.2f} lr={lr_now:.5f}")

    cell_features[:, 2:4] = pos.detach()

    # Deflate back to true sizes before legalization
    if inflate > 1.0:
        cell_features[:, 4] = initial_cell_features[:, 4]
        cell_features[:, 5] = initial_cell_features[:, 5]

    # === MULTI-PASS PIPELINE (compiler-style) ===
    from ashvin.legalize import legalize as legalize_greedy
    from ashvin.abacus import abacus_legalize
    from ashvin.wl_optimize import barycentric_refinement, targeted_scatter_reconverge

    _leg_call = [0]

    def legalize_fallback(cf, **kwargs):
        """First call: try Abacus + greedy, pick best. After: greedy only (fast)."""
        _leg_call[0] += 1

        if _leg_call[0] <= 1:
            from placement import calculate_normalized_metrics
            pf, el = pin_features, edge_list
            cf_pre = cf.clone()

            cf_a = cf_pre.clone()
            abacus_legalize(cf_a)
            repair_overlaps(cf_a, max_iterations=200)
            m_a = calculate_normalized_metrics(cf_a, pf, el)

            cf_g = cf_pre.clone()
            stats = legalize_greedy(cf_g, pin_features=pf, edge_list=el)
            repair_overlaps(cf_g, max_iterations=200)
            m_g = calculate_normalized_metrics(cf_g, pf, el)

            if m_a["overlap_ratio"] == 0 and (m_g["overlap_ratio"] > 0 or m_a["normalized_wl"] < m_g["normalized_wl"]):
                cf[:] = cf_a
            else:
                cf[:] = cf_g
            return stats
        else:
            return legalize_greedy(cf, pin_features=pin_features, edge_list=edge_list)

    skip_scatter = config.get("_skip_scatter", False) if config else False
    num_macros_det = (cell_features[:, 5] > 1.5).sum().item()

    legalize_time = 0.0
    repair_time = 0.0
    repair_before = 0
    repair_after = 0

    # Phase 1: Initial legalization (guarantee zero overlap)
    for leg_pass in range(5):
        leg_stats = legalize_fallback(cell_features, pin_features=pin_features, edge_list=edge_list)
        legalize_time += leg_stats["time"]
        rep_stats = repair_overlaps(cell_features, max_iterations=repair_iterations)
        repair_time += rep_stats["time"]
        if leg_pass == 0:
            repair_before = rep_stats["overlaps_before"]
        repair_after = rep_stats["overlaps_after"]
        if repair_after == 0:
            break

    # Phase 2: Anchor-based WL optimization loop
    # Key insight: after legalization, store positions as anchors.
    # GD optimizes WL but is tethered to the legal state via anchor loss.
    # Next legalization only needs small corrections.
    from placement import calculate_normalized_metrics
    best_wl = calculate_normalized_metrics(cell_features, pin_features, edge_list)["normalized_wl"]
    best_features = cell_features.clone()

    pipeline_passes = config.get("pipeline_passes", 3) if config else 3
    lambda_anchor = config.get("lambda_anchor", 0.1) if config else 0.1
    anchor_gd_steps = config.get("anchor_gd_steps", 80) if config else 80

    for pipe_iter in range(pipeline_passes):
        improved_this_iter = False

        # Pass A: Barycentric refinement (fast, local)
        bary_stats = barycentric_refinement(cell_features, pin_features, edge_list)

        # Pass B: Targeted scatter + reconverge (break local minima)
        if not skip_scatter and N <= 5000:
            scatter_result = targeted_scatter_reconverge(
                cell_features, pin_features, edge_list, config=config
            )
            if scatter_result is not None:
                cell_features[:] = scatter_result["final_cell_features"]

        # Pass C: Anchor-tethered GD — optimize WL while staying near legal positions
        # Store current legal positions as anchors
        anchor_pos = cell_features[:, 2:4].detach().clone()

        std_pos = cell_features[num_macros_det:, 2:4].clone().detach()
        std_pos.requires_grad_(True)
        macro_pos = cell_features[:num_macros_det, 2:4].detach()
        anchor_std = anchor_pos[num_macros_det:]

        opt_wl = optim.Adam([std_pos], lr=0.003)
        for _ep in range(anchor_gd_steps):
            opt_wl.zero_grad()
            full_pos = torch.cat([macro_pos, std_pos], dim=0)
            cf_tmp = cell_features.clone()
            cf_tmp[:, 2:4] = full_pos
            wl_l = wirelength_attraction_loss(cf_tmp, pin_features, edge_list)
            # Anchor loss: soft spring to legal positions
            anc_l = ((std_pos - anchor_std) ** 2).mean()
            total = lambda_wl * wl_l + lambda_anchor * anc_l
            total.backward()
            torch.nn.utils.clip_grad_norm_([std_pos], max_norm=1.0)
            opt_wl.step()
        cell_features[:, 2:4] = torch.cat([macro_pos, std_pos.detach()], dim=0)

        # Pass D: Re-legalize (should be small corrections thanks to anchor)
        for _lp in range(3):
            legalize_fallback(cell_features, pin_features=pin_features, edge_list=edge_list)
            rep = repair_overlaps(cell_features, max_iterations=100)
            if rep["overlaps_after"] == 0:
                break

        # Check if this iteration improved WL
        cur_m = calculate_normalized_metrics(cell_features, pin_features, edge_list)
        if cur_m["overlap_ratio"] == 0 and cur_m["normalized_wl"] < best_wl:
            best_wl = cur_m["normalized_wl"]
            best_features = cell_features.clone()
            improved_this_iter = True

        if not improved_this_iter:
            cell_features[:] = best_features  # revert to best
            break

    cell_features[:] = best_features

    # Phase 3: Detailed placement (swaps + reinsertion) — small designs only
    skip_detailed = config.get("_skip_detailed", False) if config else False
    if not skip_detailed and N <= 300:
        from ashvin.detailed import detailed_placement
        from placement import calculate_normalized_metrics
        wl_pre_dp = calculate_normalized_metrics(cell_features, pin_features, edge_list)["normalized_wl"]
        cf_backup = cell_features.clone()
        dp_stats = detailed_placement(cell_features, pin_features, edge_list)
        rep_final = repair_overlaps(cell_features, max_iterations=50)
        m_post = calculate_normalized_metrics(cell_features, pin_features, edge_list)
        if m_post["overlap_ratio"] > 0 or m_post["normalized_wl"] >= wl_pre_dp:
            cell_features[:] = cf_backup

    # Phase 4: Iterative swap engine — within-row + cross-row moves
    skip_swaps = config.get("_skip_swaps", False) if config else False
    swap_iters = config.get("swap_iterations", 20) if config else 20
    if not skip_swaps:
        from ashvin.swap_engine import swap_engine
        from placement import calculate_normalized_metrics
        wl_pre_swap = calculate_normalized_metrics(cell_features, pin_features, edge_list)["normalized_wl"]
        cf_backup = cell_features.clone()

        se_stats = swap_engine(
            cell_features, pin_features, edge_list,
            max_iterations=swap_iters, verbose=verbose,
        )

        # Verify legality
        rep_se = repair_overlaps(cell_features, max_iterations=100)
        m_se = calculate_normalized_metrics(cell_features, pin_features, edge_list)
        if m_se["overlap_ratio"] > 0 or m_se["normalized_wl"] >= wl_pre_swap:
            cell_features[:] = cf_backup
        elif verbose:
            print(f"  Swap engine: {se_stats['swaps']} swaps, {se_stats['moves']} moves, "
                  f"WL {wl_pre_swap:.4f} -> {m_se['normalized_wl']:.4f}")

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
    """Run solver with multiple strategies, pick best WL.

    Tries: original positions + spectral placement + WL-priority legalization.
    Returns the result with lowest WL (that has 0 overlap).
    """
    from placement import calculate_normalized_metrics

    N = cell_features.shape[0]
    best_result = None
    best_wl = float("inf")

    strategies = [("greedy_legal", cell_features.clone(), {})]

    # Island-clustered init
    if N <= 5000:
        from ashvin.constructive import island_init
        island_cf = cell_features.clone()
        island_init(island_cf, pin_features, edge_list, config=config, verbose=verbose)
        strategies.append(("island_init", island_cf, {}))

    # Add spectral init for small/medium designs
    if N <= 5000:
        from ashvin.init_placement import spectral_placement
        spectral_cf = cell_features.clone()
        spectral_placement(spectral_cf, pin_features, edge_list)
        strategies.append(("spectral", spectral_cf, {}))

    for name, cf, extra_config in strategies:
        if verbose:
            print(f"  Multi-start: trying {name}...")

        run_config = dict(config) if config else {}
        run_config.update(extra_config)

        result = solve(cf, pin_features, edge_list, config=run_config, verbose=False)
        m = calculate_normalized_metrics(result["final_cell_features"], pin_features, edge_list)

        if verbose:
            print(f"    {name}: overlap={m['overlap_ratio']:.4f} wl={m['normalized_wl']:.4f}")

        if m["overlap_ratio"] == 0 and m["normalized_wl"] < best_wl:
            best_wl = m["normalized_wl"]
            best_result = result

    if best_result is None:
        best_result = solve(cell_features, pin_features, edge_list, config=config, verbose=verbose)

    return best_result
