"""Instrumented training wrapper with per-phase timing."""

import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path for placement imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.optim as optim

from placement import overlap_repulsion_loss, wirelength_attraction_loss


def instrumented_train_placement(
    cell_features,
    pin_features,
    edge_list,
    num_epochs=1000,
    lr=0.01,
    lambda_wirelength=1.0,
    lambda_overlap=10.0,
    lambda_density=0.0,
    verbose=True,
    log_interval=100,
):
    """Same as train_placement() but with per-phase timing.

    Returns dict with all keys from train_placement() plus:
        timing: dict with cumulative seconds for each phase
    """
    cell_features = cell_features.clone()
    initial_cell_features = cell_features.clone()

    cell_positions = cell_features[:, 2:4].clone().detach()
    cell_positions.requires_grad_(True)

    optimizer = optim.Adam([cell_positions], lr=lr)

    loss_history = {
        "total_loss": [],
        "wirelength_loss": [],
        "overlap_loss": [],
        "density_loss": [],
    }

    # Import density loss if needed
    density_loss_fn = None
    if lambda_density > 0:
        from ashvin.density import density_loss as _density_loss
        density_loss_fn = _density_loss

    # Cumulative timing accumulators
    wl_time = 0.0
    overlap_time = 0.0
    density_time = 0.0
    backward_time = 0.0
    optimizer_time = 0.0

    train_start = time.perf_counter()

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        cell_features_current = cell_features.clone()
        cell_features_current[:, 2:4] = cell_positions

        t0 = time.perf_counter()
        wl_loss = wirelength_attraction_loss(
            cell_features_current, pin_features, edge_list
        )
        t1 = time.perf_counter()
        overlap_loss = overlap_repulsion_loss(
            cell_features_current, pin_features, edge_list
        )
        t2 = time.perf_counter()

        if density_loss_fn is not None:
            d_loss = density_loss_fn(cell_features_current)
        else:
            d_loss = torch.tensor(0.0)
        t3 = time.perf_counter()

        total_loss = (
            lambda_wirelength * wl_loss
            + lambda_overlap * overlap_loss
            + lambda_density * d_loss
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([cell_positions], max_norm=5.0)
        t4 = time.perf_counter()

        optimizer.step()
        t5 = time.perf_counter()

        wl_time += t1 - t0
        overlap_time += t2 - t1
        density_time += t3 - t2
        backward_time += t4 - t3
        optimizer_time += t5 - t4

        loss_history["total_loss"].append(total_loss.item())
        loss_history["wirelength_loss"].append(wl_loss.item())
        loss_history["overlap_loss"].append(overlap_loss.item())
        loss_history["density_loss"].append(d_loss.item())

        if verbose and (epoch % log_interval == 0 or epoch == num_epochs - 1):
            print(f"Epoch {epoch}/{num_epochs}:")
            print(f"  Total Loss: {total_loss.item():.6f}")
            print(f"  Wirelength Loss: {wl_loss.item():.6f}")
            print(f"  Overlap Loss: {overlap_loss.item():.6f}")
            if lambda_density > 0:
                print(f"  Density Loss: {d_loss.item():.6f}")

    train_end = time.perf_counter()

    final_cell_features = cell_features.clone()
    final_cell_features[:, 2:4] = cell_positions.detach()

    return {
        "final_cell_features": final_cell_features,
        "initial_cell_features": initial_cell_features,
        "loss_history": loss_history,
        "timing": {
            "wl_loss_time": wl_time,
            "overlap_loss_time": overlap_time,
            "density_loss_time": density_time,
            "backward_time": backward_time,
            "optimizer_time": optimizer_time,
            "total_train_time": train_end - train_start,
        },
    }


def _run_stage(
    cell_features, pin_features, edge_list,
    cell_positions, num_macros, optimize_macros,
    num_epochs, lr, lambda_wl, lambda_overlap, lambda_density,
    lr_schedule="constant", overlap_ramp=False,
    beta_start=None, beta_end=None,
    stage_name="", verbose=False,
):
    """Run one stage of optimization on macro or std cell positions.

    Args:
        num_macros: number of macro cells (first num_macros indices)
        optimize_macros: if True, optimize macros; if False, optimize std cells
        lr_schedule: "constant" or "cosine" (cosine annealing to 0)
        overlap_ramp: if True, ramp lambda_overlap from 10% to 100% over epochs
        beta_start: softplus beta at epoch 0 (None = ReLU throughout)
        beta_end: softplus beta at final epoch (annealed from start to end)
    """
    import math
    from ashvin.density import density_loss as density_loss_fn
    from ashvin.overlap import _pair_cache

    # Reset pair cache between stages
    _pair_cache["pairs"] = None
    _pair_cache["call_count"] = 0

    N = cell_positions.shape[0]

    # Split positions into macro and std parts
    macro_pos_frozen = cell_positions[:num_macros].detach().clone()
    std_pos_frozen = cell_positions[num_macros:].detach().clone()

    if optimize_macros:
        opt_positions = cell_positions[:num_macros].clone().detach()
    else:
        opt_positions = cell_positions[num_macros:].clone().detach()
    opt_positions.requires_grad_(True)

    optimizer = optim.Adam([opt_positions], lr=lr)

    # LR scheduler
    scheduler = None
    if lr_schedule == "cosine" and num_epochs > 0:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    wl_time = overlap_time = density_time = backward_time = optimizer_time = 0.0
    stage_start = time.perf_counter()

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Lambda ramping: overlap weight ramps from 10% to 100%
        progress = epoch / max(num_epochs - 1, 1)
        if overlap_ramp and num_epochs > 1:
            ramp = 0.1 + 0.9 * progress
            cur_lambda_overlap = lambda_overlap * ramp
        else:
            cur_lambda_overlap = lambda_overlap

        # Beta annealing: softplus sharpens over training
        if beta_start is not None and beta_end is not None:
            cur_beta = beta_start + (beta_end - beta_start) * progress
        else:
            cur_beta = None

        # Reconstruct full position tensor via cat (clean autograd)
        if optimize_macros:
            full_positions = torch.cat([opt_positions, std_pos_frozen], dim=0)
        else:
            full_positions = torch.cat([macro_pos_frozen, opt_positions], dim=0)

        cell_features_current = cell_features.clone()
        cell_features_current[:, 2:4] = full_positions

        t0 = time.perf_counter()
        wl_loss = wirelength_attraction_loss(
            cell_features_current, pin_features, edge_list
        )
        t1 = time.perf_counter()
        if cur_beta is not None:
            from ashvin.overlap import scalable_overlap_loss as _scalable_ol
            overlap_loss = _scalable_ol(cell_features_current, beta=cur_beta)
        else:
            overlap_loss = overlap_repulsion_loss(
                cell_features_current, pin_features, edge_list
            )
        t2 = time.perf_counter()
        d_loss = density_loss_fn(cell_features_current) if lambda_density > 0 else torch.tensor(0.0)
        t3 = time.perf_counter()

        total_loss = lambda_wl * wl_loss + cur_lambda_overlap * overlap_loss + lambda_density * d_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([opt_positions], max_norm=5.0)
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

        if verbose and (epoch % 100 == 0 or epoch == num_epochs - 1):
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  [{stage_name}] Epoch {epoch}/{num_epochs}: "
                  f"total={total_loss.item():.4f} wl={wl_loss.item():.4f} "
                  f"overlap={overlap_loss.item():.4f} lr={lr_now:.5f}")

    # Write optimized positions back
    if optimize_macros:
        cell_positions[:num_macros] = opt_positions.detach()
    else:
        cell_positions[num_macros:] = opt_positions.detach()

    return {
        "wl_loss_time": wl_time,
        "overlap_loss_time": overlap_time,
        "density_loss_time": density_time,
        "backward_time": backward_time,
        "optimizer_time": optimizer_time,
        "stage_time": time.perf_counter() - stage_start,
    }


def two_stage_train_placement(
    cell_features, pin_features, edge_list,
    stage_a_epochs=500, stage_a_lr=0.05,
    stage_a_lambda_wl=0.0, stage_a_lambda_overlap=100.0, stage_a_lambda_density=5.0,
    stage_a_lr_schedule="cosine", stage_a_overlap_ramp=False,
    stage_a_beta_start=None, stage_a_beta_end=None,
    stage_b_epochs=500, stage_b_lr=0.01,
    stage_b_lambda_wl=1.0, stage_b_lambda_overlap=10.0, stage_b_lambda_density=1.0,
    stage_b_lr_schedule="cosine", stage_b_overlap_ramp=False,
    stage_b_beta_start=None, stage_b_beta_end=None,
    repair_max_iterations=100, repair_epsilon=0.01,
    config=None,
    verbose=False,
):
    """Two-stage training: macros first, then std cells.

    If config dict is provided, it overrides all keyword arguments.
    Returns same dict format as instrumented_train_placement().
    """
    # Config dict overrides keyword arguments
    if config is not None:
        stage_a_epochs = config.get("stage_a_epochs", stage_a_epochs)
        stage_a_lr = config.get("stage_a_lr", stage_a_lr)
        stage_a_lambda_wl = config.get("stage_a_lambda_wl", stage_a_lambda_wl)
        stage_a_lambda_overlap = config.get("stage_a_lambda_overlap", stage_a_lambda_overlap)
        stage_a_lambda_density = config.get("stage_a_lambda_density", stage_a_lambda_density)
        stage_a_lr_schedule = config.get("stage_a_lr_schedule", stage_a_lr_schedule)
        stage_a_overlap_ramp = config.get("stage_a_overlap_ramp", stage_a_overlap_ramp)
        stage_a_beta_start = config.get("stage_a_beta_start", stage_a_beta_start)
        stage_a_beta_end = config.get("stage_a_beta_end", stage_a_beta_end)
        stage_b_epochs = config.get("stage_b_epochs", stage_b_epochs)
        stage_b_lr = config.get("stage_b_lr", stage_b_lr)
        stage_b_lambda_wl = config.get("stage_b_lambda_wl", stage_b_lambda_wl)
        stage_b_lambda_overlap = config.get("stage_b_lambda_overlap", stage_b_lambda_overlap)
        stage_b_lambda_density = config.get("stage_b_lambda_density", stage_b_lambda_density)
        stage_b_lr_schedule = config.get("stage_b_lr_schedule", stage_b_lr_schedule)
        stage_b_overlap_ramp = config.get("stage_b_overlap_ramp", stage_b_overlap_ramp)
        stage_b_beta_start = config.get("stage_b_beta_start", stage_b_beta_start)
        stage_b_beta_end = config.get("stage_b_beta_end", stage_b_beta_end)
        repair_max_iterations = config.get("repair_max_iterations", repair_max_iterations)
        repair_epsilon = config.get("repair_epsilon", repair_epsilon)

    cell_features = cell_features.clone()
    initial_cell_features = cell_features.clone()

    # Detect macros (height > 1.5 — macros are square with height >= 10)
    num_macros = (cell_features[:, 5] > 1.5).sum().item()
    N = cell_features.shape[0]

    cell_positions = cell_features[:, 2:4].clone().detach()

    train_start = time.perf_counter()

    # Stage A: optimize macros only
    if verbose:
        print(f"Stage A: {num_macros} macros, {stage_a_epochs} epochs")
    timing_a = _run_stage(
        cell_features, pin_features, edge_list,
        cell_positions, num_macros, optimize_macros=True,
        num_epochs=stage_a_epochs, lr=stage_a_lr,
        lambda_wl=stage_a_lambda_wl, lambda_overlap=stage_a_lambda_overlap,
        lambda_density=stage_a_lambda_density,
        lr_schedule=stage_a_lr_schedule, overlap_ramp=stage_a_overlap_ramp,
        beta_start=stage_a_beta_start, beta_end=stage_a_beta_end,
        stage_name="A-macros", verbose=verbose,
    )

    # Stage B: optimize std cells only (macros frozen)
    if verbose:
        print(f"Stage B: {N - num_macros} std cells, {stage_b_epochs} epochs")
    timing_b = _run_stage(
        cell_features, pin_features, edge_list,
        cell_positions, num_macros, optimize_macros=False,
        num_epochs=stage_b_epochs, lr=stage_b_lr,
        lambda_wl=stage_b_lambda_wl, lambda_overlap=stage_b_lambda_overlap,
        lambda_density=stage_b_lambda_density,
        lr_schedule=stage_b_lr_schedule, overlap_ramp=stage_b_overlap_ramp,
        beta_start=stage_b_beta_start, beta_end=stage_b_beta_end,
        stage_name="B-stdcells", verbose=verbose,
    )

    # Stage C: greedy repair pass
    final_cell_features = cell_features.clone()
    final_cell_features[:, 2:4] = cell_positions

    from ashvin.repair import repair_overlaps
    if verbose:
        print("Stage C: greedy repair")
    repair_stats = repair_overlaps(
        final_cell_features, num_macros=num_macros,
        max_iterations=repair_max_iterations, epsilon=repair_epsilon,
    )
    if verbose:
        print(f"  Repair: {repair_stats['overlaps_before']}→{repair_stats['overlaps_after']} "
              f"overlapping pairs in {repair_stats['iterations']} iterations "
              f"({repair_stats['time']:.2f}s)")

    train_end = time.perf_counter()

    return {
        "final_cell_features": final_cell_features,
        "initial_cell_features": initial_cell_features,
        "loss_history": {"total_loss": [], "wirelength_loss": [], "overlap_loss": [], "density_loss": []},
        "timing": {
            "wl_loss_time": timing_a["wl_loss_time"] + timing_b["wl_loss_time"],
            "overlap_loss_time": timing_a["overlap_loss_time"] + timing_b["overlap_loss_time"],
            "density_loss_time": timing_a["density_loss_time"] + timing_b["density_loss_time"],
            "backward_time": timing_a["backward_time"] + timing_b["backward_time"],
            "optimizer_time": timing_a["optimizer_time"] + timing_b["optimizer_time"],
            "total_train_time": train_end - train_start,
            "stage_a_time": timing_a["stage_time"],
            "stage_b_time": timing_b["stage_time"],
            "repair_time": repair_stats["time"],
            "repair_before": repair_stats["overlaps_before"],
            "repair_after": repair_stats["overlaps_after"],
        },
    }
