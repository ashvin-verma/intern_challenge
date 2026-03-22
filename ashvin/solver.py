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
        lambda_density = config.get("lambda_density", lambda_density)
        beta_start = config.get("beta_start", beta_start)
        beta_end = config.get("beta_end", beta_end)
        warmup_epochs = config.get("warmup_epochs", warmup_epochs)
        repair_iterations = config.get("repair_iterations", repair_iterations)

    cell_features = cell_features.clone()
    initial_cell_features = cell_features.clone()
    N = cell_features.shape[0]

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
    warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=max(warmup_epochs, 1)
    )

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
        if epoch < warmup_epochs:
            warmup.step()
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
