"""Post-legalization wirelength optimization.

Two approaches:
1. Gradient WL polish: run GD optimizing wirelength only, then re-legalize
2. Barycentric refinement: move cells toward connected neighbors (fast vectorized)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.optim as optim

from placement import wirelength_attraction_loss


def gradient_wl_polish(
    cell_features, pin_features, edge_list,
    epochs=200, lr=0.005,
):
    """Run gradient descent on wirelength only, then re-legalize.

    This exploits the fact that legalization is fast and deterministic.
    We optimize positions freely for WL, then snap back to legal positions.
    Iterate a few times: GD → legalize → GD → legalize.

    Returns dict with stats.
    """
    from ashvin.legalize import legalize
    from ashvin.repair import repair_overlaps

    start_time = time.perf_counter()

    num_macros = (cell_features[:, 5] > 1.5).sum().item()
    N = cell_features.shape[0]

    # Scale epochs for large designs
    if N > 10000:
        epochs = 50
    elif N > 2000:
        epochs = 100

    initial_wl = wirelength_attraction_loss(cell_features, pin_features, edge_list).item()

    for cycle in range(3):  # GD → legalize cycles
        # Gradient descent on WL only (macros frozen)
        pos = cell_features[:, 2:4].clone().detach()
        # Only optimize std cell positions
        std_pos = pos[num_macros:].clone().detach()
        std_pos.requires_grad_(True)
        macro_pos = pos[:num_macros].detach()

        optimizer = optim.Adam([std_pos], lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            full_pos = torch.cat([macro_pos, std_pos], dim=0)
            cf_cur = cell_features.clone()
            cf_cur[:, 2:4] = full_pos
            wl_loss = wirelength_attraction_loss(cf_cur, pin_features, edge_list)
            wl_loss.backward()
            torch.nn.utils.clip_grad_norm_([std_pos], max_norm=2.0)
            optimizer.step()

        # Write back and re-legalize
        cell_features[:, 2:4] = torch.cat([macro_pos, std_pos.detach()], dim=0)
        legalize(cell_features, num_macros=num_macros)
        repair_overlaps(cell_features, num_macros=num_macros, max_iterations=50)

        # Reduce LR for next cycle
        lr *= 0.5

    final_wl = wirelength_attraction_loss(cell_features, pin_features, edge_list).item()

    return {
        "time": time.perf_counter() - start_time,
        "wl_before": initial_wl,
        "wl_after": final_wl,
        "improvement": (initial_wl - final_wl) / initial_wl if initial_wl > 0 else 0,
    }
