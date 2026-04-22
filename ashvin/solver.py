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

from ashvin.device_utils import move_runtime_tensors
from ashvin.density import density_loss
from ashvin.overlap import _pair_cache, scalable_overlap_loss
from ashvin.projected_gd import project_to_legal_rows
from ashvin.repair import repair_overlaps
from placement import wirelength_attraction_loss


def _size_in_ranges(size, ranges):
    if not ranges:
        return False
    return any(lo <= size <= hi for lo, hi in ranges)


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
    config = dict(config) if config else {}
    cell_features, pin_features, edge_list, _runtime_device, _runtime_reason = move_runtime_tensors(
        cell_features, pin_features, edge_list, config=config, verbose=verbose
    )
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
    exhaustive = config.get("exhaustive_multistart", False) if config else False

    # Optional size-aware caps for expensive full-suite runs.
    if config is not None:
        if N > 50000:
            epoch_cap = config.get("epoch_cap_over_50000", None)
            warmup_cap = config.get("warmup_cap_over_50000", None)
        elif N > 10000:
            epoch_cap = config.get("epoch_cap_over_10000", None)
            warmup_cap = config.get("warmup_cap_over_10000", None)
        elif N > 2000:
            epoch_cap = config.get("epoch_cap_over_2000", None)
            warmup_cap = config.get("warmup_cap_over_2000", None)
        else:
            epoch_cap = None
            warmup_cap = None
        if epoch_cap is not None:
            epochs = min(epochs, epoch_cap)
        if warmup_cap is not None:
            warmup_epochs = min(warmup_epochs, warmup_cap)

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
    projection_time = 0.0
    projection_count = 0
    train_start = time.perf_counter()

    enable_projected_gd = config.get("enable_projected_gd", False) if config else False
    projected_gd_min_cells = config.get("projected_gd_min_cells", 0) if config else 0
    projected_gd_max_cells = config.get("projected_gd_max_cells", 3000) if config else 3000
    projected_gd_ranges = config.get("projected_gd_ranges", None) if config else None
    projection_interval = max(1, config.get("projection_interval", 50) if config else 50)
    projection_start_epoch = config.get("projection_start_epoch", warmup_epochs) if config else warmup_epochs
    projection_gap = config.get("projection_gap", 1e-3) if config else 1e-3
    projection_final = config.get("projection_final", True) if config else True
    num_macros_gd = int((cell_features[:, 5] > 1.5).sum().item())
    if projected_gd_ranges:
        projected_gd_allowed = _size_in_ranges(N, projected_gd_ranges)
    else:
        projected_gd_allowed = projected_gd_min_cells <= N <= projected_gd_max_cells
    run_projected_gd = enable_projected_gd and projected_gd_allowed

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
        d_loss = (
            density_loss(cell_features_current)
            if lambda_density > 0
            else torch.tensor(0.0, device=cell_features_current.device)
        )
        t3 = time.perf_counter()

        total_loss = lambda_wl * wl_loss + lam_ov * ov_loss + lambda_density * d_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([pos], max_norm=5.0)
        t4 = time.perf_counter()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        t5 = time.perf_counter()

        if (
            run_projected_gd
            and epoch >= projection_start_epoch
            and (epoch - projection_start_epoch) % projection_interval == 0
        ):
            proj_stats = project_to_legal_rows(
                pos,
                cell_features[:, 4].detach(),
                cell_features[:, 5].detach(),
                num_macros=num_macros_gd,
                gap=projection_gap,
            )
            projection_time += proj_stats["time"]
            projection_count += 1
            _pair_cache["pairs"] = None
            _pair_cache["call_count"] = 0

        wl_time += t1 - t0
        overlap_time += t2 - t1
        density_time += t3 - t2
        backward_time += t4 - t3
        optimizer_time += t5 - t4

        if verbose and (epoch % 200 == 0 or epoch == epochs - 1):
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch}/{epochs}: wl={wl_loss.item():.4f} "
                  f"overlap={ov_loss.item():.4f} beta={beta:.2f} lr={lr_now:.5f}")

    if run_projected_gd and projection_final:
        proj_stats = project_to_legal_rows(
            pos,
            cell_features[:, 4].detach(),
            cell_features[:, 5].detach(),
            num_macros=num_macros_gd,
            gap=projection_gap,
        )
        projection_time += proj_stats["time"]
        projection_count += 1
        _pair_cache["pairs"] = None
        _pair_cache["call_count"] = 0

    cell_features[:, 2:4] = pos.detach()

    # Deflate back to true sizes before legalization
    if inflate > 1.0:
        cell_features[:, 4] = initial_cell_features[:, 4]
        cell_features[:, 5] = initial_cell_features[:, 5]

    # === MULTI-PASS PIPELINE (compiler-style) ===
    postprocess_device = cell_features.device
    cpu_postprocess = config.get("cpu_postprocess", False) if config else False
    if cpu_postprocess and postprocess_device.type != "cpu":
        cell_features = cell_features.detach().cpu().clone()
        pin_features_post = pin_features.detach().cpu()
        edge_list_post = edge_list.detach().cpu()
    else:
        pin_features_post = pin_features
        edge_list_post = edge_list

    from ashvin.legalize import legalize as legalize_greedy
    from ashvin.abacus import abacus_legalize
    from ashvin.wl_optimize import barycentric_refinement, targeted_scatter_reconverge

    def legalize_fallback(cf, **kwargs):
        """Greedy row-pack legalization."""
        return legalize_greedy(cf, pin_features=pin_features_post, edge_list=edge_list_post)

    enable_selective_scatter = config.get("enable_selective_scatter", False) if config else False
    skip_scatter = config.get("_skip_scatter", False) if config else False
    scatter_min_cells = config.get("scatter_min_cells", 0) if config else 0
    scatter_max_cells = config.get("scatter_max_cells", 40) if config else 40
    scatter_min_wl = config.get("scatter_min_wl", 0.33) if config else 0.33
    enable_abacus_candidate = config.get("enable_abacus_candidate", False) if config else False
    abacus_candidate_min_cells = config.get("abacus_candidate_min_cells", 0) if config else 0
    abacus_candidate_max_cells = config.get("abacus_candidate_max_cells", 3000) if config else 3000
    abacus_candidate_ranges = config.get("abacus_candidate_ranges", None) if config else None
    enable_shelf_legalizer_v2 = config.get("enable_shelf_legalizer_v2", False) if config else False
    shelf_legalizer_min_cells = config.get("shelf_legalizer_min_cells", 0) if config else 0
    shelf_legalizer_max_cells = config.get("shelf_legalizer_max_cells", 3000) if config else 3000
    shelf_legalizer_ranges = config.get("shelf_legalizer_ranges", None) if config else None
    shelf_legalizer_row_limit = config.get("shelf_legalizer_row_limit", 5) if config else 5
    shelf_legalizer_gap = config.get("shelf_legalizer_gap", 1e-3) if config else 1e-3
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

    from placement import calculate_normalized_metrics

    def try_candidate_legalizer(label, legalizer_fn):
        legal_wl = calculate_normalized_metrics(cell_features, pin_features_post, edge_list_post)["normalized_wl"]
        cf_candidate = cell_features.clone()
        legalizer_fn(cf_candidate)
        repair_overlaps(cf_candidate, max_iterations=repair_iterations)
        m_candidate = calculate_normalized_metrics(cf_candidate, pin_features_post, edge_list_post)
        if m_candidate["overlap_ratio"] == 0 and m_candidate["normalized_wl"] < legal_wl:
            cell_features[:] = cf_candidate
            if verbose:
                print(
                    f"  {label} accepted: WL {legal_wl:.4f} -> {m_candidate['normalized_wl']:.4f}"
                )
            return True
        if verbose:
            print(
                f"  {label} rejected: overlap={m_candidate['overlap_ratio']:.4f} "
                f"wl={m_candidate['normalized_wl']:.4f}"
            )
        return False

    if abacus_candidate_ranges:
        run_abacus_candidate = any(lo <= N <= hi for lo, hi in abacus_candidate_ranges)
    else:
        run_abacus_candidate = abacus_candidate_min_cells <= N <= abacus_candidate_max_cells

    if enable_abacus_candidate and run_abacus_candidate:
        try_candidate_legalizer(
            "Abacus candidate",
            lambda cf_candidate: abacus_legalize(
                cf_candidate,
                num_macros=num_macros_det,
                pin_features=pin_features_post,
                edge_list=edge_list_post,
            ),
        )

    if shelf_legalizer_ranges:
        run_shelf_legalizer = _size_in_ranges(N, shelf_legalizer_ranges)
    else:
        run_shelf_legalizer = shelf_legalizer_min_cells <= N <= shelf_legalizer_max_cells

    if enable_shelf_legalizer_v2 and run_shelf_legalizer:
        from ashvin.shelf_legalizer import shelf_legalize_v2

        try_candidate_legalizer(
            "Shelf legalizer v2",
            lambda cf_candidate: shelf_legalize_v2(
                cf_candidate,
                pin_features_post,
                edge_list_post,
                num_macros=num_macros_det,
                row_limit=shelf_legalizer_row_limit,
                max_cells=shelf_legalizer_max_cells,
                gap=shelf_legalizer_gap,
            ),
        )

    # Phase 2: Anchor-based WL optimization loop
    # Key insight: after legalization, store positions as anchors.
    # GD optimizes WL but is tethered to the legal state via anchor loss.
    # Next legalization only needs small corrections.
    best_wl = calculate_normalized_metrics(cell_features, pin_features_post, edge_list_post)["normalized_wl"]
    best_features = cell_features.clone()

    pipeline_passes = config.get("pipeline_passes", 3) if config else 3
    lambda_anchor = config.get("lambda_anchor", 0.1) if config else 0.1
    anchor_gd_steps = config.get("anchor_gd_steps", 80) if config else 80
    barycentric_passes = config.get("barycentric_passes", 15) if config else 15

    if not exhaustive:
        if N > 120:
            pipeline_passes = min(pipeline_passes, 2)
            anchor_gd_steps = min(anchor_gd_steps, 40)
        elif N > 40:
            pipeline_passes = min(pipeline_passes, 3)
            anchor_gd_steps = min(anchor_gd_steps, 40)

    if config is not None:
        if N > 50000:
            pipeline_cap = config.get("pipeline_pass_cap_over_50000", None)
            anchor_cap = config.get("anchor_steps_cap_over_50000", None)
            bary_cap = config.get("barycentric_cap_over_50000", None)
        elif N > 10000:
            pipeline_cap = config.get("pipeline_pass_cap_over_10000", None)
            anchor_cap = config.get("anchor_steps_cap_over_10000", None)
            bary_cap = config.get("barycentric_cap_over_10000", None)
        elif N > 2000:
            pipeline_cap = config.get("pipeline_pass_cap_over_2000", None)
            anchor_cap = config.get("anchor_steps_cap_over_2000", None)
            bary_cap = config.get("barycentric_cap_over_2000", None)
        else:
            pipeline_cap = anchor_cap = bary_cap = None
        if pipeline_cap is not None:
            pipeline_passes = min(pipeline_passes, pipeline_cap)
        if anchor_cap is not None:
            anchor_gd_steps = min(anchor_gd_steps, anchor_cap)
        if bary_cap is not None:
            barycentric_passes = min(barycentric_passes, bary_cap)

    for pipe_iter in range(pipeline_passes):
        improved_this_iter = False

        # Pass A: Barycentric refinement (fast, local)
        bary_stats = barycentric_refinement(
            cell_features, pin_features_post, edge_list_post, num_passes=barycentric_passes
        )

        # Pass B: Targeted scatter + reconverge (break local minima)
        if (
            enable_selective_scatter
            and not skip_scatter
            and pipe_iter == 0
            and scatter_min_cells <= N <= scatter_max_cells
        ):
            wl_before_scatter = calculate_normalized_metrics(
                cell_features, pin_features_post, edge_list_post
            )["normalized_wl"]
            if wl_before_scatter >= scatter_min_wl:
                if verbose:
                    print(f"  Selective scatter: trying at WL {wl_before_scatter:.4f}")
                scatter_result = targeted_scatter_reconverge(
                    cell_features, pin_features_post, edge_list_post, config=config
                )
                if scatter_result is not None:
                    cell_features[:] = scatter_result["final_cell_features"]
                    if verbose:
                        wl_after_scatter = calculate_normalized_metrics(
                            cell_features, pin_features_post, edge_list_post
                        )["normalized_wl"]
                        print(f"  Selective scatter accepted: WL -> {wl_after_scatter:.4f}")
                elif verbose:
                    print("  Selective scatter rejected")
            elif verbose:
                print(f"  Selective scatter skipped: WL {wl_before_scatter:.4f} < threshold {scatter_min_wl:.4f}")

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
            wl_l = wirelength_attraction_loss(cf_tmp, pin_features_post, edge_list_post)
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
        cur_m = calculate_normalized_metrics(cell_features, pin_features_post, edge_list_post)
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
    detailed_max_cells = config.get("detailed_max_cells", 300) if config else 300
    detailed_passes = config.get("detailed_passes", 5) if config else 5
    if config is not None and N > 300:
        detailed_cap = config.get("detailed_pass_cap_over_300", None)
        if detailed_cap is not None:
            detailed_passes = min(detailed_passes, detailed_cap)
    if not skip_detailed and N <= detailed_max_cells:
        from ashvin.detailed import detailed_placement
        from placement import calculate_normalized_metrics
        wl_pre_dp = calculate_normalized_metrics(cell_features, pin_features_post, edge_list_post)["normalized_wl"]
        cf_backup = cell_features.clone()
        dp_stats = detailed_placement(
            cell_features,
            pin_features_post,
            edge_list_post,
            num_passes=detailed_passes,
            num_macros=num_macros_det,
        )
        rep_final = repair_overlaps(cell_features, max_iterations=50)
        m_post = calculate_normalized_metrics(cell_features, pin_features_post, edge_list_post)
        if m_post["overlap_ratio"] > 0 or m_post["normalized_wl"] >= wl_pre_dp:
            cell_features[:] = cf_backup

    # Phase 4: Iterative swap engine — within-row + cross-row moves
    skip_swaps = config.get("_skip_swaps", False) if config else False
    swap_iters = config.get("swap_iterations", 20) if config else 20
    enable_within_row_swaps = config.get("enable_within_row_swaps", False) if config else False
    within_row_window = config.get("within_row_window", 3) if config else 3
    cross_row_limit = config.get("cross_row_limit", None) if config else None
    swap_max_cells = config.get("swap_max_cells", None) if config else None
    within_row_swap_max_cells = config.get("within_row_swap_max_cells", None) if config else None
    if not exhaustive:
        if N > 120:
            swap_iters = min(swap_iters, 8)
        elif N > 40:
            swap_iters = min(swap_iters, 12)
    if swap_max_cells is not None and N > swap_max_cells:
        skip_swaps = True
    if within_row_swap_max_cells is not None and N > within_row_swap_max_cells:
        enable_within_row_swaps = False
    if not skip_swaps:
        from ashvin.swap_engine import swap_engine
        wl_pre_swap = calculate_normalized_metrics(cell_features, pin_features_post, edge_list_post)["normalized_wl"]
        cf_backup = cell_features.clone()

        se_stats = swap_engine(
            cell_features, pin_features_post, edge_list_post,
            max_iterations=swap_iters,
            enable_within_row_swaps=enable_within_row_swaps,
            within_row_window=within_row_window,
            cross_row_limit=cross_row_limit,
            verbose=verbose,
        )

        # Verify legality
        rep_se = repair_overlaps(cell_features, max_iterations=100)
        m_se = calculate_normalized_metrics(cell_features, pin_features_post, edge_list_post)
        if m_se["overlap_ratio"] > 0 or m_se["normalized_wl"] >= wl_pre_swap:
            cell_features[:] = cf_backup
        elif verbose:
            print(f"  Swap engine: {se_stats['swaps']} swaps, {se_stats['moves']} moves, "
                  f"WL {wl_pre_swap:.4f} -> {m_se['normalized_wl']:.4f}")

    # Phase 4b: Bounded row-level refinement for mid-size cases.
    enable_mid_row_refine = config.get("enable_mid_row_refine", False) if config else False
    mid_row_min_cells = config.get("mid_row_refine_min_cells", 1000) if config else 1000
    mid_row_max_cells = config.get("mid_row_refine_max_cells", 3000) if config else 3000
    if enable_mid_row_refine and mid_row_min_cells <= N <= mid_row_max_cells:
        from ashvin.mid_row_refine import mid_size_row_refine

        wl_pre_mid = calculate_normalized_metrics(cell_features, pin_features_post, edge_list_post)["normalized_wl"]
        cf_backup = cell_features.clone()
        mid_stats = mid_size_row_refine(
            cell_features,
            pin_features_post,
            edge_list_post,
            num_passes=config.get("mid_row_refine_passes", 2),
            num_macros=num_macros_det,
            min_row_cells=config.get("mid_row_refine_min_row_cells", 4),
            max_window=config.get("mid_row_refine_window", 16),
            try_row_remap=config.get("mid_row_refine_remap", True),
            verbose=verbose,
        )
        repair_overlaps(cell_features, max_iterations=100)
        m_mid = calculate_normalized_metrics(cell_features, pin_features_post, edge_list_post)
        if m_mid["overlap_ratio"] > 0 or m_mid["normalized_wl"] >= wl_pre_mid:
            cell_features[:] = cf_backup
        elif verbose:
            print(
                f"  Mid-row refine: {mid_stats['rows_changed']} rows, "
                f"{mid_stats['remaps']} remaps, WL {wl_pre_mid:.4f} -> {m_mid['normalized_wl']:.4f}"
            )

    # Phase 5: Legacy global swap pipeline — slower, but historically strong on some cases.
    skip_global_swap = config.get("_skip_global_swap", False) if config else False
    gs_passes = config.get("gs_passes", 5) if config else 5
    global_swap_max_cells = config.get("global_swap_max_cells", 3000) if config else 3000
    if global_swap_max_cells is not None and N > global_swap_max_cells:
        skip_global_swap = True
    if not skip_global_swap:
        from ashvin.global_swap import global_swap

        wl_pre_gs = calculate_normalized_metrics(cell_features, pin_features_post, edge_list_post)["normalized_wl"]
        cf_backup = cell_features.clone()
        gs_stats = global_swap(
            cell_features,
            pin_features_post,
            edge_list_post,
            num_passes=gs_passes,
            num_macros=num_macros_det,
            verbose=verbose,
        )

        rep_gs = repair_overlaps(cell_features, max_iterations=100)
        m_gs = calculate_normalized_metrics(cell_features, pin_features_post, edge_list_post)
        if m_gs["overlap_ratio"] > 0 or m_gs["normalized_wl"] >= wl_pre_gs:
            cell_features[:] = cf_backup
        elif verbose:
            print(
                f"  Global swap: {gs_stats['swaps']} row swaps, {gs_stats['cross_row_moves']} moves, "
                f"WL {wl_pre_gs:.4f} -> {m_gs['normalized_wl']:.4f}"
            )

    train_end = time.perf_counter()
    final_cell_features = (
        cell_features.to(postprocess_device)
        if cpu_postprocess and cell_features.device != postprocess_device
        else cell_features
    )

    return {
        "final_cell_features": final_cell_features,
        "initial_cell_features": initial_cell_features,
        "loss_history": {"total_loss": [], "wirelength_loss": [], "overlap_loss": [], "density_loss": []},
        "timing": {
            "wl_loss_time": wl_time,
            "overlap_loss_time": overlap_time,
            "density_loss_time": density_time,
            "backward_time": backward_time,
            "optimizer_time": optimizer_time,
            "projection_time": projection_time,
            "projection_count": projection_count,
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

    config = dict(config) if config else {}
    cell_features, pin_features, edge_list, _runtime_device, _runtime_reason = move_runtime_tensors(
        cell_features, pin_features, edge_list, config=config, verbose=verbose
    )

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

    Uses a size-aware strategy mix so we keep the helpful diversity without
    paying the runtime cost of expensive inits on large designs.
    Returns the result with lowest WL (that has 0 overlap).
    """
    from placement import calculate_normalized_metrics

    config = dict(config) if config else {}
    cell_features, pin_features, edge_list, _runtime_device, _runtime_reason = move_runtime_tensors(
        cell_features, pin_features, edge_list, config=config, verbose=verbose
    )

    N = cell_features.shape[0]
    best_result = None
    best_wl = float("inf")

    strategies = [("greedy_legal", cell_features.clone(), {})]

    exhaustive = config.get("exhaustive_multistart", False) if config else False

    if exhaustive:
        from ashvin.constructive import island_init
        from ashvin.init_placement import spectral_placement

        island_cf = cell_features.clone()
        island_init(island_cf, pin_features, edge_list, config=config, verbose=verbose)
        strategies.append(("island_init", island_cf, {}))

        spectral_cf = cell_features.clone()
        spectral_placement(spectral_cf, pin_features, edge_list)
        strategies.append(("spectral", spectral_cf, {}))

    # Tiny designs benefit most from more diverse starts, and island init is cheap enough.
    elif N <= 40:
        from ashvin.constructive import island_init
        from ashvin.init_placement import spectral_placement

        island_cf = cell_features.clone()
        island_config = dict(config) if config else {}
        island_config.setdefault("coarse_epochs", 400)
        island_init(island_cf, pin_features, edge_list, config=island_config, verbose=verbose)
        strategies.append(("island_init", island_cf, {}))

        spectral_cf = cell_features.clone()
        spectral_placement(spectral_cf, pin_features, edge_list)
        strategies.append(("spectral", spectral_cf, {}))

    # Mid-size designs: force-directed init is much cheaper than island init and
    # sometimes beats plain random starts after legalization/refinement.
    elif N <= config.get("force_directed_max_cells", 300):
        from ashvin.init_placement import force_directed_init

        force_cf = cell_features.clone()
        force_iters = config.get("force_iterations", 20) if config else 20
        force_directed_init(force_cf, pin_features, edge_list, iterations=force_iters)
        strategies.append(("force_directed", force_cf, {}))

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
