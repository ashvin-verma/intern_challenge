import subprocess

import torch


def _query_nvidia_smi():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    rows = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "utilization": float(parts[1]),
                "memory_used_mb": float(parts[2]),
                "memory_total_mb": float(parts[3]),
            }
        )
    return rows


def choose_runtime_device(config=None):
    """Pick CUDA when available and idle enough, otherwise fall back to CPU."""
    runtime_device = config.get("runtime_device", "auto") if config else "auto"
    if runtime_device != "auto":
        device = torch.device(runtime_device)
        return device, f"forced via runtime_device={runtime_device}"

    cached_device = config.get("_runtime_device") if config else None
    if cached_device:
        return torch.device(cached_device), config.get("_runtime_device_reason", "cached")

    if not torch.cuda.is_available():
        return torch.device("cpu"), "cuda unavailable"

    util_threshold = config.get("gpu_util_threshold", 20.0) if config else 20.0
    mem_used_threshold = config.get("gpu_mem_used_mb_threshold", 1024.0) if config else 1024.0
    mem_fraction_threshold = config.get("gpu_mem_fraction_threshold", 0.10) if config else 0.10

    try:
        gpu_rows = _query_nvidia_smi()
    except Exception:
        gpu_rows = []

    if gpu_rows:
        candidates = []
        for row in gpu_rows:
            mem_total = max(row["memory_total_mb"], 1.0)
            mem_fraction = row["memory_used_mb"] / mem_total
            if (
                row["utilization"] <= util_threshold
                and row["memory_used_mb"] <= mem_used_threshold
                and mem_fraction <= mem_fraction_threshold
            ):
                candidates.append((row["utilization"], mem_fraction, row["memory_used_mb"], row["index"]))

        if candidates:
            _, _, _, gpu_index = min(candidates)
            device = torch.device(f"cuda:{gpu_index}")
            try:
                torch.empty(1, device=device)
                match = next(row for row in gpu_rows if row["index"] == gpu_index)
                reason = (
                    f"cuda:{gpu_index} selected "
                    f"(util={match['utilization']:.0f}%, mem={match['memory_used_mb']:.0f}/{match['memory_total_mb']:.0f}MB)"
                )
                return device, reason
            except Exception as exc:
                return torch.device("cpu"), f"cuda probe failed: {exc}"

        busiest = min(
            gpu_rows,
            key=lambda row: (
                row["utilization"],
                row["memory_used_mb"] / max(row["memory_total_mb"], 1.0),
                row["memory_used_mb"],
            ),
        )
        reason = (
            "gpu busy, using cpu "
            f"(best gpu util={busiest['utilization']:.0f}%, "
            f"mem={busiest['memory_used_mb']:.0f}/{busiest['memory_total_mb']:.0f}MB)"
        )
        return torch.device("cpu"), reason

    try:
        torch.empty(1, device="cuda:0")
        return torch.device("cuda:0"), "cuda available (nvidia-smi unavailable)"
    except Exception as exc:
        return torch.device("cpu"), f"cuda unavailable after probe: {exc}"


def move_runtime_tensors(cell_features, pin_features, edge_list, config=None, verbose=False):
    """Move solver inputs to the selected runtime device and cache the decision."""
    had_cached_device = bool(config and config.get("_runtime_device"))
    cpu_runtime_max_cells = config.get("cpu_runtime_max_cells") if config else None
    if cpu_runtime_max_cells is not None and cell_features.shape[0] <= cpu_runtime_max_cells:
        device = torch.device("cpu")
        reason = f"forced cpu for N<={cpu_runtime_max_cells}"
    else:
        device, reason = choose_runtime_device(config)
    if config is not None:
        config["_runtime_device"] = str(device)
        config["_runtime_device_reason"] = reason

    if verbose and not had_cached_device:
        print(f"  Runtime device: {device} ({reason})")

    if cell_features.device != device:
        cell_features = cell_features.to(device)
    if pin_features.device != device:
        pin_features = pin_features.to(device)
    if edge_list.device != device:
        edge_list = edge_list.to(device)

    return cell_features, pin_features, edge_list, device, reason
