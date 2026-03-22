"""Solver configuration presets.

Each config is a dict that fully specifies a solver run.
Used by run_tests.py --config and optuna tuning.
"""

DEFAULT = {
    # Stage A: macro placement
    "stage_a_epochs": 500,
    "stage_a_lr": 0.05,
    "stage_a_lambda_wl": 0.0,
    "stage_a_lambda_overlap": 100.0,
    "stage_a_lambda_density": 5.0,
    "stage_a_lr_schedule": "cosine",  # "constant" or "cosine"
    "stage_a_overlap_ramp": False,    # ramp overlap weight from 1x to full over epochs

    # Stage B: std cell placement
    "stage_b_epochs": 500,
    "stage_b_lr": 0.01,
    "stage_b_lambda_wl": 1.0,
    "stage_b_lambda_overlap": 10.0,
    "stage_b_lambda_density": 1.0,
    "stage_b_lr_schedule": "cosine",
    "stage_b_overlap_ramp": False,

    # Repair
    "repair_max_iterations": 100,
    "repair_epsilon": 0.01,
}

# Aggressive overlap — prioritize zero overlap over wirelength
AGGRESSIVE_OVERLAP = {
    **DEFAULT,
    "stage_a_lambda_overlap": 200.0,
    "stage_a_epochs": 800,
    "stage_b_lambda_overlap": 50.0,
    "stage_b_lambda_wl": 0.5,
    "stage_b_epochs": 700,
    "stage_b_overlap_ramp": True,
}

# Balanced — try to get good wirelength too
BALANCED = {
    **DEFAULT,
    "stage_a_epochs": 400,
    "stage_b_epochs": 600,
    "stage_b_lambda_overlap": 20.0,
    "stage_b_lambda_wl": 1.0,
    "stage_b_lr_schedule": "cosine",
}

# Strategy A: annealed softplus + lambda ramp + warmup + more epochs
# Inspired by top competitors (Shashank, Brayden, Pawan)
ANNEALED = {
    # Stage A: macros — aggressive separation
    "stage_a_epochs": 500,
    "stage_a_lr": 0.05,
    "stage_a_lambda_wl": 0.0,
    "stage_a_lambda_overlap": 100.0,
    "stage_a_lambda_density": 5.0,
    "stage_a_lr_schedule": "cosine",
    "stage_a_overlap_ramp": True,
    "stage_a_beta_start": 0.1,   # soft early
    "stage_a_beta_end": 4.0,     # sharp late

    # Stage B: std cells — longer, ramped
    "stage_b_epochs": 1500,
    "stage_b_lr": 0.01,
    "stage_b_lambda_wl": 1.0,
    "stage_b_lambda_overlap": 50.0,
    "stage_b_lambda_density": 1.0,
    "stage_b_lr_schedule": "cosine",
    "stage_b_overlap_ramp": True,
    "stage_b_beta_start": 0.1,
    "stage_b_beta_end": 6.0,     # sharper than Stage A

    "repair_max_iterations": 200,
    "repair_epsilon": 0.01,
}

PRESETS = {
    "default": DEFAULT,
    "aggressive": AGGRESSIVE_OVERLAP,
    "balanced": BALANCED,
    "annealed": ANNEALED,
}
