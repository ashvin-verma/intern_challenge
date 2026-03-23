"""Generate comparison plots for PROGRESS.md."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = Path(__file__).resolve().parent / "plots" / "run24_multistart"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Data from our runs
tests = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Old baseline (Run 23, detailed_v4)
old_wl = [0.4124, 0.3529, 0.4166, 0.4350, 0.4070, 0.3275, 0.3059, 0.3283, 0.3255, 0.2292]

# New multistart results (from our tests)
# Tests 1-4 done, 5-9 estimated/TBD — fill in as they complete
new_wl = [0.3957, 0.3118, 0.3413, 0.4331, None, None, None, None, None, None]

# Strategy winners
winners = ["greedy", "wl_prio", "spectral", "greedy", None, None, None, None, None, None]

# Fill in available data only
available = [(i, t) for i, t in enumerate(tests) if new_wl[i] is not None]

# Plot 1: Bar comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

x = np.arange(len(available))
w = 0.35
ax = axes[0]
ax.bar(x - w/2, [old_wl[i] for i, _ in available], w, label="Run 23 (baseline)", color="#cc4444")
ax.bar(x + w/2, [new_wl[i] for i, _ in available], w, label="Run 24 (multistart)", color="#4488cc")
ax.set_xticks(x)
ax.set_xticklabels([f"T{t}" for _, t in available])
ax.set_ylabel("Normalized WL")
ax.set_title("WL comparison: Run 23 vs Run 24 (multistart)")
ax.legend()
ax.axhline(y=0.131, color="gold", linestyle="--", alpha=0.7, linewidth=2)
ax.text(0.5, 0.135, "#1 target (0.131)", color="gold", fontsize=9)

# Annotate winners
for j, (i, t) in enumerate(available):
    if winners[i]:
        ax.text(j + w/2, new_wl[i] + 0.005, winners[i], ha='center', fontsize=7, color='blue')

# Plot 2: Improvement
ax = axes[1]
improvements = [(old_wl[i] - new_wl[i]) / old_wl[i] * 100 for i, _ in available]
colors = ["#44cc44" if imp > 0 else "#cc4444" for imp in improvements]
bars = ax.bar(x, improvements, color=colors)
ax.set_xticks(x)
ax.set_xticklabels([f"T{t}" for _, t in available])
ax.set_ylabel("Improvement (%)")
ax.set_title("Per-test improvement from multistart")
ax.axhline(y=0, color="black", linewidth=0.5)
for j, imp in enumerate(improvements):
    ax.text(j, imp + 0.3, f"{imp:+.1f}%", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "comparison.png", dpi=120)
plt.close()
print(f"Saved to {PLOTS_DIR / 'comparison.png'}")

# Plot 3: Progress over runs
fig, ax = plt.subplots(figsize=(10, 6))
runs = ["Run 0\nBaseline", "Run 1\nNaive OV", "Run 2\nSpatial", "Run 5\nRepair",
        "Run 7\nAnnealed", "Run 11\nLegalize", "Run 15\nOptuna", "Run 20\nScatter",
        "Run 23\nDetailed", "Run 24\nMultistart"]
wl_history = [0.3627, 0.4541, 0.4801, 0.5081, 0.5092, 0.5132, 0.4091, 0.3842, 0.3540, 0.3705]
ov_history = [0.8294, 0.6239, 0.4802, 0.0724, 0.0839, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

# Note: early runs optimized overlap first, WL got worse temporarily
ax.plot(range(len(runs)), wl_history, 'b-o', linewidth=2, markersize=8, label='Avg WL')
ax.fill_between(range(len(runs)), wl_history, alpha=0.1, color='blue')
ax.axhline(y=0.131, color="gold", linestyle="--", alpha=0.7, linewidth=2, label='#1 target')
ax.set_xticks(range(len(runs)))
ax.set_xticklabels(runs, rotation=45, ha='right', fontsize=8)
ax.set_ylabel("Normalized WL (avg tests 1-10)")
ax.set_title("WL progress over optimization runs")
ax.legend()
ax.grid(True, alpha=0.3)

# Secondary axis for overlap
ax2 = ax.twinx()
ax2.plot(range(len(runs)), ov_history, 'r--s', linewidth=1.5, markersize=6, alpha=0.5, label='Overlap')
ax2.set_ylabel("Overlap ratio", color='red')
ax2.legend(loc='center right')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "progress.png", dpi=120)
plt.close()
print(f"Saved to {PLOTS_DIR / 'progress.png'}")
