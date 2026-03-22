# Experiment History

## Baseline — Placeholder Overlap Loss (2026-03-21)

**Config:** Default `train_placement()` params (1000 epochs, Adam lr=0.01, lambda_wl=1.0, lambda_overlap=10.0). `overlap_repulsion_loss()` is a placeholder returning constant 1.0.

| Test | Cells | Overlap | Norm WL | Time (s) |
|------|-------|---------|---------|----------|
| 1    | 22    | 0.9091  | 0.3435  | 16.16    |
| 2    | 28    | 0.8929  | 0.3450  | 0.62     |
| 3    | 32    | 0.9375  | 0.3492  | 0.59     |
| 4    | 53    | 0.8302  | 0.3866  | 0.93     |
| 5    | 79    | 0.9367  | 0.4173  | 0.82     |
| 6    | 105   | 0.7429  | 0.3443  | 0.83     |
| 7    | 155   | 0.7548  | 0.3403  | 0.90     |
| 8    | 157   | 0.8662  | 0.3784  | 0.89     |
| 9    | 208   | 0.6394  | 0.3787  | 0.87     |
| 10   | 2010  | 0.7846  | 0.3441  | 1.82     |

**Averages (tests 1-10):** overlap=0.8294, wl=0.3627, total_time=24.43s

**Notes:** Tests 11 (10K cells) and 12 (100K cells) not run — `calculate_cells_with_overlaps()` uses O(N^2) Python loops, too slow for large designs.

## Naive Overlap Loss — N×N Broadcasting (2026-03-21)

**Config:** Same default params. Implemented `overlap_repulsion_loss()` using pairwise broadcasting: `relu((w1+w2)/2 - |x1-x2|) * relu((h1+h2)/2 - |y1-y2|)`, upper triangle mask, normalized by pair count.

| Test | Cells | Overlap | Norm WL | Time (s) | Overlap Loss (s) |
|------|-------|---------|---------|----------|-------------------|
| 1    | 22    | 0.4091  | 0.5036  | 13.60    | 0.17              |
| 2    | 28    | 0.6429  | 0.4124  | 0.94     | 0.15              |
| 3    | 32    | 0.5000  | 0.6023  | 0.91     | 0.14              |
| 4    | 53    | 0.6038  | 0.4607  | 1.14     | 0.18              |
| 5    | 79    | 0.6076  | 0.5398  | 1.09     | 0.17              |
| 6    | 105   | 0.6476  | 0.4323  | 1.18     | 0.21              |
| 7    | 155   | 0.7097  | 0.3982  | 1.48     | 0.29              |
| 8    | 157   | 0.6815  | 0.4341  | 1.55     | 0.32              |
| 9    | 208   | 0.6202  | 0.4094  | 1.80     | 0.41              |
| 10   | 2010  | 0.8164  | 0.3486  | 67.84    | 30.79             |

**Averages (tests 1-10):** overlap=0.6239, wl=0.4541, total_time=91.53s

**vs Baseline:** overlap 0.83→0.62 (-25%), but wirelength worse 0.36→0.45 (tradeoff). Test 10 bottleneck: overlap loss 30.8s + backward 33.9s out of 66s training. O(N²) approach unusable for N>2000.

**Next:** Need hyperparameter tuning (more epochs, higher lambda_overlap, LR schedule) and scalable overlap engine for tests 11-12.
