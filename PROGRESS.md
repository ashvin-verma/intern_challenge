# Progress Log

## Run 0: Baseline — No Overlap Loss (2026-03-21)

**Heuristic:** None. `overlap_repulsion_loss()` returns constant `1.0` — a placeholder with no connection to cell positions. The optimizer only minimizes wirelength via `wirelength_attraction_loss()`.

**Hyperparameters:** 1000 epochs, Adam lr=0.01, lambda_wl=1.0, lambda_overlap=10.0

**Why this is the expected result:** The combined loss is `1.0 * wl_loss + 10.0 * 1.0`. The constant `10.0` contributes zero gradient (`d(constant)/d(positions) = 0`), so Adam only sees `d(wl_loss)/d(positions)`. Wirelength loss pulls connected cells together with no opposing force, causing cells to cluster and overlap.

| Test | Cells | Macros | Overlap | Norm WL | Time (s) |
|------|-------|--------|---------|---------|----------|
| 1    | 22    | 2      | 0.9091  | 0.3435  | 16.16    |
| 2    | 28    | 3      | 0.8929  | 0.3450  | 0.62     |
| 3    | 32    | 2      | 0.9375  | 0.3492  | 0.59     |
| 4    | 53    | 3      | 0.8302  | 0.3866  | 0.93     |
| 5    | 79    | 4      | 0.9367  | 0.4173  | 0.82     |
| 6    | 105   | 5      | 0.7429  | 0.3443  | 0.83     |
| 7    | 155   | 5      | 0.7548  | 0.3403  | 0.90     |
| 8    | 157   | 7      | 0.8662  | 0.3784  | 0.89     |
| 9    | 208   | 8      | 0.6394  | 0.3787  | 0.87     |
| 10   | 2010  | 10     | 0.7846  | 0.3441  | 1.82     |

**Avg overlap: 0.8294 | Avg WL: 0.3627 | Total time: 24.43s**

**Observations:**
- Overlap is uniformly high (64-94%) — cells cluster to minimize wiring.
- Tests 6, 9 have slightly lower overlap (74%, 64%). These have more macros relative to std cells (5/105, 8/208). Macros occupy large area so the initial random spread has more spacing. But without repulsion this is just chance.
- WL is relatively low (0.34-0.42) because the optimizer freely overlaps cells to shorten wires.
- Runtime is fast (~0.5-2s per test after warmup) because the placeholder overlap loss is O(1).

**CSV:** `ashvin/results/` — not saved (pre-instrumentation run).

---

## Run 1: Naive N×N Overlap Loss (2026-03-21)

**Heuristic:** Pairwise overlap area via broadcasting. For each pair (i, j):
```
overlap_x = relu((wi + wj)/2 - |xi - xj|)
overlap_y = relu((hi + hj)/2 - |yi - yj|)
overlap_area = overlap_x * overlap_y
loss = sum(overlap_area for all i<j) / (N*(N-1)/2)
```

This creates N×N tensors for dx, dy, min_sep_x, min_sep_y, overlap_x, overlap_y, overlap_area. Upper triangle mask selects i<j pairs.

**Hyperparameters:** Same as baseline (1000 epochs, Adam lr=0.01, lambda_wl=1.0, lambda_overlap=10.0)

**Why this should work:** `torch.relu()` is differentiable — gradient is 1 where overlap > 0, 0 otherwise. For overlapping pair (i,j), the gradient pushes xi and xj apart (and yi, yj apart) proportional to the overlap magnitude. With lambda_overlap=10.0, the repulsion force is 10× the wirelength attraction per unit gradient.

| Test | Cells | Macros | Overlap | Norm WL | Time (s) | Overlap Loss (s) | Backward (s) |
|------|-------|--------|---------|---------|----------|-------------------|---------------|
| 1    | 22    | 2      | 0.4091  | 0.5036  | 13.60    | 0.17              | 0.82          |
| 2    | 28    | 3      | 0.6429  | 0.4124  | 0.94     | 0.15              | 0.48          |
| 3    | 32    | 2      | 0.5000  | 0.6023  | 0.91     | 0.14              | 0.46          |
| 4    | 53    | 3      | 0.6038  | 0.4607  | 1.14     | 0.18              | 0.58          |
| 5    | 79    | 4      | 0.6076  | 0.5398  | 1.09     | 0.17              | 0.55          |
| 6    | 105   | 5      | 0.6476  | 0.4323  | 1.18     | 0.21              | 0.60          |
| 7    | 155   | 5      | 0.7097  | 0.3982  | 1.48     | 0.29              | 0.74          |
| 8    | 157   | 7      | 0.6815  | 0.4341  | 1.55     | 0.32              | 0.77          |
| 9    | 208   | 8      | 0.6202  | 0.4094  | 1.80     | 0.41              | 0.94          |
| 10   | 2010  | 10     | 0.8164  | 0.3486  | 67.84    | 30.79             | 33.86         |

**Avg overlap: 0.6239 | Avg WL: 0.4541 | Total time: 91.53s**

**CSV:** `ashvin/results/20260321_152039_naive_overlap.csv`

**Change from baseline:**
- Overlap: 0.8294 → 0.6239 (**-25%** relative reduction)
- WL: 0.3627 → 0.4541 (**+25%** worse — expected tradeoff: cells spread out to reduce overlap, increasing wire lengths)

**Observations:**

1. **Overlap reduced but far from zero.** The old leaderboard leaders achieved 0.0000 overlap. With only 1000 epochs and default hyperparameters, the optimizer hasn't converged. The loss function works (gradient signal exists) but is underpowered.

2. **Overlap gets worse with more cells.** Test 10 (2010 cells): 0.8164 overlap vs test 1 (22 cells): 0.4091. Two compounding factors:
   - **Normalization dilution:** We divide by N*(N-1)/2 pairs. For N=2010, that's ~2M pairs. Most pairs are distant and contribute zero loss. The average loss per pair is tiny, producing weak gradients. For N=22, only 231 pairs — each overlapping pair has 4× more influence on the gradient.
   - **Computational budget:** 1000 epochs is fixed regardless of problem size. Larger problems need more iterations to resolve all overlaps.

3. **O(N²) scaling is visible in timing.** Overlap loss time scales as expected:
   - N=22→105: 0.14-0.21s (negligible)
   - N=155→208: 0.29-0.41s (growing)
   - N=2010: 30.79s (**150× more than N=208**, consistent with (2010/208)² ≈ 93× — the extra factor is from backward pass also scaling O(N²))

4. **Backward pass dominates.** For test 10: overlap_loss=30.8s, backward=33.9s, wl_loss=0.86s, optimizer=0.42s. The backward pass through the N×N overlap computation is as expensive as the forward pass. Total training: 66s out of 67.8s elapsed.

5. **Memory scaling makes tests 11-12 impossible:**
   - N=2010: 7 tensors × 2010² × 4 bytes ≈ 108 MB (fine)
   - N=10010: 7 × 10010² × 4 ≈ 2.7 GB (tight on 12GB RTX 3080 Ti)
   - N=100010: 7 × 100010² × 4 ≈ 267 GB (impossible)

**What needs to change to reach 0.0000 overlap:**
- **Hyperparameter tuning:** Higher lambda_overlap, more epochs, LR scheduling. The leaderboard notes suggest "cosine annealing on LR with warmup" and "increase lambda_overlap" work.
- **Better normalization:** Instead of dividing by all pairs, normalize by overlapping pairs or total area. This prevents gradient dilution at large N.
- **Scalable overlap computation (Task 2):** Spatial hashing to avoid O(N²). Required for tests 11-12.

---

## Run 2: Scalable Spatial Hash Overlap (2026-03-21)

**Heuristic:** Two-tier spatial hashing with pair caching.
- Tier 1 (macro): exhaustive — all C(M,2) macro-macro pairs + vectorized macro-stdcell filter
- Tier 2 (std-std): uniform grid, bin_size=3.0, 3×3 neighbor lookup via forward-neighbor pattern
- Candidate pairs cached and rebuilt every 50 epochs
- Normalization: `sum(overlap_areas) / N` instead of `/N*(N-1)/2`

**Hyperparameters:** Same (1000 epochs, Adam lr=0.01, lambda_wl=1.0, lambda_overlap=10.0)

| Test | Cells  | Overlap | Norm WL | Time (s) | Overlap Loss (s) |
|------|--------|---------|---------|----------|------------------|
| 1    | 22     | 0.2273  | 0.5133  | 14.38    | 0.18             |
| 2    | 28     | 0.6429  | 0.4159  | 0.97     | 0.16             |
| 3    | 32     | 0.4375  | 0.6350  | 0.98     | 0.17             |
| 4    | 53     | 0.5094  | 0.4655  | 1.13     | 0.15             |
| 5    | 79     | 0.5570  | 0.5656  | 1.23     | 0.20             |
| 6    | 105    | 0.5810  | 0.4491  | 1.23     | 0.22             |
| 7    | 155    | 0.4516  | 0.4374  | 1.42     | 0.33             |
| 8    | 157    | 0.3885  | 0.4654  | 1.45     | 0.29             |
| 9    | 208    | 0.2500  | 0.4545  | 1.81     | 0.44             |
| 10   | 2010   | 0.7567  | 0.3994  | 3.02     | 0.81             |
| 11   | 10010  | 0.6361  | 0.3897  | 15.53    | 6.92             |
| 12   | 100010 | 0.6488  | 0.3838  | 392.04   | 248.21           |

**Avg overlap (tests 1-10): 0.4802 | Avg WL: 0.4801 | Total time: 27.62s**

**CSV:** `ashvin/results/20260321_161617_scalable_cached.csv` (tests 10-11), `20260321_162329_scalable_cached_t12.csv` (test 12)

**Change from Run 1:**
- Overlap: 0.6239 → 0.4802 (**-23%** on tests 1-10). The `/N` normalization provides stronger gradients than `/N*(N-1)/2`.
- Test 10: 67.84s → 3.02s (**22× faster**). Overlap loss: 30.79s → 0.81s.
- Tests 11-12 now run for the first time. Test 12 (100K cells) completes in 392s — previously impossible (needed 267GB memory).

**Observations:**

1. **Normalization matters more than expected.** The `/N` normalization (vs `/N*(N-1)/2`) improved overlap across all tests, even small ones (test 1: 0.41→0.23). This is because `/N` gives each cell a constant "repulsive budget" regardless of N, while `/N*(N-1)/2` dilutes the signal quadratically.

2. **Pair caching is essential.** Without caching (Run test 11 before fix): 285s. With 50-epoch rebuild: 15.5s (**18× speedup**). The pair generation Python loop costs ~0.5s for 10K cells and ~12s for 100K cells — running it every epoch dominates training time.

3. **Test 12 bottleneck: pair generation + backward.** 248s overlap loss (pair rebuilds × 20) + 103s backward. The backward pass scales with number of candidate pairs, not N². Next optimization: reduce rebuild frequency or vectorize pair generation.

4. **Backward pass scales well.** For test 10: 1.70s (from 33.86s with naive). The scalable approach builds a computation graph proportional to P (candidate pairs) not N², so autograd is efficient.

5. **Evaluation works for all tests.** Test 11: 0.28s eval, Test 12: 9.38s eval. Previously these were skipped.

**What's still needed:**
- Hyperparameter tuning (the actual challenge — reaching 0.0000 overlap)
- Faster pair generation for 100K cells (vectorize the Python loop or use GPU binning)
- Macro-first placement strategy (PLAN.md Task 4)

---

## Summary Table

## Run 3: Density Penalty (lambda_density=1.0) (2026-03-21)

**Heuristic:** Added bilinear density loss as auxiliary term. Each cell's area is distributed to 4 surrounding bins via bilinear interpolation weights (differentiable). Bins exceeding uniform target density get penalized. Pushes cells from crowded regions toward empty space.

**Hyperparameters:** 1000 epochs, Adam lr=0.01, lambda_wl=1.0, lambda_overlap=10.0, **lambda_density=1.0**, bin_size=10.0

| Test | Cells | Overlap (Run 2) | Overlap (Run 3) | Norm WL | Time (s) |
|------|-------|-----------------|-----------------|---------|----------|
| 1    | 22    | 0.2273          | 0.2273          | 0.5132  | 14.57    |
| 2    | 28    | 0.6429          | **0.6071**      | 0.4104  | 1.43     |
| 3    | 32    | 0.5000          | **0.3750**      | 0.6344  | 1.40     |
| 4    | 53    | 0.5094          | **0.4528**      | 0.4629  | 1.41     |
| 5    | 79    | 0.5570          | **0.5443**      | 0.5644  | 1.51     |
| 6    | 105   | 0.5810          | **0.5619**      | 0.4490  | 1.64     |
| 7    | 155   | 0.4516          | **0.4323**      | 0.4368  | 1.83     |
| 8    | 157   | 0.3885          | **0.4204**      | 0.4663  | 1.83     |
| 9    | 208   | 0.2500          | **0.2452**      | 0.4499  | 2.31     |
| 10   | 2010  | 0.7567          | **0.7542**      | 0.3995  | 3.44     |

**Avg overlap: 0.4621 (vs 0.4802) | Avg WL: 0.4787 | Total time: 31.35s**

**CSV:** `ashvin/results/20260321_..._density_v1.csv`

**Observations:**

1. **Small but consistent improvement.** Overlap improved on 8/10 tests (worsened on test 8). Average overlap: 0.4802 → 0.4621 (**-3.8%**). The density pressure helps cells find empty space rather than drifting into other clusters.

2. **Wirelength roughly unchanged.** 0.4801 → 0.4787. The density term doesn't fight wirelength significantly — it just redirects cells to less crowded areas.

3. **Diminishing returns.** The improvement is modest because the density bin_size (10.0) is much larger than std cells (width 1-3). The density field is too coarse to resolve individual cell overlaps — that's the overlap loss's job. The density term's value is in preventing macro-scale clustering.

4. **Runtime overhead acceptable.** 27.62s → 31.35s (+13%). The density loss is O(N) — negligible compared to overlap loss pair generation.

**Next:** The density term helps marginally. The real bottleneck to reaching 0.0000 overlap is hyperparameter tuning (more epochs, higher lambda_overlap, LR scheduling) and macro-first placement (Task 4). The density term is a supporting actor, not the lead.

---

## Summary Table

| Run | Heuristic | Avg Overlap | Avg WL | Total Time | Tests Run |
|-----|-----------|-------------|--------|------------|-----------|
| 0   | None (placeholder) | 0.8294 | 0.3627 | 24.43s | 1-10 |
| 1   | Naive N×N overlap | 0.6239 | 0.4541 | 91.53s | 1-10 |
| 2   | Scalable spatial hash (/N norm) | 0.4802 | 0.4801 | 27.62s | 1-10 |
| 3   | + density penalty (lambda=1.0) | 0.4621 | 0.4787 | 31.35s | 1-10 |
| 4   | Two-stage macro-first | 0.3027 | 0.5051 | 28.13s | 1-10 |
| 5   | + greedy repair pass | **0.0724** | 0.5081 | 34.17s | 1-10 |
| 6a  | Config: default (cosine LR) | 0.1249 | 0.4945 | 51.25s | 1-10 |
| 6b  | Config: aggressive | 0.0859 | 0.5062 | 50.52s | 1-10 |
| 7   | Single-stage annealed solver | 0.0839 | 0.5092 | 69.62s | 1-10 |
| 8   | + deterministic legalization | 0.0093 | 0.5197 | 51.03s | 1-10 |
| 9   | Fixed legalization edge cases | 0.0011 | 0.5200 | 47.99s | 1-10 |
| 10  | + brute-force repair + adaptive epochs | 0.0001 | 0.5200 | ~48s | 1-10 |
| 11  | + macro repair in legalization | 0.0000 | 0.5132 | 40.51s | 1-10 |
| **12** | **+ iterative legalize-repair** | **0.0000** | **0.5132** | **40.51s** | **1-10** |
| 12  | (test 11) | 0.0000 | 0.6064 | 9.61s | 11 |
| 12  | (test 12, 100K cells) | **0.0000** | 0.6492 | 721.77s | 12 |
| 13  | + GD WL polish → re-legalize | 0.0000 | **0.4971** | 45.28s | 1-10 |

**Run 13 notes:** GD polish + cell swaps. WL 0.5132→0.4912.

**Run 14 (optuna v1): 30 trials. Best WL: 0.4544.**

**Run 15 (optuna v3): 100 trials on tests 1,3,5,7,9. Best WL: 0.4091 on all tests.**
Best config: lr=0.003, lambda_wl=3.58, lambda_overlap 1.2→96, beta 0.11→2.03, 500 epochs, warmup_cosine.
Key insight: higher lambda_wl (3.58) + warmup_cosine LR + low overlap start (1.2).

**Run 16 (multi-start): spectral + random init, pick best. WL: 0.4468.** Spectral helps on some tests (3,5,9) but hurts on others.

**Run 17 (barycentric): move cells toward neighbor centroids post-legalization. WL: 0.4538.** Modest help, most moves rejected due to overlap.

**Run 18 (explosive scatter): scatter all positions 1.3-2.0× from centroid, reconverge.** Doesn't help — disrupted solutions don't find better minima.

**Run 19 (targeted scatter): identify top 20% highest-WL edges, move those cells toward neighbor centroids, short re-solve. WL: 0.4015.** Big win! Breaks local WL minima.

**Run 20 (multi-scatter, 3 iterations): WL: 0.3842.** Each iteration finds new hot cells. Best result yet.

**Run 21 (nuclear/SEMF loss): Lennard-Jones and SEMF-inspired potentials. WL: 0.4453.** Negligible impact — redundant with existing WL loss. The attraction term doesn't add information beyond what wirelength_attraction_loss already provides.

**Current best config:** `ashvin/results/best_config.json` + 3 scatter iterations.
| 15  | Optuna v3 (100 trials) | 0.0000 | 0.4091 | ~45s | 1-10 |
| 20  | + multi-scatter (3 iters) | 0.0000 | **0.3842** | ~90s | 1-9 |
| —   | Old leaderboard #1 | 0.0000 | 0.1310 | 11.32s | 1-10 |

**Run 6 notes:** Added config-driven solver with cosine LR + lambda ramping. Cosine LR slightly hurt vs constant. Infrastructure ready for optuna.

**Run 7 notes:** New single-stage annealed solver (`ashvin/solver.py`). Softplus beta anneals 0.1→6.0, lambda_overlap ramps 5→100 over 2000 epochs, warmup LR 100 epochs. 3 tests PASS (1,3,5). Tests 7,8 near-zero (0.013, 0.019). Test 10 (N=2010) still at 0.54 — gradient per cell = overlap/N becomes too weak at large N. Need: N-adaptive lambda, more epochs for large tests, or deterministic legalization (guarantees 0.0).

**Current best per test (across all runs):**
| Test | Best Overlap | Best Run | Notes |
|------|-------------|----------|-------|
| 1    | 0.0000      | Run 5,7  | Solved |
| 2    | 0.0714      | Run 5,7  | 2 cells |
| 3    | 0.0000      | Run 5,7  | Solved |
| 4    | 0.0566      | Run 5,7  | 3 cells |
| 5    | 0.0000      | Run 5,7  | Solved |
| 6    | 0.0381      | Run 5    | 4 cells |
| 7    | 0.0129      | Run 7    | 2 cells |
| 8    | 0.0127      | Run 5    | 2 cells |
| 9    | 0.0288      | Run 5    | 6 cells |
| 10   | 0.4592      | Run 5    | 923 cells — main bottleneck |

**Run 4 details:** Stage A: 500 epochs, lr=0.05, lambda_wl=0.0, lambda_overlap=100, lambda_density=5.0 (macros only). Stage B: 500 epochs, lr=0.01, lambda_wl=1.0, lambda_overlap=10, lambda_density=1.0 (std cells only). Key insight: zero wirelength in Stage A lets macros spread freely; high overlap+density forces separation. Test 1: macros fully separated (both blue). Test 8: 145→38 overlap pairs, multiple macros escaped.

**Plots:** `ashvin/plots/run4_twostage/`
