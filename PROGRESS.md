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

**Run 22 (multi-pass pipeline): Compiler-style optimization passes. WL: 0.3695 (tests 1-10).**
Pipeline: legalize → [barycentric → scatter → GD(WL-only, 100ep) → re-legalize] × 3 passes.
Best-so-far tracking with revert. Small improvement over scatter-only (0.3717→0.3695).
Bottleneck: barycentric has O(N²) overlap check, slow on test 10.

**Current best config:** `ashvin/results/best_config.json` + 3 pipeline passes + scatter.
**Run 23 (detailed placement): pair swaps + reinsertion (N≤300). WL: 0.3540 (tests 1-10). 0.0000 overlap.**
Best per-test: test 7=0.3059, test 10=0.2292. Detailed swaps help small/medium tests most.

**Current best avg WL: 0.3540 (tests 1-10), 0.0000 overlap. Rank ~9.**

**Run 24 (multistart + WL-priority legalization): WL improvement on tests 1-4.**
New strategies added:
1. **WL-priority legalization** (`ashvin/wl_legalize.py`): Places cells in WL-priority order (worst-WL first) at barycentric-optimal positions. Beats greedy row-packing on some tests by 12%.
2. **Row reordering** (`ashvin/global_swap.py`): Reorders cells within rows + cross-row reinsertion. Always-legal by construction (compaction after each swap).
3. **SA refinement** (`ashvin/sa_refine.py`): Simulated annealing with Metropolis criterion on legal moves. Small improvement (~0.1%).
4. **Multistart** via `solve_multistart()`: Tries 3 strategies (greedy, wl_priority, spectral) and keeps best. Different strategies win on different tests.

Per-test results (multistart for 1-3, greedy for 4-9):
| Test | N | Old WL | New WL | Change | Strategy |
|------|---|--------|--------|--------|----------|
| 1 | 22 | 0.4124 | 0.3957 | -4.1% | multistart (greedy won) |
| 2 | 28 | 0.3529 | 0.3118 | -11.6% | multistart (wl_priority won) |
| 3 | 32 | 0.4166 | 0.3413 | -18.1% | multistart (spectral won) |
| 4 | 53 | 0.4350 | 0.4331 | -0.4% | multistart (greedy won) |
| 5 | 79 | 0.4070 | 0.4039 | -0.8% | greedy + row reorder |
| 6 | 105 | 0.3275 | 0.3223 | -1.6% | greedy + row reorder |
| 7 | 155 | 0.3059 | 0.3050 | -0.3% | greedy + row reorder |
| 8 | 157 | 0.3283 | 0.3288 | +0.1% | greedy + row reorder |
| 9 | 208 | 0.3255 | 0.3215 | -1.2% | greedy + row reorder |

**Key insights:**
- No single strategy wins all tests. Multistart (greedy + wl_priority + spectral) guarantees we never do worse.
- Biggest wins on tests 1-3 (4-18%) from multistart, noise-level improvements on tests 5-9 from row reordering alone.
- WL-priority legalization places cells in WL-contribution order at barycentric-optimal positions — 12% better than greedy on test 2.
- Spectral init is 18% better on test 3 — the best single improvement.

**Run 24b (optuna v2, 80 trials on tests 1-3):**
Best trial 54: score 0.3739 (avg WL on tests 1-3, down from 0.3823 baseline = 2.2% improvement)
Full eval on tests 1-9: **avg WL = 0.3593** (down from 0.3679 = 2.3% improvement).
Best config saved: `ashvin/results/best_config_v2.json`
Key config changes vs old:
- lambda_wl: 3.58 → **7.51** (doubled! WL matters more)
- lr: 0.003 → **0.001** (lower, more stable)
- warmup_epochs: 200 → **50** (shorter warmup)
- beta_start: 0.11 → **0.43** (start sharper)
- pipeline_passes: 3 → **5** (more refinement)
- lambda_overlap_end: 96.2 → **140.2** (higher final overlap penalty)

Intuition: higher lambda_wl forces optimizer to prioritize WL harder. Lower LR prevents overshooting. More pipeline passes = more legalize-refine cycles.

**Run 24c (multistart + v2 config, tests 1-5):**
| Test | N | Old WL | New WL | Change |
|------|---|--------|--------|--------|
| 1 | 22 | 0.4124 | **0.3813** | -7.6% |
| 2 | 28 | 0.3529 | **0.3187** | -9.7% |
| 3 | 32 | 0.4166 | **0.3335** | -19.9% |
| 4 | 53 | 0.4350 | **0.4321** | -0.7% |
| 5-10 | | (not yet run with multistart + v2) | | |

**Estimated avg WL (tests 1-10): ~0.338** using v2+multistart for tests 1-4, old numbers for 5-10.
Still rank ~9 on old leaderboard. Need 22% more to reach #2 (0.263), 61% more for #1 (0.131).

**Run 25 (cell inflation + anchor loss): All tests improved!**
Two structural changes addressing root cause (GD→legalization WL damage):
1. **Cell inflation** (8%): inflate cell widths/heights during GD so overlap penalty spreads cells further apart. Deflate before legalization → cells have natural gaps → legalization needs minor corrections only.
2. **Anchor loss**: after legalization, GD refinement is tethered to legal positions via `lambda_anchor * ||pos - anchor||^2`. Prevents cells from drifting far from legal state. Next legalization only needs small corrections.

| Test | N | Old WL | New WL | Change |
|------|---|--------|--------|--------|
| 1 | 22 | 0.4124 | **0.3868** | -6.2% |
| 2 | 28 | 0.3529 | **0.3376** | -4.3% |
| 3 | 32 | 0.4166 | **0.3953** | -5.1% |
| 4 | 53 | 0.4350 | **0.4305** | -1.0% |
| 5 | 79 | 0.4070 | **0.4000** | -1.7% |
| 6 | 105 | 0.3275 | **0.3203** | -2.2% |
| 7 | 155 | 0.3059 | **0.3021** | -1.2% |
| 8 | 157 | 0.3283 | **0.3250** | -1.0% |
| 9 | 208 | 0.3255 | **0.3240** | -0.4% |
| **AVG** | | **0.3679** | **0.3580** | **-2.7%** |

With multistart, test 3 reaches **0.3237** (22.3% better, spectral init + inflation + anchor).

**Run 25b (topology-preserving legalization): Mixed.**
Changed legalization to re-center compacted rows at GD centroid instead of always pushing rightward.
Small improvement on tests 1,2,5 (+0.001-0.003), slight regression on tests 3,4 (-0.002-0.003).
The re-centering helps but isn't a game-changer — the cursor-push issue was less severe than expected.

**Run 26 (swap engine): Iterative within-row swaps + cross-row reinsertion.**
New engine (`ashvin/swap_engine.py`) runs up to 20 iterations of targeted cell moves after legalization.
Each move is O(degree) to evaluate. Two move types:
- Within-row swap: exchange cell ordering, recompact (always legal)
- Cross-row reinsertion: remove from source row, insert near barycentric target in dest row

First test on test 1: **0.3868 → 0.3700** (+4.3%). The cross-row reinsertion is effective — 20 moves per iteration.
Tests 2-3: minimal improvement (0-6 swaps found). The pipeline already handles these well.

Note: topology-preserving legalization (re-centering rows at GD centroid) caused REGRESSION on most tests.
Reverted to original left-to-right packing. The original legalization is already topology-preserving
(cells sorted by x within each row). The issue is the cursor push, which inflation partially addresses.

Full suite results (detailed + swap engine):
| Test | N | Orig | Prev | New | vs Prev |
|------|---|------|------|-----|---------|
| 1 | 22 | 0.412 | 0.387 | **0.369** | **+1.8%** |
| 2 | 28 | 0.353 | 0.338 | 0.347 | -1.0% |
| 3 | 32 | 0.417 | 0.395 | 0.402 | -0.7% |
| 4 | 53 | 0.435 | 0.431 | 0.432 | -0.1% |
| 5 | 79 | 0.407 | 0.400 | 0.401 | -0.1% |
| 6 | 105 | 0.328 | 0.320 | 0.321 | -0.1% |
| 7 | 155 | 0.306 | 0.302 | 0.305 | -0.3% |
| 8 | 157 | 0.328 | 0.325 | **0.322** | **+0.4%** |
| 9 | 208 | 0.326 | 0.324 | 0.330 | -0.6% |
| **AVG** | | **0.368** | **0.358** | **0.359** | **-0.1%** |

Cross-row reinsertion helps test 1 (+4.8%) and test 8 (+1.0%). Within-row swaps cause slight regressions elsewhere.
The swap engine currently evaluates only swapped/moved cells' WL, not displaced neighbors — needs fixing for within-row.

**Current best approach:** Cell inflation (8%) + anchor loss (0.1) + v2 optuna config + detailed + swap engine + multistart.

**Visual analysis (`ashvin/plots/legalize_compare/`):**
- Abacus fails on test 3 because it spreads cells across too many rows to minimize displacement — WL skyrockets
- Abacus wins on test 2 because GD positions are already good neighborhoods and Abacus preserves them
- Greedy packing is compact but topology-blind — pushes everything rightward
- Core insight: minimizing displacement ≠ minimizing WL. Need legalization that's WL-aware AND compact.

## Next Phase: Architecture Overhaul (in order)

### Step 1: Interleaved Legalize-GD — TESTED, DOESN'T HELP
Split 500 epochs into 5 rounds of GD(100) → legalize → GD(100, anchored).
Result: -0.2% to -4.8% on all tests. Mid-training legalization disrupts Adam momentum.
The current pipeline (full GD → legalize → anchored-GD-polish × 5) is already the right structure.
**Conclusion:** Interleaving at the GD level doesn't help. The bottleneck is elsewhere.

### Step 2: Legalization-Aware GD — TESTED, DOESN'T HELP
Added sin²(πy) row penalty ramped over last 60% of GD epochs (integrated, not bolted on).
Result: same pattern as everything else — T2 +6.1%, T1 -5%, T3 -1.9%, T4-5 flat.
The row penalty helps Abacus win on T2 (row-aligned GD → better displacement preservation)
but constrains GD exploration on other tests.
**Conclusion:** GD→legalize architecture has a fundamental ceiling ~0.35-0.36. Need Step 3.

### Step 3: Constructive Placement — Island Clustering (user's idea)
Build placement bottom-up via multi-level clustering:
1. **Form islands:** greedily cluster connected cells into small legal blocks (5-10 cells each)
   - Each island is internally packed (no overlaps within)
   - Place cells within island at WL-optimal positions relative to each other
2. **Promote islands to macro-like units:** treat each island as a single large cell
   - Width = island bounding box width, height = island bounding box height
3. **Coarse placement:** place island-macros using force-directed or GD with LR schedule
   - High LR initially for global exploration
   - Low LR for fine-tuning positions
4. **Uncluster:** expand islands back into individual cells
5. **Fine refinement:** local swaps + shifts to polish
No legalization needed — each level starts and stays legal.
LR schedule controls coarse→fine transition.

### Alternative Constructive: Greedy WL-Optimal
1. Sort cells by connectivity degree (most-connected first)
2. Place each cell at WL-optimal position given already-placed cells, snapped to legal row
3. No overlaps by construction (check before placing)
4. Then iterate with local swaps
Fast (O(N * degree)), starts legal, no legalization shock.

### Step 3a: Full Constructive Pipeline — TESTED, DOESN'T HELP
Islands → coarse GD (800 epochs, high overlap penalty) → uncluster → legalize → polish → swaps.
Result: worse than GD pipeline on ALL tests (0.36-0.49 vs 0.30-0.43).
Root cause: coarse GD can't spread islands apart — they're massive overlapping blocks
(visible in plots at `ashvin/plots/constructive/`). Overlap still 76-100% after 800 epochs.
The overlap loss isn't calibrated for island-sized objects.

### Step 3b: Island-clustered INIT for existing GD pipeline
Instead of replacing GD, use island clustering to create better INITIAL positions:
1. Form islands (connected cell clusters)
2. Pack internally (single-row blocks)
3. Coarse-place islands (spread apart)
4. Uncluster → use as init for existing GD pipeline (replaces random init)

This combines the connectivity-aware clustering with the proven GD optimizer.

### Step 3b Results (multistart with island init):
| Test | Orig | Best | Improve | Winner |
|------|------|------|---------|--------|
| 1 | 0.412 | **0.400** | +2.9% | island_init |
| 2 | 0.353 | **0.313** | +11.4% | island_init |
| 3 | 0.417 | **0.311** | +25.3% | spectral |
| 4 | 0.435 | **0.431** | +0.8% | greedy |
| 5 | 0.407 | **0.401** | +1.6% | greedy |

Island init beats random init AND spectral on tests 1-2. Spectral still best on test 3.
Greedy (random init) still best on tests 4-5. No single strategy dominates.

### Future idea: Hub-spoke clustering init
Highest-connectivity cells serve as hubs, less-connected cells as spokes.
Form clusters around hubs, place hub clusters as units.
Different from islands (which grow greedily) — this is degree-centric.

### Init strategy comparison (single pipeline, no multistart):
| T | random | spectral | force_dir | sequential |
|---|--------|----------|-----------|------------|
| 1 | **0.406** | 0.408 | 0.426 | 0.444 |
| 2 | **0.322** | 0.518 | 0.392 | 0.380 |
| 3 | 0.403 | **0.311** | 0.419 | 0.419 |
| 4 | **0.431** | 0.504 | 0.434 | 0.452 |
| 5 | **0.401** | 0.498 | 0.407 | 0.426 |
| 6 | 0.327 | 0.419 | **0.323** | **0.322** |
| 7 | **0.306** | 0.339 | 0.335 | 0.359 |
| **AVG** | **0.371** | 0.428 | 0.391 | 0.400 |

**Random wins 5/7 tests.** GD is robust to random init — connectivity-aware inits
cluster cells too tightly, making overlap resolution harder. Init is NOT the bottleneck.
**Legalization is the bottleneck.** Moving to fix legalization next.

### WL-aware Abacus legalization
Rewrote Abacus to optimize WL instead of displacement. Uses barycentric
target of neighbors as candidate position during cluster merge DP.

**Raw legalization comparison (no pipeline):**
WL-aware Abacus wins 7/9 tests vs greedy, by 1-4% each. Only test 3 loses.

**But in full pipeline: WORSE.** The pipeline passes (anchor GD, barycentric,
scatter) were tuned for greedy output. Abacus produces different positions →
pipeline converges to different (worse) local minima. Tried:
- Both legalizers per call (pick best): unstable, slow
- Abacus first call only: same result
- Abacus as sole legalizer: worse on 7/9 tests

**Conclusion:** The legalizer can't be improved in isolation. The entire
GD→legalize→refine pipeline is co-adapted. Changing one component without
re-adapting the others causes regression. This is the fundamental ceiling
of the bolt-on approach.

### Constructive v2: legal-from-the-start placement
No GD. No legalization. Place cells one-by-one at WL-optimal positions, then swap.
1. Place macros (spread apart)
2. Place std cells by degree (most-connected first, at barycentric target)
3. Swap refinement (cross-row moves, 50 iterations)

Results (with overlap fix — compact after every move):
| T | N | GD pipe | Constr v2 | OV |
|---|---|---------|-----------|-----|
| 1 | 22 | 0.387 | **0.343** | 0.09 |
| 2 | 28 | 0.338 | 0.352 | 0.11 |
| 3 | 32 | 0.395 | **0.359** | 0.06 |
| 4 | 53 | 0.431 | **0.387** | 0.06 |
| 5 | 79 | 0.400 | 0.426 | 0.04 |
| 6 | 105 | 0.320 | 0.337 | 0.05 |
| 7 | 155 | 0.302 | 0.362 | 0.04 |
| 8 | 157 | 0.325 | 0.388 | 0.40 |
| 9 | 208 | 0.324 | 0.346 | 0.03 |

Wins on tests 1, 3, 4 (WL 10-12% better than GD). Loses on larger tests.
Still has residual overlap (3-40%) — needs better compaction.
Runtime is fast (3-42s, vs 30-300s for GD pipeline).

**Key insight:** The constructive approach produces competitive WL on small tests
with zero GD overhead. The swap engine needs more iterations and better moves
to match GD on larger tests. This IS the right architecture — needs refinement.

**Plots:** `ashvin/plots/run24_multistart/`, `ashvin/plots/legalize_compare/`

**What didn't work (new):**
- Position-based cell swaps (global swap): Cells have different widths (1.0-3.0) in packed rows. Swapping positions always creates overlap. Fixed by switching to row-based reordering.
- Graduated row snapping (sin²(πy) penalty during GD): Actually hurt WL by fighting the WL optimization.
- SA refinement: Tiny improvement (<0.1%) because within-row swaps have limited improvement potential after row reordering.
- Row reordering on tests 5-9: 0-1.6% improvement — noise level.

**What's stopping #1 (0.13 WL):**
- Legalization adds 0.05-0.15 WL penalty per application (row packing is connectivity-blind)
- GD gets positions to ~0.25 WL but legalization bumps to ~0.35+
- #1 (Shashank) uses 5+ heuristic passes: constructive init, shelf-based refinement,
  cell swaps targeting specific high-WL edges, barycentric within size groups,
  multiple legalization+polish cycles
- This is fundamentally a different architecture — a compiler with optimization passes
  vs our single GD+legalize+scatter pipeline
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
