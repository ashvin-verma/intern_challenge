# Goal
Build a strong solver for the partcl intern placement challenge.

Primary metric: overlap ratio = number of cells involved in overlaps / total cells.
Secondary metric: normalized wirelength.

Important constraint:
The test suite includes designs up to 10 macros + 100000 standard cells.
Do NOT use O(N^2) all-pairs overlap tensors except for tiny debugging cases.

# Problem framing
This is a scalable mixed-size overlap-removal problem, not full production PnR.
The best solution will likely be:
1. macro-aware
2. coarse-to-fine
3. spatially local
4. GPU-friendly
5. driven by search over solver schedules, not raw coordinate chromosomes

# Immediate tasks

## Task 1: inspect and instrument
- Read placement.py and test.py.
- Add timing breakdowns for:
  - overlap loss
  - wirelength loss
  - optimizer step
  - total runtime
- Add per-test logging:
  - overlap_ratio
  - num_cells_with_overlaps
  - normalized_wl
  - runtime
- Add seed control and CSV logging.

## Task 2: build a scalable overlap engine
Implement a spatial-hash or uniform-grid overlap candidate generator:
- bin cells by center
- only compare cells in same or neighboring bins
- support macros and std cells
- return candidate pairs
- compute overlap penalties only on candidate pairs

Need both:
- exact overlap metric for evaluation
- differentiable overlap loss for optimization

## Task 3: add a density term
Implement a bin overflow / density penalty:
- accumulate cell area into bins
- penalize overflow above target density
- make it differentiable if practical
- start with a simple smooth penalty

## Task 4: macro-first pipeline
Add a 2-stage solver:
- stage A: place / legalize macros first
- stage B: place std cells given macro anchors
- optional stage C: allow small macro nudges if hot bins remain

For macro placement, try:
- simulated annealing on macro coordinates
- or greedy local search with restarts

## Task 5: hot-bin repair
Implement a local repair pass:
- identify bins with highest overlap / overflow
- collect cells in those bins
- try batch local moves:
  - small translations
  - nearest-low-density-bin snap
  - pair swaps
  - short local reorder
- accept moves that reduce overlap first, wirelength second

## Task 6: outer-loop search
Do NOT use GA over all cell coordinates.
Use evolutionary search over solver parameters and schedules:
- overlap weight schedule
- density weight schedule
- learning rate / temperature schedule
- bin size
- number of repair passes
- macro move radius
- restart count
- stage transition criteria

Represent one candidate as a compact config dict.
Each candidate decodes into a deterministic or semi-deterministic run.

## Task 7: GPU acceleration
Port bottlenecks first:
- binning
- sorting / grouping
- candidate pair generation
- overlap scoring
- batch move scoring

Use PyTorch or Triton if convenient.
Do not port high-level orchestration until kernels matter.

# Experiments to run

## Baseline set
1. repo baseline
2. scalable overlap only
3. scalable overlap + density
4. macro-first + scalable overlap + density
5. macro-first + hot-bin repair
6. outer-loop EA over schedules
7. macro SA + deterministic cell spreading
8. parallel multi-start SA on macro-only state

## Ablations
- no macro-first
- no density term
- no hot-bin repair
- no outer-loop search
- different bin sizes
- different overlap penalties:
  - area
  - squared area
  - softplus on overlap lengths before multiply
- different schedules:
  - fixed
  - ramped overlap weight
  - overlap-first then WL polish

# Acceptance criteria
- Solver handles all benchmark sizes without OOM
- Overlap ratio driven to ~0 on most or all tests
- Runtime remains competitive
- Wirelength improves once overlap is solved

# Guardrails
- Never introduce full NxN tensors for large cases
- Do not use GA over raw coordinates
- Do not spend time on RL or learned policies yet
- Keep every change behind a config flag
- Always run ablations and save results to CSV

# Deliverables
- clean solver code
- config-driven experiment runner
- CSV results
- short notes on what helped, what failed, and why

# Post-algo: competitive analysis & tuning

## Task 8: compile & document what we did
- Write up each heuristic, what worked, what didn't, with numbers
- Clean up PROGRESS.md into a coherent narrative
- Ensure all code is well-organized in ashvin/

## Task 9: competitor analysis
- Download competitor solutions from the old leaderboard PRs (partcleda/intern_challenge)
- Run their solutions through our test suite
- Compare: overlap, wirelength, runtime
- Plot inspections: how do their placements look vs ours?
- Identify what they got right that we missed

## Task 10: new heuristics (informed by competitor analysis + literature)

### Key competitor insights (old leaderboard, all achieved 0.0000 overlap):
- **Annealed softplus** (not ReLU): beta ramps 0.1→4.0. Smooth early, sharp late. Used by top 3.
- **Lambda ramping**: overlap weight 20→200 linear (Shashank) or 4*(e/E)^10 exponential (Brayden)
- **Warmup + cosine LR**: LinearLR 5% warmup then CosineAnnealing. Pawan's 1.74s solution.
- **Deterministic legalization**: row-packing guarantees 0.0000. Marcos, 2.3s for 100K cells.
- **Soft-Coulomb repulsion**: 1/r² global field for spreading (manuhalapeth, WL=0.2630)
- **Cell swaps on high-WL edges**: Shashank's WL secret (0.1310)

### Strategies to implement:

**Strategy A: Annealed activation + lambda ramp + more epochs**
- Replace ReLU with annealed softplus/GELU/leaky-ReLU (try all three)
- Ramp lambda_overlap from 10% to 100% over training
- Warmup LR (5%) + cosine decay
- Double epochs (1000 per stage → 2000 total or more)
- This is the common thread across ALL zero-overlap competitors

**Strategy B: Simulated annealing for macro placement (Stage A replacement)**
- SA naturally maximizes entropy → spreads macros apart
- Perturbation: random macro translations, swaps
- Energy = overlap_area + alpha * wirelength
- Temperature schedule: high→low over iterations
- Accept worse moves probabilistically → escapes local minima
- Literature: TimberWolf (Sechen 1986), Dragon (Wang+ 2000) use SA for macro placement
- Our current gradient descent on 10 macros gets stuck; SA explores better

**Strategy C: Deterministic legalization (guarantees 0.0000)**
- After gradient descent + SA, run row-based greedy packing
- Sort cells by x-coordinate, assign to rows, resolve conflicts by shifting
- Handles macros as fixed obstacles, packs std cells around them
- Marcos achieves 100K cells in 2.3s with this approach
- Eliminates need for our current greedy repair (which doesn't guarantee 0)

**Strategy D: WL-aware post-optimization**
- After legalization (overlap = 0 guaranteed), optimize wirelength
- Cell swaps: for each high-WL edge, try swapping endpoints with neighbors
- Barycentric refinement: move each cell toward weighted center of its connected cells
- Accept moves only if overlap stays at 0
- This is where Shashank gets 0.1310 WL vs everyone else's 0.26+

### Implementation order:
1. Strategy A first (quick win, config changes only)
2. Strategy C next (guarantees 0.0000, enables WL optimization)
3. Strategy B if Stage A still fails on some seeds
4. Strategy D last (WL polish, competitive edge)

## Task 11: optuna hyperparameter tuning
- Define search space over: activation type, beta schedule, lambda ramp curve,
  LR + warmup, epochs per stage, bin_size, repair params
- Objective: minimize overlap_ratio, tiebreak on normalized_wl
- Run on tests 1-10 with budget ~100-200 trials
- Also evaluate on alternate seeds (2001-2010) to prevent overfitting
- Apply best config, verify generalization
- Record best config and results

## Task 12: GPU acceleration (originally Task 7)
- Port pair generation to GPU (current bottleneck for 100K cells)
- Vectorize bin assignment + neighbor lookup
- Use torch sorting + searchsorted instead of Python defaultdict
- Target: test 12 under 60s (currently 392s)

# Longer-term plan (playing to win)

## Phase 1: Zero overlap (current priority)
- Strategy A + C should get us to 0.0000 on all tests
- Optuna tunes the exact schedule
- Validate on alternate seeds

## Phase 2: WL optimization (competitive edge)
- Strategy D (cell swaps + barycentric refinement)
- Multi-start: run solver 3-5 times with different seeds, pick best WL
- Edge sampling for large designs (Marcos: 50-80K edges/epoch)

## Phase 3: Scale + speed
- GPU acceleration for 100K cell tests
- Adaptive epoch count by problem size
- Target: all 12 tests under 60s total

## Phase 3.5: Benchmark competitors on tests 11-12
- Run top competitor solutions on the NEW test suite (tests 11-12: 10K and 100K cells)
- Most competitors used O(N²) approaches — they will OOM or timeout on 100K cells
- Document which competitors scale and which don't
- This is our competitive advantage: scalable spatial hash + legalization

## Phase 4: Outlandish ideas (if gap remains)
- Soft-Coulomb repulsion field (manuhalapeth)
- Graph neural network for initial placement prediction
- Force-directed placement with momentum
- Spectral placement (eigenvector-based initial positions)
- Reinforcement learning for schedule selection