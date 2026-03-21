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