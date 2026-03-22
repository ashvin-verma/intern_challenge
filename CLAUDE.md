# CLAUDE.md

## How to Run

This project runs under WSL. From a Windows terminal:

    wsl -d Ubuntu-24.04

Then from the repo root (`/mnt/c/Users/ashvi/Documents/intern_challenge`):

    uv run python test.py                              # upstream test suite (all 12 tests)
    uv run python ashvin/run_tests.py                  # instrumented runner (timing + CSV)
    uv run python ashvin/run_tests.py --tests 1,2,3    # run specific tests
    uv run python ashvin/run_tests.py --tag experiment1 # tag for CSV filename

## Environment

- Python 3.12 (managed by uv)
- PyTorch with CUDA 12.8 (RTX 3080 Ti)
- Package manager: uv (see pyproject.toml)
- OS: WSL Ubuntu 24.04 on Windows 11

## Project Structure

- `placement.py` — challenge code (we implement `overlap_repulsion_loss()` here)
- `test.py` — upstream test harness (DO NOT MODIFY)
- `PLAN.md` — strategic roadmap
- `HISTORY.md` — raw experiment results log
- `PROGRESS.md` — analysis of each run: what worked, why, what to try next
- `ashvin/` — all custom code
  - `ashvin/run_tests.py` — instrumented test runner with CSV output
  - `ashvin/instrumented_train.py` — training wrapper with per-phase timing
  - `ashvin/results/` — CSV output from experiments

## Conventions

1. `placement.py` is modified only for `overlap_repulsion_loss()` (the challenge). `test.py` is read-only.
2. All custom code goes in `ashvin/`.
3. Log every experiment: raw data in `HISTORY.md`, analysis in `PROGRESS.md`.
4. Primary metric: overlap_ratio (lower = better, 0.0 = perfect).
5. Secondary metric: normalized_wl (lower = better).
