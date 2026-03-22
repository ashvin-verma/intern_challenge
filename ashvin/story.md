# How the placement optimizer works

## The setup

We have N rectangular cells (circuit components) that need to be placed on a 2D chip. Each cell has:
- A center position (x, y) — **this is what we optimize**
- Fixed dimensions (width, height) — these don't change

The optimizer uses **gradient descent** (Adam) to move cells around. There is no neural network. The (x, y) positions are the parameters, just like weights in a neural net.

## Two competing forces

Each iteration, two loss functions compute a scalar penalty from the current positions:

### 1. Wirelength loss (already implemented)

Connected cells should be close together. For every wire (edge) connecting two pins:

```
wirelength = |pin1_x - pin2_x| + |pin1_y - pin2_y|   (Manhattan distance)
```

The gradient of this loss pulls connected cells toward each other. It uses a smooth approximation (`logsumexp`) instead of raw `abs()` for better gradient behavior near zero.

**Evidence:** `placement.py:249-299`. Returns `total_wirelength / num_edges`.

### 2. Overlap loss (placeholder — returns constant 1.0)

The placeholder at `placement.py:359-360`:
```python
return torch.tensor(1.0, requires_grad=True)
```

This is a **constant tensor** — it has no connection to `cell_features`. When PyTorch calls `.backward()`, the gradient of a constant w.r.t. positions is **zero**. The optimizer receives no signal about overlaps.

**Evidence:** From the training loop (`placement.py:428-431`):
```python
total_loss = lambda_wirelength * wl_loss + lambda_overlap * overlap_loss
total_loss.backward()
```
Since `overlap_loss` is a constant w.r.t. `cell_positions`, `d(overlap_loss)/d(cell_positions) = 0`. The optimizer only sees gradients from `wl_loss`, which pulls cells together. Nothing pushes them apart → 83% overlap.

## The training loop

Each of 1000 epochs (`placement.py:412-449`):
1. Zero gradients
2. Compute wirelength loss (pull together)
3. Compute overlap loss (currently: nothing)
4. Sum: `total = 1.0 * wl_loss + 10.0 * overlap_loss`
5. Backpropagate: compute `d(total)/d(positions)` for every cell
6. Clip gradients (max norm 5.0) to prevent explosion
7. Adam step: update each cell's (x, y) by a small amount in the negative gradient direction

The `lambda_overlap=10.0` means overlap is weighted 10× higher than wirelength — **if the overlap loss actually provided a gradient**.

## What we need to implement

A function that:
1. Takes current cell positions and dimensions
2. Computes how much each pair of rectangles overlaps
3. Returns a scalar penalty that is **differentiable** — PyTorch can trace the computation back to `cell_positions` and compute gradients

### Rectangle overlap geometry

Two axis-aligned rectangles overlap when they overlap in BOTH x and y:

```
Cell i at (xi, yi) with width wi, height hi
Cell j at (xj, yj) with width wj, height hj

Gap in x: |xi - xj| - (wi + wj)/2
Gap in y: |yi - yj| - (hi + hj)/2

If both gaps are negative → overlap exists
Overlap amount in x: max(0, (wi + wj)/2 - |xi - xj|)
Overlap amount in y: max(0, (hi + hj)/2 - |yi - yj|)
Overlap area = overlap_x * overlap_y
```

**Evidence:** This matches `calculate_overlap_metrics()` at `placement.py:500-520`, which is the ground-truth evaluator.

### Making it differentiable

We can't use Python `max()` or `if` statements — those don't propagate gradients. Instead:

- `torch.relu(x)` = `max(0, x)` but differentiable (gradient is 1 when x > 0, 0 otherwise)
- `torch.abs(x)` is differentiable everywhere except x=0

So: `overlap_x = torch.relu((wi + wj)/2 - torch.abs(xi - xj))`

When two cells overlap, `overlap_x > 0`, so the gradient flows. It tells each cell: "move apart in x to reduce this overlap." The magnitude tells them how fast.

### Why this produces correct gradients

Consider cell i at x=0 (width=2) and cell j at x=0.5 (width=2):
- `min_sep_x = (2+2)/2 = 2`
- `|xi - xj| = 0.5`
- `overlap_x = relu(2 - 0.5) = 1.5`

The gradient `d(overlap_x)/d(xi)`:
- `d/d(xi) relu(2 - |xi - xj|)` = `d/d(xi) relu(2 - (xj - xi))` (since xi < xj)
- = `d/d(xi) relu(2 - xj + xi)` = `+1` (since the relu is active)

So `xi` gets pushed in the **+x direction** (gradient = +1, optimizer subtracts it → xi decreases... wait). Actually, the optimizer does `xi -= lr * gradient`. The gradient of the loss w.r.t. xi is +1, so xi decreases — moving AWAY from xj? No, xi=0 and xj=0.5, so decreasing xi increases the gap. Correct!

For xj: `d(overlap_x)/d(xj) = -1`, so the optimizer does `xj -= lr * (-1) = xj + lr` — xj increases, also moving away. Both cells repel.

### Broadcasting for all pairs

Instead of a Python loop over all pairs, we use PyTorch broadcasting:

```python
# [N] → [N, 1] and [1, N] → broadcast to [N, N]
dx = torch.abs(x.unsqueeze(1) - x.unsqueeze(0))  # pairwise x-distances
```

This creates an N×N matrix where entry (i,j) is the distance between cells i and j. For N=2010, this is ~4M entries — fine. For N=100K, it's 10 billion entries — impossible (37 GB). That's why Task 2 builds a spatial-hash approach.

### Upper triangle masking

The N×N matrix counts each pair twice: (i,j) and (j,i). Also (i,i) = 0 but wastes compute. We use:

```python
mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
```

This keeps only entries where i < j.

## Summary

| Component | Current state | Effect |
|-----------|--------------|--------|
| Wirelength loss | Working | Pulls connected cells together |
| Overlap loss | Returns constant 1.0 | Zero gradient → no repulsion |
| **After implementation** | Returns sum of overlap areas | Pushes overlapping cells apart |
| Combined loss | `1.0 * wl + 10.0 * overlap` | Balance: close but not overlapping |
