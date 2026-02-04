# Proma Optimization Log

## Problem
- Proma is ~2x slower than GRPO
- Goal: Make it faster without changing the logic
- Previous attempt: Didn't train (some error)

## Instructions
- Inject changes to dp_actor.py into docker container
- Run `run_code.sh` to test after each change
- Keep going until it works

## Approach
Instead of rebuilding docker image (slow), mount modified dp_actor.py into container at runtime.

## Optimization Attempts

### Attempt 1: Mount modified dp_actor.py
Date: 2026-02-04

Analysis of slowdown in `dp_actor.py`:
1. The `proma_intra_eig` code path (lines 246-283) runs for EVERY linear layer during backward pass
2. It does expensive operations: QR decomposition, matrix multiplications
3. Uses float32 unnecessarily (lines 168-169) - could use bfloat16
4. Multiple `.clone()` calls that may be unnecessary
5. No torch.compile on the heavy computation

Key optimizations (without changing logic):
- Use bfloat16 instead of float32 for intermediate calculations
- Remove unnecessary clones
- Add torch.no_grad() where applicable
- Compile the projection function

Changes made:
1. Modified run_code.sh to mount dp_actor.py into container (avoid rebuild)
2. Changed float32 -> bfloat16 for act_in and g_out (lines 168-169)
3. Removed unnecessary .clone() calls (lines 165-166)
4. Wrapped proma_intra_eig computation in torch.no_grad() (projection basis doesn't need grads)
5. QR still uses float32 internally (required by linalg.qr), then converts back

Running test...

### Test 1 Results - PROMA (with optimizations)
- Step 1: update_actor=10.59s, step=24.65s
- Step 2-4: update_actor=~9.8-10s, step=~22-23s
- proma_intra_reduction_pct: ~10% (algorithm working)
- Training completed successfully!

### Test 2 Results - GRPO (baseline)
- Step 1: update_actor=3.50s, step=17.37s
- Step 2-5: update_actor=~2.5s, step=~15s

## Comparison Summary

| Metric | GRPO | PROMA (optimized) | Ratio |
|--------|------|-------------------|-------|
| update_actor (avg) | 2.5s | 9.8s | 3.9x slower |
| step time (avg) | 15.2s | 22.8s | 1.5x slower |
| Memory | 59.9 GB | 71.5 GB | +12 GB |

The update_actor step is ~4x slower because proma_intra_eig runs QR decomposition
for EVERY linear layer (~196 layers) in the backward pass.

## Attempt 2: Reduce k from 100 to 30

Results with k=30:
- update_actor: ~9.4s (vs 9.8s with k=100) - only 5% faster
- The QR decomposition is NOT the main bottleneck

The bottleneck is likely the per-layer overhead of:
- Python loops (unflatten, seq_grads computation)
- 196 hook calls per backward pass
- Random tensor generation

## Current Status

| Config | update_actor | vs GRPO |
|--------|--------------|---------|
| GRPO | 2.5s | 1x |
| PROMA k=100 | 9.8s | 3.9x slower |
| PROMA k=30 | 9.4s | 3.8x slower |

Training works correctly with optimizations:
- bfloat16 for activations (removed float32 cast)
- Removed unnecessary .clone() calls
- float32 only for QR (required by CUDA)

## Attempt 3: Profiling and targeted optimizations

**Profiling revealed bottlenecks:**
```
Before unflatten fix (total=15.07s):
  setup: 0.506s (3.4%)
  unflatten: 3.809s (25.3%)  <-- BIG!
  seq_grads: 0.579s (3.8%)
  grad_accum: 0.411s (2.7%)
  proma_intra: 6.945s (46.1%)  <-- BIGGEST!
  proma: 2.822s (18.7%)
```

**Fix 1: Optimized unflatten_attention_mask_list**
- Old: Create intermediate tensor, then loop with boolean indexing
- New: Directly slice flat_x using cumulative lengths

**After unflatten fix (total=12.00s, 20% faster!):**
```
  setup: 0.519s (4.3%)
  unflatten: 0.707s (5.9%)  <-- 5.4x faster!
  seq_grads: 0.546s (4.5%)
  grad_accum: 0.430s (3.6%)
  proma_intra: 6.949s (57.9%)
  proma: 2.845s (23.7%)
```

**Fix 2: Wrapped entire hook in torch.no_grad()**
- Previously only proma_intra_eig section had no_grad
- Now all hook computations avoid autograd overhead

**Current Results:**
- update_actor: 10.1s -> 9.5s (vs original ~11.7s)
- Hook overhead: 15s -> 12s (20% faster)

## Attempt 4: Vectorize sequence loops

Date: 2026-02-04

**Goal:** Remove Python loops by using vectorized tensor operations.

**Changes:**
1. Stack seq_grads into 3D tensor `seq_grads_stacked` (num_seqs, d_out, d_in)
2. Replace loop-based grad accumulation with `torch.einsum('n,nij->ij', advantages, seq_grads_stacked)`
3. Vectorize NTK computation using gram matrix
4. Vectorize proma section:
   - Compute seq_grads_normed using batch normalization
   - Replace project_to_complement loops with vectorized dot products
   - Replace project function with vectorized matrix ops

**Dtype handling:** Keep original model dtype (bfloat16) for memory efficiency, convert to float32 only for numerically sensitive ops (QR decomposition, eigenvalue decomps, matrix inversions).

**Results after vectorization:**
```
Hook timings (total=3.75s per 1000 calls):
  setup: 4.6%
  unflatten: 6.3%
  seq_grads: 5.7%
  grad_accum: 2.4%
  proma_intra: 63.1%  <-- Main bottleneck (QR decomposition)
  proma: 17.8% (down from 21.3%)
```

- update_actor: 8.84s (vs 9.14s before, vs 2.5s GRPO)
- Training works correctly, proma_intra_reduction_pct: ~10%

## Attempt 5: Optimize float conversion in proma_intra

Date: 2026-02-04

**Issue:** `act_in.float()` and `g_out.float()` were being called multiple times inside loops.

**Fix:** Convert to float32 once at the start:
```python
act_in_f = act_in.float()
g_out_f = g_out.float()
```

Then use `act_in_f` and `g_out_f` throughout the power iteration and QR computation.

**Results:**
```
Hook timings (total=3.54s per 1000 calls):
  setup: 4.8%
  unflatten: 6.7%
  seq_grads: 6.1%
  grad_accum: 2.5%
  proma_intra: 60.8% (2.155s, down from 2.368s = 9% faster)
  proma: 19.1%
```

- update_actor: 8.57s (down from 8.84s)
- Training works correctly, proma_intra_reduction_pct: ~10%

## Final Status

| Config | update_actor | vs GRPO |
|--------|--------------|---------|
| GRPO | 2.5s | 1x |
| PROMA (original) | ~11.7s | 4.7x slower |
| PROMA (optimized) | ~8.6s | 3.4x slower |

**~27% speedup achieved!** (11.7s -> 8.6s)

## Summary of Optimizations

1. **unflatten_attention_mask_list** - Direct slicing instead of boolean indexing (5.4x faster)
2. **Vectorized grad_accum** - Replace loops with einsum for weighted sums
3. **Vectorized proma section** - Batch operations for seq_grads_normed, NTK, projections
4. **Single float conversion** - Convert act_in/g_out to float32 once instead of multiple times

**Remaining bottleneck:** proma_intra (61%) which is randomized SVD + QR decomposition running on every linear layer (~196 layers). This is inherent to the algorithm and cannot be optimized further without changing the logic.

---
