# Bug: Broadcast with Rank-Mismatched Operands

**Date:** 2026-06-24  
**Status:** Known issue, workaround available

---

## Problem

When a binary elementwise op has operands of different ranks (e.g. `(256,256) + (256,)` for bias-add), the tracer correctly emits a `linalg.generic` with a broadcast indexing map:

```mlir
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
linalg.generic {indexing_maps = [#map, #map1, #map], ...}
  ins(%x : tensor<256x256xf32>, %bias : tensor<256xf32>) ...
```

This is valid linalg IR. However, a later canonicalization pass (likely MLIR upstream `--canonicalize`) rewrites the generic into a named op (`linalg.add`), which requires all operands to have the same rank:

```
error: 'linalg.add' op expected operand rank (1) to match the result rank of indexing_map #1 (2)
```

The canonicalization from `linalg.generic` → `linalg.add` is incorrect when the indexing maps include a broadcast (operand rank < result rank).

## Workaround

Use a leading-1 dimension to keep ranks equal:

```python
# Instead of:
@trace(input_specs=[((256, 256), "f32"), ((256,), "f32")])
def kernel(x, bias):
    return x + bias

# Use:
@trace(input_specs=[((256, 256), "f32"), ((1, 256), "f32")])
def kernel(x, bias):
    return x + bias
```

With `(1, 256)`, both operands are rank-2, so the canonicalization produces `linalg.add` without rank mismatch.

## Root Cause

The upstream MLIR `--canonicalize` pass includes a pattern that rewrites `linalg.generic` to named linalg ops when the body matches. This pattern doesn't account for broadcast indexing maps that reduce the operand rank relative to the result.

## Fix Plan

Either:
1. Disable the `linalg.generic → linalg.add` canonicalization pattern in our pipeline (preferred — we handle generics fine)
2. Insert a rank-expanding reshape before the generic when operand ranks differ, so the canonicalization produces valid IR

Low priority — the workaround is trivial and doesn't affect codegen quality.
