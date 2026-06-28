# Infer-layout rewrite: post-implementation bugs

**Date:** 2026-06-27
**Prereq:** [infer-layout-simplification](2026-06-27-infer-layout-simplification.md)

36 tests fail after the rewrite. 17 are FileCheck tests checking old IR
structure (need assertion updates, not pass fixes). 11 are pre-existing.
The remaining 8 are real bugs described below.

## FileCheck test assertions ✅ Fixed

17 tests in `tests/passes/infer_layout/` checked for old-style
`nkipy.layout` with a `tile_size` attribute. Updated to check for
`nkipy.tile_op` with `loop_tile_size` instead.

## Bug 1: Tile propagation doesn't work ✅ Fixed

**Affects:** test_feedforward_sbuf_compact_silu, test_rope, test_rope_3d_compound
**Error:** `[LegalizeLayout] Error: inconsistent tile sizes`

### What the user expects

```python
# "I annotate the result; intermediates should tile the same way"
gated = gate / (1.0 + np.exp(-gate)) * up
knob.knob(gated).tile_op(tile_size=[128, 128])
```

The full expression (neg, exp, add, reciprocal, mul) shares SBUF buffers
with `gated`. All ops must use the same tile because SBUF allocs have one
physical layout. The user's annotation should propagate backward through
the chain.

### What actually happens

After tracing, each op gets its own output alloc (linalg DPS on memrefs):

```mlir
%a     = memref.alloc() : memref<256x256xf32>
%b     = memref.alloc() : memref<256x256xf32>
%c     = memref.alloc() : memref<256x256xf32>
%d     = memref.alloc() : memref<256x256xf32>
%e     = memref.alloc() : memref<256x256xf32>
%gated = memref.alloc() : memref<256x256xf32>

linalg.negf       ins(%gate)     outs(%a)      ← default tile [128, 256]
linalg.exp        ins(%a)        outs(%b)      ← default tile [128, 256]
linalg.add        ins(%cst, %b)  outs(%c)      ← default tile [128, 256]
linalg.reciprocal ins(%c)        outs(%d)      ← default tile [128, 256]
linalg.mul        ins(%gate, %d) outs(%e)      ← default tile [128, 256]
linalg.mul        ins(%e, %up)   outs(%gated)  ← user tile [128, 128]
```

All intermediates are assigned SBUF by `defaultLayouts`. The conflict is on
`%e`: the first mul writes it with tile [128, 256], but the second mul (user-
annotated) reads it with tile [128, 128]. LegalizeLayout needs a single
physical tile shape per SBUF alloc, so it fails.

(HBM allocs don't have this constraint — LegalizeLayout only tiles SBUF.)

### Why propagation doesn't fire

`propagateTileOps` only copies a tile to a producer that has NO tile yet.
But `defaultTileOps` already gave every op a tile. And even if we swap the
order, `func.walk` visits top-down — only one hop propagates before the
walk moves past the earlier ops.

### Fix

Run propagation first, in reverse program order:

```cpp
void runOnOperation() override {
  propagateTileOps(func);  // user tiles flow backward through chains
  defaultTileOps(func);    // fill whatever's still unannotated
  defaultLayouts(func);
}

void propagateTileOps(func::FuncOp func) {
  SmallVector<linalg::LinalgOp> ops;
  func.walk([&](linalg::LinalgOp op) { ops.push_back(op); });

  // Reverse iteration: consumer visited before producer, so tile
  // propagates through the entire chain in one pass.
  for (auto it = ops.rbegin(); it != ops.rend(); ++it) { ... }
}
```

## Bug 2: Transpose allocs wrongly assigned SBUF

**Affects:** test_head_deconcat, test_qwen3_layer
**Errors:** `tile shape rank 2 but permutation has 4 dimensions` /
`tile rank 2 != alloc rank 3`

### What happens

```mlir
%transposed = linalg.transpose(%input) permutation=[0,2,1,3]
              → output: memref<2x128x2x128xf32>
```

`defaultLayouts` sees this alloc isn't a return value → assigns
`Sbuf, partition_dim=0`. But:
- dim 0 is batch (size 2), not a partition dim
- SBUF produces rank-2 tiles, but the transpose permutation is rank-4
- The alloc needs SharedHbm because it gets reshaped to 2D downstream

Also `isAnnotatableOp` doesn't include `linalg::TransposeOp`, so the
transpose tile_op code (lines 138-149) is dead.

### Fix

1. Add `linalg::TransposeOp` to `isAnnotatableOp`
2. In `defaultLayouts`: high-rank (>2D) transpose outputs → SharedHbm

## Bug 3: HBM-to-HBM DMA transpose

**Affects:** test_sigmoid_partition_dim_1, test_exp_partition_dim_1
**Error:** `neuronx-cc` exit code 70

### What happens

```mlir
%result = alloc : memref<128x64xf32>     ← SharedHbm (return value, correct)
%staging = alloc : memref<64x128xf32>    ← SharedHbm (defaultLayouts assigns this)

compute → %staging
dma_transpose %staging → %result         ← HBM-to-HBM: hardware can't do this
```

DMA transpose requires at least one operand in SBUF. The old pass assigned
SBUF to `%staging`. The new pass sees it's not a return value and not a
matmul output, but still gives it SharedHbm (??? — actually it should give
Sbuf by the current logic; need to investigate why it doesn't).

### Fix

The staging alloc for `canonicalize-partition-dim` transposing copies should
be SBUF. Either:
- `canonicalize-partition-dim` marks it SBUF when it creates the alloc, or
- `defaultLayouts` correctly assigns it (may already be correct if the
  alloc has no memspace — need to verify what's actually happening here)

## Bug 4: Multi-output naming (unrelated)

**Affects:** test_qkv_projection
**Error:** `nki.output_names has 1 entries but output index is 1`

Kernel returns 3 values but `nki.output_names = ["output"]`. Tracing bug,
not infer-layout.

## Summary

| Bug | Tests | Fix |
|-----|-------|-----|
| 1: propagation ordering | 3 | reverse-walk propagation before defaults |
| 2: transpose → SBUF | 2 | assign SharedHbm + fix isAnnotatableOp |
| 3: HBM-HBM DMA | 2 | staging alloc must be SBUF |
| 4: output naming | 1 | tracing layer (separate) |
| FileCheck updates | 17 | update test assertions |
| Pre-existing | 11 | not caused by rewrite |
