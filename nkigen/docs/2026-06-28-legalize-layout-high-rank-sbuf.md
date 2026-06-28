# LegalizeLayout: high-rank SBUF alloc tiling bug

**Date:** 2026-06-28
**Status:** Open
**Found during:** infer-layout rewrite ([post-implementation](2026-06-27-infer-layout-post-implementation.md), Bug 2)

## Symptom

**Affects:** test_head_deconcat, test_qwen3_layer

**Errors:**
- `'memref.copy' op requires the same shape for all operands`
- `[LegalizeLayout] Error: tile rank 2 != alloc rank 3`

## What happens

User code does reshape + transpose + reshape:

```python
y = x.reshape(2, 2, 128, 128)
z = np.transpose(y, [0, 2, 1, 3])   // → memref<2x128x2x128>
x2 = z.reshape(256, 256)            // → reinterpret_cast to rank 2
result = np.matmul(x2, w)
```

After infer-layout + knob-driven-tiling + canonicalize-reshape:

```mlir
// The transpose output alloc is rank 4, in SBUF (correct — it's an intermediate)
%alloc = memref.alloc() : memref<2x128x2x128xf32, #nkipy.mem<Sbuf>>

// Knob-driven-tiling created loops with tile [1,128,2,1]:
scf.for %i = 0 to 2 {
  scf.for %j = 0 to 128 {
    %src_tile = memref.subview %input[%i, ...] [1,2,128,1] ...
    %dst_tile = memref.subview %alloc[%i, ...] [1,128,2,1] ...
    linalg.transpose ins(%src_tile) outs(%dst_tile) permutation=[0,2,1,3]
  }
}

// canonicalize-reshape inserted a copy (SBUF→HBM) because downstream
// needs a different-rank view:
%view = memref.reinterpret_cast %alloc to memref<256x256xf32, #nkipy.mem<Sbuf>>
%hbm = memref.alloc() : memref<256x256xf32, #nkipy.mem<SharedHbm>>
memref.copy %view, %hbm

// matmul consumes %hbm
```

LegalizeLayout then processes `%alloc` (rank 4, SBUF):

```
Processing SBUF alloc: memref<2x128x2x128xf32>
  Inferred tile: [1,128,2,1], numBlocks=[2,1,1,128]
  Pass completed successfully   // ← claims success but output IR is invalid
```

The rewritten IR has a `memref.copy` where source and dest shapes
don't match. LegalizeLayout restructured the alloc but broke the
copy that canonicalize-reshape had inserted.

For qwen3 (rank-3 alloc `memref<4x128x128>`), LegalizeLayout infers
a rank-2 tile `[128,128]` from consumers behind a `reinterpret_cast`
and fails immediately with `tile rank 2 != alloc rank 3`.

## Root cause

LegalizeLayout doesn't trace through `reinterpret_cast`. When it
processes a high-rank SBUF alloc:
- It doesn't know the `reinterpret_cast` + copy exist as users
- When it rewrites the alloc's structure (inserting block loops), it
  invalidates the shape assumptions of the existing copy

## Fix

LegalizeLayout should skip SBUF allocs that have a `reinterpret_cast`
user. Those allocs are already tiled by knob-driven-tiling (via
tile_op on the linalg op that writes to them). The
`reinterpret_cast` + copy pair handles the transfer to the downstream
consumer. LegalizeLayout has nothing to add — the alloc is already
tile-sized from knob-driven-tiling's subviews.

```cpp
// In findSbufTensorsToLegalize:
for (auto allocOp : sbufAllocs) {
  // Skip allocs that are accessed via reinterpret_cast — they're
  // already tiled by knob-driven-tiling and copied out by
  // canonicalize-reshape.
  bool hasReinterpretCast = llvm::any_of(
      allocOp.getResult().getUsers(), [](Operation *user) {
        return isa<memref::ReinterpretCastOp>(user);
      });
  if (hasReinterpretCast) continue;
  ...
}
```
