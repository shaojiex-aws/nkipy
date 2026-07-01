# High-rank SBUF: LegalizeLayout + NISA emitter fixes

**Date:** 2026-06-28
**Status:** Open
**Affects:** test_head_deconcat, test_qwen3_layer

## The IR these tests produce (after knob-driven-tiling)

```mlir
// User's input reshaped to 4D
%reinterpret_cast = memref.reinterpret_cast %arg0
    : memref<4x128x128xf32, SharedHbm> to memref<2x2x128x128xf32, SharedHbm>

// Transpose output alloc (rank 4, SBUF)
%alloc = memref.alloc() : memref<2x128x2x128xf32, #nkipy.mem<Sbuf>>

// Tiling loop (iterates over identity dims of perm [0,2,1,3])
scf.for %i = 0 to 2 {
  scf.for %j = 0 to 128 {
    // Load source tile from HBM
    %subview = memref.subview %reinterpret_cast[%i,%j,...] [1,2,128,1]
    %tmp = memref.alloc() : memref<1x2x128x1xf32, Sbuf>
    memref.copy %subview, %tmp

    // Transpose into output alloc
    %dst = memref.subview %alloc[%i,%j,...] [1,128,2,1]
    linalg.transpose ins(%tmp) outs(%dst) permutation=[0,2,1,3]
  }
}

// Reshape + copy out to HBM for downstream matmul
%view = memref.reinterpret_cast %alloc to memref<256x256xf32, Sbuf>
%hbm = memref.alloc() : memref<256x256xf32, SharedHbm>
memref.copy %view, %hbm
```

## Problem 1: LegalizeLayout `tileTranspose` rewrites already-tiled transpose (head_deconcat)

The `linalg.transpose` inside the tiling loop was created by
knob-driven-tiling — it's already at tile granularity:

```mlir
scf.for %i = 0 to 2 {
  scf.for %j = 0 to 128 {
    %tmp = memref.alloc() : memref<1x2x128x1xf32, Sbuf>
    %src_slice = memref.subview %reinterpret_cast[%i,%j,...] [1,2,128,1]
    memref.copy %src_slice, %tmp

    %dst_slice = memref.subview %alloc[%i, 0, 0, %j] [1,128,2,1]
    linalg.transpose ins(%tmp) outs(%dst_slice) permutation=[0,2,1,3]
  }
}
```

LegalizeLayout's `tileTranspose` picks this up and tries to rewrite
it (the `allOne` path emits a broken `memref.copy` due to a bad
perm2D heuristic). But it shouldn't touch this op — it's already tiled.

### Root cause

`tileCopyAndTranspose` collects transposes matching:
```cpp
if (needsTiledTransfer(...) || (isSbuf(...) && isSbuf(...)))
```

The `isSbuf && isSbuf` branch catches the user's already-tiled
SBUF-to-SBUF transpose (inside `scf.for`). `tileTranspose` then
breaks it with the perm2D heuristic. But SBUF-to-SBUF transposes
are always already tiled by knob-driven-tiling — the NISA emitter
handles them directly as `nisa.dma_transpose`.

The `needsTiledTransfer` branch catches HBM↔SBUF transposes (matmul
`transpose_a`). These are top-level, NOT inside loops, and DO need
tiling by LegalizeLayout.

### Fix

Goal: remove `tileCopyAndTranspose` entirely from LegalizeLayout.
All tiling happens before it.

**Step 1 ✅: `canonicalize-partition-dim` boundary transposes**

`seedTileSizeAttr` was null with new infer-layout. Fixed: fall back
to `findTileOp(target).getLoopTileSizeAttr()`.

**Step 2 ✅: Matmul `transpose_a`**

Tiling the transpose independently doesn't work. The transpose
output alloc is shared with the matmul — the matmul reads the
full alloc without going through subviews:

K can be > 128 (e.g. matmul `[256,256] x [256,256]` with tile
`[128,128,128]` has K=256 at the block level before the K-tiling
loop subdivides it further, but the transpose happens at the full
K dimension).

Current IR (works, tileTranspose in LegalizeLayout handles it).
`%lhs` is HBM or SBUF (user's choice — if already SBUF, promotion
is a no-op):

```mlir
scf.for %block_m = 0 to M step BLOCK_M {
  %lhs_block = memref.subview %lhs[%block_m, 0] [BLOCK_M, K]
  %alloc = memref.alloc() : memref<Kx256xf32, Sbuf>
  linalg.transpose ins(%lhs_block) outs(%alloc) perm=[1,0]
  linalg.matmul_transpose_a ins(%alloc, %rhs_block) outs(%out_block)
}
```

If we try to tile the transpose with scf::tileUsingSCF({128, 0}):

```mlir
scf.for %block_m = 0 to M step BLOCK_M {
  %lhs_block = memref.subview %lhs[%block_m, 0] [BLOCK_M, K]
  %alloc = memref.alloc() : memref<Kx256xf32, Sbuf>
  scf.for %i = 0 to K/128 {
    %src = memref.subview %lhs_block[0, %i*128] [256, 128]
    %dst = memref.subview %alloc[%i*128, 0] [128, 256]
    linalg.transpose ins(%src) outs(%dst) perm=[1,0]  // writes [128, 256]
  }
  linalg.matmul_transpose_a ins(%alloc, ...) ...      // reads full [K, 256]
}
// LegalizeLayout's traceToLinalgOperands walks %alloc's users:
//   - follows subview → finds linalg.transpose → records shape [128, 256]
//   - finds linalg.matmul_transpose_a directly → records shape [K, 256]
// Two different shapes on the same alloc → "inconsistent tile sizes"
//
// The correct tile for this alloc IS [K, 256] (full size, no blocking).
// traceToLinalgOperands infers [128, 256] from the subview — wrong.
```

The matmul reads `%alloc` directly (not via subview). Any
independent tiling of the transpose creates a conflicting access
pattern on the shared alloc.

Fix: attach `sbuf_tile_size` explicitly to the alloc's `nkipy.layout`
when creating it. LegalizeLayout reads this directly instead of
inferring from consumers. See
[explicit-sbuf-tile-on-layout](2026-06-30-explicit-sbuf-tile-on-layout.md)
— done there (`NkipyTransposeMatmulOp` + `promote_tensor` attach the
explicit `[tileK, tileM]` / `[tileK, tileN]` tile).

Also ✅: deleted the dead `isMemrefMode` flag and the tensor-only
(`transform::TransposeMatmulOp`) path from
`buildMatmulBlockingTransforms`. The pipeline is memref-native (the
tracer always emits `memref` func args), so only the
`nkipy.transpose_matmul` path was ever taken.

**Step 3: Make all HBM↔SBUF copies/transposes reach LegalizeLayout
already tiled**

Goal: nothing untiled reaches LegalizeLayout, so `tileCopyAndTranspose`
becomes dead (step 4 deletes it).

Instrumenting `tileCopyAndTranspose` across the whole suite shows it
only ever tiles **two** kinds of op, both block-granularity staging
buffers, both inside loops:

1. **Matmul LHS transpose + RHS copy-in.** `buildMatmulBlockingTransforms`
   promotes the LHS (transpose) at the block-M level and the RHS (copy)
   at the block-N level — *outside* the inner tile loops. `TileUsingForOp`
   only tiles the `matmul` op itself down to `[128,128]`; the staging
   transpose/copy it inserts around the matmul stay at block size
   (e.g. `256x256`, `sbuf_map blocks:[2,2]`). The matmul *compute* is
   already tiled; only its staging ops are not.
2. **Reshape copy-out.** `canonicalize-reshape` materializes a
   `memref.copy` at a reshape/mem_space boundary (e.g. head_deconcat's
   SBUF→HBM copy-out of the 4D transpose result). It is emitted late
   and full-size, with no `tile_op`.

Everything else (elementwise promote copies, the 4D tiled transpose)
already arrives tile-sized and is skipped by `tileCopyAndTranspose`.

A `256x256` SBUF buffer with `blocks:[2,2]` is physically
`[128,2,2,128]` — 2 blocks in the *partition* dim. SBUF has only 128
partitions, so this cannot be one DMA: it must be a per-block loop.
So the work can't be deleted, only moved earlier (to where tile info
already exists). Two sub-parts:

- **3a — matmul staging (main case, every matmul kernel).**
  Prerequisite: [unify-on-linalg-copy](2026-07-01-unify-on-linalg-copy.md)
  (the RHS copy-in must be a `linalg.copy` so the builtin can tile it;
  `memref.copy` has no `TilingInterface`). Once it is, tile the LHS
  transpose and RHS copy-in with the builtin `emitTile` at leaf tile —
  same path the transpose knob already uses — no hand-rolled loop.

  Note: explicit-sbuf-tile already set the `tile_size=[128,128]`
  *attribute* on these buffers; it did not split the ops. 3a splits the
  ops — at exactly that annotated tile (must match, or the per-block
  loop and the buffer's `sbuf_map` disagree). Both LHS and RHS need it.

  What's untiled today (feedforward, first matmul), pre-legalize —
  one full `256x256` transpose and one full `256x256` copy, no loop
  over either (the buffers carry `tile_size=[128,128]`, but the ops
  move all `256x256` at once):

  ```mlir
  scf.for %block_m ... {
    %alloc_12 = memref.alloc() : memref<256x256xf32, Sbuf>
    nkipy.layout(%alloc_12) {tile_size = [128, 128]}
    linalg.transpose ins(%lhs_blk : 256x256, SharedHbm)
                     outs(%alloc_12 : 256x256, Sbuf) perm=[1,0]   // ← untiled
    scf.for %block_n ... {
      %alloc_15 = memref.alloc() : memref<256x256xf32, Sbuf>
      nkipy.layout(%alloc_15) {tile_size = [128, 128]}
      memref.copy %rhs_blk, %alloc_15
        : 256x256, SharedHbm to 256x256, Sbuf                     // ← untiled
      scf.for %tile_m ... scf.for %tile_n ... scf.for %k ...
        linalg.matmul_transpose_a ins(%a:128x128, %b:128x128) ... // compute IS tiled
    }
  }
  ```

  After 3a they should already be `[128,128]` inside block loops, so
  LegalizeLayout has nothing to split.

- **3b — reshape copy-out.** Materialize this copy before
  knob-driven-tiling with a `tile_op` attached, so tiling splits it.
  Home: a new `materializeBoundaryCopies` in InferLayout next to
  `materializeReturnCopies` (which already does alloc + layout + copy +
  copy-the-tile_op), porting `canonicalize-reshape`'s
  `hasMemSpaceConflict` check. Don't move `canonicalize-reshape` wholesale
  — its type-stamping must stay after tiling. (Covers only 3b.)

  What's untiled today (head_deconcat), pre-legalize — the 4D
  transpose result is `reinterpret_cast` to `256x256` and copied to
  HBM in one shot. Unlike 3a there is **no `tile_op`/`tile_size`** on
  this copy at all (it's materialized late by canonicalize-reshape):

  ```mlir
  %alloc = memref.alloc() : memref<2x128x2x128xf32, Sbuf>     // 4D transpose out
  %rc = memref.reinterpret_cast %alloc to sizes: [256,256]
          : ... to memref<256x256xf32, Sbuf>
  %hbm = memref.alloc() : memref<256x256xf32, SharedHbm>
  memref.copy %rc, %hbm : 256x256, Sbuf to 256x256, SharedHbm  // ← untiled, no tile_op
  ```

Do 3a first (highest impact, self-contained in `KnobDrivenTiling.cpp`),
verify LegalizeLayout tiles nothing for a pure-matmul kernel, then 3b.

**Step 4: Remove `tileCopyAndTranspose` from LegalizeLayout**

After 3a+3b, no untiled transposes or copies reach LegalizeLayout.
Remove `tileCopyAndTranspose`, `tileTranspose`, and `tileMemrefCopy`
(and the now-unused `findLayoutForValue`/`getSbufMapFor` helpers if
they have no other users).

## Problem 2: LegalizeLayout rank mismatch (qwen3)

A rank-3 alloc `memref<4x128x128>` is accessed via rank-reducing
subviews (tiling loop iterates over dim 0). `traceToLinalgOperands`
follows through the subview and records the consumer's operand type
shape `[128,128]` (rank 2) as the tile. But the alloc is rank 3.

### Fix

In `traceToLinalgOperands`, when following a `SubViewOp`, record the
subview's **static sizes** (at source rank) instead of following
through and recording the result type shape:

```cpp
if (auto subviewOp = dyn_cast<memref::SubViewOp>(user)) {
    auto staticSizes = subviewOp.getStaticSizes();
    if (llvm::none_of(staticSizes, [](int64_t s) {
          return s == ShapedType::kDynamic; })) {
        SmallVector<int64_t> tileShape(staticSizes.begin(), staticSizes.end());
        results.push_back({linalg::LinalgOp(nullptr), 0, tileShape});
    } else {
        workList.push(subviewOp.getResult());
    }
    continue;
}
```

## Problem 3: NISA emitter `_trace_access` rank mismatch

After fixing problems 1-2, the 4D `linalg.transpose` reaches the
NISA emitter. `_emit_copy` is called for the HBM→SBUF load
(`memref.copy %subview, %tmp`). `_operand_str` on `%subview` calls
`_trace_access` which walks:

```
%subview (rank 4) → %reinterpret_cast (rank 4) → %arg0 (rank 3)
```

Now `offsets` has 4 elements (from rank-4 subview) but `base_shape`
is rank 3 (from `%arg0`). `_linearize_offsets` crashes.

### Fix

`_trace_access` should stop at a rank-changing `reinterpret_cast`.
The `reinterpret_cast` result IS the coordinate system where offsets
are defined — the DMA references this view, not the underlying base.

```python
if op_name == "memref.reinterpret_cast":
    source = op.operation.operands[0]
    if up_ir.MemRefType(source.type).rank != up_ir.MemRefType(base.type).rank:
        break
    base = source
    continue
```

## Problem 4: NISA emitter permutation rank mismatch

After fixing problem 3, `_emit_transpose` emits the 4D permutation
`[0,2,1,3]` but `_operand_str` projected operands to 2D. NISA
requires permutation rank = tile rank.

### Fix

In `_emit_transpose`, reduce 4D permutation to 2D when only 2
non-unit dims are swapped:

```python
src_shape = list(up_ir.MemRefType(src.type).shape)
if len(permutation) > 2:
    non_unit = [i for i, s in enumerate(src_shape) if s != 1]
    if len(non_unit) == 2:
        i0, i1 = non_unit
        permutation = [1, 0] if permutation[i0] == i1 else [0, 1]
    else:
        raise ValueError(...)
```

If result is `[0,1]` (identity after stripping units), emit
`dma_copy` instead of `dma_transpose`.

## Problem 5: NISA emitter sbuf_map tile_str inconsistency

`_operand_str` has two paths for on-chip >2D operands:
- **sbuf_map path**: `par = tile_shape[0]`, `free = product(rest)`
- **non-sbuf_map path**: skip leading 1s, `par = first non-unit`

For the transpose, src (no sbuf_map) gets `2|128` and dst (sbuf_map)
gets `1|256`. They should be consistent — both should skip leading 1s.

### Fix

In the sbuf_map path, skip leading unit dims before computing
`par|free` (same logic as non-sbuf_map path):

```python
elif is_onchip and self._has_sbuf_map(base_type):
    sbuf_map = self._get_sbuf_map(base_type)
    skip = 0
    while skip < len(tile_shape) - 2 and tile_shape[skip] == 1:
        skip += 1
    par = tile_shape[skip]
    free = 1
    for d in tile_shape[skip + 1:]:
        free *= d
    tile_str = f"{par}| {free}"
```
