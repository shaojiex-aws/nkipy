# High-rank SBUF: LegalizeLayout + NISA emitter fixes

**Date:** 2026-06-28
**Status:** Problem 1 done (Steps 1–4). Problems 2–5 open.
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

Instrumenting `tileCopyAndTranspose` across the suite shows it only ever
tiles **two** kinds of untiled op, both inside loops:

1. **Matmul LHS transpose + RHS copy-in.** `buildMatmulBlockingTransforms`
   promotes the LHS (transpose) at the block-M level and the RHS (copy)
   at the block-N level — *outside* the inner tile loops. `TileUsingForOp`
   only tiles the `matmul` op itself down to `[128,128]`; the staging
   transpose/copy stay at block size (e.g. `256x256`, `blocks:[2,2]`).
2. **Concat / boundary copy-out.** `builder.concatenate` lowers each
   input to a full-size SBUF→HBM copy of a slice of the result.

Everything else (elementwise promote copies, the 4D tiled transpose)
already arrives tile-sized and is skipped by `tileCopyAndTranspose`.

**The tiling principle (it's about contiguity, not the op kind).** The
emitter turns an N-D tile into a 2D `partition x free` DMA by folding the
non-partition dims into one `free` span. That fold is only valid if those
dims are **contiguous** in the buffer: `stride[i] == size[i+1]*stride[i+1]`.
A tile that folds a dim which has a stride gap makes the DMA walk the wrong
addresses and read garbage. This is true for *any* op, not just copies — a
copy is only where it bites, because a copy is the op we insert writing into
a strided slice.

Concrete: concatenating two `128x4x64` halves along the last axis into a
`128x4x128` result. Each half is a strided slice — `stride = [512,128,1]`,
so dim1 steps by 128 but the half is only 64 wide (a 64-element gap where
the other half interleaves):

```mlir
%half0 = memref.subview %out[0,0,0] [128,4,64] : ... to strided<[512,128,1]>

// WRONG — folds dim1(4)·dim2(64) into free=256, but dim1 is NOT contiguous
// (128 != 64), so the DMA strides across the gap into the other half:
linalg.copy ins(%src) outs(%half0)  {loop_tile_size = [128,4,64]}
// RIGHT — tile the non-contiguous dim to 1, loop over it; each step folds
// only the contiguous tail (dim2=64):
linalg.copy ins(%src) outs(%half0)  {loop_tile_size = [128,1,64]}
```

Concat on an *earlier* axis keeps the tail contiguous (e.g. axis-1 gives
`stride=[256,64,1]`, dim1·dim2 packed) — there the full `[128,4,64]` tile is
correct. So the rule is per-buffer, driven by strides; the last-axis concat
is just the case that breaks contiguity.

The staging/boundary copies below are where untiled full copies still reach
legalize. Their tiling can't be deleted, only moved earlier. Two sub-parts:

- **3a — matmul staging (main case, every matmul kernel). ✅**
  Prerequisite ✅: [unify-on-linalg-copy](2026-07-01-unify-on-linalg-copy.md)
  (the RHS copy-in is now a `linalg.copy`, which has `TilingInterface`;
  `memref.copy` does not). Tile the LHS transpose and RHS copy-in via the
  **transform dialect** — emit `transform.structured.tile_using_for` on
  each in the generated sequence in `buildMatmulBlockingTransforms`, right
  after promoting that operand. This is the same `emitTile` helper the
  elementwise/transpose knobs already use — no hand-rolled loop.

  **Where the handle comes from — NOT `get_producer_of_operand`.** That was
  the original plan, but it fails in the memref pipeline: the producer of a
  matmul operand is the `memref.alloc`, not the copy/transpose that writes
  into it (a `linalg.copy` on memrefs has no result, so it is never an
  operand's defining op). Instead the ops that *create* the staging ops
  return handles to them — `NkipyTransposeMatmulOp` returns the inserted
  `linalg.transpose`, `PromoteTensorOp` returns the stage-in `linalg.copy`
  (empty if the value was already in SBUF). `buildMatmulBlockingTransforms`
  then `emitTile`s each.

  **Do NOT tile inside `PromoteTensorOp::apply`.** Tried it; it corrupts
  the IR (a nested `scf::tileUsingSCF` + `replaceOp` run *during* transform
  interpretation, while the outer driver holds handles to the same IR,
  produced a malformed subview — impossible stride, dropped mem_space).
  The tiler itself is fine: tiling a `linalg.copy` standalone (strided
  source, contiguous SBUF dest, with/without mem_space) works and
  preserves mem_space. The fix is to tile as a **first-class transform
  step**, not a nested rewrite.

  Tile sizes: `[tileK, tileM]` (transpose out / LHS), `[tileK, tileN]`
  (RHS) — the same `tile_size` attr explicit-sbuf-tile already stamped on
  these buffers. Must match the buffer's `sbuf_map` or the per-block loop
  and the map disagree.

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

- **3b — concat / boundary copy-out. ✅** The SBUF→HBM copies that lower a
  `np.concatenate` are the other thing legalize tiled. Let
  knob-driven-tiling tile them like any elementwise op — no new pass. A
  copy is just an elementwise op that needs no SBUF staging (it lowers to a
  DMA, which addresses HBM/SBUF directly). Three coordinated changes:

  1. `builder.concatenate` emits `linalg.copy` (tileable) instead of
     `memref.copy` (no `TilingInterface`).
  2. `InferLayout::defaultTileOps` gives a copy a **contiguity-safe tile**:
     start from the shape default, then set any output dim that is *not*
     contiguous with its successor (`stride[i] != size[i+1]*stride[i+1]`) to
     1. This tiles exactly the dims the fold can't cross and leaves the
     contiguous tail folded — correct for last-axis concat, earlier-axis
     concat, and plain contiguous copies alike, regardless of any knob.
  3. `buildElementwiseTiling`: a copy is **tiled but not promoted** — DMA
     needs no staging buffer; promoting the HBM side would add a spurious
     `SBUF→SBUF→HBM` hop.

  What's untiled today (rope_3d concat), pre-tiling — each concat half is a
  full-size `memref.copy` into a strided slice of the result, with no tile:

  ```mlir
  %out   = memref.alloc() : memref<128x4x128xf32, SharedHbm>
  %half0 = memref.subview %out[0,0,0] [128,4,64] : ... to strided<[512,128,1]>
  memref.copy %q_rot0, %half0                   // ← untiled memref.copy
  ```

  After the three changes it is a `linalg.copy` with a contiguity-safe tile
  (`[128,1,64]` here — dim1 stride 128 ≠ 64, so dim1 → 1), which the
  elementwise knob path splits into per-tile loops — no staging alloc.

  head_deconcat's reshape copy-out (a `reinterpret_cast` copy materialized
  late by canonicalize-reshape) is a *different* case, still blocked on
  Problems 2–5 below — out of scope here.

Do 3a first (highest impact, self-contained in `KnobDrivenTiling.cpp`),
verify LegalizeLayout tiles nothing for a pure-matmul kernel, then 3b.

**Step 4: Remove `tileCopyAndTranspose` from LegalizeLayout ✅**

After 3a+3b, no untiled transposes or copies reach LegalizeLayout.
Remove `tileCopyAndTranspose`, `tileTranspose`, `tileMemrefCopy`, and the
helpers that only they used (`findLayoutForValue`, `getSbufMapFor`,
`getCopyOperands`, `needsTiledTransfer`, `lookThroughCast`,
`createBlockLoopNest`). Legalize keeps only phases 1–2 (attach `sbuf_map`)
and the HBM-fill decomposition.

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
