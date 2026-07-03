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

## Problem 2: HBM→HBM transpose cannot use the DMA transpose engine

Problems 2–5 in the original analysis were symptoms of one root cause:
`buildTransposeTiling` unconditionally promotes the transpose output
to SBUF and routes it through the DMA transpose engine. This is wrong
for the head_deconcat case, where the transpose is HBM→HBM.

### Why the transpose engine doesn't work here

The DMA transpose engine swaps partition and free lanes — a hardware
operation that requires the two swapped axes to be **contiguous in
memory** (they form the 2D tile the engine flips). In head_deconcat:

```
Input:  memref<2x2x128x128xf32, SharedHbm>
Output: memref<2x128x2x128xf32, SharedHbm>
Permutation: [0, 2, 1, 3]   (swap dims 1 and 2)
```

After tiling, each tile is `[1, 2, 128, 1]` with permutation
`[0, 2, 1, 3]`. The swapped axes have sizes 2 and 128, but they are
NOT the partition/free axes of a 2D view — the engine sees them as
(W, X) and rejects: `transposing dimensions (WX) should be continuous`.

This is fundamental. No amount of emitter hacking can fix it — the
hardware path physically cannot execute this permutation.

### What `buildTransposeTiling` does today (wrong for HBM→HBM)

```cpp
// KnobDrivenTiling.cpp:471-486
void buildTransposeTiling(...) {
  Value matched = emitMatch(...);
  Value tiledOp = emitTile(...);
  emitPromoteOperand(builder, loc, tiledOp, numInputs, sbufMemSpace);
  //                                                   ^^^^^^^^^^^^
  // Always promotes OUTPUT to SBUF — forces the transpose engine path
}
```

Compare with `buildElementwiseTiling` (line 461):
```cpp
if (opName == "linalg.copy")
  return;  // skips SBUF promotion — DMA copy addresses HBM directly
```

The copy path is already proven: tiled `linalg.copy` emits
`nisa.dma_copy` with strided addressing (arbitrary access patterns,
no contiguity constraint). The transpose engine is only needed when
you want to physically reorder data **within SBUF** as a staging step
before compute (e.g., matmul `transpose_a`).

### The clean fix: lower HBM→HBM transpose to permuted-view + copy

When both source and destination are in HBM, the transpose doesn't
need to physically move data through the engine. It just needs to
read elements in permuted order and write them contiguously (or
vice-versa). The DMA copy engine handles this via strided addressing.

**Lowering:**
```
linalg.transpose ins(%src: HBM) outs(%dst: HBM) perm=[0,2,1,3]
```
becomes:
```
%permuted_view = memref.transpose %src by perm [0,2,1,3]
    : memref<2x2x128x128, HBM> → memref<2x128x2x128, strided, HBM>
linalg.copy ins(%permuted_view) outs(%dst)
```

`memref.transpose` is a zero-cost op — same memory, permuted strides.
`linalg.copy` from a strided source to a contiguous destination is
exactly what `nisa.dma_copy` does: it reads via the strided pattern
and writes contiguously.

**Where to implement:** A new pattern in `KnobDrivenTiling.cpp` (or
a small pre-pass) that rewrites `linalg.transpose` whose **output is
HBM** into `memref.transpose` + `linalg.copy`. The resulting copy
then flows through `buildElementwiseTiling` (which skips SBUF
promotion for copies) and the existing emitter handles it.

`buildTransposeTiling` remains for the SBUF-output case (matmul LHS
staging), where the engine IS correct.

### What this deletes from emit.py

All the hacky emitter workarounds for the "4D transpose through
engine" path go away:

- `_reduce_permutation_2d` — only served the engine path
- `_free_dim_stride` / `128*d1` stride hack — only served the engine
- Modified `_emit_transpose` logic for high-rank — unnecessary when
  HBM→HBM never reaches the transpose emitter

What STAYS (these fix real, independent bugs):
- `_trace_access` stopping at rank-changing `reinterpret_cast` —
  correct invariant for any high-rank DMA (the copy path uses this)
- `_resolve_materialized` — correct for any view-op chain
- 5-tuple return + `crossed_reshape` / view logic — needed whenever
  the emitter sees a `reinterpret_cast` it didn't materialize

### Implementation plan

1. **Detect HBM→HBM transpose:** In `KnobDrivenTiling`, before
   dispatch (line 703), check if the matched transpose has its output
   in HBM (not SBUF). This is exactly the case where InferLayout
   assigned SharedHbm via `getViewMemSpace` (Problem 1 fix).

2. **Lower to permuted-view + copy:** Emit a transform sequence:
   - Match the `linalg.transpose`
   - Rewrite to `memref.transpose %src` + `linalg.copy`
   - The copy inherits the knob's tile_size
   - Route through `buildElementwiseTiling` (skip promotion)

3. **Delete emitter hacks:** Remove `_reduce_permutation_2d`,
   `_free_dim_stride`, and the high-rank engine path from
   `_emit_transpose`.

4. **InferLayout fix stays:** `getViewMemSpace` correctly assigns
   SharedHbm to the 4D output alloc, which is what triggers the
   HBM→HBM detection in step 1.

### Why this is clean

- **One principle:** transpose engine for SBUF staging (compute
  prep), DMA copy for bulk data movement (HBM↔HBM).
- **No emitter hacks:** the copy emitter already handles arbitrary
  strided patterns.
- **Correct by construction:** `memref.transpose` is a view (no
  codegen), `linalg.copy` is proven.
- **Breaking change is fine:** `buildTransposeTiling` only routes
  SBUF-output cases to the engine; all other transposes become copies.
  This matches hardware capability exactly.
