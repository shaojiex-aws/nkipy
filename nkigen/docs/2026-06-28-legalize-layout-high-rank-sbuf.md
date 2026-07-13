# High-rank SBUF: LegalizeLayout + NISA emitter fixes

**Date:** 2026-06-28
**Status:** Problems 1, 3, 4, 6 done ✅. Problem 2 reimplemented (2026-07-06)
with explicit pack/split + 2D transpose lowering. Problem 5 not needed (resolved
by 3+4). Problem 7 diagnosed (2026-07-13): backend OOB on 3D sbuf with
partition_dim=0 assigned to post-tiling buffer; workaround = skip HW mode.
**Affects:** test_head_deconcat (green), test_qwen3_layer (STRING_CHECK + LLVM +
CODEGEN green; HW mode blocked on Problem 7)

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

## Problem 2: High-rank transpose is being forced through the 2D transpose engine

The emitter's `_emit_transpose` currently lowers every `linalg.transpose` to
`nisa.dma_transpose`. That is only a clean lowering for a true 2D tile. The
head_deconcat transpose is rank 4:

```mlir
linalg.transpose ins(%src: memref<2x2x128x128xf32>)
                 outs(%dst: memref<2x128x2x128xf32>)
                 permutation = [0, 2, 1, 3]
```

Trying to make this look like a 2D transpose by picking two surviving dims is
the wrong abstraction. It makes one test green, but the rule depends on which
logical dims happen to be last in the output and does not generalize to other
permutations.

### Rejected 2026-07-04 attempt

The 2026-07-04 uncommitted fix made `test_head_deconcat` green by combining:

1. `SimplifyLinalg::collapseUnitDimTransposes`, which rank-reduced tiled
   high-rank transposes with unit dims.
2. `KnobDrivenTiling::buildTransposeTiling` input promotion, so transpose input
   tiles are staged into SBUF before the engine reads them.
3. `InferLayout::defaultTileOps` high-rank heuristic: keep the output last dim
   and one swapped dim, tile the rest to 1.

Do **not** commit this as-is. The heuristic is output-oriented, while the
copy-in hazard is source-oriented. For example, `perm=[2,0,1]` can infer an
output tile `[1,2,4]`, which maps to an input tile `[2,4,1]`: the source
stride-1 dim is tiled to 1, so the HBM copy-in is exactly the kind of strided
middle-dim access the heuristic claimed to avoid. For `perm=[2,1,0]`, the
heuristic can keep only one surviving dim and still let a high-rank transpose
reach NISA as a 3D permutation over 2D operand strings.

### Clean direction

Do not add `collapseUnitDimTransposes` to `simplify-linalg`. That pass is the
wrong ownership boundary for this issue, and the long-term goal is to remove or
shrink it. Shape/view/index interpretation already belongs to the emitter.

The cleaner invariant is:

1. **`nisa.dma_transpose` is only for proven rank-2 transpose.** The emitter
   should assert/reject anything else before producing NISA.
2. **Rank >2 transpose lowers to loops around an effective-rank-2 inner op.**
   Pick two logical output dims to keep live (prefer the dimensions actually
   moved by the permutation), tile all other dims to 1, and loop over them.
3. **The effective-rank-2 inner op is either copy or transpose.** If the reduced
   permutation is identity, emit a copy. If it is `[1,0]`, stage into 2D SBUF
   and emit `nisa.dma_transpose`.
4. **Staging must be legal, not heuristic.** If the HBM/SBUF view cannot be
   represented by the current 2D `view()` syntax, lower through an explicit
   pack/unpack sequence or fail clearly. Do not silently flatten a strided
   middle dimension as if it were contiguous.

For head_deconcat, the intuitive target tile is the actual swapped pair:
`[1,128,2,1]` on the output. The matching source tile is `[1,2,128,1]`.
After dropping unit dims, this is a real 2D transpose:

```
source live shape: [2, 128]    // head, seq
dest live shape:   [128, 2]    // seq, head
reduced perm:      [1, 0]
```

That is the clean logical model. The important catch is physical staging:
`[1,2,128,1]` reads source dims `head` and `seq` while `head_dim` is fixed, so
the `seq` lane has stride 128 in HBM. The current emitter's 2D `view()` syntax
cannot express a column stride of 128; it would incorrectly read contiguous
`head_dim` elements. Therefore Option B is clean only if the implementation also
adds a correct pack step for this strided HBM source tile.

### Mental model: loop + copy vs loop + transpose

A high-rank transpose is a loop nest around smaller data movements. The inner
movement can be either a copy or a true 2D transpose, depending on how many dims
are still live inside the tile.

**Case A: effective-rank 1 tile -> copy.** This is legal and useful, but it is
not the complete target if we want a real high-rank transpose lowering.

```mlir
// Logical op:
//   out[b, s, h, d] = in[b, h, s, d]
//   perm = [0, 2, 1, 3]

// Pick an output tile with one live dim: [1, 128, 1, 1].
// The matching source tile is [1, 1, 128, 1].
scf.for %b = 0 to 2 {
  scf.for %h = 0 to 2 {
    scf.for %d = 0 to 128 {
      %src = memref.subview %in[%b, %h, 0, %d]
             [1, 1, 128, 1] [1, 1, 1, 1]
      %tmp_in = memref.alloc() : memref<1x1x128x1xf32, Sbuf>
      linalg.copy ins(%src) outs(%tmp_in)        // HBM -> SBUF

      %tmp_out = memref.alloc() : memref<1x128x1x1xf32, Sbuf>
      linalg.transpose ins(%tmp_in) outs(%tmp_out)
          permutation = [0, 2, 1, 3]

      // Emitter sees only one non-unit dim in tmp_in/tmp_out, so this
      // "transpose" is just a tensor_copy, not nisa.dma_transpose.

      %dst = memref.subview %out[%b, 0, %h, %d]
             [1, 128, 1, 1] [1, 1, 1, 1]
      linalg.copy ins(%tmp_out) outs(%dst)       // SBUF -> HBM
    }
  }
}
```

Only `%s` varies inside the tile. There is no pair of dimensions to swap, so
using the transpose engine would be unnecessary. This is simple and robust, but
it gives up the natural 2D transpose over `seq x head`.

**Case B: effective-rank 2 tile -> 2D transpose.** This is the clean complete
target, provided pack/unpack preserves the real source and destination strides.

```mlir
// Pick output tile [1, 128, 2, 1].
// Matching source tile is [1, 2, 128, 1].
%src = memref.subview %in[%b, 0, 0, %d] [1, 2, 128, 1] [1, 1, 1, 1]
%dst = memref.subview %out[%b, 0, 0, %d] [1, 128, 2, 1] [1, 1, 1, 1]

// After ignoring unit dims, this is a real 2D transpose:
//   source live shape: [2, 128]
//   dest live shape:   [128, 2]
//   reduced perm:      [1, 0]
linalg.transpose ins(%src_2d) outs(%dst_2d) permutation = [1, 0]
```

This requires careful staging and reduced-rank indexing. The rejected
2026-07-04 patch tried to get here through heuristic tile selection, but it did
not implement the missing strided pack correctly. The complete fix should make
this path explicit: pack the strided high-rank source tile into contiguous 2D
SBUF, run the 2D transpose, then unpack/copy to the high-rank destination tile.

### Implementation plan

1. **Back out/rework the 2026-07-04 code changes**:
   - replace the output-last-dim heuristic in `InferLayout.cpp`.
   - keep `KnobDrivenTiling.cpp` input staging; it provides the pack buffer.
   - remove `SimplifyLinalg.cpp::collapseUnitDimTransposes`.

2. **Change high-rank transpose default tiling** in `InferLayout::defaultTileOps`:
   - rank <= 2: keep the existing 2D tile behavior.
   - rank > 2: choose two live output dims from the permutation's moved dims,
     capped by the target limits, and set all other dims to 1.
   - For head_deconcat `[0,2,1,3]`, this should pick output dims 1 and 2,
     producing `[1,128,2,1]`.

3. **Lower high-rank transpose explicitly as pack -> 2D op -> unpack.**
   - Pack reads the high-rank source tile with correct strides into contiguous
     2D SBUF. In the current implementation this is the promoted input
     `linalg.copy`, and the NISA emitter recursively splits unsafe HBM `view()`
     copies into safe 1D DMA slices.
   - The inner 2D op is `dma_transpose` for reduced perm `[1,0]`, or copy for
     reduced identity.
   - Unpack writes the contiguous 2D result to the high-rank destination tile
     with correct strides, using the same recursive copy splitting.

4. **Make `_emit_transpose` strict.**
   - Raw `nisa.dma_transpose` emission accepts only rank-2 operands.
   - Any high-rank transpose that was not lowered through the explicit pack path
     is a compile error.

5. **Keep `simplify-linalg` out of this.** Do not add
   `collapseUnitDimTransposes`; the lowering belongs either in the tiling/DMA
   canonicalization path or in the emitter where physical indexing is known.

6. **Add focused tests before re-running e2e**:
   - head_deconcat infers `[1,128,2,1]`.
   - head_deconcat emits pack + 2D `dma_transpose` + unpack, not a raw high-rank
     `dma_transpose`.
   - a rank-3 cycle such as `perm=[2,0,1]` also lowers via the same explicit
     pack path or fails clearly if no legal two-dim tile is selected.

**Implemented 2026-07-06.** The current lowering for head_deconcat is shown
below. Important: `nkipy.tile_op` is a top-level annotation. The
`knob-driven-tiling` and `linalg-to-nisa` snippets are the body of one tiled
iteration; the real IR has outer `scf.for` loops around them.

```mlir
// infer-layout: top-level annotation, not executable loop-body IR.
nkipy.tile_op(%transpose_out) {loop_tile_size = array<i64: 1, 128, 2, 1>}

// knob-driven-tiling: outer tile loops are generated from the tile_op.
scf.for %b = 0 to 2 step 1 {
  scf.for %s0 = 0 to 128 step 128 {
    scf.for %h0 = 0 to 2 step 2 {
      scf.for %d = 0 to 128 step 1 {
        // This is the tile-loop body.
        %src_tile = subview %in[%b, %h0, %s0, %d] [1,2,128,1]
        %pack = memref.alloc() : memref<1x2x128x1xf32, Sbuf>
        linalg.copy ins(%src_tile) outs(%pack)

        %transposed = memref.alloc() : memref<1x128x2x1xf32, Sbuf>
        linalg.transpose ins(%pack) outs(%transposed) perm=[0,2,1,3]

        %dst_tile = subview %out[%b, %s0, %h0, %d] [1,128,2,1]
        linalg.copy ins(%transposed) outs(%dst_tile)
      }
    }
  }
}

// linalg-to-nisa emitter: same tile-loop body after lowering.
scf.for %b = ... {
  ...
    // pack copy is split because source seq has HBM stride 128.
    scf.for %seq = ... {
      nisa.dma_copy(...)        // safe [2,1] slice
    }

    nisa.dma_transpose(..., permutation=[1, 0])

    // unpack copy is split because destination head has HBM stride 128.
    scf.for %head = ... {
      nisa.dma_copy(...)        // safe [128,1] slice
    }
  ...
}
```

Validation:
- `pytest tests/passes/infer_layout`
- `pytest tests/passes/linalg_to_nisa`
- `pytest tests/passes/knob_driven_tiling`
- `pytest tests/e2e/test_head_deconcat.py::test_head_deconcat`

**Rejected: the LegalizeLayout "default-tile fallback."** An earlier attempt made
`legalize-layout` invent a default tile (`[min(d0,128), d1, ...]`) for an SBUF
alloc missing `tile_size`, instead of erroring. This was backed out: it does not
fix anything and is actively harmful. Its only effect was to shove
`test_qwen3_layer` past legalize-layout, where it then produced silently-wrong
output (see below) instead of a clear compile error. Legalize keeps its honest
`missing tile_size` error.

### What remains for test_qwen3_layer (Problems 3–5)

qwen3 does **not** fail on the transpose-emitter issue; Problem 2 is not its
blocker. Three bugs remain, all triggered by the interaction between
`batch_matmul` decomposition (in `canonicalize-compute`) and downstream passes.

**Pass bisection (2026-07-07, clean HEAD):**

```
canonicalize-compute        max_rel 0.0000  PASS
infer-layout                max_rel 0.0000  PASS
canonicalize-partition-dim  max_rel 0.5033  FAIL   <-- numerical divergence
…
legalize-layout             ERROR: missing tile_size on nkipy.layout
```

**Repro:** run the qwen3 kernel through
`compile_knob_pipeline(traced, stop_after=<pass>)` + `LLVMModule` for each pass,
comparing to `traced.__wrapped__(*inputs)` under `np.random.seed(42)`.

---

#### Problem 3 ✅: `canonicalize-partition-dim` — transpose inserted before producer loop

**Root cause:** `findProducerLinalgOp(input)` only finds direct DPS writers to a
buffer. When `input` is written through subviews inside an `scf.for` loop (from
`batch_matmul` decomposition), no direct linalg writer is found. The fallback
places the boundary transpose after `input.getDefiningOp()` (the alloc), which
is **before** the loop populates the buffer — transposing stale/zero data.

**Concrete scenario:** In qwen3, the attention-scores matmul (Q×K^T) is a
`batch_matmul` decomposed into:

```mlir
%alloc_45 = memref.alloc() : memref<4x128x128xf32>
scf.for %i = 0 to 4 {
  %sv = memref.subview %alloc_45[%i, 0, 0] [1,128,128] ...
  linalg.matmul ... outs(%sv)           // writes per-batch results
}
// softmax reads %alloc_45 with partition_dim=1
linalg.generic ins(%alloc_45) ...       // scale
```

The softmax component has `partition_dim=1`. Its boundary input is `alloc_45`.
`findProducerLinalgOp(alloc_45)` returns null (writes are to subviews, not to
`alloc_45` directly). So the pass inserts:

```mlir
%alloc_45 = memref.alloc() ...
%transposed = memref.alloc() : memref<128x4x128xf32>
linalg.transpose ins(%alloc_45) outs(%transposed)  // STALE DATA!
scf.for %i = 0 to 4 { ... writes into alloc_45 ... }
// softmax now reads %transposed (zeros/garbage)
```

**Fix:** Replace `findProducerLinalgOp` with a utility that traces through
view-like aliases (subview, reinterpret_cast) and returns the last
write-completion op in the buffer's definition block — typically the enclosing
`scf.for` when writes happen through subviews inside a loop.

---

#### Problem 4 ✅: `canonicalize-compute` — base alloc loses layout annotation

**Root cause:** `decomposeOneBatchMatmul` transfers layout/tile annotations from
the base output `init` to the per-batch `initSlice` subviews, then **erases all
annotations on `init`**. After decomposition, `init` has no layout. Then
`infer-layout` sees the unannotated 3D alloc, defaults it to `Sbuf` without
`tile_size`:

```mlir
%alloc = memref.alloc() : memref<4x128x128xf32>
nkipy.layout(%alloc) {mem_space = Sbuf, partition_dim = 0}  // NO tile_size!
scf.for ... { linalg.matmul ... outs(subview of %alloc) }
```

`legalize-layout` correctly errors: `missing tile_size on nkipy.layout`.

**Fix:** After the for-loop, re-attach a `LayoutOp` to the base `init` alloc
preserving the original `mem_space`. Derive `tile_size` from the per-batch tile
(prepend batch dim = 1). Example: user annotated
`tile_size=[1, 128, 128, 128]` (including reduction) → base alloc gets
`tile_size=[1, 128, 128]` (output dims only).

---

#### Fix order and dependencies

```
Problem 4 (canonicalize-compute)  — independent, fix first
Problem 3 (canonicalize-partition-dim) — independent, fix second
```

Both fixes share a common infrastructure need: **a write-completion finder
that traces through view aliases**. Implemented as a shared utility in
`IRHelpers.h/.cpp`:

```cpp
/// Collect `base` and all memref view values derived from it.
void collectMemRefAliases(Value base, SetVector<Value> &aliases);

/// Find the last op in `buffer`'s definition block after which all nested DPS
/// writes to `buffer` or any of its view aliases have completed.
Operation *findWriteCompletionOp(Value buffer);
```

#### Problem 6: `linalg-to-nisa` emitter — rank-changing SBUF alias

**Root cause:** The NISA emitter assumes every SBUF value keeps the same rank as
its allocation. After canonicalize passes, a physical 2D SBUF alloc
(`memref<128x128xf32, Sbuf>`) can be read through a rank-changing alias
(`memref.reinterpret_cast` / `memref.expand_shape` → 3D or 4D). The emitter
tries to index the value with the alias's rank and produces invalid NISA — type
mismatches or wrong offset calculations.

**Concrete scenario:** In qwen3, the attention-scores output is a 3D batch alloc
that legalize-layout tiles into a 2D physical SBUF. A downstream reshape reads
it as 4D (head deconcat). The emitter sees a 4D memref backed by a 2D
allocation and doesn't know how to express the access.

**Fix:** Two changes in `_operand_str_from_trace`:

1. **On-chip crossed_reshape:** When the access coordinate frame (from a
   `reinterpret_cast`) differs from the physical alloc, linearize all offsets in
   the alias frame into a flat element index, then split into physical par/free
   using the alloc's sbuf_map projection. Use the physical alloc's NISA type for
   `memloc_ref` so the SSA name's type stays consistent.

2. **sbuf_map leading-unit-dim strip:** The sbuf_map path now strips leading unit
   dims from the tile_shape (matching what the non-sbuf_map `>2D` path does).
   Without this, a non-rank-reducing `[1, 128, 128]` subview would emit
   `par=1, free=16384` instead of the correct `par=128, free=128`.

**Status:** ✅ Done. NISA emits valid, well-typed assembly.

#### Problem 7: Backend OOB — 3D sbuf_map with partition_dim=0 on a post-tiling buffer

**Symptom:** neuronx-cc `birverifier` rejects access pattern `[[16384,128],[1,128]]`
on a `memref<4x16384xf32, sbuf>` buffer. The tensor_copy tile `<128|128>` claims
128 partitions but the physical buffer only has 4.

**Root cause:** The attention context matmul (`attn_weights @ v`) creates a 3D
intermediate accumulator `memref<4x128x128xf32, Sbuf>` (BH × seq × head_dim).
Its `nkipy.layout` has `tile_size=[1,128,128]` (from `attn_tile`), no
`partition_dim`. Legalize-layout assigns `sbuf_map<tile:[1,128,128], blocks:[4,1,1]>`
— making dim 0 (BH=4) the partition dimension with only 4 physical partitions.

But the matmul accumulates in psum (128×128 per batch), and the psum↔sbuf
tensor_copy needs 128 partitions. The emitter correctly emits tile `<128|128>` for
the psum copy, which the backend correctly rejects: you can't access 128 partitions
on a 4-partition buffer.

**Why it happens:** The buffer is created by `knob-driven-tiling` (pass 05) AFTER
`canonicalize-partition-dim` (pass 03) has already run. So no pass reorders the
dimensions to put partition first. The `tile_size=[1,128,128]` means "1 batch per
tile, 128×128 per batch" — partition should be dim 1 (seq_len=128), not dim 0
(BH=1 per tile).

**Current workaround:** Skip HW mode on qwen3 (test CODEGEN + LLVM only).

**Cleanest fix — partition-dim enforcement in LegalizeLayout:**

When `attachSbufMapAttrs` processes a tile where `tile[0] * blocks[0] < 128` but
`tile[1] * blocks[1] >= 128`, the buffer needs partition at dim 1. The fix:

1. **Reshape the alloc** from `memref<4x128x128xf32>` to `memref<512x128xf32>`
   (collapse dims [0,1]).
2. **Assign 2D sbuf_map:** `tile:[128,128], blocks:[4,1]` — giving 128 physical
   partitions with free = 4×128 = 512. Each batch occupies one partition-block.
3. **Insert `memref.expand_shape`** for all existing users that index it as 3D.
   The expand_shape reassociation `[[0,1],[2]]` maps the 2D buffer back to 3D.
4. All subviews like `subview(%buf, %batch)[0, 0] → 128x128` become
   `subview(%buf_2d, %batch*128, 0)[128, 128]` — indexing into the correct
   partition-block.

This is invasive (requires updating all users) but architecturally clean: it
enforces the sbuf_map invariant that dim 0 has enough partitions for any DMA tile
that accesses the buffer.

**Alternative (simpler, less general):** Have `canonicalize-reshape` (pass 07) set
`partition_dim=1` on the `nkipy.layout` for buffers where `tile[0] < 128` and
`tile[1] >= 128`. Then run a **second canonicalize-partition-dim pass** after
tiling/reshape to transpose late-created buffers. This avoids reshape surgery but
adds a pass ordering dependency.

---

## Follow-up refactor: group DMA-related rewrites into `canonicalize-dma`

After Problem 2 is fixed, `canonicalize-dma` should become the home
for all DMA/data-movement canonicalization. Move into it:

```
canonicalize-dma:
  1. decomposeHbmFills       — fill on HBM → fill SBUF tile + copy (from legalize-layout)
  2. materializeCopies       — alloc+copy for cross-space views (from canonicalize-reshape)
  3. materializeReturnCopies — SBUF return value → insert SBUF→HBM copy (from infer-layout)
```

Then delete `canonicalize-reshape` (empty after moving
`materializeCopies`). Move `applyMemSpaceAnnotations` from
`canonicalize-reshape` to `legalize-layout`.

Final shapes:

```
infer-layout (Phase 1, pure annotation):
  1. propagateAnnotations  — propagate tile_size/mem_space from knobs
  2. defaultTileOps        — auto-annotate unannotated ops
  3. defaultLayouts        — assign mem_space to allocs

canonicalize-dma (Phase 1, after infer-layout):
  1. decomposeHbmFills
  2. materializeCopies
  3. materializeReturnCopies

legalize-layout (Phase 4, after fusion):
  1. applyMemSpaceAnnotations  — stamp nkipy.layout mem_space onto memref types
  2. attachSbufMapAttrs        — physical SBUF tile layout (#nkipy.sbuf_map)
  3. eraseAllLayoutOps         — consume annotation markers
```

This is a pure refactor — no behavior change, just code motion.
Do it after Problem 2 is verified green.
