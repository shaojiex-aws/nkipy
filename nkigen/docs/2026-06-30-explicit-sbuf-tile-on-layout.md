# Explicit SBUF tile on nkipy.layout (eliminate LegalizeLayout inference)

**Date:** 2026-06-30
**Status:** Implemented (steps 0–5 done)
**Unblocks:** [legalize-layout-high-rank-sbuf](2026-06-28-legalize-layout-high-rank-sbuf.md), Problem 1 Step 2

## Problem

LegalizeLayout infers the physical SBUF tile by back-tracing from
alloc → subviews → linalg consumers, reading their operand shapes.
This causes:
- "inconsistent tile sizes" when an alloc is accessed at different
  granularities (e.g. transpose writes tiles, matmul reads full)
- "tile rank mismatch" when rank-reducing subviews confuse the trace
- Complex, fragile code (`traceToLinalgOperands`, `tileTranspose`)

The tile info already exists: `nkipy.tile_op` has `loop_tile_size`.
From that + indexing maps, we can derive the physical SBUF tile
upfront. LegalizeLayout should just read it, not infer it.

## Design

### Step 0 ✅: matmul output defaults to SBUF (not SharedHbm)

`defaultLayouts` currently has `isMatmulOutput(alloc) → SharedHbm`.
This is wrong — the clean mental model is:

- **All intermediates default to SBUF** (including matmul outputs)
- **Return values → SharedHbm** (hardware requirement)
- **User annotation → respected**

The matmul's internal PSUM staging is handled by knob-driven-tiling
(PSUM alloc + copy-back to the output alloc). If the output alloc is
SBUF, copy-back is PSUM→SBUF. If it's SharedHbm (return value), the
boundary-copy pass (see
[materialize-boundary-copies](2026-06-28-materialize-boundary-copies.md))
inserts the extra SBUF→HBM step.

Fix: remove `isMatmulOutput` from the SharedHbm condition in
`defaultLayouts`. Only `isReturnValue` → SharedHbm.

The matmul output's mem_space is then determined by:
- User annotates SharedHbm → SharedHbm (respected)
- User annotates SBUF → SBUF (respected)
- Return value → SharedHbm (from `isReturnValue`)
- No annotation, not returned → SBUF (default)

Hardware flow is always PSUM → SBUF (copy-back). If the output alloc
is SharedHbm (user choice or return value), an additional SBUF→HBM
copy is needed — see
[materialize-boundary-copies](2026-06-28-materialize-boundary-copies.md)
(not a separate pass; done as part of knob-driven-tiling).

Order: step 0 → materialize-boundary-copies → step 1.

### Step 1 ✅: infer-layout computes `sbuf_tile_size` for all SBUF allocs

infer-layout already emits `nkipy.tile_op` (loop_tile_size) and
`nkipy.layout` (mem_space, partition_dim) for every op/alloc. Extend
it to also compute `sbuf_tile_size` on SBUF allocs.

#### How to derive `sbuf_tile_size`

`loop_tile_size` is in **iterator space** (one entry per linalg
iterator). Each operand's physical tile is determined by applying
the op's **indexing map** to the loop tile. This is exactly what
MLIR's `TileUsingForOp` does internally to compute subview sizes.

For the output alloc, we need: find the linalg op that writes to
it, get its output indexing map, and apply to `loop_tile_size`.

```
output_indexing_map: (iterators) → (output_dims)
sbuf_tile_size[i] = loop_tile_size[map.getDimPosition(i)]
```

Examples:
- Elementwise `(d0,d1) → (d0,d1)`: sbuf_tile = [tile[0], tile[1]]
- Matmul out `(m,n,k) → (m,n)`: sbuf_tile = [tile[0], tile[1]]
- Reduction out `(d0,d1) → (d0,0)` (keepdims): sbuf_tile = [tile[0], 1]
  (result dim is constant 0 in the map → size 1)
- Broadcast `(d0,d1) → (d0)`: sbuf_tile = [tile[0]]

#### Implementation

```cpp
void computeSbufTileSizes(func::FuncOp func) {
  func.walk([&](memref::AllocOp allocOp) {
    // Only SBUF allocs.
    auto layout = findLayoutWithMemSpace(alloc);
    if (!layout || layout.getMemSpace() != Sbuf) return;
    if (layout.getTileSizeAttr()) return;  // already set

    // Find tile_op on this alloc.
    auto tileOp = findTileOp(alloc);
    if (!tileOp) return;
    auto loopTile = tileOp.getLoopTileSizeAttr().asArrayRef();

    // Find the linalg op that writes to this alloc (DPS init).
    linalg::LinalgOp producer = findProducerLinalgOp(alloc);
    if (!producer) return;

    // Get output indexing map and derive sbuf_tile_size.
    AffineMap outMap = producer.getMatchingIndexingMap(
        producer.getDpsInitOperand(0));
    SmallVector<int64_t> sbufTile;
    for (unsigned i = 0; i < outMap.getNumResults(); i++) {
      auto expr = outMap.getResult(i);
      if (auto dimExpr = dyn_cast<AffineDimExpr>(expr))
        sbufTile.push_back(loopTile[dimExpr.getPosition()]);
      else
        sbufTile.push_back(1);  // constant expr (e.g. reduction keepdims)
    }
    layout.setTileSizeAttr(DenseI64ArrayAttr::get(ctx, sbufTile));
  });
}
```

```mlir
// Result:
nkipy.tile_op(%alloc) {loop_tile_size = array<i64: 128, 256>}
nkipy.layout(%alloc) {mem_space = Sbuf, partition_dim = 0,
                       sbuf_tile_size = array<i64: 128, 256>}
```

#### What step 1 covers

Any SBUF alloc that exists at infer-layout time with a tile_op and
a linalg producer:
- Elementwise outputs
- Reduction outputs (keepdims)
- Matmul outputs (now SBUF after step 0)

`sbuf_tile_size` of an alloc is determined by the op that **writes**
to it. For `C = A @ B`, C's `sbuf_tile_size` comes from the matmul:
output map `(m,n,k)→(m,n)` + `loop_tile_size=[M_t,N_t,K_t]` →
`sbuf_tile_size=[M_t, N_t]`. If C is later consumed by `d = exp(c)`,
that doesn't change C's tile — C's physical layout is fixed by its
producer.

Consumers may access the alloc at full size or via subviews. That's
fine — the alloc's tile is set once, consumers access within it.

The bug this eliminates: `traceToLinalgOperands` sees one consumer
accessing via `[128, M]` subviews and another reading the full
`[K, M]` directly. It can't tell "full-size = no tiling needed for
this consumer" from "full-size = different tile" → errors with
"inconsistent tile sizes". With explicit `sbuf_tile_size` from the
producer, LegalizeLayout never infers from consumers.

#### What step 1 does NOT cover

The matmul `transpose_a` alloc — created by `NkipyTransposeMatmulOp`
during knob-driven-tiling (after infer-layout runs). Step 2 handles
this.

### Step 2 ✅: `NkipyTransposeMatmulOp` attaches tile to its output

`NkipyTransposeMatmulOp::apply` creates the transpose output alloc
`[K, M]` during transform interpretation (after infer-layout). It
attaches `sbuf_tile_size` directly:

```mlir
nkipy.layout(%transposedInit) {mem_space = Sbuf, partition_dim = 0,
              sbuf_tile_size = array<i64: min(K,128), M>}
```

> **Superseded by the step-3 fix below.** `[min(K,128), M]` is wrong:
> the transpose output is consumed at tile granularity `[K_t, M_t]`,
> not block granularity `[K, M]`. The fix passes the correct tile
> explicitly (see step 3).

### Step 3: Simplify LegalizeLayout ✅

`traceToLinalgOperands` removed. LegalizeLayout errors if
`sbuf_tile_size` is missing. All SBUF allocs now get explicit tile:

1. **infer-layout** `computeSbufTileSizes` ✅
2. **`NkipyTransposeMatmulOp`** ✅: explicit `[tileK, tileM]` from caller
3. **`PromoteTensorOp`** ✅: explicit tile from caller, else
   `[min(shape[0], 128), shape[1]...]`
4. **`canonicalize-reshape`** ✅: `[min(shape[0], 128), shape[1]...]`
5. **Test IR** ✅: updated

#### The matmul-operand regression (fixed)

Two tests — `test_feedforward_sbuf` and
`test_feedforward_sbuf_compact_silu` — previously failed with
neuronx-cc `[NCC_IBIR243] Access pattern out of bounds`. Here is why,
grounded in the dumped IR.

The two buffers that broke are both **matmul block-level operands**,
promoted by `buildMatmulBlockingTransforms` /
`NkipyTransposeMatmulOp`. Here is the matmul loop nest as it appears
pre-legalize (`mm_gup = x @ gate_up_weight`, trimmed; the `x` matmul
becomes `matmul_transpose_a(transpose(x), gate_up_weight)`):

```mlir
%alloc = memref.alloc() : memref<256x512xf32, Sbuf>            // mm_gup output
scf.for %block_m = 0 to 1 step 1 {                            // block-M (degenerate, M=256)
  %x_blk = memref.subview %arg0[...] [256, 256]               // x block from HBM

  // ── %alloc_12: transpose output [K, M] = 256x256 (NkipyTransposeMatmulOp) ──
  %alloc_12 = memref.alloc() : memref<256x256xf32, Sbuf>
  nkipy.layout(%alloc_12) {tile_size = [128, 256]}            // ← WRONG (step 2 heuristic)
  linalg.transpose ins(%x_blk) outs(%alloc_12) perm = [1, 0]  // x^T, lives in SBUF

  scf.for %block_n = 0 to 2 step 1 {                          // block-N, step BLOCK_N=256
    %w_blk = memref.subview %arg1[0, %block_n*256] [256, 256]  // gate_up_weight block (HBM)

    // ── %alloc_15: RHS copy-in 256x256 (block-N promote_tensor) ──
    %alloc_15 = memref.alloc() : memref<256x256xf32, Sbuf>
    nkipy.layout(%alloc_15) {tile_size = [128, 256]}          // ← WRONG (promote heuristic)
    memref.copy %w_blk, %alloc_15                             // HBM → SBUF block copy

    scf.for %tile_m = 0 to 2 step 1 {                        // tile-M, step TILE_M=128
      %lhsT_m = memref.subview %alloc_12[0, %tile_m*128] [256, 128]   // [K, TILE_M]
      scf.for %tile_n = 0 to 2 step 1 {                      // tile-N, step TILE_N=128
        %rhs_n = memref.subview %alloc_15[0, %tile_n*128] [256, 128]  // [K, TILE_N]
        %psum  = memref.alloc() : memref<128x128xf32, Psum>
        scf.for %k = 0 to 2 step 1 {                         // reduction, step TILE_K=128
          %a = memref.subview %lhsT_m[%k*128, 0] [128, 128]  // ← reads %alloc_12 as [TILE_K, TILE_M] = 128x128
          %b = memref.subview %rhs_n[%k*128, 0]  [128, 128]  // ← reads %alloc_15 as [TILE_K, TILE_N] = 128x128
          linalg.matmul_transpose_a ins(%a, %b) outs(%psum)
        }
        memref.copy %psum, ...                               // PSUM → %alloc tile
      }
    }
  }
}
```

`%alloc_12` (`x^T`, LHS) and `%alloc_15` (`gate_up_weight`, RHS) are
both allocated/copied at the **block** level (`256x256`) but read by
the innermost `matmul_transpose_a` at the **tile** level
(`[TILE_K, *] = [128, 128]`). The `[min(shape[0], 128), shape[1]]`
heuristic only caps the partition dim, giving `[128, 256]` →
`numBlocks = [2, 1]`, which neuronx-cc rejects. The correct tile is
`[128, 128]` → `numBlocks = [2, 2]`.

The heuristic is fine for **elementwise/reduction/transpose**
promotion, where `promote_tensor` runs *after* `emitTile`, so the
operand is already leaf-sized (free dim never tiled). It only breaks
for **matmul**, where LHS/RHS/transpose promotion happens at the
*block* level, *before* the inner tiling splits the free dimension.

#### Fix: pass the correct tile explicitly from the matmul builder

No tracing, no new pass, no extra inference. `buildMatmulBlockingTransforms`
already knows the iterator tile (`tileM, tileN, tileK`), so it knows
each operand's physical tile directly:

- transpose output / LHS `A^T`: `[tileK, tileM]`
- RHS `B`: `[tileK, tileN]`

`promote_tensor` and `nkipy.transpose_matmul` each take an optional
`tile_size` attribute. The matmul builder passes these tiles; the ops
attach them to `nkipy.layout` verbatim. When the attribute is absent
(elementwise path), the ops fall back to the existing
`[min(shape[0], 128), shape[1]]` default — so that path is unchanged.

This keeps the design's "one authoritative tile per alloc, set
upfront" model: the producer of the buffer (here, the tiling transform
that creates it) states the physical tile, and LegalizeLayout just
reads it. Re-introducing `traceToLinalgOperands` (even as a fallback)
was rejected — it brings back the post-tiling trace and the
"inconsistent tile sizes" failure mode this redesign exists to remove.

### Step 4 ✅: Tile transposes/copies using the explicit tile

Already satisfied by step 3 — no code change needed.

`tileTranspose` / `tileMemrefCopy` get their tile dims from
`info->tileSize`, which phase 1 (`findSbufTensorsToLegalize`) reads
directly from `nkipy.layout`'s `tile_size`. They never read tile dims
out of `sbuf_map`: `getSbufMapFor` is used only as a presence gate
(tile *whether*, not *how*), and `SbufMapAttr::getTileSize()` is never
called in the pass. So once step 3 made every SBUF alloc carry an
explicit, authoritative `tile_size`, these functions were already
reading it.

These functions still need to exist: matmul promotion emits the LHS
transpose and RHS staging copy at *block* granularity (outside the
inner tile loops), so legalize-layout is what splits them into the
per-block loops. They cannot be removed until those ops are generated
tile-sized at promotion time (parent doc, step 3).

### Step 5 ✅: Matmul transpose when input is already SBUF

Already satisfied by step 3 — no code change needed, and verified
end to end.

`NkipyTransposeMatmulOp` always allocates a fresh SBUF output `%alloc`
and transposes the LHS *directly* into it — the LHS itself is never
promoted, so its mem space is irrelevant to how `%alloc` is tiled:

```mlir
scf.for %block_m = 0 to M step BLOCK_M {
  %lhs_block = memref.subview %lhs[%block_m, 0] [BLOCK_M, K]  // HBM or SBUF
  %alloc = memref.alloc() : memref<KxBLOCK_M, Sbuf>
  linalg.transpose ins(%lhs_block) outs(%alloc) perm=[1,0]     // →SBUF
  linalg.matmul_transpose_a ins(%alloc, ...) ...
}
```

Two things make this work for either input mem space:
- Step 3 attaches the explicit `[tileK, tileM]` tile to `%alloc`
  irrespective of where `%lhs_block` lives.
- `tileTranspose` in legalize-layout handles both HBM→SBUF and
  SBUF→SBUF (`needsTiledTransfer(...) || (isSbuf(in) && isSbuf(out))`),
  so the transpose is tiled correctly regardless.

(The operand-0 promote of `%alloc` is a no-op via
`findExistingMemSpace` since `%alloc` is already SBUF — but that holds
in both cases and is unrelated to the input's mem space.)

`test_feedforward_sbuf_compact_silu` exercises both cases in one
kernel: the first matmul's LHS (`x`) is HBM → transpose is HBM→SBUF;
the second matmul's LHS (`gated`) is an SBUF intermediate → transpose
is SBUF→SBUF. Both transpose outputs get `tile_size = [128, 128]` and
the test passes.

## Why this is cleaner

| Before | After |
|--------|-------|
| Infer tile bottom-up from consumers | Read tile top-down from annotation |
| Multiple consumers can disagree | One authoritative tile per alloc |
| `traceToLinalgOperands` fragile | Direct attribute read |
| `tileTranspose` uses sbuf_map | Uses explicit `sbuf_tile_size` |
| "inconsistent tile sizes" possible | Impossible by construction |

## Pipeline flow

```
infer-layout       → assigns mem_space, partition_dim, tile_op
infer-layout       → also computes sbuf_tile_size for SBUF allocs
                     (computeSbufTileSizes, from tile_op + indexing maps)
knob-driven-tiling → generates loops; promote_tensor /
                     NkipyTransposeMatmulOp attach sbuf_tile_size to
                     matmul operands (explicit [tileK, tileM]/[tileK, tileN])
legalize-layout    → reads sbuf_tile_size, attaches sbuf_map, tiles copies
```
