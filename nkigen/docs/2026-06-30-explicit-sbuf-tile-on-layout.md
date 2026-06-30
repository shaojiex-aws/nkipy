# Explicit SBUF tile on nkipy.layout (eliminate LegalizeLayout inference)

**Date:** 2026-06-30
**Status:** Proposed
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

This is the only alloc that needs special handling — everything else
is covered by step 1.

### Step 3: Simplify LegalizeLayout ✅ (with 2 known regressions)

`traceToLinalgOperands` removed. LegalizeLayout errors if
`sbuf_tile_size` is missing. All SBUF allocs now get explicit tile:

1. **infer-layout** `computeSbufTileSizes` ✅
2. **`NkipyTransposeMatmulOp`** ✅
3. **`PromoteTensorOp`** ✅: tile = `[min(shape[0], 128), shape[1]...]`
4. **`canonicalize-reshape`** ✅: same tile formula
5. **Test IR** ✅: updated

#### Known regressions (2 tests)

`test_feedforward_sbuf` and `test_feedforward_sbuf_compact_silu`
fail with neuronx-cc `[NCC_IBIR243] Access pattern out of bounds`.

**What happens:**

The `PromoteTensorOp` code attaches `tile_size=[128, 256]` to a
`256x256` promoted alloc (using `min(shape[0], 128), shape[1]`).
LegalizeLayout sees `numBlocks=[2, 1]`, attaches sbuf_map, and
`tileMemrefCopy` tiles the HBM→SBUF copy. The resulting access
pattern is rejected by neuronx-cc.

**Why the old code worked:**

`traceToLinalgOperands` found `[128, 128]` subview accesses from
the elementwise tiling loops AND the full-size `[256, 256]` from the
`memref.copy`. It filtered out full-size accesses, kept `[128, 128]`,
giving `numBlocks=[2, 2]`. This tiling pattern was valid.

**The conflict:**

`PromoteTensorOp` computes tile as `[min(shape[0], 128), shape[1]]`
= `[128, 256]`. But `traceToLinalgOperands` finds `[128, 128]` from
actual subview accesses. These are DIFFERENT tiles for the same alloc.
The `[128, 128]` tile (from subview tracing) is correct — it matches
how the data is actually accessed in the tiling loops.

**Root issue:**

The correct physical tile for a promoted alloc can only be determined
AFTER tiling creates subview access patterns. `computeSbufTileSizes`
(infer-layout, runs before tiling) uses indexing maps which give the
wrong answer for allocs whose access pattern changes after promotion.
`PromoteTensorOp` uses `[min(shape[0], 128), shape[1]]` which is
also wrong (doesn't match actual subview access).

**Fix needed:**

Don't attach `tile_size` from `PromoteTensorOp`. Keep
`traceToLinalgOperands` as fallback for allocs without explicit
`tile_size`. Alternatively, move tile computation to the beginning
of LegalizeLayout (Phase 0) where subviews exist — same tracing
logic as `traceToLinalgOperands` but stored on the layout upfront.

### Step 4: Tile transposes/copies using the explicit tile

`tileTranspose` and `tileMemrefCopy` read the tile from `nkipy.layout`
instead of from sbuf_map inference. Since the tile is set once by the
producer, it's always consistent.

Eventually (after step 3 of the parent doc), `tileCopyAndTranspose`
can be removed entirely.

### Step 5: Matmul transpose when input is already SBUF

If `%lhs` is already SBUF (user placed input there), the transpose
inside the block-M loop is SBUF→SBUF:

```mlir
scf.for %block_m = 0 to M step BLOCK_M {
  %lhs_block = memref.subview %lhs[%block_m, 0] [BLOCK_M, K]  // SBUF
  %alloc = memref.alloc() : memref<KxBLOCK_M, Sbuf>
  linalg.transpose ins(%lhs_block) outs(%alloc) perm=[1,0]     // SBUF→SBUF
  linalg.matmul_transpose_a ins(%alloc, ...) ...
}
```

The promotion sees `%alloc` is already SBUF → no-op. With this
refactor, `NkipyTransposeMatmulOp` always attaches `sbuf_tile_size`
to `%alloc` (the SBUF output). LegalizeLayout reads it and tiles
the transpose correctly — whether the input is HBM or SBUF doesn't
matter for how the output alloc is tiled.

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
knob-driven-tiling → generates loops, NkipyTransposeMatmulOp attaches
                     sbuf_tile_size to transpose output
[new pass or extension] → computes sbuf_tile_size for remaining allocs
                          from tile_op + indexing maps
legalize-layout    → reads sbuf_tile_size, attaches sbuf_map, tiles copies
```
