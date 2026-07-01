# Unify all copies on `linalg.copy` (retire `memref.copy`)

**Date:** 2026-07-01
**Status:** Proposed
**Blocks:** step 3a/3b of
[legalize-layout-high-rank-sbuf](2026-06-28-legalize-layout-high-rank-sbuf.md)

## Why

The staging copies that knob-driven-tiling emits (matmul RHS copy-in,
reshape copy-out) must be tiled into per-block loops. The clean way to
tile them is the builtin `transform.structured.tile_using_for` /
`scf::tileUsingSCF`, which needs `TilingInterface`. `linalg.copy`
implements it; `memref.copy` does **not** (it only has `CopyOpInterface`).
That's the sole reason legalize hand-rolls an `scf.for` in
`tileMemrefCopy`.

Rather than keep two copy ops with a split "tileable vs not" rule, the
clean end-state is **one copy op everywhere: `linalg.copy`**. Then every
copy is uniformly tileable, canonicalizable, and lowered by one emitter
path. `memref.copy` is retired from the passes we own.

## Scope

`memref.copy` create sites we own (≈14):

| File | count | role | after |
|------|-------|------|-------|
| `NkipyTransformOps.cpp` (PromoteTensorOp) | 3 | stage-in / copy-back | `linalg.copy` |
| `LegalizeLayout.cpp` | 6 | tiled copies + HBM-fill DMA | deleted (step 4) / `linalg.copy` |
| `InsertSpillReload.cpp` | 2 | spill / reload | `linalg.copy` |
| `CanonicalizeReshape.cpp` | 1 | boundary copy | `linalg.copy` |
| `InferLayout.cpp` | 1 | SBUF→HBM return copy | `linalg.copy` |
| `InlineNkipyReference.cpp` | 1 | ref-impl result copy | `linalg.copy` |

Match/consume sites (2): `LegalizeLayout.cpp:405` (deleted in step 4),
`SimplifyLinalg.cpp:213` (dead-copy elimination — update to match
`linalg.copy`).

Emitter (3): `emit_memory.py` (`dispatch["memref.copy"]`),
`nisa/emit.py:210`, `__init__.py` (`_dst_operand`). All must accept
`linalg.copy`.

## Caveats

1. **Operand order.** `memref.copy(source, target)` vs
   `linalg.copy ins(source) outs(target)`. `_emit_copy` already swaps
   to `(dst, src)` for the kb call; `_dst_operand` special-cases
   `memref.copy` as `operands[1]`. For `linalg.copy` the dst is the
   `outs` operand (last). Keep both correct during migration.
2. **Canonicalizations.** `memref.copy` folds (self-copy removal, etc.)
   don't all apply to `linalg.copy`. Verify the passes that relied on
   folding (e.g. dead-copy cleanup) still converge — `SimplifyLinalg`'s
   dead-copy path may need to do the removal explicitly.
3. **PSUM.** `linalg.copy` on PSUM operands must still lower via the
   DMA-can't-touch-PSUM two-hop in `_emit_copy`. Lowering is by
   mem_space, unchanged — just triggered from the `linalg.copy` case.
4. **`linalg.copy` has a region/body.** Named op; builder is
   `linalg::CopyOp` with `ins`/`outs`. Simpler than a generic — no
   explicit region needed.

## Plan

1. ✅ **Emitter first (accept both).** `linalg.copy` → same `_emit_copy`
   body (both nisa + kernelbuilder emitters).
2. ✅ **Flip producers.** All create sites now emit `linalg::CopyOp`
   (transform ops, spill/reload, reshape, return, ref-impl, and the
   interim legalize copies). `SimplifyLinalg` dead-copy match handles
   both. Fixed two latent bad-tile bugs uncovered by validation (return
   copy + boundary transpose — see below). Full suite back to baseline.
3. **Do 3a/3b** (the other doc) on top: staging copies are now
   `linalg.copy`, tiled by the builtin `emitTile` — no hand-rolled loops.
4. **Step 4** deletes `tileCopyAndTranspose` + the legalize copy sites.
5. **Retire `memref.copy`** from our code paths; drop the emitter's
   `memref.copy` case once no pass emits it.

Breaking changes are acceptable: golden `.mlir` dumps under
`tests/e2e/outputs/` will show `linalg.copy`; only `test_elementwise.py`
asserts on copy text and will be updated.

## Order vs the other doc

Do steps 1–2 here first (unify the op), then return to
legalize-layout-high-rank-sbuf 3a/3b/4, which become trivial once every
copy is a tileable `linalg.copy`.

## Discovered issue during step 2: bad tile on a reduction's return path

Flipping producers to `linalg.copy` broke `test_reduce.py` (3D
`np.sum(axis=-1, keepdims)`, `8x128x64 → 8x128x1`). Investigation showed
**one bug class with two instances** on the reduction's return path.

### The shared root cause

The failing attribute is `loop_tile_size` (`nkipy.tile_op`), not
`sbuf_tile_size`. Key distinction:

- A **reduction**'s `loop_tile_size` is **iterator-space**: one entry
  per loop, including the reduced dim. The reduction here is `[1,128,64]`
  — axis-2 = 64 is the reduction loop over K. This is **correct** and
  must stay (dropping the 64 would drop the reduction loop).
- A **pure-parallel** op (transpose, copy, elementwise) has no reduction
  loop, so its `loop_tile_size` matches its **own shape**.

The bug: two places propagate the reduction's `[1,128,64]` onto a
downstream **pure-parallel** op whose shape is `...x1` (axis-2 = 1),
where a `64` is invalid:

1. **InferLayout `materializeReturnCopies`** copied the producer's
   `tile_op` verbatim onto the return copy buffer.
2. **CanonicalizePartitionDim `insertOutputTransposes`** seeds the
   inserted return-orientation transpose's tile from the component's
   `seedTileSizeAttr` (= the reduction knob `[1,128,64]`), not from the
   transpose's own `8x128x1` output.

Both yield a pure-parallel op with `tile[2]=64` vs `dim[2]=1`. When it's
later validated, `tile[2]=64 > dim[2]=1` errors. (The failing op_id=5 is
the return transpose `128x8x1 → 8x128x1`; nothing in it has a `64`.)

### Why it was latent before this migration

At HEAD, the final return-boundary op was a `memref.copy`, which has no
`TilingInterface` — knob-driven-tiling **skips** it, so its bad tile was
never validated. The `linalg.copy` migration reorganized the return path
so the boundary op is now a `linalg.transpose` (validated) → the
long-standing bad tile is finally exposed. It is *not* a classification
bug: a user `np.copy` is legitimately a knobbable elementwise op
(verified: `np.copy` + `.tile_op` lowers to `linalg.copy` and tiles
fine).

### Fix: a pure-parallel op derives its tile from its own shape

Only the **pure-parallel** boundary ops are wrong; the reduction's
iterator-space tile is untouched. For a pure-parallel op the loop tile
= its own shape, so use the shape-based defaults `defaultTileOps`
already uses:

1. `materializeReturnCopies`: give the copy
   `[min(shape[0],128), shape[1], ...]` (elementwise rule) from its own
   shape instead of copying the producer's `tile_op`. ✅ done.
2. `insertOutputTransposes`: clamp the boundary transpose's tile per-dim
   to its own output shape (an inherited reduction dim of 64 becomes 1
   on a size-1 axis), instead of using the reduction's `seedTileSizeAttr`
   as-is. ✅ done.

Both are always shape-valid, don't touch the reduction, and leave user
`np.copy` / knobbed ops untouched.

Rejected alternatives:
- Remove `linalg.copy` from `isNamedUnaryElementwiseOp` — breaks user
  `np.copy` knobs and papers over the bad tile.
- Make `validateElementwiseTileSize` tolerate the mismatch — hides a
  genuinely wrong tile.
