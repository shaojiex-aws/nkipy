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

## Gotchas

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

1. **Emitter first (accept both).** Add `linalg.copy` → same `_emit_copy`
   body (delegate by mem_space), fix `_dst_operand`. Now IR with either
   op lowers. No behavior change yet.
2. **Flip producers.** Change the ≈8 non-legalize create sites to
   `linalg::CopyOp`. Update `SimplifyLinalg` match. Run full suite.
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
