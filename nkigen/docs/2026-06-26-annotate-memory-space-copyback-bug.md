# RFC: Remove `annotate-memory-space` — Inline Mem-Space Into Existing Passes

**Date:** 2026-06-26
**Status:** Proposed
**Affects:** batch_matmul e2e, any kernel where output is promoted HBM→SBUF

## 1. Problem Statement

`annotate-memory-space` is a 350-line pass doing four unrelated jobs:

1. Stamp `SharedHbm` on func args/results (trivial type rewriting)
2. Consume `nkipy.layout` → set mem_space on target memref types
3. **Create allocs + copies when mem_space conflicts** (the actual bug source)
4. Propagate mem_space through view chains (subview, collapse, expand, cast)

Job 3 is where the bug lives: it creates a staging SBUF alloc and a
copy-in (HBM→SBUF) when a subview is annotated as SBUF but its parent is
HBM. It never creates the copy-back (SBUF→HBM), so writes to the SBUF
buffer are lost. This breaks any kernel where an output subview is
promoted to SBUF (e.g., batch_matmul decomposition).

### Why does this pass even exist?

Historically: the pipeline traced into tensor IR, bufferized, and THEN
needed to stamp mem_space on the resulting memrefs (since tensor IR
can't carry mem_space). In the memref-native pipeline, the frontend
already emits `memref.alloc()` — we can assign mem_space much earlier
or let existing passes handle it.

## 2. Proposed Fix: Delete `annotate-memory-space`

The pass can be fully replaced by distributing its work into passes that
already exist:

| Current job | Move to | Rationale |
|-------------|---------|-----------|
| 1. Stamp SharedHbm on func args | Frontend (`builder.py`) | Emit `memref<...xf32, #nkipy.mem<SharedHbm>>` directly at trace time |
| 2. Consume `nkipy.layout` → type | `legalize-layout` prologue | Already walks all allocs; can set mem_space from layout annotations |
| 3. Create SBUF staging allocs + copies | **Delete entirely** | `knob-driven-tiling` already creates these via `PromoteTensorOp`. The conflict-resolution alloc+copy was a bufferization workaround |
| 4. Propagate mem_space through views | **Delete entirely** | `knob-driven-tiling` already attaches mem_space to subviews it creates; no propagation needed |

### Why Job 3 should be deleted (not moved)

The conflict-resolution logic (Phase 3 of `annotate-memory-space`)
exists because `infer-layout` annotates a subview with `Sbuf` while its
parent is `SharedHbm`. This creates a "conflict" that the pass resolves
by materializing an alloc+copy.

But this is the **wrong place** to do promotion. SBUF promotion should
happen during **tiling** (Phase 2), where `PromoteTensorOp` already
creates `memref.alloc(Sbuf) + memref.copy` for inputs AND outputs. The
reason it fails for batch_matmul outputs is a separate bug:
`PromoteTensorOp` doesn't find the DPS consumer through nested subviews
(see Section 3.2).

With proper tiling-time promotion, there is no mem_space conflict left
for `annotate-memory-space` to resolve — `knob-driven-tiling` already
attaches the correct mem_space to subviews it creates.

## 3. Step-by-Step Implementation Plan

### 3.1 Frontend: emit `nkipy.layout(SharedHbm)` on func args and outputs ✅

All memref types are bare throughout the pipeline (no mem_space on types).
Memory placement is expressed solely via `nkipy.layout` annotations:

- `begin_function()`: emits `nkipy.layout(mem_space=SharedHbm)` on each arg
- `finish_function()`: emits `nkipy.layout(mem_space=SharedHbm)` on return
  values (skips values that already have a user-provided layout)
- User code: `knob(x).layout(mem_space="Sbuf")` for intermediates

All annotations are applied to memref types in one shot by
`canonicalize-reshape`'s prologue (`applyMemSpaceAnnotations`), which
also propagates mem_space through view ops (subview, cast, etc.).

`mlir_utils.py` owns the canonical `MEM_SPACE_MAP` dict and
`mem_space_attr(name)` helper.

### 3.2 Fix `PromoteTensorOp` copy-back for outputs written through subviews ✅

Currently `PromoteTensorOp` inserts copy-back only when it finds a
direct `dpsConsumer` (a linalg op that has the promoted value as DPS
init). After batch_matmul decomposition, the matmul writes through
nested subviews — no direct DPS consumer is found.

Fix: when promoting a value that is NOT read (`!needsCopyIn`, meaning
it's a pure output), always insert copy-back before the block
terminator. The promoted buffer is written through subviews; the
copy-back flushes it to the original location after all writes complete.

```cpp
if (dpsConsumer) {
  rewriter.setInsertionPointAfter(dpsConsumer);
} else if (!needsCopyIn) {
  // Pure output — written through subviews, flush at end of scope.
  rewriter.setInsertionPoint(value.getParentBlock()->getTerminator());
}
auto copyBack = rewriter.create<memref::CopyOp>(...);
```

### 3.3 Fix `RemoveZeroFillBeforeMatmul` to trace through subviews ✅

After batch_matmul decomposition:
```
linalg.fill(0) → %alloc (3D)
scf.for {
  %slice = memref.subview %alloc[%b, 0, 0]
  linalg.matmul ... outs(%slice_of_slice)
}
```

The pattern looks for `fill(0)` whose output feeds a matmul directly.
It doesn't trace through subviews.

Fix: walk users of the fill output recursively through subviews. If ALL
terminal users are matmul-like (or subviews that eventually reach
matmul-like ops), the fill is redundant.

```cpp
static bool allUsersAreMatmulLikeThroughViews(Value fillOutput) {
  SmallVector<Value> worklist = {fillOutput};
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    for (Operation *user : v.getUsers()) {
      if (isMatmulLikeOp(user)) continue;
      if (auto sv = dyn_cast<memref::SubViewOp>(user)) {
        worklist.push_back(sv.getResult());
        continue;
      }
      return false;  // non-matmul, non-subview user
    }
  }
  return true;
}
```

### 3.4 Move `nkipy.layout` consumption to `canonicalize-reshape`; erase knobs in `knob-driven-tiling` ✅

Two small changes:

1. **`legalize-layout` prologue:** Walk `nkipy.layout` ops, set the
   target Value's memref type to include the annotated `mem_space`,
   then erase the `nkipy.layout` ops. (~20 lines, replaces Phase 2.)

2. **`knob-driven-tiling` cleanup:** Erase `nkipy.tile_op` and
   `nkipy.cache_op` knobs after consuming them — the pass that reads
   these annotations should own their removal.

No propagation step is needed because `knob-driven-tiling` already
stamps mem_space on all subviews it creates during tiling.

### 3.5 Delete `annotate-memory-space` ✅

- Remove from pipeline (`pipeline.py`)
- Remove from `Passes.td`, `Passes.h`, `CMakeLists.txt`
- Delete `AnnotateMemorySpace.cpp`
- Update tests that use `stop_after='annotate-memory-space'`

### 3.6 Update `decomposeOneBatchMatmul` — don't erase layout annotations ✅

Keep the user's `nkipy.layout(mem_space=SharedHbm)` on the output
buffer `%alloc`. This ensures `legalize-layout`'s prologue stamps
SharedHbm on it, and propagation carries SharedHbm to all subviews.
The inner matmul output goes to PSUM (via tiling promotion), gets
copied back to the SharedHbm subview — no SBUF intermediate needed.

## 4. Expected Outcome

- Pipeline: 14 → 13 passes (remove `annotate-memory-space`)
- batch_matmul HW tests pass (copy-back present, no SBUF OOM)
- No more "silent data loss" class of bugs from promotion without
  copy-back
- Cleaner separation: tiling handles promotion, legalize-layout
  handles physical factorization, no pass does both

## 5. Migration Checklist

- [x] 3.1 Frontend emits `nkipy.layout(SharedHbm)` on args and outputs
- [x] 3.2 Fix PromoteTensorOp copy-back (pure output → flush at block end)
- [x] 3.3 Remove fill(0) in decomposeOneBatchMatmul (PSUM zeros intrinsically)
- [x] 3.4a Apply `nkipy.layout` mem_space in `canonicalize-reshape` prologue
- [x] 3.4b Erase `tile_op`/`cache_op` knobs in `knob-driven-tiling`
- [x] 3.5 Delete `annotate-memory-space` pass (source, Passes.td, CMake, pipeline)
- [x] 3.6 N/A — layout annotations naturally preserved (no pass erases them early)
- [x] Run full test suite: 20 failed, 310 passed (matches pre-change baseline)

## 6. Remaining: batch_matmul HW accuracy failure

### The problem

`test_bmm_e2e` compiles to NISA but produces wrong results on HW.
Root cause: `decomposeOneBatchMatmul` erases the `nkipy.layout` on the
output alloc but doesn't transfer it to the decomposed output subview.

IR flow:

```mlir
// Before decomposition (00_input.mlir):
%alloc = memref.alloc() : memref<2x256x256xf32>
linalg.batch_matmul ... outs(%alloc)
nkipy.tile_op(%alloc) {loop_tile_size = [1, 128, 128, 128]}
nkipy.layout(%alloc) {mem_space = SharedHbm}     ← user says output is HBM

// After decomposition (01_canonicalize-compute.mlir):
%alloc = memref.alloc() : memref<2x256x256xf32>
scf.for %b {
  %subview_1 = memref.subview %alloc[%b, 0, 0] ...
  linalg.matmul ... outs(%subview_1)
  nkipy.tile_op(%subview_1) {loop_tile_size = [128, 128, 128]}  ← transferred ✓
  // nkipy.layout(SharedHbm) is MISSING on %subview_1            ← erased, not transferred ✗
}

// After infer-layout (02_infer-layout.mlir):
// infer-layout sees %subview_1 has tile_op but no layout
// → infers mem_space=Sbuf (wrong! it's a view of the HBM output)
nkipy.layout(%subview_1) {mem_space = Sbuf}       ← incorrect inference
```

Then `applyMemSpaceAnnotations` stamps `Sbuf` on `%subview_1`'s type,
but its source `%alloc` gets `SharedHbm` → verifier rejects the
mem_space mismatch on the subview.

### Why does infer-layout infer Sbuf?

`infer-layout` defaults to `Sbuf` for any value that has a `tile_op`
but no explicit `layout`. This is correct for standalone matmul (where
the output gets promoted to PSUM/SBUF by tiling). But for decomposed
batch_matmul, the output subview is a slice of the HBM return buffer —
it should stay `SharedHbm`.

### The fix

`decomposeOneBatchMatmul` should transfer the `nkipy.layout` annotation
to `initSlice` (just like it already transfers `nkipy.tile_op`). The
layout's `tile_size` drops the batch dim (same as tile_op); `mem_space`
carries through unchanged.

This way `infer-layout` sees that `%subview_1` already has
`mem_space=SharedHbm` and won't override it with `Sbuf`.

After that, tiling promotes the matmul output to PSUM (separate alloc
inside the tiled loop), computes there, and copies back to the
SharedHbm subview — which is exactly what the hardware needs.

### Second bug: linalg-to-nisa `view()` access pattern for >2D HBM

After the layout fix above, the pipeline compiles to NISA successfully
but neuronx-cc rejects it with an access-pattern-out-of-bounds assertion.

**Background:** `linalg.matmul` requires 2D operands, so the batch_matmul
decomposition MUST use rank-reducing subviews (`memref<2x256x256> →
memref<256x256, strided>`). The NISA codegen then flattens the >2D HBM
base buffer into a 2D `view()` for DMA ops.

For `memref<2x256x256xf32>` accessed through a rank-reducing subview,
the codegen produces:

```
view(memref<2x256x256xf32, ...>, f32, [2, 65536])[%iv + d0, col + d1]
```

The problem: `[2, 65536]` means dim0 has size 2 (the batch dim). But
`%iv + d0` adds the batch index (0 or 1) to the partition coordinate
`d0` (range 0..127). Result: index goes up to 128, exceeding size 2.

**Root cause:** `_emit_access_pattern` uses a `first_accessed` heuristic
to determine which base dim becomes the "row" of the 2D view. It scans
`tile_shape` for the first dim > 1. But `tile_shape` is `[128, 128]`
(already rank-reduced to 2D by the subview) — so `first_accessed = 0`
maps to base dim 0 (batch, size 2) instead of base dim 1 (M, size 256).

**The issue:** `_trace_access` returns `offsets` with one entry per base
dim (3 entries), but `tile_shape` is the val's type (2D after rank
reduction). There's no tracking of which base dims the tile dims
correspond to.

**Fix:** Two cases based on whether `_trace_access` collected more
offsets than tile dims (indicating a rank-reducing subview):

```python
if len(offsets) > len(tile_shape):
    # Rank-reducing subview dropped leading base dims.
    first_accessed = len(offsets) - len(tile_shape)
else:
    # Same rank — find first tile dim > 1 (skip unit dims).
    first_accessed = next(i for i, t in enumerate(tile_shape) if t > 1)
```

For bmm: `offsets=[%iv, row, col]` (3), `tile_shape=[128,128]` (2) →
`first_accessed = 3-2 = 1`, view becomes `[512, 256]`. ✓

For 3D SBUF temp: `offsets=[b, r, c]` (3), `tile_shape=[1,128,64]` (3)
→ else branch, first dim > 1 at index 1 → `first_accessed=1`,
view `[4, 8192]`. ✓
