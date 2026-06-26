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

### 3.1 Frontend: emit SharedHbm on func args

In `builder.py`'s `begin_function()`:
- Func args: `memref<...xf32, #nkipy.mem<SharedHbm>>`
- All allocs (intermediates and outputs): no mem_space at trace time

Output allocs get `SharedHbm` later via `annotate-memory-space` Phase 1
(return type rewriting) until that pass is deleted in step 3.5.

Also: `mlir_utils.py` now owns the canonical `MEM_SPACE_MAP` dict and
`mem_space_attr(name)` helper. `knob.py` uses these instead of a local
copy. SubView/ReinterpretCast ops propagate the source's memory_space.

**mem_space assignment flow:** func args are born with SharedHbm.
Intermediate allocs are born bare — passes (`infer-layout`,
`knob-driven-tiling`) assign them to SBUF/PSUM via `nkipy.layout`.
Return values get SharedHbm from `annotate-memory-space` Phase 1.

We should NOT:
1. Stamp SharedHbm on intermediate or output allocs at trace time —
   `annotate-memory-space` still handles outputs and will conflict.
2. Propagate mem_space from func args into pass-created staging allocs
   (e.g. transpose buffers) — those are intermediates destined for SBUF.

### 3.2 Fix `PromoteTensorOp` copy-back for outputs written through subviews

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

### 3.3 Fix `RemoveZeroFillBeforeMatmul` to trace through subviews

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

### 3.4 Move `nkipy.layout` consumption to `legalize-layout`; erase knobs in `knob-driven-tiling`

Two small changes:

1. **`legalize-layout` prologue:** Walk `nkipy.layout` ops, set the
   target Value's memref type to include the annotated `mem_space`,
   then erase the `nkipy.layout` ops. (~20 lines, replaces Phase 2.)

2. **`knob-driven-tiling` cleanup:** Erase `nkipy.tile_op` and
   `nkipy.cache_op` knobs after consuming them — the pass that reads
   these annotations should own their removal.

No propagation step is needed because `knob-driven-tiling` already
stamps mem_space on all subviews it creates during tiling.

### 3.5 Delete `annotate-memory-space`

- Remove from pipeline (`pipeline.py`)
- Remove from `Passes.td`, `Passes.h`, `CMakeLists.txt`
- Delete `AnnotateMemorySpace.cpp`
- Update tests that use `stop_after='annotate-memory-space'`

### 3.6 Update `decomposeOneBatchMatmul` — don't erase layout annotations

Keep the user's `nkipy.layout(mem_space=SharedHbm)` on the output
buffer `%alloc`. This ensures `legalize-layout`'s prologue stamps
SharedHbm on it, and propagation carries SharedHbm to all subviews.
The inner matmul output goes to PSUM (via tiling promotion), gets
copied back to the SharedHbm subview — no SBUF intermediate needed.

## 4. Expected Outcome

- Pipeline: 13 → 12 passes (remove `annotate-memory-space`)
- batch_matmul HW tests pass (copy-back present, no SBUF OOM)
- No more "silent data loss" class of bugs from promotion without
  copy-back
- Cleaner separation: tiling handles promotion, legalize-layout
  handles physical factorization, no pass does both

## 5. Migration Checklist

- [x] 3.1 Frontend emits SharedHbm on args (outputs via annotate-memory-space until step 3.5)
- [ ] 3.2 Fix PromoteTensorOp copy-back
- [ ] 3.3 Fix RemoveZeroFillBeforeMatmul subview tracing
- [ ] 3.4a Move `nkipy.layout` consumption to legalize-layout prologue
- [ ] 3.4b Erase `tile_op`/`cache_op` knobs in knob-driven-tiling
- [ ] 3.5 Delete annotate-memory-space pass
- [ ] 3.6 Keep layout annotations in decomposeOneBatchMatmul
- [ ] Run full test suite, update FileCheck patterns
