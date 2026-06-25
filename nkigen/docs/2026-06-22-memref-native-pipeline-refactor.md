# RFC: Remove Tensor Dialect — Memref-Native Pipeline

**Date:** 2026-06-22
**Status:** Accepted
**Priority:** Work Item #1 (prerequisite for KV cache / in-place aliasing)

## Motivation

The current pipeline traces user code into **tensor SSA IR**, then relies on
`one-shot-bufferize` to convert tensors into memrefs. This creates fundamental
problems:

1. **No aliasing control.** Tensor semantics are pure/functional — every
   `tensor.insert_slice` produces a new SSA value. The bufferizer *infers*
   in-place reuse but cannot be forced. KV cache (read + in-place update of a
   persistent buffer) is impossible to express.

2. **Bufferization surprises.** Unexpected copies appear when the alias
   analysis can't prove safety. We have 3 post-bufferize cleanup passes
   (`eliminate-uninitialized-copies`, `eliminate-same-memspace-copy`,
   `canonicalize-reshape`) working around this.

3. **Complexity.** The `promote_tensor` transform op uses
   `bufferization.alloc_tensor` + `bufferization.materialize_in_destination` —
   abstractions that exist solely because we're in tensor-land. In memref
   world this is just `memref.alloc` + `memref.copy`.

## Proposed New Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRACING (Python)                             │
│  User NumPy code → linalg ops on MEMREF + memref.alloc/subview      │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1: CANONICALIZATION                                          │
│  • canonicalize-linalg-for-nisa (matmul prep, arithmetic prep,      │
│    batch-matmul decomposition)                                      │
│  • infer-layout (propagate mem_space + partition_dim)               │
│  • canonicalize-partition-dim (insert transposes for pdim=0)        │
│  • assign-linalg-op-ids                                             │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 2: LOOP TILING                                               │
│  • knob-driven-tiling (emit scf.for + memref.subview;               │
│    includes loop-step canonicalization as final stage)              │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 3: FUSION                                                    │
│  • knob-driven-fusion (fuse sibling scf.for loops)                  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 4: LAYOUT LEGALIZATION                                       │
│  • annotate-memory-space (HBM / SBUF / PSUM assignment)             │
│  • insert SBUF promotion (memref.alloc in SBUF + memref.copy)       │
│  • legalize-layout (2D → 4D physical layout for SBUF)               │
│  • canonicalize-reshape                                             │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 5: SCHEDULING                                                │
│  • simplify-linalg (decompose high-rank transpose, etc.)            │
│  • insert-spill-reload (SBUF memory pressure management)            │
│  • insert-memref-dealloc (lifetime endpoints)                       │
│  • CSE + canonicalize                                               │
│  • (future: instruction scheduling / software pipelining)           │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 6: CODEGEN                                                   │
│  • Backend A: linalg-to-nisa (→ NKI NISA instructions)              │
│  • Backend B: linalg-to-kernelbuilder (→ nb.compiler.* Python)      │
└─────────────────────────────────────────────────────────────────────┘
```

## What Changes

### Frontend (Python tracing layer)

| Current | New |
|---------|-----|
| `tensor.empty` → `TensorHandle` | `memref.alloc` → `MemrefHandle` |
| `linalg.GenericOp([tensor_type], [inputs], [output_tensor])` | `linalg.GenericOp([memref_type], [inputs], [output_memref])` |
| `tensor.insert_slice` / `tensor.extract_slice` | `memref.subview` (returns a view, no copy) |
| `scf.for` with tensor iter_args (loop-carried SSA) | `scf.for` operating on memrefs in-place (no iter_args needed for buffers) |
| Function signature: `(tensor<...>, ...) → tensor<...>` | Function signature: `(memref<...>, ...) → ()` (side-effecting) |

### Tiling (KnobDrivenTiling.cpp)

| Current | New |
|---------|-----|
| Uses `transform.structured.tile_using_for` | **Same** — `tile_using_for` works on memref ops (zero results → no insert_slice) |
| Produces `tensor.extract_slice` / `tensor.insert_slice` around tiled ops | Produces `memref.subview` of input/output memrefs (automatic via TilingInterface) |
| Needs `promote_tensor` (alloc_tensor + materialize) for SBUF promotion | `promote_memref` or dual-mode promotion: `memref.alloc(sbuf)` + `memref.copy` |

### Passes Deleted

These become unnecessary:

- `one-shot-bufferize` (no tensors to bufferize)
- `eliminate-uninitialized-copies` (we control allocation explicitly)
- `eliminate-same-memspace-copy` (promotion is explicit, no redundant copies)
- `promote_tensor` transform op (replaced by direct memref.alloc + copy)
- `NkipyTransformOps.cpp::findExistingMemSpace` (no need to walk alias chains)
- The DPS-output-rewiring hack in `builder.py:1634-1656`

### Passes Modified

- `InferLayout.cpp` — operates on linalg ops; these work on memref too, minimal change
- `CanonicalizeReshape.cpp` — already post-bufferize, just remove tensor handling
- `LegalizeLayout.cpp` — already operates on memref
- `AnnotateMemorySpace.cpp` — already operates on memref

## Detailed Work Items

### WI-1: Memref-native frontend (`builder.py` + `traced_array.py`) ✅

**Scope:** Replace `TensorHandle` with `MemrefHandle`. Change all op builders to
produce linalg-on-memref.

**Sub-tasks:**
1. Replace `make_empty()` → `memref.AllocOp` (returns `memref<...>`) ✅
2. Replace `ranked_tensor_of()` → `MemRefType.get()` throughout ✅
3. Change `linalg.GenericOp` / named linalg calls to accept memref operands ✅
4. Change `begin_function()` signature to `(memref<...>, ...) → memref<...>` ✅
5. Replace `__setitem__` (`tensor.insert_slice`) with `memref.subview` + copy ✅
6. Replace `__getitem__` (`tensor.extract_slice`) with `memref.subview` ✅
7. Remove `scf.for` tensor iter_args — loops just mutate memrefs in-place ✅
8. DPS-rewiring: `finish_function` uses `p._value.type` — no separate hack needed ✅

**Implementation:** Gated behind `backend="memref"` flag on `@trace()`. Tensor
path unchanged — all 303 passing tests remain green. Dual-mode via thin
abstraction layer (`_make_output`, `_linalg_result_types`, `_linalg_result`) that
dispatches on global `_backend_mode`. The `take` (GatherOp) is not yet dual-mode
since it's a custom dialect op lowered directly by NISA — deferred to WI-6.

**Files changed:** `mlir_utils.py`, `frontend/builder.py`, `frontend/trace.py`

**Risk:** ~~Linalg on memref doesn't use DPS (result = init buffer), so tiling
interface behavior differs.~~ **RESOLVED:** `tileUsingSCF` handles zero-result
(memref) ops correctly — it only emits `tensor.insert_slice` for loop-carried
values, which don't exist for memref ops. `getTiledImplementation` already emits
`memref.subview` for memref operands via `makeTiledShapes`.

### WI-2: Adapt tiling pipeline for memref ✅

**Scope:** Keep `KnobDrivenTiling.cpp` and the Transform dialect approach.
`tile_using_for` already works on memref linalg ops (zero results → no
`tensor.insert_slice` needed, `makeTiledShapes` emits `memref.subview`). The
main work is replacing tensor-based promotion with memref promotion.

**Sub-tasks:**
1. `PromoteTensorOp` made dual-mode: memref path emits `memref.alloc(sbuf)` +
   `memref.copy` (copy-in) + `memref.copy` (copy-back after DPS consumer) ✅
2. `TransposeMatmulOp` skipped in memref mode (upstream op is tensor-only);
   matmul blocking works without the LHS transpose ✅
3. `KnobDrivenTiling` knob extraction updated: for zero-result ops (memref
   linalg), match on DPS init operands instead of op results ✅
4. `findExistingMemSpace` updated to walk through `memref.alloc` and
   `memref.subview` aliasing chains ✅
5. End-to-end verified: elementwise and matmul tiling produce correct
   `scf.for` + `memref.subview` + SBUF promotion IR ✅

**Deferred:**
- Remove `one-shot-bufferize` and post-bufferize cleanup passes from memref
  pipeline path (WI-4/WI-5 — needs a pipeline flag to gate tensor-only passes)

**Files changed:** `NkipyTransformOps.cpp`, `KnobDrivenTiling.cpp`,
`mlir/lib/TransformOps/CMakeLists.txt`

**Risk:** Low. Tensor path unchanged (all 303 tests pass). Memref path produces
correct tiled IR but downstream passes (annotate-memory-space, legalize-layout,
NISA emit) not yet adapted.

### WI-3: Simplify fusion pass ✅

**Scope:** `KnobDrivenFusion.cpp` fuses `scf.for` loops. Adapted for memref
(zero-result loops, no tensor iter_args).

**Sub-tasks:**
1. `findProducingForLoop` extended: memref path walks users to find the
   outermost enclosing `scf.for` (since memref ops don't produce values) ✅
2. `hoistSetupOpsBetween` made dominance-aware: only hoists ops whose
   operands all dominate the hoist point (needed for memref.alloc between
   loops that must move above the fused loop) ✅
3. `eraseExtraYields`: new helper removes duplicate `scf.yield` ops left
   by `fuseIndependentSiblingForLoops` when fusing zero-result loops ✅
4. Verified: tensor fusion tests unchanged (4/4 pass), memref fusion
   end-to-end with LLVM correctness check ✅

**Files changed:** `mlir/lib/Transforms/KnobDrivenFusion.cpp`

### WI-4: Clean up layout legalization

**Scope:** Remove tensor-related dead code from post-bufferize passes.

**Sub-tasks:**
1. Delete `eliminate-uninitialized-copies` pass
2. Delete `eliminate-same-memspace-copy` pass  
3. Simplify `AnnotateMemorySpace` (no tensor→memref boundary to reason about)
4. Simplify `CanonicalizeReshape` (only memref reshape ops remain)

### WI-5: Delete bufferization infrastructure

**Scope:** Remove all tensor/bufferization dialect usage.

**Sub-tasks:**
1. Remove `BufferizableOpInterface` from `NkipyOps.cpp` (LayoutOp, TileOp, GatherOp)
2. Remove `bufferization.alloc_tensor` / `materialize_in_destination` from
   `NkipyTransformOps.cpp`
3. Remove `one-shot-bufferize` from pipeline
4. Remove `bufferization` dialect registration from `nkipy-opt.cpp`
5. Remove `_zero_fill_empty_tensors_ir()` from `execution/llvm.py`

### WI-6: Update codegen backends ✅

**Scope:** Both NISA and KernelBuilder backends already consume memref IR.
Verify they still work with the new pipeline output.

**Sub-tasks:**
1. `codegen/nisa/emit.py` — unchanged (already takes memref linalg) ✅
2. `codegen/kernelbuilder/` — unchanged (already reads memref.alloc, subview, etc.) ✅
3. Custom `nkipy.transpose_matmul` transform op for memref matmul path ✅
4. Fix `RemoveZeroFillBeforeMatmul` for memref (in-place fill, no results) ✅
5. End-to-end verified: memref matmul emits `nisa.matmul`, produces identical
   NISA IR to tensor path ✅

**Implementation:**
- `NkipyTransposeMatmulOp` in `NkipyTransformOps.cpp`: rewrites `linalg.matmul`
  → `linalg.transpose` + `linalg.matmul_transpose_a` for memref operands
  (allocates transpose buffer in SBUF). Upstream `TransposeMatmulOp` only
  handles tensor, so this custom op fills the gap.
- `MatmulPrep.cpp` memref fill removal: `linalg.fill(0)` on memref has no
  results (writes in-place). The pattern now checks that a matmul-like DPS init
  consumes the same memref, then erases the fill.
- HW execution (`Mode.HW`) passes for elementwise. Standalone matmul HW fails
  due to a pre-existing neuronx-cc limitation (affects tensor path identically).

**Files changed:** `NkipyTransformOps.td`, `NkipyTransformOps.cpp`,
`KnobDrivenTiling.cpp`, `MatmulPrep.cpp`, `test_memref_backend.py`

### WI-7: Enable KV cache / in-place aliasing

**Scope:** With memref-native IR, in-place buffer aliasing is trivial.

**Sub-tasks:**
1. Allow function args to be marked as "in-place" (read + write to same memref)
2. `memref.subview` at dynamic offset for KV cache update
3. No copy needed — caller and kernel share the same buffer

## Migration Strategy

**Incremental, phase by phase:**

1. **WI-1** ✅ (frontend). Produce memref IR from tracing, gated behind
   `backend="memref"`. Tensor path unchanged.
2. **WI-2** next (tiling). Much simpler than originally scoped — `tile_using_for`
   already works on memref linalg ops. Main work: replace `PromoteTensorOp` with
   memref-aware promotion (`memref.alloc(sbuf)` + `memref.copy`).
3. **WI-3 + WI-4** together (fusion + cleanup).
4. **WI-5** (delete bufferization). Only after all passes work on memref.
5. **WI-6 + WI-7** (codegen verification + KV cache).

Run the full test suite (`uv run pytest tests/ -n auto`) after each WI.
The 312 existing tests serve as the correctness oracle throughout.

## Open Questions

1. **Linalg TilingInterface on memref** — **RESOLVED:** `tileUsingSCF` handles
   zero-result (memref) ops correctly. It only creates `tensor.insert_slice` for
   loop-carried values — memref linalg ops have zero results, so that code path
   doesn't execute. `getTiledImplementation` + `makeTiledShapes` emit
   `memref.subview` for memref operands. **We can keep `tile_using_for` as-is.**
   No bypass needed — the original concern was wrong.

2. **scf.for without iter_args** — currently loop-carried values (accumulators)
   use tensor iter_args. With memref, the accumulator is just a memref that gets
   written in the loop body. The `scf.for` has no results. Does this interact
   badly with any downstream pass that expects loop results?

3. **Function return convention** — currently `func.func` returns a tensor
   (the output). With memref-native, the output is a function argument (memref)
   that's written in-place. The function returns void. Does this affect the
   driver's handling of results?

4. **Custom ops** — `custom_op.py` emits `func.func private` with tensor
   signatures and inlines NISA bodies. These need to be migrated to memref
   signatures too.
