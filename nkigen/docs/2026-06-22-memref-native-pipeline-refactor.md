# RFC: Remove Tensor Dialect — Memref-Native Pipeline

**Date:** 2026-06-22
**Status:** Proposal
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
| Uses `transform.structured.tile_using_for` (tensor-only) | Custom tiling: emit `scf.for` + `memref.subview` + tiled linalg op directly |
| Produces `tensor.extract_slice` / `tensor.insert_slice` around tiled ops | Produces `memref.subview` of input/output memrefs |
| Needs `promote_tensor` (alloc_tensor + materialize) for SBUF promotion | Direct `memref.alloc(sbuf)` + `memref.copy` |

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

### WI-1: Memref-native frontend (`builder.py` + `traced_array.py`)

**Scope:** Replace `TensorHandle` with `MemrefHandle`. Change all op builders to
produce linalg-on-memref.

**Sub-tasks:**
1. Replace `make_empty()` → `memref.AllocOp` (returns `memref<...>`)
2. Replace `ranked_tensor_of()` → `MemRefType.get()` throughout
3. Change `linalg.GenericOp` / named linalg calls to accept memref operands
   (linalg already supports this — the `outs` operand just needs to be memref)
4. Change `begin_function()` signature to `(memref<...>, ...) → ()`
5. Replace `__setitem__` (`tensor.insert_slice`) with `memref.subview` + store/copy
6. Replace `__getitem__` (`tensor.extract_slice`) with `memref.subview`
7. Remove `scf.for` tensor iter_args — loops just mutate memrefs in-place
8. Kill the DPS-rewiring hack (builder.py:1634-1656)

**Risk:** Linalg on memref doesn't use DPS (result = init buffer), so tiling
interface behavior differs. Verify `TilingInterface` implementations work on
memref-typed linalg ops.

### WI-2: Rewrite tiling to emit memref.subview

**Scope:** Rewrite `KnobDrivenTiling.cpp` to directly emit `scf.for` +
`memref.subview` instead of going through the Transform dialect's
`tile_using_for`.

**Sub-tasks:**
1. For each linalg op with a knob annotation:
   - Emit `scf.for` with the tile bounds
   - Use linalg's `makeTiledShapes` / `computeSliceParameters` (from
     `Linalg/Utils/Utils.cpp`) to compute subview offsets — these already
     emit `memref.subview` for memref-typed operands
   - Clone the linalg op with subviewed operands
   - Fold loop-step canonicalization as a final cleanup stage
2. Remove the Transform dialect dependency for tiling
3. Remove `apply-and-strip-transforms` pass
4. Implement SBUF promotion inline: after subview, emit
   `memref.alloc(sbuf)` + `memref.copy` for inputs that need promotion

**Risk:** We bypass the upstream `TileUsingSCFForOp` (which is tensor-only in
its loop generation), but reuse the shape computation utilities. Our tiling is
already heavily customized (knob-driven, not heuristic), so the upstream pass
was mostly a dispatch mechanism.

### WI-3: Simplify fusion pass

**Scope:** `KnobDrivenFusion.cpp` already fuses `scf.for` loops. Verify it
works without tensor iter_args and simplify any tensor-specific alias tracking.

**Sub-tasks:**
1. Remove any `tensor.extract_slice` → source walk logic
2. Verify loop fusion with memref subviews (shared base memref detection)

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

### WI-6: Update codegen backends

**Scope:** Both NISA and KernelBuilder backends already consume memref IR.
Verify they still work with the new pipeline output.

**Sub-tasks:**
1. `codegen/nisa/emit.py` — should be unchanged (already takes memref linalg)
2. `codegen/kernelbuilder/` — should be unchanged (already reads memref.alloc, subview, etc.)
3. Update test expectations for the new IR shapes

### WI-7: Enable KV cache / in-place aliasing

**Scope:** With memref-native IR, in-place buffer aliasing is trivial.

**Sub-tasks:**
1. Allow function args to be marked as "in-place" (read + write to same memref)
2. `memref.subview` at dynamic offset for KV cache update
3. No copy needed — caller and kernel share the same buffer

## Migration Strategy

**Incremental, phase by phase:**

1. **WI-1** first (frontend). Produce memref IR from tracing. Run existing
   passes with `--allow-unknown-ops` to verify linalg-on-memref propagates.
2. **WI-2** next (tiling). This is the hardest piece — replace tile_using_for.
   Keep old tiling behind a flag until new tiling passes all tests.
3. **WI-3 + WI-4** together (fusion + cleanup).
4. **WI-5** (delete bufferization). Only after all passes work on memref.
5. **WI-6 + WI-7** (codegen verification + KV cache).

Run the full test suite (`uv run pytest tests/ -n auto`) after each WI.
The 312 existing tests serve as the correctness oracle throughout.

## Open Questions

1. **Linalg TilingInterface on memref** — **RESOLVED:** upstream's
   `TileUsingSCFForOp` (the outer loop generation in `TileUsingInterface.cpp`)
   is hard-coded to tensor: it emits `tensor.insert_slice` for yield values and
   uses `tensor::getOrCreateDestinations`. However, linalg's *inner*
   `getTiledImplementation` + `makeTiledShapes` / `materializeTiledShape` DO
   support memref — they emit `memref.subview` via a TypeSwitch. So WI-2 must
   bypass the upstream `tile_using_for` transform op, but can reuse
   `makeTiledShapes` / `computeSliceParameters` from `Linalg/Utils/Utils.cpp`
   to compute subview offsets/sizes. The tiling pass emits `scf.for` +
   `memref.subview` + cloned linalg op directly (no insert_slice, no yield of
   tensor results).

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
