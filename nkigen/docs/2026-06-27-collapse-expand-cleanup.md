# Proposal: Eliminate collapse_shape/expand_shape from the pipeline

**Date:** 2026-06-27
**Status:** Proposed

## 1. Design Principle

All reshapes should be `memref.reinterpret_cast` — a zero-cost view
with explicit offset/sizes/strides. No `memref.collapse_shape` or
`memref.expand_shape` in the IR.

Why:
- `collapse_shape`/`expand_shape` require reassociation maps that
  constrain which dims can be merged/split (contiguity rules)
- They complicate shape analysis: downstream passes must understand
  the reassociation semantics
- `reinterpret_cast` is fully explicit: new shape + strides, no hidden
  constraints
- The NISA emitter already handles `reinterpret_cast` natively (traces
  through it in `_trace_access`)

## 2. Current Status

**Frontend (builder.py):** clean. Only emits `reinterpret_cast`.

**C++ passes that still create collapse/expand:**

### CanonicalizeReshape (line 212, 222, 354)

Converts `memref.reshape` (from the old bufferization path) into
`collapse_shape(→1D) + expand_shape(→target)`. This exists because
`memref.reshape` requires a shape tensor operand and can't be directly
lowered. The comment says "FoldHbmReshapePattern in linalg-to-nisa
handles these."

**Can we eliminate?** Yes. We no longer bufferize — the frontend emits
`reinterpret_cast` directly. If any `memref.reshape` ops still enter
the pipeline, it's from an old code path that should be updated.

Also creates `expand_shape` (line 354) for the "returned expand_shape
of func args needs alloc+copy" pattern. This can be replaced with
`reinterpret_cast` + alloc+copy.

### SimplifyLinalg (line 163, 409)

Collapses >2D SBUF allocs to 2D for `linalg.transpose` (NISA
dma_transpose only supports 2D). Then expands back after.

```
%collapsed = memref.collapse_shape %sbuf_3d [[0,1],[2]]  // 3D → 2D
linalg.transpose ... outs(%collapsed)
%expanded = memref.expand_shape %collapsed [[0,1],[2]]   // 2D → 3D
```

**Can we eliminate?** Probably. The NISA emitter's `_trace_access`
already handles >2D SBUF tiles by projecting to 2D (par × free). We
could emit the transpose directly on the >2D memref and let the emitter
handle the 2D projection. Or use `reinterpret_cast` instead of
collapse/expand.

### LegalizeLayout (line 649, 650, 724, 725)

Collapses >2D SBUF tiles to 2D before emitting tiled `memref.copy` or
`linalg.transpose` in the block loop:

```
%in2D = memref.collapse_shape %inTile [[0,1],[2]]
%out2D = memref.collapse_shape %outTile [[0,1],[2]]
memref.copy %in2D, %out2D
```

**Can we eliminate?** Same as SimplifyLinalg — the collapse is a
workaround because `memref.copy` between 3D SBUF tiles was thought to
need 2D operands for NISA lowering. If the emitter handles >2D, we can
skip the collapse.

## 3. reinterpret_cast vs memref.reshape

| | `reinterpret_cast` | `memref.reshape` |
|---|---|---|
| Operands | static offset/sizes/strides | shape tensor (runtime) |
| Constraints | None — explicit strides | Source must be contiguous |
| Verifier | Minimal | Checks dynamic shape tensor |
| NISA emitter | Already handled | Needs conversion first |
| Use case | All contiguous reshapes | Dynamic shapes (not needed) |

**Recommendation: keep `reinterpret_cast`.** It's what the frontend
already emits, the NISA emitter traces through it, and it requires no
special handling in passes. `memref.reshape` adds a shape tensor operand
that complicates the IR for no benefit (we always know shapes statically).

## 4. Action Items

1. ✅ **Audit CanonicalizeReshape**: confirmed no `memref.reshape`,
   `collapse_shape`, or `expand_shape` enters the pipeline. Gutted the
   pass from 440→170 lines. Only keeps: apply mem_space annotations +
   materialize output allocs for returned views of func args.

2. [ ] **SimplifyLinalg**: try removing the collapse/expand around
   transpose. If NISA emitter handles >2D SBUF tiles correctly for
   transpose, the collapse is unnecessary.

3. [ ] **LegalizeLayout**: same — try removing collapse before tiled copy.
   The emitter should handle >2D copy operands.

4. [ ] **Long term**: add a verifier/lint that rejects `collapse_shape` /
   `expand_shape` in the IR after legalize-layout, to prevent future
   regressions.
