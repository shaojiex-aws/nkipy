# Materialize boundary copies

**Date:** 2026-06-28
**Status:** Proposed

## Problem

Several cases need copy + tile_op insertion that currently aren't handled:

1. **SBUF return value**: user annotates result as SBUF, but hardware
   requires return values in SharedHbm. Need SBUF→HBM copy.

2. **HBM-to-HBM transpose**: `canonicalize-partition-dim` can produce
   a transpose where both source and dest are SharedHbm. Hardware DMA
   transpose requires at least one side in SBUF. Need to stage through
   SBUF (load tile → SBUF, transpose → HBM).

3. **HBM-to-HBM copy** (general): any `memref.copy` between two HBM
   buffers needs to stage through SBUF with tiling.

## Where in the pipeline

```
infer-layout          ← annotates mem_space, tile_op
canonicalize-partition-dim ← may insert transposes
knob-driven-tiling    ← generates scf.for loops from tile_ops
                      ← ★ materialize-boundary-copies HERE ★
canonicalize-reshape
legalize-layout
```

**After knob-driven-tiling** is the right place because:
- By then, all user-facing tile_ops have been consumed into loops
- We can see which copies/transposes are untiled (no enclosing loop)
- We can insert new tile_ops + loops for the boundary copies
- It runs before legalize-layout, which needs everything properly tiled

## What the pass does

Walk all `linalg.transpose` and `memref.copy` ops that are NOT inside
an `scf.for` (i.e., untiled). For each:

### Case 1: SBUF return value

```mlir
// Before:
%sbuf_result = memref.alloc() : memref<128x64xf32, #nkipy.mem<Sbuf>>
linalg.reciprocal ... outs(%sbuf_result)
return %sbuf_result

// After:
%sbuf_result = memref.alloc() : memref<128x64xf32, #nkipy.mem<Sbuf>>
linalg.reciprocal ... outs(%sbuf_result)
%hbm_out = memref.alloc() : memref<128x64xf32, #nkipy.mem<SharedHbm>>
memref.copy %sbuf_result, %hbm_out  // tiled by this pass
return %hbm_out
```

### Case 2: HBM-to-HBM transpose (untiled)

```mlir
// Before:
linalg.transpose ins(%hbm_src) outs(%hbm_dst) permutation=[1,0]

// After (tile into loop, stage through SBUF):
scf.for %i ... {
  %sbuf_tile = memref.alloc() ...
  %src_slice = memref.subview %hbm_src[%i, ...] [tile] ...
  memref.copy %src_slice, %sbuf_tile           // HBM → SBUF load
  %dst_slice = memref.subview %hbm_dst[%i, ...] [tile] ...
  linalg.transpose ins(%sbuf_tile) outs(%dst_slice)  // SBUF → HBM transpose
}
```

### Case 3: HBM-to-HBM copy (untiled)

Same as case 2 but without permutation — just stage through SBUF.

## Interaction with existing passes

- **infer-layout** stays simple: just annotates. If a user says
  result is SBUF, that's fine.
- **builder.py `finish_function`**: currently annotates unannotated
  return values as SharedHbm. Keep this — it handles the common case.
  The new pass handles the case where the user explicitly overrides.
- **canonicalize-partition-dim**: can freely insert transposes without
  worrying about mem_space staging — this pass cleans it up.
- **legalize-layout**: by the time it runs, all SBUF allocs are
  properly tiled and all boundary copies go through SBUF.

## Func args

Func args are already annotated as SharedHbm by `builder.py` at trace
time (line 194). infer-layout doesn't touch them. No change needed.
