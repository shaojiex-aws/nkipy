# Materialize boundary copies

**Date:** 2026-06-28
**Status:** Proposed

## Problem

When mem_space boundaries are crossed or output aliasing is needed,
copies must be inserted:

1. **SBUF return value**: user annotates result as SBUF, but hardware
   requires return values in SharedHbm. Need SBUF→HBM copy.

2. **HBM-to-HBM transpose**: `canonicalize-partition-dim` can produce
   a transpose where both source and dest are SharedHbm. Hardware DMA
   transpose requires at least one side in SBUF.

3. **Matmul output to HBM**: matmul computes in PSUM, copies to SBUF.
   If the output needs to be in HBM (return value or user annotation),
   an additional SBUF→HBM copy is needed.

4. **HBM-to-HBM copy**: e.g. `return x.reshape(128, 256)` — the
   reshaped view aliases the input. `canonicalize-reshape` inserts a
   contiguous HBM alloc + `nisa.dma_copy` (HBM DMA engine handles
   this directly, no SBUF staging needed). Already works today.

## Principle

No separate pass. Each pass that creates a boundary-crossing op is
responsible for tiling it:
- Either tile directly (explicit `scf.for` + subviews), or
- Attach `nkipy.tile_op` so knob-driven-tiling handles it

## Implementation steps

### Step A ✅: SBUF return value → insert copy in infer-layout

infer-layout's `defaultLayouts` marks return values as SharedHbm.
If a user explicitly annotates a return value as SBUF, infer-layout
should detect this (SBUF alloc flowing to `func.return`) and insert
an HBM alloc + copy + tile_op:

```mlir
// User wrote: knob(result).layout(mem_space="Sbuf"); return result
// infer-layout inserts:
%hbm_out = memref.alloc() : memref<...xf32>
nkipy.layout(%hbm_out) {mem_space = SharedHbm}
memref.copy %result, %hbm_out
nkipy.tile_op(%hbm_out) {loop_tile_size = ...}
return %hbm_out
```

knob-driven-tiling then tiles the copy.

### Step B ✅: HBM-to-HBM transpose (already handled)

`nisa.dma_transpose` requires one side in SBUF. An HBM-to-HBM
transpose must be decomposed into:
1. `dma_transpose` HBM → SBUF (transpose + load into temp buffer)
2. `dma_copy` SBUF → HBM (store to destination)

This is already handled by knob-driven-tiling's promotion. When the
user writes `np.transpose(hbm_input)` with output in SharedHbm, the
tiling + promotion produces exactly this sequence. The `knob.py`
validation (no `partition_dim` on non-SBUF) prevents
`canonicalize-partition-dim` from creating HBM-to-HBM transposes.
No additional work needed.

Generated NISA (2 ops per tile — `buildTransposeTiling` only
promotes the output, so `dma_transpose` reads HBM directly):

```mlir
scf.for %i = ... {
  nisa.dma_transpose(dst=sbuf %buf, src=hbm %arg0[tile], perm=[1,0])  // HBM→SBUF
  nisa.dma_copy(dst=hbm %out[tile], src=sbuf %buf)                    // SBUF→HBM
}
```

### Step C ✅: Matmul SBUF→HBM copy (already handled)

knob-driven-tiling's promotion generates all copies at tile size
inside the tiling loops. The per-tile flow:

```mlir
scf.for %block_m = ... {
  scf.for %block_n = ... {
    scf.for %tile_m = ... {
      scf.for %tile_n = ... {
        // Load output init HBM→SBUF→PSUM
        nisa.dma_copy(dst=sbuf %init, src=hbm %out[tile])
        nisa.tensor_copy(dst=psum %acc, src=sbuf %init)
        // Matmul accumulate
        scf.for %k = ... {
          nisa.matmul(dst=psum %acc, ...)
        }
        // Store PSUM→SBUF→HBM
        nisa.tensor_copy(dst=sbuf %buf, src=psum %acc)
        nisa.dma_copy(dst=hbm %out[tile], src=sbuf %buf)
      }
    }
  }
}
```

All copies are already at tile size — `tileMemrefCopy` in
LegalizeLayout is a no-op for this case (tile-sized SBUF allocs
don't have sbuf_map, so it skips them).

`tileMemrefCopy` is still needed for `canonicalize-reshape` copies
(large SBUF allocs with sbuf_map copied to HBM). Removal deferred
until copy insertion moves before knob-driven-tiling.

### Step D: HBM-to-HBM copy (canonicalize-reshape)

`canonicalize-reshape` inserts a copy when a non-contiguous HBM view
(e.g. `return x.reshape(...)`) needs to be materialized as a
contiguous output. The copy needs SBUF staging (HBM→SBUF→HBM per
tile).

Currently `canonicalize-reshape` runs after knob-driven-tiling and
the copy is tiled by `tileMemrefCopy` in LegalizeLayout. When we
move `canonicalize-reshape` before knob-driven-tiling, it attaches a
tile_op to the copy and knob-driven-tiling tiles it with SBUF staging
(same as any HBM↔SBUF copy). Then `tileMemrefCopy` can be removed.
