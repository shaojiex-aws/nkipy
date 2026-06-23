# Strided HBM Copy Bug (Concat)

**Date:** 2026-06-23  
**Status:** Done ✅  
**Test:** `test_rope_3d_multi_partition` — PASSES

---

## The Problem

`np.concatenate([a, b], axis=-1)` produces wrong results on hardware.

The kernel computes two tensors of shape `(4, 128, 64)` and concatenates them
along the last axis into a `(4, 128, 128)` output.

## How Concat is Lowered (Before NISA Emit)

The pipeline lowers `np.concatenate` into two `memref.copy` operations that
write into subviews of the output allocation:

```mlir
// Output: contiguous (4, 128, 128) buffer
%out = memref.alloc() : memref<4x128x128xf32, hbm>

// First half: a → out[:, :, 0:64]
%sv0 = memref.subview %out[0, 0, 0] [4, 128, 64] [1, 1, 1]
     : memref<4x128x128xf32> to memref<4x128x64xf32, strided<[16384, 128, 1]>>
memref.copy %a, %sv0

// Second half: b → out[:, :, 64:128]
%sv1 = memref.subview %out[0, 0, 64] [4, 128, 64] [1, 1, 1]
     : memref<4x128x128xf32> to memref<4x128x64xf32, strided<[16384, 128, 1], offset: 64>>
memref.copy %b, %sv1
```

**Key point: the destination subview is non-contiguous (strided).**

## Memory Layout Visualization

The output buffer `(4, 128, 128)` is laid out in row-major order. Each
partition (dim 0) has 128 rows, and each row has 128 columns:

```
Partition 0, Row 0:  [ a_col0 a_col1 ... a_col63 | b_col0 b_col1 ... b_col63 ]
Partition 0, Row 1:  [ a_col0 a_col1 ... a_col63 | b_col0 b_col1 ... b_col63 ]
...
Partition 0, Row 127: [ a_col0 ... a_col63 | b_col0 ... b_col63 ]
Partition 1, Row 0:   [ a_col0 ... a_col63 | b_col0 ... b_col63 ]
...
```

The source tensor `a` has shape `(4, 128, 64)`, laid out contiguously:
- Stride: [8192, 64, 1] — each row is 64 elements, tightly packed.

The destination subview for `a` in `out[:, :, 0:64]` has:
- Stride: [16384, 128, 1] — each row is still 64 elements, but the NEXT
  row starts 128 elements later (skipping `b`'s columns).

## What Goes Wrong

Our NISA emitter flattens both src and dst to 2D using `view()`:

```mlir
// What we emit (WRONG):
src<4| 8192>=view(memref<4x128x64xf32> %a, f32, [4, 8192])[%c0 + d0, %c0 + d1]
dst<4| 8192>=view(memref<4x128x128xf32> %out, f32, [4, 16384])[%c0 + d0, %c0 + d1]
```

This treats BOTH as contiguous flat arrays. The DMA copies 8192 contiguous
elements from `a` and writes them to 8192 CONTIGUOUS positions in `out`
starting at offset 0.

But the correct behavior is to write 64 elements, skip 64 positions, write
the next 64, skip 64, etc. — interleaving with `b`'s space.

```
What we write (wrong):     [aaaa...8192 elements...aaaa][bbbb...8192...]
                           Fills columns 0-63 of rows 0-127 AND columns 64-127 of rows 0-63

What we should write:      Row 0: [aaa...64][___64 gap___]
                           Row 1: [aaa...64][___64 gap___]
                           ...128 rows...
```

## Why the Old Backend Worked

The old pattern-based NISA backend passed multi-dim shapes directly to
NISA's affine map builder. NISA knows that `memref<4x128x64xf32>` has stride
64 per row and `memref<4x128x128xf32>` has stride 128 per row — it computes
the correct addresses from the memref type, not from a flattened view.

Our emitter loses this information by flattening to 2D via `view()`.

## The Fix: Native >2D (No View) Except for HBM↔SBUF

NISA supports >2D natively. The `view()` hack is only needed for one case:
**HBM↔SBUF transfers**, because BIR requires src and dst to have the same
rank, and SBUF is always 2D.

For all other cases, we can use native >2D tile shapes and subscripts:

```mlir
// HBM-to-HBM copy (concat's second half): both sides are 3D, no view needed
nisa.dma_copy(
  dst<4| 128, 64>=memref<4x128x128xf32, hbm> %out[%c0 + d0, %c0 + d1, %c64 + d2],
  src<4| 128, 64>=memref<4x128x64xf32, hbm> %a[%c0 + d0, %c0 + d1, %c0 + d2]
) engine=dma
```

NISA sees that `%out` has 128 columns and `%a` has 64 columns. It knows
the stride per row differs between src and dst. It generates the correct
strided DMA. No flattening, no information loss.

### Emitter Rule

```
if both src and dst are HBM:
    → native >2D (per-dim tile shape, per-dim subscripts, no view)

if one side is HBM and the other is SBUF:
    → view() on the HBM side to make it 2D (BIR rank-matching)

if both sides are on-chip (SBUF/PSUM):
    → already 2D (projected by _memref_type_str_nisa), no view needed
```

This keeps the mental model simple:
- **`view()` only exists to solve the BIR rank-matching constraint** between
  HBM (>2D) and SBUF (always 2D).
- Everything else uses native shapes — NISA handles strides automatically.

### Step-by-Step Implementation

**Step 1: Add `_operand_str_multidim()` helper** ✅

New method that emits an operand with its full rank — no flattening, no view:
- Tile shape: `<par| d1, d2, ..., dN>` (commas between free dims)
- Subscripts: `[off0 + d0, off1 + d1, ..., offN + dN]` (one per dim)
- Memloc ref: `memref<original_shape> %name` (original type as-is)

**Step 2: Use it in `_emit_copy()` for HBM↔HBM** ✅

`_emit_copy()` already checks `src_is_hbm` and `dst_is_hbm`. When both are
True and rank > 2, call `_operand_str_multidim()` instead of `_operand_str()`.

The existing `_operand_str()` (which uses view()) continues to handle the
HBM↔SBUF case unchanged.

**Step 3: Test** ✅

- `test_rope_3d_multi_partition` — PASSES
- All 243 unit/pass tests — PASS (no regressions)

