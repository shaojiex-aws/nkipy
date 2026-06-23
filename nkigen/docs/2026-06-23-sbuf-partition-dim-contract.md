# SBUF Partition Dim Contract

**Date:** 2026-06-23  
**Status:** Design agreed

---

## Contract

1. **Dim 0 = partition** for on-chip tensors (enforced by `canonicalize-partition-dim`).
2. **SBUF allocs are 2D** at the hardware level: `[partition, free]`.
3. The IR may have >2D SBUF allocs. The mapping to physical 2D is: first non-unit dim = partition, rest combined into free.
4. `.tile_op(tile_size=[...])` = **loop tile**. Controls iteration structure.
5. `.layout(tile_size=[...])` = **SBUF resident size**. How much data is physically in SBUF at once. Only meaningful for `mem_space="Sbuf"`. Not needed for HBM.
6. For user-specified SBUF (`.layout(mem_space="Sbuf")`): shape is exactly what the user specifies.
7. For compiler-generated temp SBUF (staging for HBM compute): strip leading unit dims, then collapse to 2D.

---

## Mapping >2D SBUF to Physical 2D

Any SBUF shape maps to `[partition, free]` by:
1. Strip leading dims of size 1
2. First remaining dim = partition
3. Product of all remaining dims after that = free

Examples:
- `(128, 64)` → par=128, free=64
- `(1, 128, 64)` → strip → `(128, 64)` → par=128, free=64
- `(1, 128, 64, 2, 1)` → strip → `(128, 64, 2, 1)` → par=128, free=128
- `(1, 1, 128, 64)` → strip → `(128, 64)` → par=128, free=64
- `(1, 128)` → already 2D, no strip → par=1, free=128

This applies uniformly to sbuf_map tile shapes AND temp SBUF allocs. No restriction on "middle dims must be unit". LegalizeLayout should NOT convert SBUF to HBM because of complex tile shapes — it should just compute the physical 2D mapping.

---

## tile_op vs layout tile_size

They are different things:

- `.tile_op(tile_size=[128, 128, 128])` on a matmul: the loop iterates M=128, N=128, K=128.
- `.layout(tile_size=[128, 128])` on the matmul output: the output C has 128×128 in SBUF at once.

For HBM targets, `.layout(tile_size=...)` is not needed. The compiler creates temp SBUF buffers shaped by slicing operands with the loop tile. Those temps follow the "strip leading 1s" rule above.

---

## User-Specified SBUF: Large Tensor

```python
# x is (256, 256) — user wants it in SBUF but only 128×128 resident at once
y = np.exp(x)
knob.knob(y).tile_op(tile_size=[128, 128]).layout(mem_space="Sbuf", tile_size=[128, 128])
```

- The logical alloc is `memref<256x256, sbuf>`
- `.layout(tile_size=[128, 128])` says only 128×128 is resident at once
- `LegalizeLayout` attaches `sbuf_map<tile: [128, 128], blocks: [2, 2]>`
- The tiling pass generates loops over the 2×2 blocks
- Each iteration, one 128×128 tile is physically in SBUF
- The emitter handles folding if partition > 128

---

## The Problem (Temp SBUF for HBM)

```python
@trace(input_specs=[((4, 128, 64), "f32"), ((1, 128, 64), "f32")])
def kernel(q, freqs_cos):
    t = q * freqs_cos
    knob.knob(t).tile_op(tile_size=[1, 128, 64]).layout(mem_space="SharedHbm")
    return t
```

After tiling:

```mlir
scf.for %i = 0 to 4 step 1 {
  %q_slice = memref.subview %q[%i, 0, 0] [1, 128, 64] [1, 1, 1]
      : memref<4x128x64xf32, hbm> to memref<1x128x64xf32, hbm>

  // Current (WRONG): copies 3D shape verbatim
  %sbuf_buf = memref.alloc() : memref<1x128x64xf32, sbuf>

  // Correct: strip leading 1, result is 2D
  %sbuf_buf = memref.alloc() : memref<128x64xf32, sbuf>
  // DMA reads from hbm[%i, 0:128, 0:64] into sbuf[0:128, 0:64]
}
```

---

## Implementation Plan

### Step 4a: Emitter handles >2D SBUF mechanically ✅

Temp SBUF allocs (compiler-generated for HBM staging) can be >2D due to loop tiling artifacts — e.g., `tile_size=[1, 128, 64]` on a 3D tensor produces `memref<1x128x64, sbuf>`. The leading 1 is meaningless (it's just the batch iteration variable held constant inside the loop body).

The emitter strips these leading unit dims, then collapses to 2D: `[dim0, product(dims[1:])]`. Dim 0 after stripping is always partition (guaranteed by `canonicalize-partition-dim`).

This applies to temp SBUF only (no `sbuf_map` attribute). For user-specified SBUF with `sbuf_map`, the emitter uses the physical dimensions from the map directly.

Files: `nkigen/codegen/nisa/emit.py`

### Step 4b: Temp SBUF allocs strip leading unit dims

The pass that creates HBM→SBUF copies must produce 2D allocs. Take the source subview shape, strip leading 1s, collapse to 2D. The HBM side keeps its original rank with offsets.

Files: need to identify which pass creates these (bufferization or annotate-memory-space)

### Step 4c: Remove LegalizeLayout "middle dims must be unit" restriction ✅

Removed the block that converted SBUF allocs to SharedHBM when middle tile dims were non-unit. Any tile shape now gets `sbuf_map` attached and goes through copy-tiling normally. The emitter handles the physical 2D mapping regardless of tile shape.

Files: `mlir/lib/Transforms/LegalizeLayout.cpp`

Files: `mlir/lib/Transforms/LegalizeLayout.cpp` (lines ~397-449)
