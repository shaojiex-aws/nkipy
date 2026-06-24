# SBUF & Knob Contract

**Date:** 2026-06-23 (updated 2026-06-24)  
**Status:** Design agreed

---

## Contract

1. **Dim 0 = partition** for on-chip tensors (enforced by `canonicalize-partition-dim`).
2. **SBUF allocs are 2D** at the hardware level: `[partition, free]`. Partition ≤128.
3. `.tile_op(tile_size=[...])` = **loop tile**. Controls iteration structure.
4. `.layout(mem_space="Sbuf")` = **full SBUF residency**. The entire tensor lives in SBUF. Physical factorization (tile) is auto-derived from the consuming `tile_op` + indexing maps.
5. `.cache(input, axis=[...])` = **SBUF staging for HBM tensors**. Cache buffer shape is derived from loop bounds at the specified axis level, with loop-iteration dims squeezed (dims whose extent = loop step at that level are not allocated).
6. No "strip leading 1s" heuristic — buffer shapes are always precisely derived from the cache level and indexing maps.

---

## SBUF Buffer Shape Derivation

SBUF buffers are always 2D at the hardware level: `[partition, free]`.

The buffer shape is derived, never specified by the user:

- **`.layout(mem_space="Sbuf")`**: physical factorization derived from consuming `tile_op` + indexing maps. `LegalizeLayout` attaches `sbuf_map` with the derived tile/blocks.
- **`.cache(input, axis=[...])`**: buffer shape = input's tile shape at the specified loop level, with loop-iteration dims squeezed. Dims whose extent equals the loop step (=1) at that cache level are not stored — they're the loop induction variable, not data.

Example: `q` is `[4, 128, 64]`, `tile_op=[1, 128, 64]`, `.cache(q, axis=[-1])`:
- Loop at axis 0 iterates over q's dim 0 (batch=4, step=1)
- Inside that loop, q's slice is `[1, 128, 64]`
- Dim 0 has extent 1 (= loop step) → squeezed
- Cache buffer: `memref<128x64, sbuf>` (partition=128, free=64)

Final 2D projection: dim 0 = partition, product(remaining) = free.

---

## tile_op, layout, and cache

Three knobs, each with a single clear role:

- `.tile_op(tile_size=[...])` = **loop tile**. Controls iteration structure. One entry per iterator.
- `.layout(mem_space=...)` = **full SBUF residency**. The entire tensor lives in SBUF. No `tile_size` parameter — physical factorization is auto-derived from the consuming `tile_op` via indexing maps.
- `.cache(input, axis=[...])` = **SBUF staging for HBM inputs**. Place a cache buffer at the specified post-tiling loop levels.

### Examples

```python
# Matmul: C[M,N] = A[M,K] @ B[K,N]
knob(C).tile_op(tile_size=[128, 128, 64])
       .layout(mem_space="Sbuf")
       .cache(A, axis=[0, 4])
       .cache(B, axis=[2, 4])
```

The compiler derives:
- C's SBUF factorization tile: [128, 128] (from tile_op, projecting M,N onto C)
- A's cache buffer shape at each level (from tile_op + indexing maps for A[M,K])
- B's cache buffer shape at each level (from tile_op + indexing maps for B[K,N])

```python
# Reduction: y[M] = sum(x[M, N], axis=-1)
knob(y).tile_op(tile_size=[128, 512])
       .layout(mem_space="Sbuf")
       .cache(x, axis=[-1])
```

```python
# Broadcast: y[B, M, N] = x[B, M, N] + bias[N]
knob(y).tile_op(tile_size=[1, 128, 512])
       .cache(x, axis=[-1])
       .cache(bias, axis=[0])    # bias reused across all iterations
```

---

## Full SBUF Residency (`.layout(mem_space="Sbuf")`)

```python
# x is (256, 256) — user wants the ENTIRE tensor resident in SBUF.
y = np.exp(x)
knob(y).tile_op(tile_size=[128, 128]).layout(mem_space="Sbuf")
```

- The logical alloc is `memref<256x256, sbuf>` — the **whole** 256×256 lives in SBUF
- Physical factorization tile [128, 128] is auto-derived from `tile_op` via indexing maps
- `LegalizeLayout` attaches `sbuf_map<tile: [128, 128], blocks: [2, 2]>`
- Physical shape becomes `[128, 2, 2, 128]` (partition=128, 2×2 block grid, free=128)
- Compute ops access individual 128×128 blocks via subview + collapse_shape

---

## SBUF Staging for HBM (`.cache()`)

```python
@trace(input_specs=[((4, 128, 64), "f32"), ((1, 128, 64), "f32")])
def kernel(q, freqs_cos):
    t = q * freqs_cos
    knob(t).tile_op(tile_size=[1, 128, 64])
           .layout(mem_space="SharedHbm")
           .cache(q, axis=[-1])
           .cache(freqs_cos, axis=[-1])
    return t
```

After tiling, `.cache(q, axis=[-1])` produces:

```mlir
scf.for %i = 0 to 4 step 1 {
  %q_slice = memref.subview %q[%i, 0, 0] [1, 128, 64] [1, 1, 1]
      : memref<4x128x64xf32, hbm> to memref<1x128x64xf32, hbm>

  // Cache buffer: loop-iteration dim (extent=1) squeezed → 2D
  %sbuf_buf = memref.alloc() : memref<128x64xf32, sbuf>
  // DMA reads from hbm[%i, 0:128, 0:64] into sbuf[0:128, 0:64]
}
```

The buffer is `128x64` (not `1x128x64`) because dim 0 has extent = loop step at this cache level — it's the loop IV, not data.

---

## Implementation Plan

### Step 4 (legacy, will be removed): Emitter >2D SBUF workarounds ✅

Steps 4a/4b/4c were workarounds for the "strip leading 1s" heuristic when temp SBUF allocs were naively copied from the tile shape. Once `.cache()` is implemented (Step 6), the cache pass produces clean 2D buffers directly — these workarounds become dead code and should be removed.

- 4a: emitter strips leading unit dims (`nkigen/codegen/nisa/emit.py`) — remove
- 4b: alloc pass strips leading 1s — remove
- 4c: LegalizeLayout "middle dims must be unit" restriction removed ✅ — keep (still correct)

### Step 5: Remove `tile_size` from `.layout()` API

Remove the `tile_size` parameter from `_KnobBuilder.layout()`. Physical factorization is always derived from the consuming `tile_op` + indexing maps by `InferLayout`. Update all existing `.layout(tile_size=...)` calls in tests to just `.layout(mem_space=...)`.

Files: `nkigen/frontend/knob.py`, `mlir/include/nkipy/Dialect/NkipyOps.td`, all tests with `.layout(tile_size=...)`

### Step 6: Implement `.cache()` primitive

Add `.cache(input, axis=[], prefetch=False)` to `_KnobBuilder`. Emit a new MLIR op (`nkipy.cache`) carrying the input reference, axis list, and prefetch flag. The cache pass replaces the current hardcoded input staging in knob-driven-tiling.

Buffer shape derivation: at each specified axis level, compute the input's tile shape from loop bounds and indexing maps, squeeze dims whose extent = loop step at that level.

Files: `nkigen/frontend/knob.py`, `mlir/include/nkipy/Dialect/NkipyOps.td`, new pass or modification to knob-driven-tiling

### Step 7: Remove hardcoded caching from knob-driven-tiling

Remove the fixed input-caching logic (matmul A-row / B-col caching). Default behavior becomes `axis=[-1]` (minimal staging) for all inputs without explicit `.cache()`. Add `.cache()` annotations to existing tests to preserve behavior.

Files: knob-driven-tiling pass, all matmul/attention tests
