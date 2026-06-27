# Fix: matmul e2e — transpose tiling OOB and matmul row_pos

## Problem

`test_memref_e2e[matmul]` failed with neuronx-cc exit code 70 (two
cascading issues).

## Context

The tiled matmul for `C[256,256] = A[256,128] @ B[128,256]` is lowered
as `C += A^T · B`. The hardware matmul engine reads both operands from
SBUF, so A and B must be loaded from HBM into SBUF first. Since the
engine needs A in transposed layout, the load of A is a DMA transpose.

```python
# Pseudocode of the full tiled computation (tile_size=[128, 128, 64])
A_t = sbuf_alloc([128, 256])   # A transposed, sbuf_map<tile:[64,128], blocks:[2,2]>
B_s = sbuf_alloc([128, 256])   # B copy,       sbuf_map<tile:[64,128], blocks:[2,2]>

# Step 1: Load A^T into SBUF (DMA transpose A[256,128] → A_t[128,256])
for iv0 in range(2):           # iterate over SBUF's 2×2 block grid
    for iv1 in range(2):
        A_t[iv0*64 : iv0*64+64, iv1*128 : iv1*128+128] = \
            transpose(A[iv1*128 : iv1*128+128, iv0*64 : iv0*64+64])

# Step 2: Load B into SBUF (DMA copy B[128,256] → B_s[128,256])
for iv0 in range(2):
    for iv1 in range(2):
        B_s[iv0*64 : iv0*64+64, iv1*128 : iv1*128+128] = \
            B[iv0*64 : iv0*64+64, iv1*128 : iv1*128+128]

# Step 3: Tiled matmul — iterate over M, N, K blocks
for m in range(2):             # M tiles of 128
    for n in range(2):         # N tiles of 128
        psum = load_from_hbm(C[m*128:(m+1)*128, n*128:(n+1)*128])
        for k in range(2):     # K tiles of 64
            nisa.matmul(
                dst=psum,
                stationary=A_t[k*64 : k*64+64, m*128 : m*128+128],
                moving=    B_s[k*64 : k*64+64, n*128 : n*128+128],
                row_pos=k*64,  # ← must match stationary's partition offset
            )
        store_to_hbm(psum, C[m*128:(m+1)*128, n*128:(n+1)*128])
```

## Root Causes

### 1. Wrong tile size in Step 1 (DMA transpose load of A)

`LegalizeLayout::tileTranspose()` generates the Step 1 loop. The SBUF
block size is [64, 128] (from `sbuf_map<tile:[64,128], blocks:[2,2]>`).
The old code permuted the tile with the transpose permutation [1,0],
giving [128, 64] instead of [64, 128]:

```
# BUG: used permuted tile [128, 64] for the SBUF subview
A_t[iv0*128 : iv0*128+128, iv1*64 : iv1*64+64] = transpose(...)
     ^--- when iv0=1: accesses partitions [128,256) on a 128-partition buffer → OOB!
```

Fix: use the SBUF's tile size [64, 128] directly — don't permute it.
The permutation only affects which HBM source region maps to which SBUF
block, not the SBUF block sizes themselves.

### 2. Hardcoded `row_pos=0` in Step 3 (matmul emission)

In Step 3, the `nisa.matmul` instruction needs `row_pos` to tell the
hardware which partition the stationary operand (A_t) starts at. When
iterating over K blocks, the stationary slice starts at partition 0 for
`k=0` and partition 64 for `k=1`.

The emitter hardcoded `row_pos=0` for all iterations. The backend
verifier rejected k=1 because the stationary starts at partition 64 but
`row_pos` claimed 0.

Fix: derive `row_pos` from the stationary operand's actual partition
offset.

### 3. Inconsistent partition folding in `_remap_sbuf_offsets`

Two functions in the NISA emitter disagreed on when to fold partition
blocks into the free dimension:
- `_memref_type_str_nisa`: folds only when `logical_par > 128`
- `_remap_sbuf_offsets`: folded whenever `num_par_blocks > 1`

In this case `logical_par = 128` (not > 128), so the type said "128
partitions" but the offset function folded as if there were only 64 —
producing offsets that exceeded the declared buffer size.

Fix: only fold when `logical_par > 128` in both places.

## Files Changed

- `mlir/lib/Transforms/LegalizeLayout.cpp` — transpose tiling fix
- `nkigen/codegen/nisa/emit.py` — row_pos and partition folding fixes

## Verification

All 13 tests in `tests/unit/test_memref_backend.py` pass. Full suite:
313 passed, 15 failed (all pre-existing, unrelated).
