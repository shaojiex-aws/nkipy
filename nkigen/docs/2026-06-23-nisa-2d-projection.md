# NISA >2D Support: Findings & Plan

**Date:** 2026-06-23  
**Status:** DONE — implemented via `view()` syntax, all non-pre-existing tests pass

---

## Findings from NISA Source Code (`~/private-nki-staging/`)

### NISA supports >2D memrefs natively

The NISA dialect supports up to 8 dimensions (`NisaAsmHelpers.h:40` defines
`d0..d7`). The constraints are per-operand:

1. **Tile shape rank == memref rank == subscript (affine map) result count**
   - Source: `NisaDatapathOpInterface.cpp:94-97`
   - If memref is 3D, tile shape must be 3D, and subscript must have 3 entries
2. **Tile shape minimum is 2D** (at least partition + one free)
   - Source: `NisaDatapathOpInterface.cpp:120-125`
3. **`|` separates partition dims from free dims** in tile shape
   - `<128| 1, 128>` = 1 partition dim (128), 2 free dims (1, 128) → total rank 3
   - Source: `NisaAsmHelpers.h:143-218` (parseTileShape)
4. **Src and dst can have different ranks** — each operand is validated independently
5. **Commas required between dimensions**: `<128| 1, 128>` not `<128| 1 128>`

### What was actually wrong

Our emitter wrote `<128| 1 128>` (no comma) instead of `<128| 1, 128>`. That's
why the parser failed with `expected '>'`. It wasn't a rank limitation.

The second error (`map results (2) != shape rank (3)`) happened because we then
"fixed" it by projecting HBM to 2D, but the underlying memref was still 3D in
some paths.

### The implemented design

```
HBM (>2D, user's rank)  --[view() → 2D]--[dma_copy]-->  SBUF tile (2D: par x free)
                                                              |
                                                         [compute]
                                                              |
HBM (>2D, user's rank)  <--[dma_copy]--[view() → 2D]--  SBUF tile (2D: par x free)
```

- **HBM keeps its original rank** in function signatures and allocs.
- **SBUF is 2D** (`partition x free`) — projected by `_memref_type_str_nisa()`.
- **dma_copy** uses `view()` on >2D HBM operands to present them as 2D,
  matching the SBUF side. BIR requires src/dst to have the same rank.
- **Offsets** into the view are linearized from the multi-dim loop IVs.

Example of emitted NISA for a `(4, 128, 64)` HBM → `(1, 8192)` SBUF copy:
```mlir
nisa.dma_copy(
  dst<1| 8192>=memref<1x8192xf32, #nisa.mem<sbuf>> %tile[%c0 + d0, %c0 + d1],
  src<1| 8192>=view(memref<4x128x64xf32, #nisa.mem<shared_hbm>> %arg0, f32, [4, 8192])[%iv + d0, %off + d1]
) engine=dma
```

**Why view() instead of native >2D?** NISA supports >2D natively, but the BIR
backend (neuronx-cc) rejects dma_copy when src and dst have different ranks.
Since SBUF is always 2D, `view()` reinterprets HBM as 2D at the access site.

## Step-by-Step Fix Plan

### Step 1: Fix function signature emission ✅

Stop flattening HBM args to 2D. Emit the original shape:
- `memref<256x2x256xf32, #nisa.mem<shared_hbm>>` not `memref<256x512xf32, ...>`

File: `nkigen/codegen/nisa/emit.py` — `_memref_type_str_nisa()` only projects
SBUF/PSUM to 2D, leaves HBM as-is.

### Step 2: Fix tile shape emission for HBM operands ✅

Used `view()` syntax instead of emitting native >2D tile shapes on HBM.
The BIR backend requires dma_copy src/dst to have matching rank, so >2D HBM
operands are reinterpreted as 2D at the access site:

```mlir
view(memref<4x128x64xf32, #nisa.mem<shared_hbm>> %arg0, f32, [4, 8192])
```

Tile shape is emitted as 2D: `<par| flat_free>` (e.g. `<1| 8192>`).

File: `emit.py` — `_operand_str()` emits `view()` for >2D HBM operands.

### Step 3: Fix subscript emission for HBM operands ✅

With `view()` flattening dims 1..N into a single free dim, multi-dim offsets
are linearized into a single flat offset via `_linearize_offsets()`. This is
necessary because the view presents a 2D interface — the linearization maps
`[off1, off2, ..., offN]` → `off1*stride1 + off2*stride2 + ... + offN`.

File: `emit.py` — `_linearize_offsets()` computes the flat free-dim offset.

### Step 4: Keep SBUF operands at 2D ✅

SBUF tiles remain 2D: `<128| 128>=memref<128x128xf32, #nisa.mem<sbuf>> %tile[%c0 + d0, %c0 + d1]`

`_memref_type_str_nisa()` projects SBUF/PSUM to 2D, no `view()` needed.

### Step 5: Fix return type ✅

Output memref keeps original rank: `memref<256x2x256xf32, ...>`.

Removed the reshape hack in `tests/harness.py`.

### Step 6: Run tests ✅

- All 119 passes tests: PASS
- test_3d_add_hbm_only: PASS
- test_sigmoid (4): PASS
- test_partition_dim (2): PASS
- Full e2e suite: no regressions (all failures are pre-existing)

## What Was Changed

- `_memref_type_str_2d()` → `_memref_type_str_nisa()` (projects only SBUF/PSUM)
- `_operand_str()` emits `view()` for >2D HBM operands
- `_linearize_offsets()` retained — needed to map multi-dim offsets into flat view
- Reshape hack in `tests/harness.py` removed

## Final Test Status

### Passing
- test_3d_add_hbm_only
- test_sigmoid (4 tests)
- test_partition_dim (2 tests)
- All 119 passes tests

### Pre-existing Failures (unrelated to this work)

| Category | Tests | Cause |
|----------|-------|-------|
| neuronx-cc crash/OOM | matmul_add (16), reduce (2), attention, qwen3 | Exit code 70 / segfault |
| KB partition alignment | test_3d_add_chain, auto_layout (2), feedforward (2) | KB backend |
| Multi-output | test_multi_output (2) | Emitter limitation |
| Custom op | test_custom_op (2) | Missing silu dispatch |
| reverse_operands | test_scalar_minus_tensor | Missing flag |
| LLVM dialect | rope (4), rmsnorm, head_deconcat | memref.alloc not lowered |
