# NISA Emitter: SBUF Mapping and HBM View Design

**Date:** 2026-06-23  
**Status:** Design agreed, implementation pending

---

## 1. Concrete Failures

`test_bmm_e2e` fails with neuronx-cc birverifier "access out of bounds". Two root causes:

**A)** SBUF alloc emitted as `memref<256x256xf32, sbuf>` — exceeds 128 partition hardware limit.

**B)** 3D HBM `memref<2x256x256>` view shape is `[2, 65536]`, but DMA tile `<128|128>` tries to read 128 rows from a 2-row view.

---

## 2. SBUF Mapping Design

### User's perspective

The user specifies a tile size for their operation. For example, `tile_size=[256, 256]` means "I want a 256×256 tile resident in SBUF."

### Compiler's job

SBUF has 128 partitions. If the user's tile partition dimension exceeds 128, the compiler folds the excess into the free dimension:

- User tile: `[256, 256]`
- Partition dim (256) > 128 → fold factor = `256 / 128 = 2`
- Physical SBUF alloc: `[128, 2*256]` = `[128, 512]`
- The physical layout is `[128, 2, 256]` conceptually: 128 partitions, 2 partition-blocks, 256 free elements per block

### Index mapping (full-resident)

Logical `[i, j]` on a `256×256` tile mapped to physical `[128, 512]`:
- Physical partition offset = 0 (tiling guarantees `i` is always a multiple of 128)
- Physical free offset = `(i / 128) * 256 + j` (exact division — no modulo needed)

The partition dim must be either ≤128 (fits directly) or a multiple of 128 (folds cleanly). In either case, all partition offsets are multiples of 128, so `i / 128` is always exact integer division. No mod operation ever reaches NISA.

This is purely a compiler-inferred layout from the user's tile size + hardware constraint. No explicit user annotation needed beyond the tile size.

### Tile-reuse case

If the user specifies `tile_size=[128, 128]` (fits in 128 partitions directly):
- Physical SBUF alloc: `[128, 128]` — no folding needed
- The tiling pass generates iteration loops to cover the full logical tensor
- No sbuf_map attribute needed at all

### What `sbuf_map` encodes

`sbuf_map<tile: [T0, T1, ...]>` means: **`tile` is the amount of data resident in SBUF at once.**

Physical SBUF allocation:
- If `T0 ≤ 128`: physical alloc = `[T0, T1]` directly
- If `T0 > 128`: fold partition overflow into free dim. Physical alloc = `[128, (T0/128) * T1]`

Reuse (derivable from logical shape and tile):
- `blocks[i] = logical_shape[i] / tile[i]` — how many times the tile is reused per dim
- The tiling pass generates the block iteration loops

Examples on `memref<16384x16384xf32, sbuf>`:

```mlir
// Small tile, heavily reused. Physical alloc: 128×128.
#nkipy.sbuf_map<tile: [128, 128]>
// blocks = [128, 128]. Reused 16384 times.

// Larger tile, partition folded. Physical alloc: 128×(2*128) = 128×256.
#nkipy.sbuf_map<tile: [256, 128]>
// 256 > 128 → fold. blocks = [64, 128]. Reused 8192 times.

// Even larger tile. Physical alloc: 128×(4*256) = 128×1024.
#nkipy.sbuf_map<tile: [512, 256]>
// 512 > 128 → fold. blocks = [32, 64]. Reused 2048 times.
```

The tile size is the user's choice: larger tile = more SBUF used, less reuse (fewer block iterations). The compiler validates that the physical alloc fits in SBUF memory.

---

## 3. HBM View

### Background

HBM is flat memory. NISA's `view(memref, elem_type, [R, C])` reinterprets flat bytes as a 2D grid. `C` is the stride between rows (in elements). The DMA accesses a 2D rectangle from this grid.

### Mental model

The DMA reads a 2D rectangle from a >2D HBM tensor. The subview sizes tell us which dims are accessed with a range (size > 1) and which are fixed indices (size = 1). The two accessed dims become the view's rows and columns. Fixed dims just contribute to the flat offset.

### Rule

Given a >2D base `memref<D0 x D1 x ... x D_{N-1}>` and subview sizes `[S0, S1, ..., S_{N-1}]`:

1. Find the first dim with size > 1 — this is the **row dim**
2. `C = product(base_shape[row_dim + 1:])` — the row-major stride of that dim
3. `R = total_elements / C`
4. `view([R, C])`
5. `row_offset = linearize(offsets[:row_dim + 1], base_shape[:row_dim + 1])`
6. `col_offset = linearize(offsets[row_dim + 1:], base_shape[row_dim + 1:])`

The subview sizes are already known in `_trace_access` (from `static_sizes`). No extra tracking needed — just inspect the sizes to find the row dim.

### Examples

**`memref<256x2x256>` with sizes `[128, 1, 128]`:**
- First size > 1 is dim 0 → row_dim = 0
- C = product([2, 256]) = 512, R = 131072/512 = 256
- view = `[256, 512]`
- row = offsets[0], col = offsets[1]*256 + offsets[2]

**`memref<2x256x256>` with sizes `[1, 128, 128]`:**
- First size > 1 is dim 1 → row_dim = 1
- C = product([256]) = 256, R = 131072/256 = 512
- view = `[512, 256]`
- row = offsets[0]*256 + offsets[1], col = offsets[2]

---

## 4. Proposed Architecture: Move Block-Loop Generation to Tiling Pass

### Current state (wrong separation of concerns)

`legalize-layout` currently does two things:
1. Attaches `sbuf_map` to the SBUF alloc
2. **Generates block-iteration loops** for HBM↔SBUF copies

Before legalize-layout, the IR has a simple `memref.copy %hbm_256x256, %sbuf_256x256`. Legalize-layout tiles this into a 2×2 loop of 128×128 copies. This is conceptually **loop tiling** — it doesn't belong in legalize-layout.

### Proposed clean separation

**Tiling pass** handles ALL loop generation uniformly:
1. HBM↔SBUF copy ops get tile annotations (same mechanism as compute ops)
2. Knob-driven tiling splits them into block loops of tile-sized copies
3. SBUF allocs shrink to tile size: `memref<128x128xf32, sbuf>`

**Legalize-layout** becomes a validation/annotation pass:
- Validates that SBUF allocs fit hardware (≤128 partitions)
- If tile partition dim > 128: folds into free dim, records on the type (sbuf_map or inferred)
- No loop generation

### Result

- Tiling pass = all loops. Legalize-layout = physical constraint enforcement.
- In the common case (tile fits in 128 partitions), no sbuf_map needed at all — the alloc IS the physical shape.
- For oversized tiles (user explicitly wants >128 partitions resident), legalize-layout folds and the emitter handles the physical layout.

---

## 5. Decisions Made

1. **SBUF folding:** `sbuf_map<tile: [T0, T1]>` records the tile size. The emitter infers physical alloc: if `T0 > 128`, fold → `[128, (T0/128)*T1]`; otherwise `[T0, T1]` directly. No mod/remainder operations in NISA — partition dim is always ≤128 or a multiple of 128, so `i / 128` is exact.

2. **Block-loop ownership:** Move from legalize-layout to the tiling pass. Legalize-layout becomes validation + folding annotation only.

3. **HBM view:** `C = product(base_shape[first_kept_dim + 1:])`, `R = total/C`. `_trace_access` must return kept-dim info so the view can be derived from the actual access pattern.

4. **`sbuf_map` fields:** `tile` is the only user-specified field. `blocks` is derivable (`logical / tile`). Whether to store `blocks` explicitly or derive it is an implementation choice — the semantic meaning is just "how many iterations the tiling pass generates."

---

## 6. Implementation Plan

### Step 1: Fix emitter — SBUF physical alloc ✅
- Read `sbuf_map` tile size from the memref type
- If `tile[0] > 128`: emit physical alloc as `[128, (tile[0]/128)*tile[1]]`
- If `tile[0] ≤ 128`: emit `[tile[0], tile[1]]` directly

### Step 2: Fix emitter — SBUF offset remapping ✅
- For folded SBUFs, remap logical `[i, j]` → physical `[0, (i/128)*tile[1] + j]`
- Partition offset always 0 (tiling guarantees alignment)

### Step 3: Fix emitter — HBM view for >2D ✅
- Find `first_accessed` = first dim where tile_shape > 1
- `par = tile_shape[first_accessed]`, `free = product(tile_shape[first_accessed+1:])`
- `C = product(base_shape[first_accessed + 1:])`, `R = total / C`
- `row_offset = linearize(offsets[:first_accessed+1], base_shape[:first_accessed+1])`
- `col_offset = linearize(offsets[first_accessed+1:], base_shape[first_accessed+1:])`
- Also: stop tracing through `collapse_shape`/`expand_shape` (they change rank, offsets don't translate)

### Step 4: SBUF 2D contract enforcement
- **4a** ✅: Emitter handles >2D SBUF mechanically (strip leading 1s, collapse to 2D) — legacy, will be removed by `.cache()` implementation
- **4b** ✅: Temp SBUF allocs strip leading unit dims — legacy, will be removed by `.cache()` implementation
- **4c** ✅: Remove LegalizeLayout "middle dims must be unit" restriction — support arbitrary tile shapes
- Full design: `docs/2026-06-23-sbuf-knob-contract.md` (renamed from sbuf-partition-dim-contract)

### Step 5: Fix BMM IR cross-tile access
- Current IR has `sbuf_map<tile: [128, 128]>` but accesses span `[256, 128]` subviews
- Either restructure compute to stay within tile boundaries, or size tile to match actual access patterns
