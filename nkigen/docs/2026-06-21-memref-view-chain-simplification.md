# Memref View-Chain Simplification

**Date:** 2026-06-21  
**Status:** Proposal  
**Motivation:** The IR between tiling and NISA lowering contains deep chains of `subview(collapse_shape(expand_shape(subview(...))))` that force both backends (KB codegen and linalg-to-nisa) to maintain ~1400 lines of fragile chain-composition logic. This complexity is the root cause of the qwen3 CODEGEN failure and a recurring source of bugs.

---

## 1. Problem Statement

After `legalize-layout` and bufferization, SBUF memrefs live in a physical `(partTile, numBlocks_0, ..., freeTile)` shape that differs from the logical shape the compute ops expect. The compiler bridges this gap by inserting chains of view ops:

```mlir
// User writes: out = matmul(x, w)  with tile_size=[128, 128]
// The logical tile shape is 128x128 (what compute ops see).
// But SBUF physically stores it as 4D: (partTile=128, blocks_row=4, blocks_col=4, freeTile=128)
// because the full 512x512 result is partitioned across 128 partition lanes,
// with 4x4 blocks of 128-element free-dim tiles.

// legalize-layout allocates in physical shape:
%alloc = memref.alloc : memref<128x4x4x128xf32, #sbuf>

// To let compute ops index it as 2D (512x512), collapse physical dims:
//   dim0 = partTile * blocks_row = 128 * 4 = 512
//   dim1 = blocks_col * freeTile = 4 * 128 = 512
%collapsed = memref.collapse_shape %alloc [[0,1],[2,3]]
                    : memref<128x4x4x128> -> memref<512x512>

// Now a tiled loop can take a 128x128 slice in logical coordinates:
%tile = memref.subview %collapsed [%i, %j] [128, 128] [1, 1]
                    : memref<512x512> -> memref<128x128>
```

The problem: to actually emit code (DMA load, compute address), the backend must **undo** this collapse — walk from `%tile` back through `%collapsed` to `%alloc` and figure out that logical index `[%i, %j]` on the 512x512 view corresponds to physical index `[%i / 4, %i % 4, %j / 128, %j % 128]` on the 4D alloc. That's a reverse index decomposition through every view op in the chain.

In complex kernels (attention, qwen3), these chains grow deeper:

```mlir
// Example: attention score output reshaped for multi-head layout
//   alloc in physical 4D (partition structure)
//   → expand_shape to expose head dimension
//   → collapse_shape to merge batch+head for GEMM
//   → subview to select one tile within the loop
//
// The backend sees:
%tile = memref.subview %collapsed [%i, %j] [128, 128] [1, 1]
// and must walk: subview → collapse → expand → subview → alloc
// composing index arithmetic at each step.
```

Both backends must walk this entire chain backwards, composing indices through each view to arrive at `(base_alloc, concrete_indices)`. This is where the complexity lives.

### 1.1 Complexity Metrics

| Component | Lines | Role |
|-----------|-------|------|
| `emit_indexing.py` (KB backend) | 558 | ~360 lines of chain composition (`_compose_chain`, `_merge_dims`, `_cross_collapse`, `_cross_expand`, `_fold_subview_through_*`) |
| `access.py` (NISA backend) | 426 | `_get_base_and_offsets()` — identical chain-walking with arith op emission |
| `finalize.py` (NISA backend) | 175 | HBM reshape folding (multi-pass DCE) |
| `CanonicalizeReshape.cpp` | 367 | View-vs-copy classification for partition-dim splits |
| **Total** | **~1400** | View-chain resolution spread across 4 files |

### 1.2 Specific Failure Patterns

**Pattern A — Deep chains with rank mismatch (qwen3 failure):**

```mlir
// Context: qwen3 attention output write. Logical tile is (seq=128, hidden=512).
// After tiling, legalize-layout maps this to 4D physical SBUF:
%alloc = memref.alloc : memref<128x1x4x128xf32, #sbuf>
//   128 = partition lanes (hardware), 1 = partition blocks,
//   4 = free-dim blocks (512/128), 128 = free tile

// User reshape: split hidden=512 into (nheads=4, head_dim=128).
// In physical space, this splits the free-dim-blocks axis: 4 → (4, 1)
%expanded = memref.expand_shape %alloc [[0],[1],[2,3],[4]]
            : memref<128x1x4x128> -> memref<128x1x4x1x128>

// User reshape: transpose + merge to get (nheads, seq, head_dim) for per-head write.
// Merges (partition=128, pblocks=1) and picks from the nheads axis:
%collapsed = memref.collapse_shape %expanded [[0,1],[2],[3],[4]]
            : memref<128x1x4x1x128> -> memref<128x4x1x128>

// Loop: select one head
%head = memref.subview %collapsed [0, %h, 0, 0] [128, 1, 1, 128] [1, 1, 1, 1]
            : memref<128x4x1x128> -> memref<128x128>  // rank-reducing (drops 2 unit dims)
```

The backend walks: `%head → %collapsed → %expanded → %alloc` (4 ops deep). At the collapse→expand boundary, composed dims have rank 4 but expand's result has rank 5. The walker bails: `if len(src_dims) != len(src_shape): return value, None` → emits `UNSUPPORTED_RESHAPE` → runtime error "DMA shape mismatch".

**Pattern B — Subview indexing a reshape result (head_deconcat):**

```mlir
// Context: deconcat splits a concatenated heads tensor back into individual heads.
// The alloc holds all heads merged; expand_shape exposes the head dimension.

%src = memref.alloc : memref<512x128xf32, #sbuf>  // (seq*nheads, head_dim)

// Expand to expose heads: (512, 128) → (4 heads, 128 per head, 128 head_dim)
%expanded = memref.expand_shape %src [[0,1],[2]]
            : memref<512x128> -> memref<4x128x128>

// Select head #2 — a subview in the EXPANDED space:
%head = memref.subview %expanded [2, 0, 0] [1, 128, 128] [1, 1, 1]
            : memref<4x128x128> -> memref<128x128>  // rank-reducing, drops head dim
```

The subview says "offset=2 in dim 0" but that's in the 3D expanded space. The base alloc is 2D (`512x128`). To emit `base[256:384, 0:128]` the backend must:
1. Recognize that dim 0 of the expanded result maps to source dim 0 (group `[[0,1],[2]]`, group 0 = `[0,1]`)
2. Compute: source offset = `2 * 128 = 256` (head_idx * head_size)
3. Source size = `1 * 128 = 128`

This is what `_fold_subview_through_expand` does — it inverts the expand's index map. When groups have multiple non-unit dims (e.g., `[[0,1],[2,3]]` where both dim 0 and dim 1 are > 1), the inversion requires div/mod decomposition and the code becomes brittle.

**Pattern C — Rank-reducing subview through collapse (attention scores):**

```mlir
// Context: attention stores scores as (batch, seq, seq). After computing one batch,
// collapse (seq, seq) for a flat DMA, then take one batch slice.

%alloc = memref.alloc : memref<8x256x256xf32, #sbuf>  // (batch, seq_q, seq_k)

// Collapse last two dims for contiguous DMA: (8, 256, 256) → (8, 65536)
%collapsed = memref.collapse_shape %alloc [[0],[1,2]]
            : memref<8x256x256> -> memref<8x65536>

// Select batch %b — rank-reducing (drops the batch dim from result):
%batch = memref.subview %collapsed [%b, 0] [1, 65536] [1, 1]
            : memref<8x65536> -> memref<65536>  // result is 1D, source was 2D

// Later, another subview takes a tile from this 1D view:
%tile = memref.subview %batch [%k] [256] [1]
            : memref<65536> -> memref<256>
```

Now `_compose_chain` must compose `%tile`'s 1 dim with `%batch`'s 2 dims (one of which was squeezed). Then compose THAT with the collapse's 3 source dims. The `_merge_dims` function sees `outer_dims=[_Dim(%b, 1, squeeze=True), _Dim(None, 65536)]` and `inner_dims=[_Dim(%k, 256)]` — different lengths! It must figure out that the single inner dim maps to the non-squeezed outer dim, not the batch dim. Getting this wrong (as it did before our fix) produces offset `%b` where it should produce `%k`, causing "source tile shape rank 3 but permutation has 2" errors.

---

## 2. Root Cause Analysis

The complexity is NOT inherent to the problem of addressing multi-dimensional tiles. It arises from a specific design choice:

> **`legalize-layout` encodes the physical-to-logical mapping as collapse/expand *ops*, which destroys the mapping information and forces every downstream consumer to reverse-engineer it.**

This is information destruction followed by information recovery. The layout pass *knows* that `(d0, d1) -> (partition, block_row, block_col, free)` — it just doesn't record that knowledge anywhere the backends can read. Instead it emits ops that *implement* the mapping, and the backends must re-derive the mapping by walking those ops backwards.

The fix is obvious: **don't destroy the information in the first place.**

### 2.1 What Other Compilers Do

- **XLA/StableHLO:** Uses a `layout` field on tensors (not separate reshape ops). Physical layout is metadata, not IR ops.
- **Triton:** Uses block pointers with explicit `strides` and `offsets` attributes. No view chains.
- **TVM/TIR:** Uses `BufferLoad`/`BufferStore` with affine index expressions. Buffer shape is decoupled from access pattern.
- **IREE:** Uses `hal.interface.binding.subspan` + affine maps. View ops are canonicalized early; by codegen time, only flat offsets remain.

The common pattern: **separate the "what shape the user thinks it is" from "how it's physically laid out."**

---

## 3. Proposal: Lower `nkipy.layout()` to `#nkipy.sbuf_map` (Eliminate Chains at Source)

### 3.1 The Lowering Pipeline

Today's pipeline:
```
nkipy.layout(tile_size=[128,128], partition_dim=0, mem_space="Sbuf")   // user intent
    ↓  legalize-layout READS this, then ERASES it
    ↓  Rewrites alloc to physical shape + inserts collapse/expand chains
memref<128x4x4x128, #sbuf> + collapse_shape + expand_shape + ...       // information destroyed
    ↓  backends must RECOVER the layout info by walking chains
~1400 lines of chain-composition code
```

Proposed pipeline:
```
nkipy.layout(tile_size=[128,128], partition_dim=0, mem_space="Sbuf")   // user intent (unchanged)
    ↓  legalize-layout CONSUMES this, LOWERS it to a type attribute
memref<512x512xf32, #nkipy.sbuf_map<...>, #sbuf>                      // resolved layout on type
    ↓  backends READ the attribute directly
~30 lines of address computation
```

The key difference: `legalize-layout` no longer rewrites the alloc shape or inserts view ops. It **lowers** the high-level `nkipy.layout()` op into a **resolved, validated layout attribute** on the memref type. The alloc stays in logical shape. The attribute tells backends exactly how to compute physical addresses.

### 3.2 Why `#nkipy.sbuf_map` Is Necessary (Not Redundant)

`nkipy.layout()` is a **high-level user hint** — it may have defaults to fill in, conflicts with tile_op to resolve, or inference to do. It's a side-channel op (attached via use-def, not part of the type).

`#nkipy.sbuf_map` is the **resolved, validated, lowered result** — it lives on the memref type itself, so every op consuming that memref inherently knows the layout. It's the contract between `legalize-layout` and the backends.

Think of it like: `nkipy.layout()` is "what the user asked for" → `#nkipy.sbuf_map` is "what the compiler decided."

### 3.3 The `#nkipy.sbuf_map` Attribute

The attribute encodes a **dim factorization**: how each logical dim is split into physical sub-dims for SBUF storage. This is exactly what `legalize-layout` currently computes — it just records it as an attribute instead of emitting reshape ops.

```mlir
// Syntax: #nkipy.sbuf_map<factors_for_dim0, factors_for_dim1, ...>
// Each factor list describes how one logical dim is split for physical storage.
// Product of factors == logical dim size.
// Physical shape = concatenation of all factor lists.

// 2D: logical (512, 512), tile_size=[128, 128], partition_dim=0
// legalize-layout resolves: partition needs 128 lanes with 4 blocks, free needs 4 blocks of 128
#nkipy.sbuf_map<[128, 4], [4, 128]>
//  dim0 = 512 = 128 * 4  → physical (128 partition_lanes, 4 partition_blocks)
//  dim1 = 512 = 4 * 128  → physical (4 free_blocks, 128 free_elements)
//  Physical shape: (128, 4, 4, 128) — same as current memref<128x4x4x128>

// 2D: logical (128, 512), tile_size=[128, 128], partition_dim=0
// No partition blocking needed (128 fits in 128 lanes)
#nkipy.sbuf_map<[128], [4, 128]>
//  dim0 = 128 → (128) — single physical dim
//  dim1 = 512 = 4 * 128 → (4, 128)

// 3D: logical (8, 256, 128), tile_size=[1, 128, 128], partition_dim=1
// batch=8 is a block dim (tile=1), seq=256 is partition, hidden=128 is free
#nkipy.sbuf_map<[8], [128, 2], [128]>
//  dim0 (batch=8)   → (8) — 8 blocks, all resident in SBUF
//  dim1 (seq=256)   → (128, 2) — 128 partition lanes, 2 partition blocks
//  dim2 (hidden=128)→ (128) — free dim, no splitting
```

**Roles are positional, not labeled.** The `canonicalize-partition-dim` pass (which runs before `legalize-layout`) guarantees that the partition dim is always logical dim 0. So in the factorization:

```
#nkipy.sbuf_map<[128, 4], [4, 128]>
                 ^^^                    first factor of dim 0 = partition lanes (always ≤128)
                      ^                 remaining factors of dim 0 = partition blocks
                         ^              factors of middle dims = block dims
                            ^^^         last factor of last dim = free elements per lane
```

No need to label partition/free/block explicitly — the physical layout convention is always `(partition, blocks..., free)`, and dim 0 is always partition by the time this attr is attached.

**Semantics:** Given logical index `(i, j)` and layout `#nkipy.sbuf_map<[A, B], [C, D]>`:
- Physical index = `(i / B, i % B, j / D, j % D)`
- Physical shape = `(A, B, C, D)`

### 3.4 How `legalize-layout` Changes

```
// Current (complex, ~500 lines of graph rewriting):
1. Read nkipy.layout() op for tile_size + partition_dim
2. Compute physical shape                             → (128, 4, 4, 128)
3. Rewrite alloc type to physical shape               → memref<128x4x4x128>
4. Insert collapse_shape to bridge back to logical    → memref<512x512>
5. Rewire ALL users through the collapse
6. Handle interactions with user reshapes already in IR
7. Erase nkipy.layout() op

// New (trivial, ~50 lines):
1. Read nkipy.layout() op for tile_size + partition_dim
2. Compute factorization                              → [[128, 4], [4, 128]]
3. Attach #nkipy.sbuf_map to the alloc's memref type → memref<512x512, #nkipy.sbuf_map<...>>
4. Erase nkipy.layout() op
5. Done. No shape rewrite. No new ops. No rewiring.
```

### 3.5 Hardware Constraint: NISA Has No Modulo

The factorization implies div/mod when computing physical addresses. NISA can't do arbitrary mod at runtime. But this constraint is **not new** — tiling already ensures alignment:

```mlir
// %i steps by 128 (tile-aligned). Layout is #nkipy.sbuf_map<[128, 4], [4, 128]>.
//   physical[0] = %i / 4   →  exact division (128/4=32, 256/4=64, ...)
//   physical[1] = %i % 4   →  always 0 (128 is a multiple of 4)
// No actual mod instruction ever reaches NISA.
```

| Access pattern | How div/mod resolves |
|---------------|---------------------|
| Tile-aligned loop var (`%i` = 0, 128, 256, ...) | Exact division, mod = 0 |
| Constant (head_idx=2 → offset=256) | Folds at compile time |
| Dynamic-but-aligned (`%h * 128`) | Strength reduction: `%h * 32`, mod = 0 |
| Unaligned | Rejected by verifier → emit alloc+copy instead |

### 3.6 How This Resolves the Failures

**Pattern A (qwen3) — no chain exists:**
```mlir
// Logical shape: (seq=128, hidden=512). Layout splits hidden into (4 blocks, 128 free):
%alloc = memref.alloc : memref<128x512xf32, #nkipy.sbuf_map<[128], [4, 128]>, #sbuf>

// User reshape (nheads split): select head %h. Just a subview in logical space:
%head = memref.subview %alloc [0, %h * 128] [128, 128] [1, 1]
        : memref<128x512> -> memref<128x128>

// Backend reads sbuf_map attr, computes physical from logical offset:
//   dim1 offset = %h * 128 → physical[1] = (%h*128) / 128 = %h, physical[2] = (%h*128) % 128 = 0
//   → physical indices: [0, %h, 0]
// One subview, trivial address math. No chain. No rank mismatch.
```

**Pattern B (head_deconcat) — no expand needed:**
```mlir
// Logical (seq*nheads=512, head_dim=128). Layout: partition dim split into (128, 4):
%alloc = memref.alloc : memref<512x128xf32, #nkipy.sbuf_map<[128, 4], [128]>, #sbuf>

// Select head #2: just a subview at logical offset 256:
%head = memref.subview %alloc [256, 0] [128, 128] [1, 1]
        : memref<512x128> -> memref<128x128>

// Backend: dim0=256, factors=[128, 4] → physical[0] = 256/4 = 64, physical[1] = 256%4 = 0
// No expand_shape. No inversion. Just factorization arithmetic.
```

**Pattern C (attention scores) — no collapse needed:**
```mlir
// Logical (batch=8, seq_q=256, seq_k=256):
%alloc = memref.alloc : memref<8x256x256xf32, #nkipy.sbuf_map<[8], [128, 2], [256]>, #sbuf>

// Select batch %b, then tile row %k — just subviews in logical space:
%batch = memref.subview %alloc [%b, 0, 0] [1, 256, 256] [1, 1, 1]
         : memref<8x256x256> -> memref<256x256>
%tile = memref.subview %batch [%k, 0] [1, 256] [1, 1]
         : memref<256x256> -> memref<256>

// Subview-of-subview composes trivially (add offsets in logical space).
// Backend reads sbuf_map attr once to emit physical address. No collapse walking.
```

### 3.7 What Changes

| Component | Current | After |
|-----------|---------|-------|
| `legalize-layout` (C++) | ~500 lines: rewrite alloc shape, insert collapse/expand, rewire users | ~50 lines: compute factorization, attach `#nkipy.sbuf_map` attr |
| `emit_indexing.py` (KB) | 360 lines of `_compose_chain` | ~30 lines: read sbuf_map attr, apply factorization to subview offset |
| `access.py` (NISA) | 426 lines of `_get_base_and_offsets` | ~40 lines: read sbuf_map attr, emit physical address arith |
| `finalize.py` (NISA) | 175 lines of HBM reshape folding | Deleted (HBM has no sbuf_map attr / uses identity) |
| `CanonicalizeReshape.cpp` | 367 lines of view-vs-copy classification | Simplified: check if reshape aligns to factorization boundaries |
| User reshape ops | Encoded as expand/collapse chains in physical space | Subviews in logical space (or alloc+copy if non-aligned) |

---

## 4. Implementation Plan

### Phase 1: Define `#nkipy.sbuf_map` Attr + `nkipy.slice` Op (1-2 weeks)

1. **Define `NkipySbufMapAttr`** in ODS/C++:
   - Storage: list of factor-lists, one per logical dim (e.g. `[[128, 4], [4, 128]]`)
   - Verifier: product of each factor list == corresponding logical dim size
   - Utility: `applyLayout(logical_offsets) -> physical_offsets` (does the div/mod factorization)
   - Utility: `physicalShape()` → flat concatenation of all factor lists

2. **Modify `legalize-layout`** to emit logical-shaped allocs with sbuf_map attr:
   - Current: compute physical shape → rewrite alloc → insert collapse/expand → rewire users
   - New: compute factorization → attach `#nkipy.sbuf_map<...>` to alloc type → done
   - The tiling decisions (partition size, block counts) go INTO the attr instead of into reshape ops

3. **Simplify `CanonicalizeReshape`**:
   - Current: classifies expand/collapse as view-vs-copy based on partition dim analysis
   - New: checks if a user reshape aligns to the sbuf_map attr's factor boundaries
   - Aligned → subview in logical space (the sbuf_map attr propagates)
   - Unaligned → emit alloc+copy with a new sbuf_map attr (same as today, simpler check)

### Phase 2: Update Backends (1-2 weeks)

4. **Update KB backend** (`emit_indexing.py`):
   - `memref_expr()` reads `#nkipy.sbuf_map` attr from alloc type
   - Applies factorization to subview offsets → physical indices
   - Emits `base[phys_0:..., phys_1:..., ...]`
   - Delete: `_compose_chain`, `_cross_collapse`, `_cross_expand`, `_fold_subview_through_*`, `_merge_dims`

5. **Update NISA backend** (`access.py`):
   - `_get_base_and_offsets()` reads `#nkipy.sbuf_map` attr, applies factorization, emits address arith
   - Delete: the entire chain-walking loop

6. **Delete `finalize.py` HBM reshape folding** — HBM allocs have no sbuf_map attr (logical == physical).

### Phase 3: Handle NISA Constraints (1 week)

7. **Ensure no runtime mod reaches NISA**:
   - Add a verifier after tiling: subview offsets must be multiples of the layout block sizes
   - If tiling produces a non-aligned access (which it shouldn't — tiling is layout-aware), error early with a clear message instead of silently generating bad code
   - For point indices (constant offsets like `head_idx=2`), the mod resolves at compile time — emit the result as a constant

8. **Transpose handling**: NISA `dma_transpose` needs explicit stride info. With `#nkipy.sbuf_map`, the backend knows the physical strides from the factorization — it can decide transpose legality without walking chains.

### Phase 4: Cleanup & Harden (1 week)

9. **Delete dead code**: `_compose_chain` (KB), `_get_base_and_offsets` chain loop (NISA), the `_PASSTHROUGH_VIEW_OPS` mechanism, `_SILENT_SKIP` entries for collapse/expand
10. **Add verifier**: reject any `memref.collapse_shape` / `memref.expand_shape` in the IR after legalize-layout (they should no longer exist)
11. **Migration tests**: run full test suite, compare generated code before/after


---

## 5. MLIR Integration: The Layout Slot

MLIR's `MemRefType` has a built-in layout slot: `memref<shape, layout, memspace>`. The layout parameter accepts any attribute implementing `MemRefLayoutAttrInterface`. Our `#nkipy.sbuf_map` attr fits directly into this slot:

```mlir
memref<512x512xf32, #nkipy.sbuf_map<[128, 4], [4, 128]>, #sbuf>
//                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  layout slot
//                                                        ^^^^  memspace slot (already used)
```

**Interface requirements for `NkipySbufMapAttr`:**
- `getAffineMap()` → returns the affine map encoding the factorization (div/mod expression)
- `verifyLayout(shape, emitError)` → validates factor products == logical dim sizes
- `isIdentity()` → false (unless trivial single-element factor lists)

**Gotcha: `memref.subview` strips custom layouts.** MLIR's built-in subview result-type inference converts custom layouts back to `StridedLayoutAttr`. Two options:

1. **Use our own slice op (`nkipy.slice`)** that preserves the sbuf_map attr on its result. Backends handle only this op. Cleaner — full control, no fighting MLIR built-ins.
2. **Use `memref.subview` + canonicalization** that reconstructs the sbuf_map attr. More fragile — every pass that creates subviews must not break the pattern.

**Recommendation: Option 1.** Define `nkipy.slice` that takes an sbuf_map memref + logical offsets/sizes and produces an sbuf_map memref result. This is the single op backends need to handle. `memref.subview` / `collapse_shape` / `expand_shape` are no longer emitted for SBUF after legalize-layout.

```mlir
// The only view op that exists after legalize-layout:
%tile = nkipy.slice %alloc [%i, %j] [128, 128]
        : memref<512x512xf32, #nkipy.sbuf_map<[128, 4], [4, 128]>, #sbuf>
        -> memref<128x128xf32, #nkipy.sbuf_map<[128], [128]>, #sbuf>
// Result has simplified factors (single tile = no split needed per dim).
// Backend: reads source sbuf_map attr + offsets → computes physical address.
```

---

## 6. Open Questions

1. **User reshapes: new factorization or copy?** If the user writes `np.reshape(x, (batch, heads, seq, dim))`, should this produce an `nkipy.slice` with an updated sbuf_map attr, or an alloc+copy with a new factorization? Rule of thumb: if the reshape aligns to factor boundaries → slice (zero cost). Otherwise → copy.

2. **HBM:** No partition structure → no `#nkipy.sbuf_map` needed. HBM memrefs keep plain `memref.subview` (logical == physical, no factorization).

3. **Transpose:** Currently `SimplifyLinalg` inserts subview+collapse for >2D transposes. With sbuf_map attrs, transpose could be expressed as a factor permutation (swap physical dim order in the attr) rather than new ops. Worth exploring.
