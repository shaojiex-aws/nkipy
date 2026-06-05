# KernelBuilder codegen vs. legalize-layout's 4D physical layout

**Date**: 2026-06-05
**Status**: Proposal
**Context**: Phase 3 of the textual kernel_builder codegen plan
(`2026-06-04-textual-kernelbuilder-codegen-plan.md`). Round-trip
(`Mode.CODEGEN`) works for elementwise + activation kernels but fails for
matmul and reductions. This documents why and proposes a clean fix.

---

## TL;DR

`legalize-layout` rewrites SBUF allocations from their logical rank-R shape into
a **(R+2)-D physical layout** `[partTile, numBlocks…, freeTile]`. The linalg
*compute* ops still see a clean **2D** operand, because each use is wrapped in a
`memref.subview` (pick one block) followed by a `memref.collapse_shape`
(`[[0,1,…,R],[R+1]]`) that folds the physical layout back to `[partTile,
freeTile]`.

The NISA backend understands this: it walks the subview+collapse chain and
builds a flat affine map, so the 4D storage never leaks into the emitted op.

The kernelbuilder backend currently treats `collapse_shape` as a no-op
pass-through and renders the *raw 4D subview* as a Python slice — e.g.
`sbuf[:, i:i+1, j:j+1, :]`. That 4D slice is shape `(128,1,1,128)`, which does
not match kb's 2D tile model, so `nisa.matmul` / `nisa.dma_transpose` /
reductions get a rank-mismatched operand and either fail verification or
mis-simulate.

The fix: make the codegen treat the **`collapse_shape` result as the operand
boundary** and lower the (4D alloc → block subview → 2D collapse) chain into
kb's natural per-block tile indexing, dropping the unit block dims — the same
information the NISA backend already extracts.

---

## Why legalize-layout produces a 4D layout

(From `mlir/lib/Transforms/LegalizeLayout.cpp`, the `LayoutInfo` struct.)

NKI's SBUF has a hard hardware constraint: the **partition dimension (dim 0)
must be ≤ 128**. A logical tensor like `256×256` cannot live in SBUF as-is — its
partition dim is 256.

So legalize-layout tiles each SBUF tensor and stores it **block-major**. For a
rank-R tensor `[d0, …, d_{R-1}]` with tile `[t0, …, t_{R-1}]`:

```
numBlocks[i]   = d_i / t_i
physical shape = [t0, numBlocks[0], …, numBlocks[R-1], t_{R-1}]   (rank R+2)
```

For the matmul example, a `256×256` SBUF tensor with tile `[128,128]` becomes:

```
logical  256 x 256
tile     128 x 128      -> numBlocks = [2, 2]
physical 128 x 2 x 2 x 128     (memref<128x2x2x128xf32, sbuf>)
         ^t0  ^nB0 ^nB1 ^t1
```

Invariant the pass maintains (`LayoutInfo` doc comment): **only dim 0 (partition)
and dim R-1 (free) may have tile > 1; all middle dims have tile = 1.** So a
single-block slice has shape `[partTile, 1, …, 1, freeTile]`, which collapses
*directly* to 2D `[partTile, freeTile]`.

This is why every compute-op operand in the legalized IR looks like:

```mlir
%sv  = memref.subview %alloc[0, %i, %j, 0] [128, 1, 1, 128] [1,1,1,1]
         : memref<128x2x2x128xf32, sbuf> to memref<128x1x1x128xf32, …>
%cs  = memref.collapse_shape %sv [[0, 1, 2], [3]]
         : memref<128x1x1x128xf32, …> into memref<128x128xf32, …>
linalg.matmul_transpose_a ins(%cs_a, %cs_b) outs(%cs_c)   // all 2D
```

The **block indices** `%i, %j` are loop induction variables; the **collapse**
turns the chosen `128×1×1×128` block into the `128×128` tile the op consumes.

## Why this conflicts with kb's model

kernel_builder tiles are addressed in their **logical tile shape**. A
`nb.compiler.alloc((128, 2, 2, 128), …)` is a 4-D tile, and slicing it
`tile[:, i:i+1, j:j+1, :]` yields a 4-D `(128,1,1,128)` view — kb has no
implicit "collapse the unit block dims to 2D" step. `nisa.matmul`,
`nisa.dma_transpose`, and `nisa.tensor_reduce_arith` expect 2-D tile operands,
so they reject / mis-handle the 4-D view.

Concretely, the current emitter does:

| IR | NISA backend | kernelbuilder backend (current) |
|----|--------------|--------------------------------|
| `memref.alloc : 128x2x2x128` | tracked as base | `nb.compiler.alloc((128,2,2,128), …)` |
| `memref.subview [0,%i,%j,0][128,1,1,128]` | offsets folded into affine map | `sbuf[:, i:i+1, j:j+1, :]` ← 4-D slice |
| `memref.collapse_shape [[0,1,2],[3]]` | merged into affine map (→ 2D) | **pass-through (ignored)** ← bug |
| `linalg.matmul ins(2D) outs(2D)` | `nisa.matmul(2D operands)` | `nisa.matmul(4-D slice)` ← rank mismatch |

The root error is the `collapse_shape` pass-through in
`emit_indexing.memref_expr`: it drops the reassociation information that
*defines* the 2D view, then renders the 4-D physical subview verbatim.

For elementwise/activation kernels this never bit us because those kernels (in
the tests that pass) keep tensors in 2D HBM or use single-block SBUF tiles
where the slice happens to be benign; matmul is the first kernel that exercises
multi-block 4-D SBUF tiles end-to-end.

---

## Fix proposals

### Option A (recommended): collapse the chain to a logical 2-D block tile

Teach the indexing layer that **`subview(4D physical) + collapse_shape` is one
logical operation**: "select block `(i, j)` and view it as a 2-D tile." Emit kb
indexing that matches kb's tile model.

Two sub-options for the kb spelling:

- **A1 — keep the 4-D alloc, drop unit dims in the slice.** Recognize the
  collapse and emit `sbuf[:, i, j, :]` (integer index, not `i:i+1` range) so
  the unit block dims are *squeezed* rather than kept as size-1 ranges. If kb's
  `TileView.__getitem__` squeezes integer-indexed dims (needs confirming — see
  Open Questions), this yields a clean `128×128` operand with minimal emitter
  change. The alloc stays `(128, 2, 2, 128)`.

- **A2 — allocate per-logical-shape and index by block offset.** Allocate the
  tile in its *logical* tiled shape and address blocks via element offsets
  (`sbuf[i*128:(i+1)*128_partition_block, …]`), reconstructing the 2-D tile the
  way kb examples normally tile. Closer to how a human writes kb, but requires
  re-deriving block→offset math the legalize pass already encoded.

Mechanically (either spelling), `memref_expr` stops treating `collapse_shape`
as a pass-through. Instead, when it sees `collapse_shape(subview(base))` where
the subview's non-collapsed dims are all unit blocks, it:
1. reads the reassociation `[[0..R],[R+1]]` to know which dims collapse;
2. maps the subview's per-dim offsets (loop IVs for block dims, 0/`:` for tile
   dims) onto the logical 2-D `[partition, free]` indices;
3. emits the 2-D tile expression.

This mirrors `codegen/nisa/access.py::_get_base_and_offsets`, which already
computes `dropped_dims` for exactly this collapse+subview pattern — the logic
can be ported rather than reinvented.

### Option B: generate kb code from *before* legalize-layout

Stop the pipeline before `legalize-layout` (rather than before
`linalg-to-nisa`), so SBUF tensors are still in their **logical 2-D shape** and
no 4-D physical layout exists. kb itself would be responsible for the SBUF
partition-dim legalization at its own compile time.

- **Pro**: the codegen never sees 4-D layout; indexing is naturally 2-D and
  matches how a human writes kb. Simplest emitter.
- **Con**: changes the contract — we'd emit *pre-tiled-for-SBUF* code and lean
  on kb's own layout legalization, which may tile differently than nkigen's
  knobs intended. We lose the "emit exactly what nkigen decided" property, and
  spill/reload (inserted after legalize-layout) would be absent. Risk: the
  generated kb may not honor the user's `tile_op`/`layout` knobs.

### Option C: emit an explicit reshape in the generated code

Keep the 4-D alloc and emit a `.rearrange(...)` / `.view(...)` (kb's TileView
has `rearrange`/`view`/`repeat` methods) to collapse `(128,1,1,128) →
(128,128)` at each use site.

- **Pro**: faithful to the physical layout; localized change.
- **Con**: verbose, less readable generated code (a reshape at every operand);
  depends on kb reshape semantics matching MLIR collapse exactly.

---

## Recommendation

**Option A1** if kb integer-indexing squeezes dims (verify first); otherwise
**Option A**'s general "collapse the chain" approach, porting the
`dropped_dims` logic from `codegen/nisa/access.py`. It keeps the generated code
faithful to nkigen's layout decisions (unlike B), stays readable (unlike C),
and concentrates the change in one place (`emit_indexing.memref_expr` +
`subview_slice`).

Option B is worth a spike if we later decide the backend's job is "emit
human-style kb and let kb optimize," but that is a different product contract
than "show what nkigen produced," which the plan's motivation section calls
for.

## Open questions

1. **Does `kb` `TileView.__getitem__` squeeze integer-indexed dims?**
   ✅ **CONFIRMED YES** (2026-06-05). `a[:, 0, 0, :]` on a
   `alloc((128,2,2,128))` tile yields a 2-D `128×128` `TileView` that
   `nisa.dma_copy` round-trips correctly through `simulate_kernel`. This makes
   **A1 viable**: emit integer indices `sbuf[:, i, j, :]` for the collapsed
   block dims instead of `i:i+1` ranges, and the 4-D alloc collapses to the 2-D
   operand the compute ops need.
2. **Multi-block operands.** Some subviews keep a non-unit block dim
   (`[128, 2, 1, 128]` in the matmul accumulation loop). The fix must handle
   "this operand spans N blocks," not only single-block. The NISA backend
   represents this in the affine map; the kb spelling needs an equivalent
   (likely an inner loop over blocks, or a 3-D tile operand if kb supports it).
3. **PSUM tiles.** The matmul output `alloc : 128x128 psum` is *not* 4-D
   (PSUM tiles aren't block-tiled the same way). Confirm the fix only triggers
   on the 4-D SBUF collapse pattern and leaves 2-D PSUM/HBM untouched.

## Validation

Use `Mode.CODEGEN` on the existing e2e matmul / reduction tests as the
acceptance check — they already compute a NumPy reference and simulate the
generated kb code. "Fixed" = those tests pass round-trip with the same
tolerances as `Mode.HW`.

---

## Status (2026-06-05): Option A1 implemented

`emit_indexing` now composes the `subview*…+collapse_shape` chain and squeezes
the unit block dims to integer indices (`sbuf[:, i_k, i_n, :]`). Round-trip
(`Mode.CODEGEN`) results:

- ✅ elementwise (add/sub/mul, 2-output), activation (sigmoid: exp + reciprocal
  + scalar), and **single-tile matmul (128×128×128)** — numerically exact.
- ❌ **multi-block matmul (256×256)** — *runs* now (the rank mismatch is gone),
  but the result is wrong. This is a **separate** problem from the 4D layout:
  **PSUM accumulation across the K-reduction loop.**

### The remaining problem: K-loop PSUM accumulation

For a tiled matmul the PSUM accumulator is allocated *outside* the K-loop and
each K-block iteration must *add* into it:

```mlir
%psum = memref.alloc() : memref<128x128, psum>      // outside K-loop
scf.for %k = 0 to 2 {                               // K-reduction loop
  ... linalg.matmul_transpose_a ins(A_k, B_k) outs(%psum_block)
}
```

In NISA the accumulation is implicit — the matmul op carries
`psum_accumulate_flags`/`psum_zero_region` and the *backend* pipeline
(rotate-numbering / scheduler, which runs after this IR) sequences the
zero-then-accumulate. The kernelbuilder emitter currently hardcodes
`nisa.matmul(..., accum=False)`, so every K iteration **overwrites** PSUM and
only the last K-block survives → wrong result.

kb's `accum` is an explicit per-call bool: the first K iteration should be
`accum=False` (zero PSUM) and the rest `accum=True` (accumulate). Open issues:

1. **Detecting the K-reduction loop.** The matmul nests in an `scf.for` whose IV
   indexes the contraction block dim of *both* operands but not the output. The
   emitter must identify that loop and key `accum` on its IV
   (`accum=(k != 0)`).
2. **kb loop unrolling.** A first experiment with a Python `for k in range(2):
   nisa.matmul(..., accum=(k!=0))` unrolled and failed NIR verification —
   the interaction between kb's loop handling and `accum` needs investigation
   (possibly requires `nb.fori_loop` with the accumulator as a loop-carried
   value, which is Phase 4 territory).

This is logged as the next concrete task for matmul/attention/feedforward
round-trip. Elementwise + activation + single-tile matmul are done.
