# Infer-layout rewrite: post-implementation bugs

**Date:** 2026-06-27
**Prereq:** [infer-layout-simplification](2026-06-27-infer-layout-simplification.md)

36 tests fail after the rewrite. 17 are FileCheck tests checking old IR
structure (need assertion updates, not pass fixes). 11 are pre-existing.
The remaining 8 are real bugs described below.

## FileCheck test assertions ✅ Fixed

17 tests in `tests/passes/infer_layout/` checked for old-style
`nkipy.layout` with a `tile_size` attribute. Updated to check for
`nkipy.tile_op` with `loop_tile_size` instead.

## Bug 1: Tile propagation doesn't work ✅ Fixed

**Affects:** test_feedforward_sbuf_compact_silu, test_rope, test_rope_3d_compound
**Error:** `[LegalizeLayout] Error: inconsistent tile sizes`

### What the user expects

```python
# "I annotate the result; intermediates should tile the same way"
gated = gate / (1.0 + np.exp(-gate)) * up
knob.knob(gated).tile_op(tile_size=[128, 128])
```

The full expression (neg, exp, add, reciprocal, mul) shares SBUF buffers
with `gated`. All ops must use the same tile because SBUF allocs have one
physical layout. The user's annotation should propagate backward through
the chain.

### What actually happens

After tracing, each op gets its own output alloc (linalg DPS on memrefs):

```mlir
%a     = memref.alloc() : memref<256x256xf32>
%b     = memref.alloc() : memref<256x256xf32>
%c     = memref.alloc() : memref<256x256xf32>
%d     = memref.alloc() : memref<256x256xf32>
%e     = memref.alloc() : memref<256x256xf32>
%gated = memref.alloc() : memref<256x256xf32>

linalg.negf       ins(%gate)     outs(%a)      ← default tile [128, 256]
linalg.exp        ins(%a)        outs(%b)      ← default tile [128, 256]
linalg.add        ins(%cst, %b)  outs(%c)      ← default tile [128, 256]
linalg.reciprocal ins(%c)        outs(%d)      ← default tile [128, 256]
linalg.mul        ins(%gate, %d) outs(%e)      ← default tile [128, 256]
linalg.mul        ins(%e, %up)   outs(%gated)  ← user tile [128, 128]
```

All intermediates are assigned SBUF by `defaultLayouts`. The conflict is on
`%e`: the first mul writes it with tile [128, 256], but the second mul (user-
annotated) reads it with tile [128, 128]. LegalizeLayout needs a single
physical tile shape per SBUF alloc, so it fails.

(HBM allocs don't have this constraint — LegalizeLayout only tiles SBUF.)

### Why propagation doesn't fire

`propagateTileOps` only copies a tile to a producer that has NO tile yet.
But `defaultTileOps` already gave every op a tile. And even if we swap the
order, `func.walk` visits top-down — only one hop propagates before the
walk moves past the earlier ops.

### Fix

Run propagation first, in reverse program order:

```cpp
void runOnOperation() override {
  propagateTileOps(func);  // user tiles flow backward through chains
  defaultTileOps(func);    // fill whatever's still unannotated
  defaultLayouts(func);
}

void propagateTileOps(func::FuncOp func) {
  SmallVector<linalg::LinalgOp> ops;
  func.walk([&](linalg::LinalgOp op) { ops.push_back(op); });

  // Reverse iteration: consumer visited before producer, so tile
  // propagates through the entire chain in one pass.
  for (auto it = ops.rbegin(); it != ops.rend(); ++it) { ... }
}
```

## Bug 2: Alloc assigned SBUF but is actually a return value (via view) ✅ Fixed (infer-layout part)

**Affects:** test_head_deconcat, test_qwen3_layer

**Errors:**
- head_deconcat: `'nisa.dma_transpose' op source tile shape rank 2 but
  permutation has 4 dimensions`
- qwen3: `[LegalizeLayout] Error: tile rank 2 != alloc rank 3`

### What happens

User code reshapes + transposes, then returns the result:

```python
y = np.transpose(x.reshape(2, 2, 128, 128), [0, 2, 1, 3])
out = y.reshape(2, 128, 256)
knob.knob(out).tile_op(tile_size=[1, 128, 256]).layout(mem_space='SharedHbm')
return out
```

After tracing:

```mlir
%alloc = memref.alloc() : memref<2x128x2x128xf32>
nkipy.layout(%alloc) {mem_space = #nkipy.mem<Sbuf>, partition_dim = 0}   ← defaultLayouts

linalg.transpose ins(%in) outs(%alloc) permutation = [0, 2, 1, 3]

%view = memref.reinterpret_cast %alloc to memref<2x128x256xf32>
nkipy.layout(%view) {mem_space = #nkipy.mem<SharedHbm>}                  ← user annotation
return %view
```

The same underlying memory has **two conflicting layouts**: the alloc
says SBUF, the view says SharedHbm. `defaultLayouts` checks
`isReturnValue(%alloc)` — but it's `%view` that flows to `return`,
not `%alloc` directly. So the check fails and the alloc gets SBUF.

LegalizeLayout then processes this SBUF alloc (rank 4). It traces
through `reinterpret_cast` to find consumers and infers a tile from
the rank-3 view → rank mismatch.

### Fix

In `defaultLayouts`, when checking `isReturnValue(alloc)`, also trace
through `reinterpret_cast` chains. If any view of the alloc reaches
`func.return`, treat the alloc as a return value → SharedHbm.

Also: `defaultTileOps` has a branch for `linalg::TransposeOp` that
computes a tile, but it's guarded by `isAnnotatableOp(linalgOp)` at
the top of the walk. `isAnnotatableOp` only returns true for
elementwise/reduction/matmul, so the transpose branch never fires.

We should add `TransposeOp` to `isAnnotatableOp`. The tile logic for
transpose is: for perm `[0, 2, 1, 3]` on output shape `[2, 128, 2, 128]`:

```
dim 0: perm[0]=0 (identity) → tile=1    ← loop over this dim
dim 1: perm[1]=2 (swapped)  → tile=128  ← full, part of the 2D transpose
dim 2: perm[2]=1 (swapped)  → tile=2    ← full, part of the 2D transpose
dim 3: perm[3]=3 (identity) → tile=1    ← loop over this dim
```

This decomposes a 4D transpose into looped 2D transposes.

Tests still fail due to a downstream LegalizeLayout bug — see
[legalize-layout-high-rank-sbuf](2026-06-28-legalize-layout-high-rank-sbuf.md).

## Bug 3: partition_dim on SharedHbm is invalid ✅ Fixed (knob.py validation)

**Affects:** test_sigmoid_partition_dim_1, test_exp_partition_dim_1
**Error:** `neuronx-cc` exit code 70

The test specifies `partition_dim=1` on a SharedHbm layout. HBM has no
partitions — this is meaningless and causes `canonicalize-partition-dim`
to produce an HBM-to-HBM DMA transpose (hardware unsupported).

Fix: `knob.py` now raises an error if `partition_dim` is specified with
non-SBUF mem_space. Test updated to use SBUF for the compute annotation.

Remaining issue: the test still needs a way to express "compute in SBUF
with partition_dim=1, then copy result to HBM for return." This requires
a boundary copy pass — see
[materialize-boundary-copies](2026-06-28-materialize-boundary-copies.md).

## Bug 4: Multi-output naming (unrelated) ✅ Fixed

**Affects:** test_qkv_projection, test_add_and_mul_hw, test_add_and_sub_hw
**Error:** `nki.output_names has 1 entries but output index is 1`

NISA emitter always emitted `nki.output_names = ["output"]` regardless
of how many return values. Fixed to emit `["output_0", "output_1", ...]`.

## Cleanup: consolidate layout annotation in infer-layout ✅ Done

Removed layout annotations from `builder.py` (func args and return
values). All default layout decisions now live in `defaultLayouts`:
- Func args → SharedHbm (always)
- Return values / matmul outputs → SharedHbm
- Intermediates → Sbuf, partition_dim=0

## Summary

| Bug | Tests | Fix |
|-----|-------|-----|
| 1: propagation ordering | 3 | reverse-walk propagation before defaults |
| 2: transpose allocs | 2 | isReturnValue + isAnnotatableOp |
| 3: partition_dim on HBM | 2 | knob.py validation + test update |
| 4: output naming | 3 | NISA emitter fix |
| FileCheck updates | 17 | update test assertions |
| Pre-existing | ~9 | neuronx-cc failures, not caused by rewrite |
