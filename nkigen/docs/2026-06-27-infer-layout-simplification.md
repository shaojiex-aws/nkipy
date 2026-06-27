# Rewrite infer-layout: simple, predictable annotation inference

**Date:** 2026-06-27
**Status:** Proposed

## 1. Goal

Users write minimal knobs. `infer-layout` fills in everything else with
obvious, predictable defaults. No magic — the user should be able to
look at their kernel and know exactly what annotations will be generated
without reading the pass source.

## 2. What knob-driven-tiling needs

For each linalg op that should be tiled, ONE thing is required:

```
nkipy.tile_op(%value) {loop_tile_size = [t0, t1, ...]}
```

The tile_size is in iterator-space form:
- Elementwise: one per output dim `[M_t, N_t]`
- Reduction: one per input dim `[M_t, N_t]` (parallel + reduction)
- Matmul: `[M_t, N_t, K_t]`

Additionally `nkipy.layout` carries `mem_space` and optionally
`partition_dim` + `tile_size` (value-shape form) for legalize-layout.

## 3. What infer-layout should do (rewritten)

Three simple steps, no BFS, no "seeding", no multi-phase propagation:

### Step 1: Default knobs for unannotated ops

Walk all linalg ops. If an op has no `nkipy.tile_op`:

- **Matmul**: `tile_op([min(M,128), min(N,512), min(K,128)])`
- **Elementwise**: `tile_op([min(dim[0],128), dim[1], ..., dim[R-1]])`
  (partition dim tiled to 128, free dims full-size)
- **Reduction**: `tile_op([min(par_dim[0],128), par_dim[1]..., red_dim...])`
  (same logic, one entry per iterator)
- **Transpose (>2D)**: insert `tile_op` that tiles identity dims to 1,
  producing a series of 2D transposes in the tiled loop

### Step 2: Default `nkipy.layout` for unannotated allocs

Walk all `memref.alloc` ops. If an alloc has no `nkipy.layout`:

- Traces back to func arg → `layout(mem_space=SharedHbm)`
- Is a return value → `layout(mem_space=SharedHbm)`
- Otherwise → `layout(mem_space=Sbuf, partition_dim=0)`

For matmul outputs specifically → `layout(mem_space=SharedHbm)` because
the matmul result is promoted to PSUM internally by knob-driven-tiling.

### Step 3: Propagate tile_op across adjacent elementwise ops

If an elementwise op's output has a `tile_op` but its producer
(another elementwise op) doesn't, copy the tile to the producer.
One pass, one direction (backward from annotated to unannotated).

This handles the common "annotate the final result, intermediates
inherit" pattern:

```python
y = np.exp(x)        # ← gets tile from z
z = y + bias         # ← user annotates this
knob.knob(z).tile_op(tile_size=[128, 128])
```

## 4. What the user should expect

| Scenario | User writes | infer-layout adds |
|----------|-------------|-------------------|
| Matmul, no knob | nothing | `tile_op([128, 512, 128])`, `layout(SharedHbm)` |
| Elementwise, no knob | nothing | `tile_op([128, N])`, `layout(Sbuf)` |
| Chain `exp→add`, only add annotated | `knob(add).tile_op(...)` | same tile_op on exp |
| Explicit knob | `knob(x).tile_op(...).layout(...)` | nothing (respect user) |

## 5. What infer-layout should NOT do

- Complex multi-phase BFS with conflict detection
- "Seeding" from matmul operands with different partition dims
- Matmul operand-specific partition_dim inference (A=1, B=0 etc.)
  — knob-driven-tiling handles operand promotion internally
- Modifying user-provided annotations (never override)

## 6. Implementation plan

Delete the current 1000-line InferLayout.cpp. Replace with ~150 lines:

```cpp
void runOnOperation() override {
  func::FuncOp func = getOperation();
  defaultTileOps(func);
  propagateTileOps(func);
  defaultLayouts(func);
}
```

Each function is a single walk, no fixpoint, no multi-pass.

### Steps

1. Write `defaultTileOps(func)`: walk linalg ops, skip those with
   existing `nkipy.tile_op`. Emit default tile based on op type:
   - Matmul: `[min(M,128), min(N,512), min(K,128)]`
   - Elementwise: `[min(dim0,128), dim1, ..., dimR-1]`
   - Reduction: same, one entry per iterator
   - Transpose (>2D, exactly 2 dims swapped): tile identity dims to 1

2. Write `propagateTileOps(func)`: single backward pass. If an
   elementwise op has tile_op but its elementwise producer doesn't,
   copy the tile to the producer.

3. Write `defaultLayouts(func)`: walk all `memref.alloc` ops. If no
   `nkipy.layout` with mem_space exists:
   - Return value or matmul output → `layout(SharedHbm)`
   - Else → `layout(Sbuf, partition_dim=0)`
   (func args already get layout from `finish_function` in builder.py)

4. Delete old InferLayout.cpp, replace with new.

5. Delete `tileCopyAndTranspose` from LegalizeLayout.cpp — with
   infer-layout inserting tile_ops on multi-block copies and
   transposes, knob-driven-tiling generates the block loops.
   Everything arrives at LegalizeLayout already tile-sized.

6. Fix tests that relied on old BFS behavior — add explicit knobs
   where needed.

### Current failures (16) and expected outcome

Fixable by this rewrite (missing annotations):
- test_head_deconcat, test_qwen3_layer (transpose output unannotated)
- test_rope, test_rope_3d_compound (similar pattern)

Not fixable (separate issues):
- test_attention (4): loop/bmm patterns needing SubViewOp fix
- test_custom_op (2): custom op resolution
- test_multi_output (2): BIR emission
- test_matmul_add, test_rmsnorm, test_softmax: neuronx-cc issues
- test_sigmoid: codegen assertion
- test_memref_e2e[matmul]: neuronx-cc compilation

Target after rewrite: ~12 failed (fix 4 from annotation issues).
