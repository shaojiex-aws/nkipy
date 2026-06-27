# InferLayout Pass Simplification

## Problem Statement

The InferLayout pass (`mlir/lib/Transforms/InferLayout.cpp`) does one conceptual
thing: ensure every linalg op has `tile_size`, `mem_space`, and `partition_dim`
before KnobDrivenTiling runs. But the implementation uses 8 phases, 3 BFS rounds,
an iter-space/value-space tile duality, seed-ID-based conflict tracking, and ~1000
lines of C++ to accomplish this.

The original idea: "attach knobs to ops the user didn't annotate." That should be
simple. This doc proposes a design that preserves all current behavior with less
machinery.

## Why Is It Complex Today?

The current pass uses **priority-by-ordering**: it runs three BFS rounds so that
user annotations propagate first (highest priority), then matmul hardware defaults,
then elementwise fallbacks. Each round only touches ops that earlier rounds didn't
reach. This creates:

1. **Three identical BFS passes** differing only in seed sources.
2. **Seed-ID tracking** — every layout carries a `seedId` so the conflict checker
   knows whether two annotations came from the same or different seeds.
3. **Iter-space tile duality** — `LayoutEntry` stores tiles in iterator order
   (rank = num iterators, including reduction dims). But `nkipy.layout` needs
   value-space tiles (rank = output rank). So every propagation and materialization
   step must project between these two forms via `projectIterTileToValueShape`.
4. **`computePropagatedLayout` does too much** — broadcast clamping, matmul K-dim
   lifting, and reduction iter-type expansion are interleaved in one 130-line
   function.

## Proposed Simplified Design

Replace "priority by ordering" with **explicit priority levels**: user > propagated
> default. One seeding pass, one BFS, one materialization.

### Phase 1: Collect User Annotations

Same as today. Walk `nkipy.layout` + `nkipy.tile_op` ops, build map. Mark these
as `priority=USER`.

### Phase 2: Assign Defaults to Everything Else

Single walk over all `isAnnotatableOp` linalg ops. For each unannotated op, compute
a default and insert it into the map with `priority=DEFAULT`:

- **Matmul**: hardware-derived tile `[min(M,128), min(N,512), min(K,128)]`,
  partition_dim=0. Operands A/B get their specialized layouts directly here
  (no need to "discover" them during BFS backward propagation).
- **Reduction**: iter-space tile from input shape, partition_dim=0.
- **Elementwise**: `[min(dim0,128), ..., dim_last]`, partition_dim=0, middle=1.

After this phase, **every op has a layout**. The "error if anything is unannotated"
phase becomes unnecessary — it's structurally impossible to miss an op.

### Phase 3: Single BFS from User Annotations

Run one BFS (forward + backward) starting from user-annotated values only.
Propagation **overwrites** entries with `priority=DEFAULT` but **never overwrites**
entries with `priority=USER`. If propagation reaches another user-annotated op with
an incompatible layout, emit an error (true conflict).

Propagated entries get `priority=PROPAGATED` — they can be overwritten by a later
BFS wavefront from a different user annotation (closer user annotation wins).

This single BFS replaces today's three rounds:
- Round 1 (user BFS) → this phase.
- Round 2 (matmul seed BFS) → unnecessary; matmul operands get defaults in Phase 2,
  and if they're reachable from a user annotation, Phase 3 overwrites the default.
- Round 3 (fallback seed BFS) → unnecessary; all ops already have defaults.

### Phase 4: Post-process

- Return values → `SharedHbm` (override mem_space).
- Fill missing `mem_space` via `tracesToFuncArg` check.

### Phase 5: Materialize

Create `nkipy.layout` + `nkipy.tile_op` for ops that didn't already have user
annotations. Update user-annotated ops that were missing fields.

## Key Simplifications

| Current | Proposed | Why simpler |
|---------|----------|-------------|
| 3 BFS rounds | 1 BFS round | Defaults assigned eagerly; BFS only refines them |
| `seedId` per entry | `priority` enum (USER/PROPAGATED/DEFAULT) | Intent is explicit, no ID bookkeeping |
| `describeConflict` at every BFS step | Conflict check only when hitting `priority=USER` | Overwriting a DEFAULT is always fine |
| Error-check phase | Gone | Every op gets a default in Phase 2 |
| `computePropagatedLayout` handles matmul/reduction lifting | Matmul/reduction operands seeded directly in Phase 2 | BFS only propagates through elementwise — no iter-space lifting during propagation |

## Eliminating the Iter-Space Tile Duality

Today, `LayoutEntry.tileSize` is in iter-space form (includes reduction dims).
This means:
- Propagation from a matmul result must "lift" by appending K.
- Propagation from a reduction result must "lift" by inserting reduction dims.
- Materialization must "project" back to value-space for `nkipy.layout`.

**Proposal**: store value-space tiles in the map. Derive iter-space tiles only at
materialization (Phase 5), since that's the only place `nkipy.tile_op` needs them.

Derivation at materialization is trivial:
- **Elementwise**: iter-space = value-space (they're the same).
- **Matmul result [M,N]**: iter-space = `[M_tile, N_tile, K_tile]`. K_tile comes
  from operand A's last dim tile (already computed in Phase 2).
- **Reduction result [M,1]**: iter-space = `[M_tile, red_dim_full]`. Red dim comes
  from input shape.

This eliminates `projectIterTileToValueShape` entirely and makes propagation
simpler: just copy/clamp value-space tiles between same-shape elementwise ops.

## Simplifying `computePropagatedLayout`

With iter-space lifting removed, propagation only needs to handle **elementwise
chains** (same rank, same or broadcast shapes). The function becomes:

```cpp
// Propagate through elementwise: just copy the tile, clamping for broadcasts.
SmallVector<int64_t> propagateElementwise(
    ArrayRef<int64_t> sourceTile, ArrayRef<int64_t> targetShape) {
  SmallVector<int64_t> tile(sourceTile);
  for (size_t i = 0; i < tile.size(); i++)
    tile[i] = std::min(tile[i], targetShape[i]);
  return tile;
}
```

The broadcast partition-dim clamping (lines 342-359 of the current code) folds into
this naturally. The matmul and reduction branches disappear from propagation because
those ops get their layouts directly in Phase 2.

## What About Matmul → Elementwise Propagation?

Today the pass propagates forward from a matmul result to its elementwise consumers.
Example: `matmul → exp → result`. The matmul result has tile `[128, 128]`
(value-space). BFS forward-propagates `[128, 128]` to `exp`.

In the new design, this still works: Phase 2 gives `exp` a default tile. Phase 3
BFS from a user-annotated matmul (or from a user-annotated elementwise op upstream)
overwrites the default. If the matmul has no user annotation either, both keep their
defaults — which are consistent because `computeElementwiseTileSize` uses the same
capping rules.

The one case where propagation from matmul matters: the matmul has a user annotation
with a non-default tile. BFS forward from the matmul result (which is user-annotated)
reaches the downstream elementwise ops and gives them the user's tile instead of the
default. This works exactly as before, but now it's just one BFS handling it.

## What About Matmul Operand Backward Propagation?

Today: BFS backward from a matmul result discovers operand A and calls
`computeMatmulOperandLayout` to generate A's specialized layout. Then BFS continues
backward through A's elementwise producers.

In the new design: Phase 2 directly assigns matmul operand layouts (A gets
partition_dim=1/K, B gets partition_dim=0/K). If a user annotated the matmul result,
Phase 3 BFS backward from the result reaches A's producer chain — but since A itself
isn't a linalg op result (it's a function arg or produced by an elementwise chain),
the BFS propagates the user's partition_dim to the elementwise chain feeding A.

Wait — there's a subtlety. Today, backward BFS from the matmul result uses
`computeMatmulOperandLayout` to generate A's layout. If A is produced by an
elementwise chain (`x → exp → matmul`), the matmul operand layout is assigned to
`exp`'s result, and BFS continues backward from there.

In the new design, Phase 2 handles this: when seeding matmul defaults, also seed the
**producing ops of matmul inputs** with the operand layout. Walk backward from each
matmul input: if the input is produced by a linalg op, give that op the matmul-
operand layout. Continue walking backward through elementwise chains until you hit a
non-elementwise op or function arg.

This is a simple linear walk (not BFS), specific to matmul operand seeding. It
replaces the general-purpose BFS backward mechanism for matmul operands.

## Implementation Plan

### Step 1: Add priority field, collapse seeding

Replace `seedId` with a `Priority` enum. Merge Phase 2 (current) and Phase 3/5
(matmul/fallback seeding) into a single "assign defaults" walk.

### Step 2: Reduce to one BFS

Remove the matmul-seed and fallback-seed BFS calls. Keep only the user-annotation
BFS. Change `tryInsertLayout` to check priority instead of seed IDs.

### Step 3: Store value-space tiles

Change `LayoutEntry.tileSize` to value-space. Move iter-space derivation to
materialization. Remove `projectIterTileToValueShape` from propagation.

### Step 4: Simplify propagation to elementwise-only

Remove the matmul/reduction branches from `computePropagatedLayout`. Matmul operands
are seeded directly. Reductions get their iter-space tile at materialization.

### Step 5: Remove dead phases

Delete the error-check phase (structurally impossible to have gaps). Delete the
three-round seeding logic. The pass becomes:

```
collect → seed_defaults → bfs_from_user → post_process → materialize
```

## Risk / Compatibility

All existing tests should pass unchanged — the observable output (what `nkipy.layout`
and `nkipy.tile_op` ops are created) is identical. The simplification is internal
to the pass structure.

One behavioral difference: today, if the user annotates op Z in chain A→B→C→Z, BFS
backward from Z reaches A. Then if A happens to feed a matmul, the matmul-seeding
phase would see A already annotated and validate compatibility. In the new design,
Phase 2 would seed A with a matmul-operand default, and Phase 3 BFS from Z would
overwrite it (since DEFAULT < PROPAGATED). The matmul-operand layout for A is lost.

**Fix**: when Phase 3 BFS overwrites a matmul-operand default, validate that the
user-propagated layout is compatible with the matmul's requirements. If not, emit a
diagnostic. This is the same conflict-detection as today, just triggered differently.

## Summary

The current pass is complex because it encodes priority through execution order.
Making priority explicit (a 3-value enum) lets us collapse 3 BFS rounds into 1,
eliminate iter-space tile tracking during propagation, and remove the error-check
phase. The pass shrinks from ~1000 LOC / 8 phases to ~400 LOC / 5 phases with
clearer invariants at each step.
