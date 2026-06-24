# Knob API Redesign

**Date:** 2026-06-24  
**Status:** Proposal

---

## 1. Simplify `.layout()` — remove `tile_size`

### Problem

`.layout(tile_size=...)` is always derivable from `.tile_op()` + the op's indexing maps. The compiler projects the iteration tile onto each tensor's dimensions mechanically:

- C[M,N] with tile_op=[128, 128, 64] → layout tile = [128, 128]
- A[M,K] with tile_op=[128, 128, 64] → layout tile = [128, 64]
- bias[N] with tile_op=[1, 128, 512] → layout tile = [512]

There is no case where you'd want a physical factorization misaligned with the access pattern. Exposing `tile_size` in `.layout()` only confuses users and duplicates information.

### Change

Remove `tile_size` from `.layout()`. The API becomes:

```python
knob(C).tile_op(tile_size=[128, 128, 64]).layout(mem_space="Sbuf")
knob(A).layout(mem_space="Sbuf")
knob(B).layout(mem_space="Sbuf")
```

`.layout()` retains only:
- `mem_space` — where the tensor lives ("Sbuf", "SharedHbm", etc.)
- `partition_dim` — which dimension is the partition axis (default 0)

The compiler derives the SBUF physical factorization tile from the consuming op's `tile_op` + indexing maps. Partition dim folding (keeping partition ≤128) is handled automatically.

### Migration

All existing `.layout(tile_size=...)` calls in tests become just `.layout(mem_space=...)`. The `InferLayout` pass handles the projection. No behavioral change — the same physical factorization is produced, just derived instead of user-specified.

---

## 2. `.cache()` primitive

### Motivation

Knob-driven-tiling currently hardcodes input caching decisions (e.g., for matmul: cache A's row, cache B's column at fixed loop levels). This is not generalizable and not overridable by users.

Users need to control SBUF caching behavior because:
- The compiler's default choice may be wrong for their performance target
- Different ops reuse inputs in different patterns
- The same tensor may be consumed by multiple ops with different caching needs

---

### The primitive

```python
knob(C).tile_op(tile_size=[128, 128, 64]).cache(A, axis=[0, 4])
```

`.cache(input, axis=...)` on the consumer's knob means: "place an SBUF cache buffer for this input at these loop axis levels."

### Key principles

1. **Cache is per-consumer, not per-tensor.** A may be cached differently for C vs D.
2. **axis refers to the post-tiling loop levels.** Tiling splits each original iterator in place (outer, inner), so `tile_op=[M, N, K]` produces 6 levels.
3. **The compiler derives everything else**: buffer shape at each level (from loop bounds and tile sizes), DMA insertion, reuse pattern.

### Loop levels after tiling

`tile_op=[128, 128, 64]` on a matmul (iterators M, N, K) produces:

```
axis 0: M0 (outer M)
axis 1: M1 (inner M)
axis 2: N0 (outer N)
axis 3: N1 (inner N)
axis 4: K0 (outer K)
axis 5: K1 (inner K, innermost = axis -1)
```

Each original iterator becomes two adjacent levels (outer, inner), in the same order as `tile_op`'s list. No reordering.

### Semantics of `axis`

Each entry in `axis` is a post-tiling loop level where an SBUF cache buffer for this input will exist. The buffer size at each level is auto-derived from the loop bounds and tile sizes.

`axis=[-1]` = innermost loop body (minimal staging, one tile, no reuse across iterations).

### Default behavior

Without `.cache()`, the compiler inserts `axis=[-1]` staging for all inputs (hardware requires SBUF staging for compute). Equivalent to:

```python
knob(C).tile_op(tile_size=[128, 128, 64])
       .cache(A, axis=[-1])   # implicit default
       .cache(B, axis=[-1])   # implicit default
```

Minimal one-tile buffer, loaded right before compute, no reuse. Correct but potentially slow.

### Examples

**Matmul with reuse**

```python
C = A @ B   # A[M,K], B[K,N], C[M,N]

knob(C).tile_op(tile_size=[128, 128, 64])
       .cache(A, axis=[0, 4])   # cache at M0-level and K0-level
       .cache(B, axis=[2, 4])   # cache at N0-level and K0-level
```

A doesn't depend on N, so the M0-level cache persists through both N-loops. The K0-level cache is a smaller buffer that persists across K1 iterations.

Replaces the current hardcoded matmul caching in knob-driven-tiling.

**Attention (Q reused across K/V)**

```python
attn = Q @ K.T    # Q[seq, dim], K[seq, dim]

knob(attn).tile_op(tile_size=[128, 128, 64])
          .cache(Q, axis=[0])   # Q cached at outermost level, reused across all inner iterations
```

**Elementwise with broadcast**

```python
# y[B, M, N] = x[B, M, N] + bias[N]
knob(y).tile_op(tile_size=[1, 128, 512])
       .cache(bias, axis=[0])   # bias cached at outermost level (reused across all B, M iterations)
```

**Pure streaming (no reuse, explicit)**

```python
y = np.exp(x)   # x is huge
knob(y).tile_op(tile_size=[128, 512])
       .cache(x, axis=[-1])   # one tile, innermost body, no reuse (same as default)
```

---

### Migration from hardcoded caching

1. Remove hardcoded input caching from knob-driven-tiling pass
2. Default: implicit `.cache(input, axis=[-1])` for all inputs (minimal staging)
3. User overrides with explicit `.cache()` to control reuse
4. Existing tests get `.cache()` annotations added to preserve current behavior

---

### Interaction with `.layout(mem_space="Sbuf")`

- `.layout(mem_space="Sbuf")` = **entire tensor** is permanently SBUF-resident. No staging needed, no `.cache()` needed.
- `.cache(A, axis=...)` = A lives in HBM; the op's loop nest **stages parts of A** into SBUF with reuse controlled by axis placement.

They don't conflict. `.layout(mem_space="Sbuf")` is for small tensors that fit entirely. `.cache()` is for large HBM tensors.

---

### Prefetch

`.cache()` accepts a `prefetch=True` flag to enable double-buffering (load next tile while computing current):

```python
knob(C).tile_op(tile_size=[128, 128, 64])
       .cache(A, axis=[0, 4], prefetch=True)
       .cache(B, axis=[2, 4], prefetch=True)
```

Default is `prefetch=False` (blocking DMA before compute).

---

---

## Open questions

1. **SBUF overflow.** If the user requests more caching than SBUF can hold, should the compiler error or silently downgrade to fewer levels?
2. **Negative indexing.** `axis=-1` = innermost, `axis=-2` = second-innermost, etc. — standard Python convention.
