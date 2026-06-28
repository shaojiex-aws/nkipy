# Traced Function API Redesign

## Problem

Users currently need awkward imports to compile a traced function:

```python
from nkigen.apis import knob as knob_fn, fori_loop
from nkigen.driver.pipeline import apply_complete_knob_pipeline

@trace
def my_kernel(a, b):
    c = a + b
    knob_fn(c).tile_op(tile_size=[128, 128]).layout(mem_space="Sbuf")
    return c

# Compile — user has to know about driver internals
mlir = my_kernel.to_mlir()
nisa_ir = apply_complete_knob_pipeline(mlir, target="trn2")
```

Issues:
1. `apply_complete_knob_pipeline` is driver plumbing exposed as public API
2. Users must manually chain `to_mlir()` → pipeline (error-prone, leaks abstraction)
3. `from nkigen.apis import knob as knob_fn` — the rename hints the ergonomics are off
4. `fuse(a, b)` is a standalone function but logically belongs on the knob builder

## Goal

All compilation flows hang off the traced function object. The knob builder is the
single surface for all optimization hints.

```python
from nkigen import trace, knob

@trace
def my_kernel(a, b):
    c = a + b
    d = c * 2
    knob(c).tile_op(tile_size=[128, 128]).layout(mem_space="Sbuf")
    knob(c, d).fuse()
    return d

# Traced linalg MLIR (no passes)
mlir = my_kernel.to_mlir()

# Full pipeline → NISA assembly
nisa = my_kernel.to_nisa(target="trn2")

# NKI kernel_builder Python source
nki_code = my_kernel.to_nki(target="trn2")
```

## Design

### Traced function methods

| Method | What it returns |
|--------|----------------|
| `.to_mlir()` | Traced linalg-on-memref MLIR (before any passes) |
| `.to_nisa(target="trn2")` | NISA MLIR assembly (full pipeline output) |
| `.to_nki(target="trn2")` | kernel_builder Python source code |

`to_nisa()` is the "just compile this" entry point. Clean signature:

```python
def to_nisa(
    target: str = "trn2",
    *,
    dump_dir: str | None = None,
) -> str:
    """Run the full knob pipeline. Returns NISA MLIR assembly."""
```

No `stop_after` / `stop_before` — that's an internal testing concern, not a user API.
Tests that need intermediate IR keep importing `apply_complete_knob_pipeline` from
`nkigen.driver.pipeline` directly (it stays available, just not promoted).

### Rename `to_kernel_builder` → `to_nki`

Same semantics, better name — "NKI" is what users know the output format as.
`to_kernel_builder` becomes a deprecated alias for one release cycle.

### `knob()` accepts multiple tensors for `.fuse()`

Current `fuse()` is a standalone function. Move it onto the builder:

```python
# Single tensor — existing builder methods
knob(c).tile_op(tile_size=[128, 128]).layout(mem_space="Sbuf")

# Multiple tensors — fusion hint
knob(c, d).fuse()

# Can still chain: tile both, then fuse
knob(c).tile_op(tile_size=[128, 128])
knob(d).tile_op(tile_size=[128, 128])
knob(c, d).fuse()
```

Implementation: `_KnobBuilder.__init__` accepts `*tensors`. Single-tensor methods
(`tile_op`, `layout`, `cache`) error if multiple tensors are passed. `.fuse()` errors
if fewer than 2 tensors are passed.

### Fix `knob` top-level export

`from nkigen import knob` gives the *function* directly (not the module).
Drop the standalone `fuse` export since it now lives on the builder.

## Step-by-Step Plan

### Phase 1: `knob(*tensors).fuse()` builder pattern ✅ Done

1. **Edit `nkigen/frontend/knob.py`**:
   - Change `_KnobBuilder.__init__` to accept `*tensors` (store as list).
   - Single-tensor methods (`tile_op`, `layout`, `cache`) validate `len(self._tensors) == 1`.
   - Add `.fuse()` method: validates `len >= 2`, emits `nkipy_d.FuseOp`.
   - Change `knob()` entry point signature to `def knob(*tensors)`.
   - Delete standalone `fuse()` function entirely.

2. **Update `nkigen/apis.py`** — export list: `knob`, `fori_loop` (remove any `fuse` reference).

3. **Update tests** — `knob.fuse(c, d)` / `fuse(c, d)` → `knob(c, d).fuse()`.

### Phase 2: `to_nisa()` and `to_nki()` methods ✅ Done

4. **Edit `nkigen/frontend/trace.py`**:
   - Add `to_nisa()` closure:
     ```python
     def to_nisa(target="trn2", *, dump_dir=None) -> str:
         from ..driver.pipeline import apply_complete_knob_pipeline
         mlir_text = to_mlir()
         return apply_complete_knob_pipeline(mlir_text, target=target, dump_dir=dump_dir)

     wrapper.to_nisa = to_nisa
     ```
   - Rename `to_kernel_builder` → `to_nki` (delete old name).

5. **Edit `nkigen/codegen/kernelbuilder/__init__.py`**:
   - Rename `trace_to_kernelbuilder` → `trace_to_nki` (delete old name).

6. **Add test** — `tests/unit/test_traced_to_nisa.py`: trace a simple kernel,
   call `.to_nisa()`, assert output contains NISA ops.

### Phase 3: Fix top-level exports ✅ Done

7. **Edit `nkigen/__init__.py`**:
   - Replace `from .frontend import knob` (module) with:
     ```python
     from .frontend.knob import knob
     from .frontend.control_flow import fori_loop
     ```
   - Remove `apis` from top-level exports (or keep as internal).
   - Update `__all__` — just `trace`, `knob`, `fori_loop`, `TracedArray`, `CustomOp`.

8. **Update test imports** — `from nkigen import knob` then `knob(x).tile_op(...)`
   (already works in most tests; fix the few that use `knob.knob(x)`).

### Phase 4: Cleanup

9. **Update `nkigen/driver/__init__.py` docstring** — note that `to_nisa()` is the
   preferred entry point.

10. **Update test harness** (`tests/harness.py`) — use `traced_func.to_nisa(...)` in
    the full-pipeline branch.

## Migration

| Before | After |
|--------|-------|
| `from nkigen.driver.pipeline import apply_complete_knob_pipeline` | `traced_fn.to_nisa()` |
| `traced_fn.to_kernel_builder()` | `traced_fn.to_nki()` |
| `knob.fuse(c, d)` / `fuse(c, d)` | `knob(c, d).fuse()` |
| `from nkigen import knob; knob.knob(x)` | `from nkigen import knob; knob(x)` |

All changes are clean breaks — no aliases, no deprecation warnings.

## Final API Surface

```python
from nkigen import trace, knob, fori_loop

@trace
def my_kernel(a, b):
    c = a + b
    d = c * 2
    knob(c).tile_op(tile_size=[128, 128]).layout(mem_space="Sbuf")
    knob(d).tile_op(tile_size=[128, 128]).layout(mem_space="Sbuf")
    knob(c, d).fuse()
    return d

# Inspect traced IR
my_kernel.to_mlir()

# Compile to NISA assembly
my_kernel.to_nisa(target="trn2")

# Generate NKI Python source
my_kernel.to_nki(target="trn2")
```
