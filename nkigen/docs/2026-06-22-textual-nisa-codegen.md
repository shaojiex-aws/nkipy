# Textual NISA Codegen

**Date:** 2026-06-22  
**Status:** Done

## Problem

The NISA codegen (`linalg_to_nisa`) previously required a "parallel context" — cloning the entire IR from the nkipy context into the NKI wheel's context so the nisa Python builders could reference SSA Values. This forced ~250 lines of type translation hacks, value maps, and proxy objects.

All that complexity existed to satisfy one requirement: the nisa builders want `nk_ir.Value` objects. We don't need those builders.

## Solution: Emit Text

Both backends (kernel_builder and NISA) now follow the same pattern:

1. Parse the tiled IR in the nkipy context (upstream `mlir.ir`)
2. Walk it top-down
3. Emit target text (Python for KB, MLIR assembly for NISA)

No second context. No NKI wheel dependency during codegen.

```
tiled nkipy IR (linalg + memref + scf + arith, with #nkipy.sbuf_map)
    ↓ parse with upstream mlir.ir
    ↓ walk top-down, emit text
NISA MLIR assembly → handed to nki.compiler for compilation
```

## Shared Structure

```
codegen/
    irutils.py              # shared: memref inspection, const folding, backing_alloc
    access.py               # shared: subview/collapse/expand chain tracing → (base, offsets)
    kernelbuilder/
        __init__.py         # entry: linalg_to_kernelbuilder()
        emitter.py          # text builder (indent, names, lines → Python source)
        emit_compute.py     # linalg.add → nb.tensor_tensor(...)
        emit_memory.py      # memref.copy → nb.dma_copy(...)
        emit_control_flow.py
    nisa/
        __init__.py         # entry: linalg_to_nisa()
        emitter.py          # text builder (indent, names, lines → MLIR assembly)
        emit.py             # walk + all patterns in one file
```

Both backends share:
- **`irutils.py`** — `memref_shape`, `memref_memspace`, `const_int`, `backing_alloc`, etc.
- **`access.py`** — traces subview/collapse/expand chains, returns `(base, offsets, dropped_dims)`. KB renders offsets as Python slices; NISA renders them as MLIR affine expressions. Same walk logic.

## NisaEmitter

```python
class NisaEmitter:
    names: dict          # nkipy Value → "%varname" string
    lines: list[str]     # accumulated output
    indent: int          # current nesting

    def emit_func(self, func): ...
    def emit_scf_for(self, op): ...
    def emit_alloc(self, op): ...
    def emit_copy(self, op): ...          # trace chain → nisa.dma_copy
    def emit_elementwise(self, op): ...   # trace operands → nisa.tensor_tensor_arith
    def emit_matmul(self, op): ...
    ...
```

The emitter assigns `%0`, `%1`, ... to each result as it walks. Dynamic offsets (loop IVs, arith results) are just string variable names — no Value objects needed.

## Access Tracing

Same chain walk as today, but returns strings:
- `base_name: str` — the `%varname` of the base alloc
- `offsets: list[str]` — one expression per dim (`"%iv"`, `"0"`, `"%3"`)
- `dropped_dims: list[bool]` — which dims are rank-reduced

When composing offsets needs arithmetic, the emitter appends arith ops as text and returns the new `%varname`.

## Memspace Translation

At emission time:
- `3 : i32` → `#nisa.mem<sbuf>`
- `4 : i32` → `#nisa.mem<shared_hbm>`
- `1 : i32` → `#nisa.mem<hbm>`
- `2 : i32` → `#nisa.mem<psum>`

## sbuf_map

`#nkipy.sbuf_map` is read directly from the memref type during access tracing. It never appears in the output — by the time we emit NISA ops, the tiling info has been consumed to produce correct access patterns.

## Output

The NKI compiler receives plain NISA MLIR text:

```mlir
module attributes {nisa.target = #nisa.target<trn2>} {
  func.func @kernel(%arg0: memref<256x256xf32, #nisa.mem<shared_hbm>>) -> memref<256x256xf32, #nisa.mem<shared_hbm>> {
    %0 = nisa.alloc : memref<128x128xf32, #nisa.mem<sbuf>>
    nisa.dma_copy src(%arg0[...]) dst(%0[...]) ...
    nisa.tensor_tensor_arith ...
    ...
  }
}
```

Compiled via:
```python
from nki.compiler._internal import ir, register_all_dialects
from nki.compiler.ncc_driver import compile_mlir_to_neff

ctx = ir.Context()
register_all_dialects(ctx)
module = ir.Module.parse(nisa_text, ctx)
compile_mlir_to_neff(module, func_name, opts)
```

## Migration Plan

1. ✅ Move `irutils.py` to shared `codegen/irutils.py` (KB already has it — NISA will use it too)
2. ✅ Rewrite `access.py` to return strings instead of `nk_ir.Value`s, move to shared location
3. ✅ Implement `nisa/emitter.py` (MLIR text builder, same shape as KB's `emitter.py`)
4. ✅ Implement `nisa/emit.py` — single walk + all patterns as methods
5. ✅ Delete: `context.py`, `walk.py`, `finalize.py`, `patterns.py`, old per-pattern files
6. Remove NKI wheel dependency from codegen (only needed at compile time) — partially done (`custom_ops.py` still uses `_vendor.py`/`nk_ir` for custom op resolution at compile time)
