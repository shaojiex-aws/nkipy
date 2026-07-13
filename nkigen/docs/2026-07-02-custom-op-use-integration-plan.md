# Plan: user-defined NKI kernels via `knob().use()`

**Date:** 2026-07-02  
**Status:** Phase 1 done (2026-07-13); phases 2–7 proposed  
**Working checkout:** `/home/ubuntu/nkipy/nkigen`

---

## Goal

Let a user write a kernel_builder kernel and splice it into a `@trace` program
with `knob(inputs..., output).use(kernel)`. nkigen extracts the subgraph between
the named boundary tensors, verifies (by CPU simulation) that the kernel computes
the same thing, and inlines its NISA body into the pipeline.

No `CustomOp` import, no manual specs.

---

## 1. What was broken (fixed in Phase 1, 2026-07-13)

`CustomOp` wraps a pre-compiled NISA function. Lifecycle: trace → emit
declaration → `_resolve_custom_ops` inlines the body. Four defects blocked this
— three named in the original plan, plus one the plan missed (d):

**(a) Crash on declarations.** ✅ Function passes (`CanonicalizeReshape`, etc.)
assumed every `func::FuncOp` had a body. Body-less `nkipy.custom_op` decl →
empty region deref → segfault. **Fixed:** `if (func.isDeclaration()) return;`
guard in the 10 per-function passes (used `isDeclaration()`, not the narrower
`hasAttr("nkipy.custom_op")` — a function pass on a body-less decl is always a
no-op).

**(b) `_resolve_custom_ops` never called.** ✅ **Fixed:** appended to
`linalg_to_nisa()` (see §Phase 1 below), no new pipeline entry.

**(c) NISA emitter drops everything.** ✅ **Fixed:** `emit_module` re-emits the
`nkipy.custom_op_bodies` attr, `_emit_func` handles body-less decls, `_emit_op`
has a `func.call` branch.

**(d) Call↔decl type mismatch (NOT in the original plan).** ✅ Once (a) was
fixed, MLIR's `-verify-each` rejected the module: `canonicalize-reshape` stamps
`#nkipy.mem<SharedHbm>` onto the buffer feeding the `func.call` (from the user's
`knob().layout()`), but the body-less decl's signature is frozen at trace time.
The decl type is inherently *dynamic* — it must track whatever mem-space the
layout passes give the call boundary. **Fixed:** converted `CanonicalizeReshape`
from a `func::FuncOp` pass to a `ModuleOp` pass (a func pass may not legally
mutate a sibling decl under MLIR's threaded pass manager), which then re-syncs
each custom-op decl's `FunctionType` to its call site. Also had to stamp
custom-op **call results** as `SharedHbm` before the view-propagation loop, so a
chained `op(op(x))` keeps the inner result's type consistent with the outer
call operand (call results are neither func args, return operands, nor view
results, so propagation otherwise missed them).

**Not yet supported:** passing a *sliced/reshaped* tensor into a custom op
(`op(x.reshape(...))`) — the emitter raises a clear `NotImplementedError`; the
whole-tensor path is the supported common case. Phase 3–4's `.use()` extraction
subsumes this.

---

## 2. The `.use()` design

### 2.1 API

```python
import numpy as np
import nki.compiler.kernel_builder as nb
from nki.compiler.kernel_builder import Tensor, build_kernel
from nki.compiler.kernel_builder import isa as nisa
from nkigen import trace, knob

# A plain kernel_builder kernel — nothing nkigen-specific.
def matmul_silu_kernel(x_hbm: Tensor, w_hbm: Tensor, out_hbm: Tensor):
    x_sb  = nb.compiler.alloc((128, 128), nb.float32, nb.sbuf)
    w_sb  = nb.compiler.alloc((128, 128), nb.float32, nb.sbuf)
    psum  = nb.compiler.alloc((128, 128), nb.float32, nb.psum)
    o_sb  = nb.compiler.alloc((128, 128), nb.float32, nb.sbuf)
    bias  = nb.compiler.alloc((128, 1),   nb.float32, nb.sbuf)
    scale = nb.compiler.alloc((128, 1),   nb.float32, nb.sbuf)

    nisa.dma_copy(dst=x_sb, src=x_hbm)
    nisa.dma_copy(dst=w_sb, src=w_hbm)
    nisa.matmul(dst=psum, stationary=x_sb, moving=w_sb, accum=False)
    nisa.memset(dst=bias, value=0.0)
    nisa.memset(dst=scale, value=1.0)
    nisa.activation(dst=o_sb, src=psum, bias=bias, scale=scale,
                    op=nisa.activation_function.silu)
    nisa.dma_copy(dst=out_hbm, src=o_sb)

# Use it inside a nkigen @trace program.
@trace(input_specs=[((128, 128), "f32"), ((128, 128), "f32")])
def model(x, w):
    mm = np.matmul(x, w)
    y  = mm * (1.0 / (1.0 + np.exp(-mm)))       # SiLU
    knob(x, w, y).use(matmul_silu_kernel)
    return y

nisa_text = model.to_nisa(target="trn2")
```

Consistent with existing `knob()` usage — all boundary tensors positional:

```python
knob(result).tile_op(...)         # one tensor
knob(a, b).fuse()                 # multiple peers
knob(q, k, v, ctx).use(kernel)   # boundaries of a region
```

**The system classifies inputs vs outputs from the graph:**
- A tensor that is a block arg or produced *outside* the boundary set → input.
- A tensor produced by ops reachable from those inputs → output.

If the tensors don't form a connected region, or the shapes don't match the
kernel signature → immediate error telling you what's wrong. No guessing.

**Multi-output:** `knob(x, a, b).use(two_output_kernel)` — `a` and `b` are both
outputs if they're produced between the boundaries.

### 2.2 Architecture — no new pipeline passes

Extraction absorbs into **post-trace** (after all ops are emitted, before the
pipeline starts). Resolution absorbs into the existing **`py:linalg-to-nisa`**.

```
   trace time                         post-trace (module complete, pre-pipeline)
   ──────────                         ──────────────────────────────────────────
   knob(x, w, y).use(kernel)          extract_use_regions(module):
     │  records marker:                 1. for each nkipy.use marker:
     │  (boundary tensors, kernel)         classify inputs/outputs from graph
     │  — no surgery yet                2. bridge kernel → NISA-MLIR
     ▼                                  3. simulate-verify region vs kernel
   nkipy.use op on the                  4. emit func.call, stash body,
   boundary values                         delete region ops + marker
                                         ▼
                               pipeline runs unchanged;
                               at end of py:linalg-to-nisa,
                               call _resolve_custom_ops to inline the body
```

**Zero new pass entries.** Extraction is a post-trace hook (like existing module
finalization). Resolution is a function call appended to `linalg_to_nisa()`.

### 2.3 Region extraction

```python
def extract_region(boundaries, func):
    """Classify boundary tensors into inputs/outputs, find region ops."""
    inputs, outputs = [], []
    for v in boundaries:
        if _is_block_arg(v) or v.owner not in _ops_of(func):
            inputs.append(v)
        else:
            outputs.append(v)

    if not inputs:
        raise ValueError("No inputs found — at least one boundary tensor "
                         "must be a trace input or produced outside the region.")
    if not outputs:
        raise ValueError("No outputs found — at least one boundary tensor "
                         "must be produced by ops between the inputs.")

    # Walk forward from inputs, backward from outputs, intersect
    reachable_fwd = _forward_reachable(inputs)
    region = _backward_walk(outputs, stop_at=set(inputs))

    for op in region:
        if op not in reachable_fwd:
            raise ValueError(
                f"Output depends on ops not reachable from the specified "
                f"inputs. Check your boundary tensors.")

    # Any region result used outside must be in outputs
    for op in region:
        for res in op.results:
            if res not in set(outputs) and _has_external_use(res, region):
                raise ValueError(
                    f"Tensor escapes the region but wasn't listed in knob(). "
                    f"Add it to the boundary tensors.")

    return region, inputs, outputs
```

Every error tells you exactly what to fix. No silent mis-slicing.

### 2.4 Bridge: kernel_builder → NISA-MLIR

```python
from nki.compiler.kernel_builder import build_kernel, Tensor

def _kernel_to_nisa_mlir(kernel_fn, input_specs, output_specs, target="trn2"):
    module = build_kernel(kernel_fn, input_specs=input_specs,
                          output_specs=output_specs, target=target)
    return module.operation.get_asm(print_generic_op_form=True)
```

Produces `func.func @k(memref<...shared_hbm>...)` with a flat block of `nisa.*`
ops — exactly what `_resolve_custom_ops` already inlines. Verified working on
this box with `nki` 0.4.0 (this is the same path `CustomOp.from_kernel_builder`
already uses).

### 2.5 Verification by simulation

```
                         same inputs
                              │
              ┌───────────────┴───────────────┐
              │                               │
  ┌───────────┴───────────┐       ┌───────────┴───────────┐
  │ REGION REFERENCE      │       │ KERNEL REFERENCE      │
  │ extracted ops, run    │       │ nb.simulate_kernel()  │
  │ eagerly in NumPy      │       │ kernel_builder sim    │
  └───────────┬───────────┘       └───────────┬───────────┘
              │                               │
              └───────────────┬───────────────┘
                   np.allclose(rtol, atol)
                              │
             mismatch → error at trace time
```

Opt-out with `verify=False`. Mismatch raises `ValueError` naming the site and
max abs diff.

---

## 3. Implementation phases

| Phase | Deliverable | Done-when | Status |
|---|---|---|---|
| **1** | Fix compile path: passes skip declarations; decl↔call types reconciled; emitter preserves call+decl+stash; resolution called inside `linalg_to_nisa` | Existing e2e custom-op tests pass | ✅ **done 2026-07-13** (also on HW) |
| **2** | Bridge: `CustomOp.from_kernel_builder()` | kernel_builder fn → NISA-MLIR that `_resolve_custom_ops` accepts | ✅ pre-existing + exercised by Phase 1 tests |
| **3** | Marker op + `.use()`: `Nkipy_UseOp` + registry | `knob(x, w, y).use(k)` traces to marker; registry populated | ⬜ not started |
| **4** | Extraction (post-trace hook) + verification | Multi-op, multi-output, and error cases all work | ⬜ not started |
| **5** | Replace `nkipy.gather` with built-in custom op | `np.take` goes through `.use()` path; `Nkipy_GatherOp` removed | ⬜ not started |
| **6** | Tests + docs | All tests pass; new e2e green | ⬜ not started |

### Phase 1 — fix the compile path ✅ DONE

**1a.** ✅ Guard empty-body deref in the 10 per-function passes:

```cpp
func::FuncOp func = getOperation();
if (func.isDeclaration()) return;   // per-func passes are no-ops on decls
```

Passes guarded: `CanonicalizeReshape`, `CanonicalizePartitionDim`,
`AssignLinalgOpIds`, `LegalizeLayout`, `SimplifyLinalg`, `InferLayout`,
`InsertSpillReload`, `InsertMemRefDealloc`, `KnobDrivenFusion`,
`InlineNkipyReference`. (The two module-scoped passes, `CanonicalizeCompute` and
`KnobDrivenTiling`, already tolerated decls via `walk`.)

**1a′.** ✅ (unforeseen — see §1(d)) `CanonicalizeReshape` promoted to a
`ModuleOp` pass; after applying mem_space annotations it re-syncs each custom-op
decl's `FunctionType` to its call site, and stamps custom-op call *results* as
`SharedHbm` so chained calls stay type-consistent.

**1b.** ✅ NISA emitter: `emit_module` re-emits `nkipy.custom_op_bodies` (via
`str(DictAttr)`, which round-trips through the `nk_ir` re-parse); `_emit_func`
dispatches body-less decls to `_emit_func_declaration`; `_emit_op` has a
`func.call` branch (`_emit_call`) that resolves operands through the view chain
and errors on unsupported sliced/reshaped inputs.

**1c.** ✅ `linalg_to_nisa()` emits NISA text (upstream `mlir` bindings), then —
if the text carries `nkipy.custom_op_bodies` — `_resolve_custom_ops_text()`
re-parses it in an `nk_ir` (NKI-wheel) context and runs `_resolve_custom_ops`.
No new pipeline entry.

**Tests:** 4 original e2e + 12 unit + 3 resolve-pass + 1 new chained-call e2e
all green; the e2e ran on a real Trainium device (numerically verified). Full
suite `pytest tests/ -n auto`: 327 passed, 1 xpassed, 6 failed — the 6 are
pre-existing on the `kb_codegen` branch (unrelated mem-space conflict in
`canonicalize-reshape`'s normal path; confirmed by stash+rebuild+rerun on a
clean baseline), not custom-op related.

### Phase 2 — bridge

- `_kernel_to_nisa_mlir(kernel_fn, input_specs, output_specs, target)` in
  `frontend/custom_op.py` — wraps `build_kernel`.
- `CustomOp.from_kernel_builder(kernel_fn, input_specs, output_specs, target)` —
  calls bridge, reads shapes/dtypes from the compiled func type.

### Phase 3 — marker op + `.use()`

- `Nkipy_UseOp` in `NkipyOps.td` (variadic operands = boundary tensors,
  `site_id` attribute).
- `.use(impl, *, verify=True)` on `_KnobBuilder`: emit `nkipy.use`, record
  `(site_id, impl, boundaries, verify)` in registry.
- Eager mode: run kernel via `nb.simulate_kernel`, return result.

### Phase 4 — extraction (post-trace hook) + verification

Runs after tracing completes, before the pipeline starts (alongside existing
module finalization). For each `nkipy.use` marker:

1. `extract_region(boundaries, func)` — classify inputs/outputs, validate.
2. Bridge kernel → `CustomOp`; validate shapes match.
3. Simulate-verify (region NumPy vs `nb.simulate_kernel`).
4. Emit `func.call`, stash body, erase region + marker.

### Phase 5 — replace `nkipy.gather` with a built-in custom op

`nkipy.gather` is an ad-hoc dialect op that does the same thing `.use()` does:
substitutes `np.take` semantics with `nisa.dma_copy_indirect`. It has its own
`TilingInterface` impl and `reference_impl` region — bespoke machinery for one
op.

1. Write an internal kernel_builder kernel that emits `dma_copy_indirect`.
2. Change the `np.take` trace path (`builder.py:take`) to emit a standard
   `.use()` marker pointing at that kernel instead of `nkipy_d.GatherOp`.
3. Remove `Nkipy_GatherOp` from `NkipyOps.td` and all its C++ support
   (`KnobDrivenTiling` special-casing, `SimplifyLinalg`, `NkipyOps.cpp`).

*Done-when:* `np.take` tests pass, `nkipy.gather` no longer exists in the
dialect, and the gather lowering goes through the custom op path.

### Phase 6 — tests + docs

- Existing e2e tests (Phase 1 gate).
- New `.use()` e2e: multi-op region, multi-output, error cases.
- `np.take` tests still pass via the custom op path.
- `pytest tests/ -n auto` green.

### Phase 7 — pluggable op registry

The same mechanism generalizes: register a callable with both a **NumPy
reference** (for eager mode + verification) and a **kernel_builder kernel** (for
NISA). During tracing, nkigen intercepts the call and emits a `.use()` marker
automatically — no manual `knob().use()` at the call site.

```python
import numpy as np
import nki.compiler.kernel_builder as nb
from nki.compiler.kernel_builder import Tensor
from nki.compiler.kernel_builder import isa as nisa
from nkigen import trace, register_op

# 1. Define the NumPy reference (what it computes).
def sigmoid_ref(x):
    return 1.0 / (1.0 + np.exp(-x))

# 2. Define the kernel_builder kernel (how hardware runs it).
def sigmoid_kernel(x_hbm: Tensor, out_hbm: Tensor):
    x_sb  = nb.compiler.alloc((128, 128), nb.float32, nb.sbuf)
    o_sb  = nb.compiler.alloc((128, 128), nb.float32, nb.sbuf)
    bias  = nb.compiler.alloc((128, 1), nb.float32, nb.sbuf)
    scale = nb.compiler.alloc((128, 1), nb.float32, nb.sbuf)
    nisa.dma_copy(dst=x_sb, src=x_hbm)
    nisa.memset(dst=bias, value=0.0)
    nisa.memset(dst=scale, value=1.0)
    nisa.activation(dst=o_sb, src=x_sb, bias=bias, scale=scale,
                    op=nisa.activation_function.sigmoid)
    nisa.dma_copy(dst=out_hbm, src=o_sb)

# 3. Register: callable + reference + kernel.
sigmoid = register_op(reference=sigmoid_ref, kernel=sigmoid_kernel)

# 4. Use it in a trace — looks like a normal function call.
@trace(input_specs=[((128, 128), "f32")])
def model(x):
    return sigmoid(x)         # emits .use() marker during tracing
                              # runs sigmoid_ref in eager/numpy mode

nisa_text = model.to_nisa(target="trn2")
```

No torch, no external package imports inside `@trace`. The user defines their
own op library (could ship as a pip package like `nkigen-ops`) with pairs of
(numpy reference, kernel_builder kernel). During tracing it emits a `.use()`
marker; in eager mode it runs the reference; verification cross-checks both.

This is the same path `np.take` takes internally after Phase 5 — the built-in
gather is just `register_op(reference=numpy_take, kernel=dma_indirect_kernel)`
registered by nkigen itself.

Implementation: `register_op` returns a callable that:
- In eager mode: calls `reference(*args)`, returns numpy.
- During tracing: emits ops from the reference (to get the linalg subgraph),
  then emits a `nkipy.use` marker pointing at the kernel. Same
  extraction/bridge/resolve path as manual `.use()`.

---

## 4. Verified facts (`~/.venv/nkidev`, `nki` 0.4.0)

- **Crash cause:** empty region deref on body-less decl → SIGSEGV. ✅ fixed.
- **Decl↔call type drift:** confirmed real — mem_space stamping mutates the
  call boundary while the decl signature is frozen. ✅ fixed by reconciling
  decls in `canonicalize-reshape` (now a `ModuleOp` pass) + stamping call
  results as HBM.
- **`_resolve_custom_ops`** already handles multi-output (loops `num_results`).
- **Bridge** (`build_kernel`) produces `func.func` with `nisa.*` ops; re-parses cleanly.
- **`nb.simulate_kernel`** returns numpy; exact match confirmed.

## 5. Risks

| Risk | Status / Mitigation |
|---|---|
| More passes crash on declaration | ✅ resolved — guarded all 10 per-func passes |
| Decl type drifts from call site as layout passes run | ✅ resolved — `canonicalize-reshape` reconciles decls at module scope (this risk materialized and was the main surprise of Phase 1) |
| Bridge uses `nki.compiler.kernel_builder` internals | Verified on 0.4.0; pin wheel |
| Emitter round-trip breaks stashed body | ✅ `str(DictAttr)` round-trips; checked by `Module.parse` in resolve pass |
| Sliced/reshaped tensor passed into a custom op | Errors clearly today; subsumed by Phase 4 `.use()` extraction |
| Extraction reorders ops (Phase 4) | Erase in reverse-topo; `replace_all_uses_with` first |
