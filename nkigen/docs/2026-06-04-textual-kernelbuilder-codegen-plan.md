# Textual KernelBuilder Code Generation for NKIGen

**Date**: 2026-06-04
**Status**: In Progress

## Progress

- ✅ **Phase 0: Package Restructuring** — complete (commit `5f27d4d`). All 312 tests pass.
  Followed by a cleanup pass: dropped the `codegen_` filename prefix
  (`nisa/elementwise.py` etc.), added `nisa/_vendor.py` as the single source of
  truth for the nki-wheel import paths, and relocated misplaced shared helpers
  (`_pad_shape_to_2d` → `patterns`, `_fold_reinterpret_casts` → `finalize`,
  deleted the duplicate `_index_const` in favor of `access._emit_const_index`).
- 🔄 **Phase 1: Foundation — IR Analysis & Code Emitter Skeleton** — in progress
- ⬜ Phase 2: Memory Operations
- ⬜ Phase 3: Compute Operations
- ⬜ Phase 4: Control Flow
- ⬜ Phase 5: Integration & Pipeline Hookup
- ⬜ Phase 6: Variable Naming & Readability
- ⬜ Phase 7: Validation & Testing
- ⬜ Phase 8: Documentation & Examples

## Goal

Add a new code generation backend to nkigen that emits readable Python
`kernel_builder` source code from the compiled IR. Instead of only lowering to
NISA MLIR (consumed by `ncc_driver`), nkigen will also be able to produce
equivalent `kernel_builder` Python code that users can read, modify, and
compile independently through the standard NKI compilation flow.

## Motivation

- **Readability**: Users can inspect what the compiler produced in familiar
  Python/kernel_builder terms rather than raw MLIR.
- **Hand-tuning**: Generated code serves as a starting point for manual
  optimization — users get an optimized skeleton from nkigen and refine it.
- **Alternative compilation path**: nkigen → kb Python → standard NKI
  `compile_kernel()` flow, enabling use of the full NKI optimization stack.
- **Debugging**: Easier to reason about correctness at the kernel_builder
  abstraction level vs raw NISA ops.
- **Education**: Shows users how high-level NumPy maps to low-level tile ops.

## Architecture Overview

```
@trace NumPy function
    ↓
nkigen pipeline (passes 1–23, C++ via nkipy-opt)
    ↓
Post-Phase-4 IR (memref + scf + linalg + arith, with nkipy annotations)
    ↓
    ├──→ [existing] codegen/nisa/ → NISA MLIR → NEFF
    │
    └──→ [NEW] codegen/kernelbuilder/ → kernel_builder Python source code
```

The new backend consumes the same post-Phase-4 IR that the NISA codegen
uses (after pass 23, before the `py:linalg-to-nisa` pass). It walks the IR
and emits equivalent `kernel_builder` API calls as Python text.

## Reference: kernel_builder API Surface (from private-nki-staging)

Key APIs to emit:

| Category | APIs |
|----------|------|
| Memory | `nb.compiler.alloc(shape, dtype, space)`, `nb.compiler.release(tile)`, `nb.compiler.rotate(tile)` |
| Data movement | `nisa.dma_copy(dst, src)` |
| Compute | `nisa.matmul(dst, stat, mov)`, `nisa.tensor_tensor_arith(dst, lhs, rhs, op)`, `nisa.activation(dst, op, src, bias, scale)` |
| Control flow | `nb.fori_loop(start, stop, step, body_fn)`, `nb.if_else(cond, true_fn, false_fn)` |
| Indexing | Tile slicing via `tile[par_slice, free_slice]` |

---

## Step-by-Step Plan

### Phase 0: Package Restructuring ✅ DONE

The current `nkigen/nkigen/` layout is flat and uses a `transforms/` folder
name that conflates three different roles: compiler driver orchestration,
MLIR→NISA code generation, and pass coordination. Before adding a new backend,
reorganize into clear sub-packages.

#### Current layout (problem)

```
nkigen/nkigen/
├── __init__.py
├── apis.py
├── builder.py          (1809 lines — IR construction)
├── compile.py          (74 lines — NEFF compilation entry)
├── control_flow.py     (163 lines — fori_loop tracing)
├── custom_op.py        (228 lines — kernel_builder CustomOp wrapper)
├── execution.py        (26 lines — LLVM/NumPy dispatch)
├── knob.py             (265 lines — @knob annotation API)
├── llvm.py             (642 lines — LLVM CPU runner)
├── mlir_utils.py       (122 lines — MLIR type helpers)
├── op_vtable.py        (411 lines — NumPy op dispatch)
├── pass_manager.py     (119 lines — legacy pass manager)
├── trace.py            (164 lines — @trace decorator)
├── traced_array.py     (350 lines — TracedArray wrapper)
├── utils.py            (217 lines — misc utilities)
├── _mlir/              (generated dialect bindings)
└── transforms/
    ├── __init__.py
    ├── nkipy_opt.py    (467 lines — compiler driver + pass pipeline)
    └── linalg_to_nisa_py.py (2767 lines — NISA codegen, monolith)
```

Issues:
- `transforms/` mixes driver logic (`nkipy_opt.py`) with codegen (`linalg_to_nisa_py.py`)
- `linalg_to_nisa_py.py` is a 2767-line monolith with 6+ logical sections
- Top-level is flat — tracing, IR building, compilation, and execution all at same level
- No clear place for a second codegen backend

#### Target layout

```
nkigen/nkigen/
├── __init__.py
├── apis.py
│
├── frontend/                    # NumPy → linalg MLIR (tracing)
│   ├── __init__.py
│   ├── trace.py                 (from: trace.py)
│   ├── traced_array.py          (from: traced_array.py)
│   ├── op_vtable.py             (from: op_vtable.py)
│   ├── builder.py               (from: builder.py)
│   ├── control_flow.py          (from: control_flow.py)
│   ├── knob.py                  (from: knob.py)
│   └── custom_op.py             (from: custom_op.py)
│
├── driver/                      # Pipeline orchestration
│   ├── __init__.py
│   ├── pipeline.py              (from: transforms/nkipy_opt.py)
│   └── pass_manager.py          (from: pass_manager.py)
│
├── codegen/                     # Backends (IR → target)
│   ├── __init__.py
│   ├── nisa/                    # NISA MLIR codegen (split from linalg_to_nisa_py.py)
│   │   ├── __init__.py          (entry point linalg_to_nisa + @pattern side-effect imports)
│   │   ├── _vendor.py           (single source of truth for nki wheel imports: nk_ir, nisa)
│   │   ├── context.py           (module parsing, memspace rewrite, _to_nki_module)
│   │   ├── access.py            (_Access class, _get_base_and_offsets, offset/const helpers)
│   │   ├── affine_map.py        (_create_standard_nisa_map, _build_nisa_map, _operand_kwargs)
│   │   ├── patterns.py          (pattern registry, _RewriteContext, predicates, shape helpers)
│   │   ├── elementwise.py       (_rewrite_elementwise)
│   │   ├── copy.py              (_rewrite_memref_copy, _rewrite_linalg_copy)
│   │   ├── alloc.py             (_rewrite_memref_alloc, _rewrite_memref_dealloc)
│   │   ├── transpose.py         (_rewrite_linalg_transpose)
│   │   ├── matmul.py            (_rewrite_matmul_transpose_a)
│   │   ├── activation.py        (_emit_activation, _rewrite_linalg_activation, _rewrite_reciprocal)
│   │   ├── fill.py              (_rewrite_linalg_fill)
│   │   ├── reduction.py         (_classify_reduction, _rewrite_linalg_generic_reduction, generic dispatch)
│   │   ├── gather.py            (_rewrite_nkipy_gather, _emit_gather_iteration, DMA indirect)
│   │   ├── custom_ops.py        (_resolve_custom_ops)
│   │   ├── finalize.py          (_fold_reinterpret_casts, _fold_hbm_reshapes — post-pass cleanups)
│   │   └── walk.py              (_walk_and_rewrite + _dce_dead_view_ops — top-level orchestration)
│   │   # _finalize_for_nki lives in __init__.py alongside the entry point.
│   │
│   └── kernelbuilder/                 # [NEW] kernel_builder Python codegen
│       ├── __init__.py          (exports: linalg_to_kernelbuilder)
│       ├── emitter.py           (Emitter — indent, naming, imports, line output)
│       ├── api.py               (API surface abstraction — maps logical ops to kb API calls)
│       ├── emit_memory.py       (alloc, release, DMA copy)
│       ├── emit_compute.py      (arithmetic, activation, matmul, reduction)
│       ├── emit_control_flow.py (fori_loop, if_else)
│       └── emit_indexing.py     (subview → slice expressions)
│
├── execution/                   # Runtime (LLVM, compile, run)
│   ├── __init__.py
│   ├── compile.py               (from: compile.py)
│   ├── llvm.py                  (from: llvm.py)
│   └── execution.py             (from: execution.py)
│
├── _mlir/                       (unchanged — generated dialect bindings)
├── mlir_utils.py                (stays — shared MLIR helpers)
└── utils.py                     (stays — shared utilities)
```

#### Task 0.1: Create sub-package directories ✅

- Create `frontend/`, `driver/`, `codegen/`, `codegen/nisa/`, `codegen/kernelbuilder/`, `execution/`
- Add `__init__.py` for each.

#### Task 0.2: Move frontend files ✅

- Move `trace.py`, `traced_array.py`, `op_vtable.py`, `builder.py`,
  `control_flow.py`, `knob.py`, `custom_op.py` → `frontend/`
- Update `frontend/__init__.py` to re-export public symbols.

#### Task 0.3: Move driver files ✅

- Move `transforms/nkipy_opt.py` → `driver/pipeline.py`
- Move `pass_manager.py` → `driver/pass_manager.py`
- Update `driver/__init__.py`.

#### Task 0.4: Split `linalg_to_nisa_py.py` into `codegen/nisa/` modules ✅

Split the 2767-line monolith along its natural section boundaries:

| Section (line range) | Target module | Contents |
|---------------------|---------------|----------|
| 1–110 | `context.py` | `_rewrite_memspace_text`, `_to_nki_module`, imports |
| 112–202 | `access.py` | `_Access`, `_reassoc_groups`, `_const_int`, `_emit_const_index`, `_emit_addi/muli/divui`, `_get_base_and_offsets` |
| 536–640 | `affine_map.py` | `_create_standard_nisa_map`, `_build_nisa_map`, `_operand_kwargs`, `_empty_operand_kwargs`, `_scalar_operand_kwargs` |
| 643–746 | `patterns.py` | `pattern` decorator, `_RewriteContext` class, helper predicates (`_is_memspace`, `_is_hbm`, etc.) |
| 748–783 | `codegen_elementwise.py` | `_rewrite_elementwise` + generic dispatch in `_rewrite_linalg_generic` |
| 784–898 | `codegen_copy.py` | `_rewrite_memref_copy`, `_rewrite_linalg_copy` |
| 900–974 | `codegen_alloc.py` | `_rewrite_memref_alloc`, `_fold_reinterpret_casts`, `_rewrite_memref_dealloc` |
| 975–1099 | `codegen_transpose.py` | `_rewrite_linalg_transpose` |
| 1100–1166 | `codegen_matmul.py` | `_rewrite_matmul_transpose_a` |
| 1167–1260 | `codegen_activation.py` | `_rewrite_reciprocal`, `_emit_activation`, `_rewrite_linalg_activation` |
| 1261–1332 | `codegen_fill.py` | `_rewrite_linalg_fill` |
| 1334–1913 | `codegen_reduction.py` | All reduction analysis/rewriting + generic body classification |
| 1914–2271 | `codegen_gather.py` | `_build_dma_copy_indirect_op`, gather iteration, `_rewrite_nkipy_gather` |
| 2272–2390 | `finalize.py` | `_fold_hbm_reshapes`, `_try_fold_hbm_reshape_alloc/arg` |
| 2392–2443 | `walk.py` | `_dce_dead_view_ops`, `_walk_and_rewrite` |
| 2444–2730 | `custom_ops.py` | `_resolve_custom_ops`, `_clone_op_with_map` |
| 2731–2767 | `__init__.py` | `linalg_to_nisa` entry point, `_finalize_for_nki` |

#### Task 0.5: Move execution files ✅

- Move `compile.py`, `llvm.py`, `execution.py` → `execution/`
- Update `execution/__init__.py`.

#### Task 0.6: Update all internal imports ✅

- Grep for all `from .transforms import`, `from .trace import`, etc.
- Update to new paths (`from .frontend.trace import`, `from .driver.pipeline import`, etc.)
- Add compatibility re-exports in top-level `__init__.py` if needed for external consumers.

#### Task 0.7: Update test imports ✅

- Update all `tests/` imports to match new package structure.
- Run full test suite to verify nothing broke.

#### Task 0.8: Delete old `transforms/` directory ✅

- Once all tests pass with new structure, remove `transforms/` and the orphaned top-level files.

---

### Phase 1: Foundation — IR Analysis & Code Emitter Skeleton

#### Task 1.1: Create the `codegen/kernelbuilder/` module structure

- Create `codegen/kernelbuilder/__init__.py` with public entry point:
  ```python
  def linalg_to_kernelbuilder(mlir_text: str, kernel_name: str, target: str = "trn2") -> str
  ```
- Return type is a string containing valid Python source code.

#### Task 1.2: Build IR walker infrastructure

- Parse the post-Phase-4 MLIR using upstream `mlir.ir` (same as NISA codegen)
- Walk the `func.func` to extract:
  - Function signature (args → HBM inputs/outputs with shapes and dtypes)
  - `memref.alloc` → SBUF/PSUM tile allocations
  - `scf.for` → loop nests
  - `linalg.generic` / `linalg.matmul` / `linalg.fill` → compute ops
  - `memref.copy` → DMA copies (HBM↔SBUF)
  - `memref.subview` → tile slicing
  - `memref.dealloc` → release calls

#### Task 1.3: Implement `emitter.py` (Emitter)

- Create an `Emitter` class that manages:
  - Indentation tracking
  - Variable name generation (SSA values → readable Python names)
  - Import statement collection
  - Line output with proper formatting
- Output format:
  ```python
  import nki.compiler.kernel_builder as nb
  from nki.compiler.kernel_builder import isa as nisa

  def kernel_name(input_0: nb.Tensor, output_0: nb.Tensor):
      tile_0 = nb.compiler.alloc((128, 512), dtype=nb.float32, space=nb.sbuf)
      ...
  ```

#### Task 1.4: Implement `api.py` (API surface abstraction)

- Define an abstraction layer that maps logical operations (alloc, copy,
  matmul, etc.) to concrete kernel_builder API call strings.
- All kernel_builder API knowledge lives here — function names, argument
  order, enum values, import paths.
- The `emit_*.py` modules call into `api.py` rather than hardcoding API strings.
- **Why**: The kernel_builder API will evolve. When it does, we adapt `api.py`
  (or create a versioned variant) without touching the IR-walking logic in
  `emit_*.py` or the formatting logic in `emitter.py`.
- Design for future versioning:
  - `api.py` exports a class or protocol (e.g., `KernelBuilderAPI`) with methods
    like `alloc(shape, dtype, space) -> str`, `dma_copy(dst, src) -> str`, etc.
  - A future `api_v2.py` can implement the same interface for a newer API version.
  - The entry point selects the API version via a parameter:
    ```python
    linalg_to_kernelbuilder(mlir_text, kernel_name, target="trn2", api_version="v1")
    ```

---

### Phase 2: Memory Operations (`emit_memory.py`)

#### Task 2.1: Emit `alloc` / `release`

- Map `memref.alloc` with memspace annotations to `nb.compiler.alloc(shape, dtype, space=...)`:
  - memspace 3 → `nb.sbuf`
  - memspace 2 → `nb.psum`
- Map `memref.dealloc` → `nb.compiler.release(tile)`
- Track SSA value → variable name mapping.

#### Task 2.2: Emit DMA copies

- Map `memref.copy` between different memspaces to `nisa.dma_copy(dst, src)`.
- Handle subview chains: trace back to base memref and emit appropriate tile
  slicing syntax (`tile[par_start:par_end, free_start:free_end]`).

#### Task 2.3: Emit tile indexing/slicing (`emit_indexing.py`)

- Map `memref.subview` to Python slice expressions on tiles.
- Handle affine offset computations (from loop IVs) as index expressions.
- Produce readable index math: `input_0[i*128:(i+1)*128, :]`

---

### Phase 3: Compute Operations (`emit_compute.py`)

#### Task 3.1: Emit arithmetic operations

- Map `linalg.generic` with known body patterns to `nisa.tensor_tensor_arith`:
  - `arith.addf` → `nisa.arith_op.Add`
  - `arith.mulf` → `nisa.arith_op.Multiply`
  - `arith.subf` → `nisa.arith_op.Subtract`
  - `arith.maximumf` → `nisa.arith_op.Maximum`
- Handle broadcast patterns (scalar-tensor, tensor-tensor).

#### Task 3.2: Emit activation/unary operations

- Map unary `linalg.generic` bodies to `nisa.activation`:
  - `math.exp` → `nisa.activation_function.exp`
  - `math.tanh` → `nisa.activation_function.tanh`
  - `math.log` → `nisa.activation_function.log`
  - `math.sqrt` → `nisa.activation_function.sqrt`

#### Task 3.3: Emit matmul

- Map `linalg.matmul` / batched-matmul patterns to `nisa.matmul(dst, stationary, moving)`.
- Handle accumulate flag based on whether dst is pre-zeroed.

#### Task 3.4: Emit reductions

- Map reduction `linalg.generic` ops (with iterators `[parallel, reduction]`)
  to appropriate NISA reduce operations.

---

### Phase 4: Control Flow (`emit_control_flow.py`)

#### Task 4.1: Emit `fori_loop`

- Map `scf.for(lb, ub, step) { body }` to `nb.fori_loop(start, stop, step, body_fn)`.
- Emit loop body as a nested function or lambda.
- Handle loop-carried values (iter_args → function parameters + return).

#### Task 4.2: Emit nested loops

- Handle multi-level loop nests with proper indentation.
- Track induction variable names per nesting level.

#### Task 4.3: Emit conditionals (if applicable)

- Map `scf.if` to `nb.if_else(cond, true_fn, false_fn)`.

---

### Phase 5: Integration & Pipeline Hookup

#### Task 5.1: Add pipeline entry point in `driver/pipeline.py`

- Add `"py:linalg-to-kernelbuilder"` as a Python-phase pass.
- Wire it so the user can call:
  ```python
  from nkigen import trace
  kb_code = trace(my_func).to_kernel_builder(target="trn2")
  ```
- Internally: `apply_complete_knob_pipeline(stop_after="canonicalize:6")` +
  call `linalg_to_kernelbuilder()` on the result.

#### Task 5.2: Expose in public API (`nkigen/__init__.py`)

- Add `to_kernel_builder()` method or parameter to the traced function interface.
- Document the new output mode.

#### Task 5.3: Add dump support

- When `dump_dir` is set, also save `24_kernelbuilder.py` alongside the MLIR dumps.
- The file should be directly executable (with appropriate imports).

---

### Phase 6: Variable Naming & Readability

#### Task 6.1: Intelligent variable naming

- Use op metadata (`nkipy.op_id`, source location) to generate meaningful names.
- Name tiles by their role: `sbuf_matmul_lhs`, `psum_acc`, `hbm_output`, etc.
- Name loop variables: `i_tile`, `j_block`, etc.

#### Task 6.2: Comment generation (optional, behind flag)

- Optionally annotate generated code with:
  - Original source line references
  - Tile shapes and memory space info
  - Which nkigen pass produced each pattern

#### Task 6.3: Code formatting

- Run output through a formatter (or emit pre-formatted).
- Ensure generated code passes `ruff check` / basic linting.

---

### Phase 7: Validation & Testing

#### Task 7.1: Round-trip correctness tests

- For each e2e test in `tests/e2e/`:
  1. Trace the NumPy function
  2. Generate kernel_builder code
  3. Execute the generated code via `nb.compile_and_execute()` or `nb.simulate_kernel()`
  4. Compare numerical output against NumPy reference
- This proves the generated code is semantically equivalent.

#### Task 7.2: Unit tests for individual op mappings

- Test each IR pattern → kb code mapping in isolation.
- Use FileCheck-style assertions on the generated Python text.
- Cover: alloc, copy, matmul, arithmetic, activation, loop, slice.

#### Task 7.3: Syntax validity tests

- `compile(generated_code, "<test>", "exec")` — ensure generated code parses.
- Import-check: verify all referenced `nb.*` / `nisa.*` symbols exist.

---

### Phase 8: Documentation & Examples

#### Task 8.1: User documentation

- Add usage guide showing the `to_kernel_builder()` workflow.
- Include before/after examples (NumPy → generated kb code).

#### Task 8.2: Developer documentation

- Document how to add new op mappings.
- Document the IR patterns recognized by each emitter function.

---

## File Manifest (New/Modified)

### Phase 0 — Restructuring

| Action | From | To |
|--------|------|----|
| Move | `trace.py` | `frontend/trace.py` |
| Move | `traced_array.py` | `frontend/traced_array.py` |
| Move | `op_vtable.py` | `frontend/op_vtable.py` |
| Move | `builder.py` | `frontend/builder.py` |
| Move | `control_flow.py` | `frontend/control_flow.py` |
| Move | `knob.py` | `frontend/knob.py` |
| Move | `custom_op.py` | `frontend/custom_op.py` |
| Move | `transforms/nkipy_opt.py` | `driver/pipeline.py` |
| Move | `pass_manager.py` | `driver/pass_manager.py` |
| Split | `transforms/linalg_to_nisa_py.py` | `codegen/nisa/*.py` (14 modules) |
| Move | `compile.py` | `execution/compile.py` |
| Move | `llvm.py` | `execution/llvm.py` |
| Move | `execution.py` | `execution/execution.py` |
| Delete | `transforms/` | (after migration complete) |

### Phases 1–8 — New Files

| File | Purpose |
|------|---------|
| `codegen/kernelbuilder/__init__.py` | Entry point: `linalg_to_kernelbuilder()` |
| `codegen/kernelbuilder/emitter.py` | Emitter class (indent, naming, imports, line output) |
| `codegen/kernelbuilder/api.py` | API surface abstraction (versionable kb API mapping) |
| `codegen/kernelbuilder/emit_memory.py` | alloc, release, DMA copy emission |
| `codegen/kernelbuilder/emit_compute.py` | arithmetic, activation, matmul, reduction |
| `codegen/kernelbuilder/emit_control_flow.py` | fori_loop, if_else |
| `codegen/kernelbuilder/emit_indexing.py` | subview → slice expressions |
| `tests/e2e/test_kernelbuilder_codegen.py` | Round-trip correctness tests |
| `tests/unit/test_emitter.py` | Unit tests for Emitter class |
| `tests/unit/test_kernelbuilder_op_mappings.py` | Unit tests for IR→code patterns |

## Dependencies

- Upstream `mlir` Python bindings (already used by NISA codegen)
- `nki.compiler.kernel_builder` (for round-trip validation tests)
- No new C++ code required — this is a pure Python backend

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Restructuring breaks imports | Do Phase 0 as atomic commit; run full test suite before/after |
| `linalg_to_nisa_py.py` split introduces bugs | Split is purely mechanical (move functions); validate with existing NISA codegen tests |
| IR patterns not 1:1 with kb API | Start with the subset nkigen actually produces; add mappings incrementally |
| Spill/reload patterns complex | Emit explicit alloc+copy for spills; mark with comments |
| Loop-carried values tricky to emit | Use `fori_loop` with explicit return values matching scf.for iter_args |
| Generated code may not be optimal | Goal is correctness first; add optimization/cleanup pass later |
| kernel_builder API may evolve | All API knowledge isolated in `api.py`; swap/branch for new versions without touching IR logic |

## Success Criteria

1. Phase 0: All existing tests pass with zero changes to test logic (only import paths change)
2. Generated code compiles without errors via `nb.build_kernel()`
3. Generated code produces numerically correct results for all existing e2e tests
4. Generated code is human-readable (meaningful names, proper indentation, no redundant ops)
5. Round-trip latency < 2s for typical kernels (code generation is fast)
