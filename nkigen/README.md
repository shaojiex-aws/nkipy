# nkigen

A Python-to-NISA MLIR compiler for AWS Trainium. Write kernels as ordinary
NumPy functions, annotate placement and tiling with `knob`, and `nkigen`
traces them to linalg MLIR and lowers them through a pipeline of C++ and
Python passes to the NISA dialect for the Neuron compiler.

Part of the **NKIPy** monorepo (a uv workspace), alongside `nkipy` (the
NumPy-like front end) and `spike` (NRT runtime bindings).

## Setup

```bash
source scripts/setup_nki.sh   # venv + LLVM/MLIR + Neuron deps
pip install -e .              # builds nkipy-opt & _mlir bindings via CMake
python setup.py build_ext    # compiles Cython/C extensions in-place
```

Requirements: Python >= 3.10, a pre-built LLVM/MLIR, `clang-22`/`clang++-22`,
and the Neuron wheels (`nki`, `neuronx-cc`). Pure-Python changes take effect
immediately; C++ changes require `pip install -e .` again.

Environment overrides: `LLVM_INSTALL_PREFIX`, `NKIPY_VENV`, `CC`/`CXX`,
`BUILD_WITH` (ninja/make), `NUM_THREADS`.

## Quick Start

```python
import numpy as np
from nkigen import trace, knob

@trace(input_specs=[((128, 256), "f32")])
def add_scalar(x):
    result = x + 2.0
    knob(result).tile_op(tile_size=[128, 128]).layout(mem_space="SharedHbm")
    return result

# Compile to NISA assembly
nisa_ir = add_scalar.to_nisa(target="trn2")

# Or generate kernel_builder Python source
nki_code = add_scalar.to_nki(target="trn2")
```

## The `knob` API

`knob(tensor)` returns a chainable builder; each method emits a `nkipy.*`
marker op and returns `self`. All the verbs on one kernel:

```python
@trace(input_specs=[((512, 256), "bf16"), ((256, 512), "bf16"), ((512, 512), "f32")])
def fused_ops(a, b, c):
    mm = a @ b
    (knob(mm)
        .tile_op(tile_size=[128, 128, 128])  # one entry per linalg iterator:
                                             #   matmul [M_t, N_t, K_t]
                                             #   elementwise: output rank
                                             #   reduction: input rank
        .cache(a, axis=[-1]))                # SBUF staging for an input;
                                             # axis = post-tiling loop levels
                                             # (needs the .tile_op before it)
    d = c * 2.0
    (knob(d)
        .tile_op(tile_size=[128, 128])
        .layout(mem_space="Sbuf",            # Hbm | Psum | Sbuf | SharedHbm
                partition_dim=0))            # Sbuf-only (HBM has no partitions)

    o = mm + d
    knob(mm, d).fuse()                       # fuse the two scf.for loops
                                             # (each needs a matching .tile_op)
    return o
```

Unannotated intermediates get tiling/placement inferred by `infer-layout`.

### `.use()`: offload subgraphs to tuners, agents, or hand-written kernels

> **Status:** in development — see
> [docs/2026-06-09-nki-autotune-backend-plan.md](docs/2026-06-09-nki-autotune-backend-plan.md).

The verbs above steer nkigen's own pipeline. One more, `.use()`, carves out
a subgraph and hands it to an external backend — and each subgraph of one
program can go to a different one:

```python
@trace(input_specs=[((1024, 128), "bf16")] * 3
                   + [((128, 512), "bf16"), ((512,), "f32")])
def attn_block(q, k, v, w_o, bias):
    # ===
    # attention core → hand-written flash-attention IP, spliced verbatim
    # ===
    s = q @ k.T / np.sqrt(128.0)
    p = np.exp(s - np.max(s, axis=-1, keepdims=True))
    ctx = (p / np.sum(p, axis=-1, keepdims=True)) @ v
    knob(q, k, v, ctx).use(flash_attn)

    # ===
    # output projection → a custom backend (e.g. HLO, Marlin) compiles
    # this region itself — no tuning involved
    # ===
    proj = ctx @ w_o
    knob(ctx, w_o, proj).use(MarlinBackend())

    # ===
    # epilogue → schedule searched offline by a tuner: NKI Gym,
    # nki-autotune, or LLM agents
    # ===
    out = np.maximum(proj + bias, 0.0)
    knob(proj, bias, out).use(AgenticTuner(), key="attn_epilogue")
    return out

# offline: run the tuners, record best kernel per region in the DB
attn_block.tune(time_limit=3600, db="tune_db/")

# fast + deterministic: splice tuned kernels from the DB, no searching
nisa_ir = attn_block.to_nisa(target="trn2", tune_db="tune_db/")
```

`knob(...)` captures the subgraph by its boundary tensors — inputs and
result — and everything between them joins the region: `knob(q, k, v, ctx)`
grabs the whole softmax core, intermediates (`s`, `p`) included. Inputs
become the spliced kernel's parameters, the result its return value. `impl`
is a hand-written NKI kernel spliced verbatim, a custom backend that
compiles the region itself, or any `Tuner` (`tune(region, deadline)`) —
NKI Gym, nki-autotune, and LLM agents all plug into the same seam.

Tuning is offline and searches; compilation only reads the DB, never
searches (same input → same NEFF):

```
                       traced program (linalg MLIR)
┌──────────────────────┬──────────────────────┬──────────────────────┐
│ attention core       │ output projection    │ epilogue             │
│ .use(flash_attn)     │ .use(MarlinBackend)  │ .use(AgenticTuner)   │
└──────────┬───────────┴──────────┬───────────┴──────────┬───────────┘
           │                      │                      │
 (hand-    │           (backend   │     .tune(time_limit, db) — offline;
  written  │            compiles  │     tuner regions search in parallel
  kernel — │            the       │     across the Trn2 devices
  no tuning│            region    │            ┌─────────▼─────────┐
  needed)  │            itself —  │            │   AgenticTuner    │
           │            no tuning │            │ LLM-guided search │
           │            needed)   │            └─────────┬─────────┘
           │                      │              ┌───────▼───────┐
           │                      │              │   tuning DB   │ best kernel
           │                      │              └───────┬───────┘ per region
           │                      │                      │
┌──────────▼──────────────────────▼──────────────────────▼───────────┐
│ .to_nisa(target, tune_db) — fast, deterministic, never searches    │
│ NKI kernel → splice · backend → compile · DB hit → splice,         │
│ miss → warn+fallback · unmarked ops → normal nkigen knob pipeline  │
└─────────────────────────────────┬──────────────────────────────────┘
                                  ▼
                                NEFF
```

Spliced regions bypass nkigen's tiling/layout passes (the backend owns its
scheduling); shapes, dtypes, and numerics are validated against the region's
NumPy reference for every payload.

## Compilation Pipeline

Entry point: `traced_fn.to_nisa(target="trn2")`. Internally defined in
`nkigen/driver/pipeline.py` -> `apply_complete_knob_pipeline()`.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRACING (Python)                             │
│  User NumPy code → linalg ops on MEMREF + nkipy annotations         │
│  (nkipy.layout, nkipy.tile_op, nkipy.cache, nkipy.fuse_op)          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1: CANONICALIZATION                        (C++ / nkipy-opt) │
│  • canonicalize-compute (div→recip*mul, batch-matmul decomp,        │
│    zero-fill removal)                                               │
│  • infer-layout (propagate mem_space + partition_dim + tile_size)   │
│  • canonicalize-partition-dim (insert transposes for pdim=0)        │
│  • assign-linalg-op-ids                                             │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 2: TILING + PROMOTION                                        │
│  • knob-driven-tiling (tile_op → scf.for loops, SBUF/PSUM           │
│    promotion via cache knobs, apply + strip transforms)             │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 3: FUSION                                                    │
│  • knob-driven-fusion (fuse sibling loops, normalize loop steps,    │
│    canonicalize)                                                    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 4: LAYOUT LEGALIZATION                                       │
│  • canonicalize-reshape (apply mem_space, materialize SBUF reshapes)│
│  • legalize-layout (attach #sbuf_map, tile HBM↔SBUF copies)         │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 5: SCHEDULING                                                │
│  • simplify-linalg (decompose high-rank transposes, etc.)           │
│  • insert-spill-reload (SBUF pressure management, Belady's MIN)     │
│  • insert-memref-dealloc (lifetime endpoints + canonicalize)        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 6: CODEGEN                                       (Python)    │
│  • Backend A: py:linalg-to-nisa (→ NISA MLIR text)                  │
│  • Backend B: linalg-to-kernelbuilder (→ nb.compiler.* Python)      │
└─────────────────────────────────────────────────────────────────────┘
```

## Inspecting Intermediate IR

```python
# Full pipeline
nisa_ir = my_kernel.to_nisa(target="trn2")

# Dump all intermediate passes
from nkigen.driver.pipeline import apply_complete_knob_pipeline
apply_complete_knob_pipeline(mlir_str, dump_dir="debug_outputs/")
apply_complete_knob_pipeline(mlir_str, stop_after="legalize-layout")
```

From a test: `pytest tests/e2e/test_rope.py::test_rope --dump-ir -v -s`

With `nkipy-opt` directly: `nkipy-opt --legalize-layout input.mlir`

## Project Structure

```
nkigen/
├── nkigen/
│   ├── frontend/           # @trace, knob, TracedArray, op_vtable, CustomOp
│   ├── driver/             # pipeline.py (pass orchestration), pass_manager.py
│   ├── codegen/
│   │   ├── nisa/           # textual NISA MLIR emitter (emit.py, custom_ops.py)
│   │   └── kernelbuilder/  # kernel_builder Python source emitter
│   ├── execution/          # verify_against_numpy, LLVM JIT, Neuron compile
│   └── _mlir/              # generated MLIR Python bindings
├── mlir/                   # nkipy dialect + C++ passes (TableGen, lib/Transforms/)
├── tests/                  # passes/ (FileCheck), e2e/, unit/
├── scripts/                # setup_nki.sh
└── pyproject.toml
```

## Public API

```python
from nkigen import trace, knob, fori_loop, TracedArray, CustomOp, verify_against_numpy
```

## Testing

```bash
source scripts/setup_nki.sh

pytest tests/ -n auto           # all tests in parallel (uses all cores)
pytest tests/passes/            # per-pass FileCheck tests
pytest tests/e2e/               # end-to-end (auto-skips without Trainium)
pytest tests/unit/              # Python-level unit tests
pytest -k test_add_2d           # name substring match
pytest tests/ -n auto -q --tb=short  # parallel, quiet, short tracebacks
```

### Test Modes

`Mode` flags in `tests/harness.py` control how each test verifies its kernel.
Modes can be combined with `|`, e.g. `Mode.HW | Mode.STRING_CHECK`.

| Mode | Meaning |
|------|---------|
| `LLVM` | LLVM JIT execution, compare to NumPy. Requires `stop_after`. |
| `HW` | Trainium hardware execution. Auto-skips when no device is detected. |
| `STRING_CHECK` | Assert compiled IR contains/excludes specific strings. |
| `FILECHECK` | Run LLVM FileCheck against the compiled IR. |
| `CODEGEN` | KernelBuilder codegen round-trip: gen KB Python, simulate, compare to NumPy. |

### Dumping IR

Pass `--dump-ir` to any test to save intermediate MLIR after every compiler pass:

```bash
pytest tests/e2e/test_rope.py::test_rope --dump-ir -v -s
```

## License

Apache License 2.0 — see repo-root `LICENSE`.
