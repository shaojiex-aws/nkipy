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
python setup.py build_ext    
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
    knob.knob(result).tile_op(tile_size=[128, 128]).layout(mem_space="SharedHbm")
    return result

module = add_scalar.to_mlir()

from nkigen.driver.pipeline import apply_complete_knob_pipeline
nisa_ir = apply_complete_knob_pipeline(str(module), target="trn2")
```

## The `knob` API

`knob.knob(tensor)` returns a chainable builder. Methods emit `nkipy.*` ops
and return `self`:

```python
knob.knob(x).tile_op(tile_size=[64, 64]).layout(mem_space="Sbuf")
```

- **`.tile_op(tile_size=[...])`** — loop tile for the producing op. One entry
  per linalg iterator:
  - Elementwise: matches output rank.
  - Reduction: matches *input* rank (compiler knows which axis reduces).
  - Matmul `A[M,K] @ B[K,N] -> C[M,N]`: `[M_t, N_t, K_t]`.
- **`.layout(mem_space=..., partition_dim=...)`** — memory placement.
  `mem_space` in `{"Hbm", "Psum", "Sbuf", "SharedHbm"}`. The physical
  factorization tile for SBUF is auto-derived from the consuming `tile_op`
  via indexing maps (no manual `tile_size` param).
- **`.cache(axis=[...])`** — SBUF staging hint (requires a preceding
  `.tile_op()`). Lists post-tiling loop levels where a cache buffer is
  allocated.
- **`knob.fuse(a, b, ...)`** — fuse sibling `scf.for` loops (each must have
  a matching `.tile_op`).

Unannotated intermediates get tiling/placement inferred by `infer-layout`.

## Compilation Pipeline

Defined in `nkigen/driver/pipeline.py` -> `apply_complete_knob_pipeline()`.

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
│  • annotate-memory-space (assign HBM / SBUF / PSUM)                 │
│  • canonicalize-reshape (materialize SBUF partition-dim reshapes)   │
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
from nkigen import trace, knob, TracedArray, CustomOp, verify_against_numpy
from nkigen.apis import knob as knob_fn, fori_loop
from nkigen.driver.pipeline import apply_complete_knob_pipeline
```

## Testing

```bash
source scripts/setup_nki.sh

pytest tests/passes/    # per-pass FileCheck tests
pytest tests/e2e/       # end-to-end (auto-skips without Trainium)
pytest tests/unit/      # Python-level unit tests
```

Test modes: `Mode.LLVM` (JIT vs NumPy), `Mode.HW` (on-device), `Mode.STRING_CHECK`,
`Mode.FILECHECK`. See `tests/README.md`.

## License

Apache License 2.0 — see repo-root `LICENSE`.
