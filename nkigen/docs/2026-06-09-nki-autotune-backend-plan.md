# Plan: subgraph delegation (`knob().use()`) with nki-autotune as a tuner backend

**Date:** 2026-06-09 (rev. 2026-06-10: `use()` API, Tuner protocol, offline tuning DB; bridge spike verified)
**Status:** Proposed
**Working checkout:** `/home/ubuntu/nkipy/nkigen` (package `nkigen`)
**Backend repo:** `/home/ubuntu/nki-autotune` (packages `nkigym` + `autotune`, branch `dev_1`)

---

## 1. Goal

nkigen's knobs (`knob(x).tile_op(...)`, `.layout(...)`, `fuse(...)`) are
human-set and **instant** — a knob call only emits a marker op at trace time.
nki-autotune sits one level lower: a fine-grained transform space
(Split / Fuse / Reorder / ComputeAt / ReverseComputeAt / SoftwarePipeline)
meant to be driven by a search agent, taking minutes-to-hours of compiles and
on-hardware runs.

This plan adds one knob verb — **`knob(x, y, ...).use(impl, key=...)`** — that
delegates the subgraph rooted at the marked tensors to an external
implementation:

```python
knob(a, b).use(flash_attn)                       # kernel object: splice this hand-written IP
knob(a, b).use(AgenticTuner())                   # region tuned by an LLM agent
knob(c, d).use(AutotuneTuner(policy="greedy"))   # region tuned by nki-autotune
```

`impl` is either a **kernel object** (e.g. a `CustomOp`) spliced directly, or a
**`Tuner`** whose kernel is discovered by an explicit offline tuning step and
recorded in a tuning DB. Both arrive through the same marker op and the same
resolve pass, are spliced via the **existing CustomOp / `resolve-custom-ops`
machinery**, and are validated identically and unconditionally: live-in/out
shapes and dtypes against the kernel signature (loud compile error on
mismatch), numerics against the region's NumPy reference.

| Layer | Knob surface | Driver | Granularity |
|---|---|---|---|
| **nkigen** | `tile_op`, `layout`, `fuse`, **`use`** | human | coarse, few knobs |
| **nki-autotune (one Tuner)** | Split/Fuse/Reorder/ComputeAt/SoftwarePipeline | agent/search | fine, many knobs |

### 1.1 Compile never searches

Compilation must stay fast and deterministic (same input → same NEFF; CI boxes
without Neuron devices can compile; the tuned kernel is a reviewable
artifact). So the lifecycle is the TVM/Halide split:

- **Tune (offline, long-running):** `prog.tune(time_limit=..., db=...)`
  slices the marked regions, runs each region's chosen Tuner, profiles
  candidates on hardware, records the winner per region in the DB.
- **Compile (always fast):** the `use-offload` pass looks up the DB. Hit →
  splice; miss → that region falls back to nkigen's own pipeline with a
  warning naming the region ("run `prog.tune()`"). Kernel-object `use()` needs
  no DB. Correctness never depends on tuning.

### 1.2 Tuner protocol, per-site choice, global budget

Tuner choice is **per site** — it *is* the knob value; different subgraphs of
one program use different tuners or policies. Time budget and DB are global:
one deadline shared by all (region, tuner) work, fanned across this box's 16
Neuron devices; a finished tuner frees devices for the rest — there are no
per-tuner limits, and nothing idles while the deadline runs.

```python
class Tuner(Protocol):
    name: str
    def tune(self, region: TuneRegion, deadline: Deadline) -> list[Candidate]
```

All candidates from all tuners are scored through the same
`autotune.runner.api.profile()`; best measured MFU per region wins and is
recorded with provenance (tuner name, policy, MFU, date).

### 1.3 Cache identity vs. `key=`

The DB **identity** is structural — canonical region IR + input shapes/dtypes
+ target + nki version — so an unchanged program always re-hits its result and
a stale entry can never be spliced into a non-matching region. `key=` is a
human-readable **folder label only**: results live under
`<db>/<key>/<struct_hash>/` (kernel source + result.json) so people can browse
"attn_qk" instead of a hash. Omitted, the key defaults to an op summary
(`matmul_1024x512x512_bf16`); a wrong or duplicated key is harmless. Pinning a
known result = exporting it and passing it back as a kernel object via
`use()`.

### 1.4 User flows

```python
@trace(input_specs=[((1024, 512), "bf16"), ((512, 512), "bf16"), ((512,), "f32")])
def prog(a, b, bias):
    c = a @ b
    knob(a, b).use(AutotuneTuner(), key="mm_proj")   # instant, like every knob
    return c + bias

prog.tune(time_limit="2h", db=DB)                    # offline, fills the DB
prog.compile(target="trn2", tune_db=DB)              # fast → NEFF
```

`prog.compile()` and `prog.tune()` are new methods on the `@trace` wrapper
(which already carries `to_mlir()`). `compile()` wraps the internal
`to_mlir()` → pass-pipeline → NEFF chain and infers input/output specs from
the trace; `tune()` shares the same front half through region slicing, then
drives the tuners. Without any `use()` site, `prog.compile()` is exactly
today's flow with a friendlier entry point — internals like the pass pipeline
stay internal.

---

## 2. Feasibility verdict: **doable**, and it reuses more shipped machinery than expected

Three integration seams already exist; the work is mostly *wiring*:

1. **Marking** rides the existing `knob()` builder. `use()` is a new method on
   `_KnobBuilder` ([nkigen/frontend/knob.py:44](../nkigen/frontend/knob.py#L44))
   emitting a new `nkipy.use` marker op, exactly parallel to how `tile_op`
   emits `nkipy.tile_op`.

2. **The tuner backend accepts source, not in-memory IR.** nki-autotune
   ingests a kernel as the *Python source* of an `@nkigym_kernel` function — it
   literally does `inspect.getsource()` + `ast.parse()`
   ([nkigym/src/nkigym/ir/dimension_analysis.py:319,348](/home/ubuntu/nki-autotune/nkigym/src/nkigym/ir/dimension_analysis.py)).
   So nkigen's job is to **emit `@nkigym_kernel` Python text** for the marked
   subgraph — a string-generation task, identical in spirit to the existing
   `codegen/kernelbuilder/` backend that already emits readable Python from the
   same IR level.

3. **Splice-back rides the existing CustomOp path.** nkigen already has a
   `CustomOp` ([nkigen/frontend/custom_op.py:138](../nkigen/frontend/custom_op.py#L138))
   that emits a `func.call @__custom_op__<name>(...)` during tracing, stashes an
   opaque NISA-MLIR body, and the **`resolve-custom-ops`** pass
   ([nkigen/codegen/nisa/custom_ops.py:8](../nkigen/codegen/nisa/custom_ops.py#L8))
   inlines that body at every call site in Phase 5. It already supports the
   *return-value style* (`func @f(%in) -> %out`) that an optimized subkernel
   produces. **This is the splice mechanism for both kernel objects and tuned
   results.**

The full tuning loop runs **on this machine**: a `trn2.48xlarge` (16 Neuron
devices, `/dev/neuron0..15`), with venv `~/.venv/nkidev` (Python 3.11.15)
carrying a real `nki` 0.4.0 + `neuronxcc`. `nkigym`/`autotune` are pure-Python
(`requires-python>=3.10`) and install cleanly into that venv.

### The bridge (spiked 2026-06-10: works)

nki-autotune `render()`s an optimized kernel as **`@nki.jit` NKI source**
(imports `nki`, calls `nisa.nc_matmul` / `nisa.dma_copy`, etc. —
[nkigym/src/nkigym/codegen/render.py:17](/home/ubuntu/nki-autotune/nkigym/src/nkigym/codegen/render.py)),
whereas the CustomOp path wants a **NISA-MLIR string** (generic form). The
bridge — **`@nki.jit` source → NISA-MLIR generic asm** — was spiked on this
box against the installed `nki` 0.4.0 wheel and works end-to-end:

```python
exec(nki_jit_source, ns)
with nki_ir_context() as ctx:                    # nki.compiler.driver
    res = TracerFrontend().compile(ctx, ns["k"], inputs=..., target="trn2")
    asm = res.module.operation.get_asm(print_generic_op_form=True)
# → func.func over HBM memrefs, single block: nisa.alloc/dma_copy/matmul.
# Round-trips through Module.parse(); exactly resolve-custom-ops's input shape.
```

This is the same `frontend.compile()` autotune itself calls inside
`compile_to_bir` — used here without the BIR step. Spike notes: must run
inside `nki_ir_context()` (plain `ir.Context` fails arg binding); PSUM drains
to SBUF before DMA-to-HBM (nkigym's drain gadget already does this). Residual
risk is only API stability of `nki.compiler.frontend` — pin the wheel (§7).

---

## 3. End-to-end flow

```
            nkipy source (user writes this)
            ┌───────────────────────────────────────────────┐
            │  @trace(...)                                  │
            │  def prog(a, b, bias):                        │
            │      c = a @ b                                │
            │      knob(a, b).use(AutotuneTuner(), key=...) │
            │      d = c + bias                             │
            │      return d                                 │
            └───────────────────────────────────────────────┘
                              │ trace  (frontend/)
                              ▼
        linalg-on-tensors MLIR + nkipy.use marker on the region's live-ins
                              │
      ┌───────────────────────┴────────────────────────────────────────┐
      │ OFFLINE  prog.tune(time_limit, db)                             │
      │   slice region → emit @nkigym_kernel source → site's Tuner     │
      │   searches under shared deadline → profile on Trn2 → best MFU  │
      │   → bridge to NISA-MLIR → record in db/<key>/<hash>/           │
      ├────────────────────────────────────────────────────────────────┤
      │ COMPILE  use-offload pass (Phase 1, pre-tiling)                │
      │   kernel object: validate sig + splice                         │
      │   tuner: DB hit → validate + splice                            │
      │          DB miss → leave region to normal pipeline + warn      │
      │   splice = func.call @__custom_op__<name> + stash NISA body    │
      └───────────────────────┬────────────────────────────────────────┘
                              │
            rest of nkigen pipeline runs UNCHANGED on remaining ops
                              ▼
            Phase 5: py:linalg-to-nisa + resolve-custom-ops (inline)
                              ▼
                            NEFF
```

Spliced regions **bypass nkigen's tiling/layout passes** — the tuner does its
own scheduling, and we hand it the workload at linalg-on-tensors level, where
live-ins/outs are plain tensors that map directly to nkigym HBM params.

---

## 4. Design details

### 4.1 The `use()` marking API

`knob(*tensors)` records the live-in set; `.use(impl, key=None)` emits one
**`nkipy.use`** op (variadic operands = marked tensors, no results — same
shape as `nkipy.fuse_op`, [nkigen/frontend/knob.py:210](../nkigen/frontend/knob.py#L210);
add `Nkipy_UseOp` to [NkipyOps.td](../mlir/include/nkipy/Dialect/NkipyOps.td)).
The impl (kernel or Tuner) and key are recorded site-side in a registry, like
CustomOp's.

**Subgraph inference.** Grow a convex region forward from the marked tensors:
include an op iff every input is marked or produced inside the region; values
consumed outside become live-outs. Live-ins → kernel params, live-out →
kernel return. Optional `out=` bounds the region from below when the closure
is larger than intended.

**Multi-output:** nki-autotune supports a single stored output today, so a
region with >1 live-out is a loud error in v1; relax when nki-autotune gains
multi-output support.

### 4.2 Slicing + nkigym source emission

1. **Slice** the convex region: collect ops, live-in and live-out SSA values
   (mechanical at linalg level; kernelbuilder does the same reasoning).

2. **Emit `@nkigym_kernel` source:** `NKILoad` per live-in, compute ops
   (`NKIMatmul`, elementwise via `NKIActivation`/`NKITensorScalar`,
   `NKITensorReduce`), `NKIStore` for the live-out; `input_specs` from live-in
   types. **Deterministic emitter, agentic fallback:** the emitter covers
   matmul/elementwise/reduce/transpose; otherwise fall back to nki-autotune's
   `compile_numpy_to_nkigym(f_numpy, input_specs)`
   ([numpy_to_nkigym.py:339](/home/ubuntu/nki-autotune/nkigym/src/nkigym/synthesis/numpy_to_nkigym.py))
   from a reconstructed numpy reference. v1 ships matmul-family + clear
   "unsupported op" error. The mapping table is small — nkigym has ~11 NKIOps,
   and nkigen's NISA pattern registry has the inverse map to crib from.

### 4.3 `prog.tune()` — driving Tuners

For each marked region, run its site's Tuner against the shared deadline:

```python
exec(nkigym_source, ns)
env = KernelMDP(ns["f_nkigym"], input_specs,
                transforms=[Split(), Fuse(), Reorder(),
                            ComputeAt(), ReverseComputeAt(), SoftwarePipeline()])
# AutotuneTuner.tune(): enumerate legal_actions / fixed trace, render() candidates
# score ALL tuners' candidates via autotune.runner.api.profile(..., "trn2")
# record best-by-MFU in db/<key>/<struct_hash>/
```

- `KernelMDP`/transforms: [mdp.py](/home/ubuntu/nki-autotune/nkigym/src/nkigym/environment/mdp.py),
  [transforms/](/home/ubuntu/nki-autotune/nkigym/src/nkigym/transforms/)
- Profiling: [api.py:24](/home/ubuntu/nki-autotune/autotune/src/autotune/runner/api.py)

**Scheduler:** all (region, tuner) pairs share one deadline and the 16-device
pool; finished work frees devices, no per-tuner limits. **v1 policy:** fixed
trace (`examples/tune_matmul_lhsT_rhs.py`-style) or bounded greedy rollout —
the deliverable is plumbing; AgenticTuner and pluggable policies come later.

### 4.4 Bridge: tuned `@nki.jit` source → NISA-MLIR (spiked, works)

The §2 spike proved the path on this box: `exec` the tuned source, then inside
`nki_ir_context()` call `TracerFrontend().compile(ctx, kernel, inputs=...,
target="trn2")` and take `res.module.operation.get_asm(print_generic_op_form=True)`.
The result is return-value-style `func.func` over HBM memrefs (flat single
block of `nisa.*`), which `Module.parse` accepts — exactly what
`resolve-custom-ops` inlines ([custom_ops.py:54](../nkigen/codegen/nisa/custom_ops.py#L54)).
Phase 4 is a small adapter wrapping this as a `CustomOp`
(`emit_custom_op_declaration` + body stash).

`nki.compiler.frontend` is internal API, so pin the wheel. **Fallbacks if it
shifts:** (1) render nkigym output in `kernel_builder` form so
`nb.build_kernel(...).get_asm(...)` — the call `CustomOp.from_kernel_builder`
already makes — yields NISA-MLIR; (2) standalone NEFF + external call.

### 4.5 Pipeline + validation

`py:use-offload` runs **before `knob-driven-tiling`**, right after
`assign-linalg-op-ids` ([pipeline.py:290](../nkigen/driver/pipeline.py)), as a
Python pass via `_run_python_pass` ([pipeline.py:466](../nkigen/driver/pipeline.py#L466)).
`prog.compile(tune_db=...)` threads the DB path down through the (internal)
pipeline entry to this pass.

Validation is unconditional for both payload kinds: signature (live-in/out
shapes, dtypes) → compile error; numerics vs. region NumPy reference → tested
at tune time, re-checkable at compile.

---

## 5. Phased implementation

| Phase | Deliverable | Done-when |
|---|---|---|
| **0. Env + smoke** | Install `nkigym`+`autotune` into `~/.venv/nkidev`; smoke-test matmul example + `profile()` on this Trn2 box. (Bridge spike done 2026-06-10, §4.4.) | matmul example runs; `profile()` returns real MFU. |
| **1. Marking + slicing** | `nkipy.use` op; `knob(*t).use(impl, key=)`; convex slicer; live-in/out detection; multi-live-out error. | Test program yields expected sliced ops + live-ins/outs. |
| **2. Kernel-object `use()`** | Resolve pass splices kernel objects with signature validation (refactor of CustomOp machinery — no tuner needed). | Hand-written matmul spliced e2e, numerics match. |
| **3. nkigym emission** | `linalg → @nkigym_kernel` emitter (matmul-family) + `input_specs`; elementwise/reduce/transpose next; agentic fallback. | Emitted source `exec`s; `simulate_fp32` matches region numpy. |
| **4. Bridge adapter** | Wrap the spiked §4.4 path as `nki_jit_source -> CustomOp`. | `Module.parse` accepts a tuned kernel's MLIR; spliced via resolve-custom-ops. |
| **5. Tune + DB** | `Tuner` protocol; `AutotuneTuner` (fixed trace/greedy); `prog.tune(time_limit, db)`; shared-deadline scheduler; `db/<key>/<hash>/` records. | Matmul region tunes measurably faster than canonical, recorded in DB. |
| **6. Compile lookup** | `prog.compile(target, tune_db)` wrapper; `use-offload` DB hit→splice / miss→fallback+warn. | Same program: instant compile post-tune; falls back cleanly pre-tune. |
| **7. Tests + docs** | Unit + e2e tests; examples; doc updates. | All 312 existing tests pass; new e2e green. |

Phases 0–3 are independent. Phase 4 (de-risked by the completed spike) gates
5/6.

---

## 6. Scope guards (v1)

- Single live-out regions only (loud error) until nki-autotune supports multi-output.
- Fixed-trace/greedy policy only; AgenticTuner and pluggable search later.
- Consumer of nki-autotune's public surface only (`compile_numpy_to_nkigym`,
  `build_initial_ir`, `KernelMDP`, `render`, `profile`); no changes there.

---

## 7. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| `@nki.jit → NISA-MLIR` bridge rides internal `nki.compiler.frontend` API | Low | Spiked working on `nki` 0.4.0 (§4.4); pin wheel; fallbacks in §4.4. |
| Convex slicing ambiguous on non-trivial programs | Med | Precise closure definition; `out=` override; single-live-out v1. |
| nkigym op set can't express region | Med | Deterministic emitter + agentic fallback; loud error otherwise. |
| venv / `nki` wheel mismatch between repos | Low | Verified on this box (`nki` 0.4.0, pure-Python backends); pin in Phase 0. |
| Stale DB entry after program change | Low | Structural identity (region IR+shapes+target+nki ver) can't match changed region → clean fallback + warn. |
| dtype gap: nkigym validates fp32, kernel is bf16 | Low | True dtypes in `input_specs`; fp32 is correctness-check only. |

---

## 8. Environment setup (Phase 0)

```bash
cd /home/ubuntu/nkipy/nkigen && source scripts/setup_nki.sh
pip install -e /home/ubuntu/nki-autotune/nkigym -e /home/ubuntu/nki-autotune/autotune
python /home/ubuntu/nki-autotune/examples/matmul_lhsT_rhs.py --cache /tmp/autotune_cache
```

This box is a Trn2 with a working `nki` 0.4.0 — the full loop runs locally.

---

## 9. Key file references

**nkigen:** `frontend/knob.py:44` (`_KnobBuilder` → `use()`), `frontend/custom_op.py:138`
(`CustomOp`), `codegen/nisa/custom_ops.py:8` (`resolve-custom-ops`),
`driver/pipeline.py:290,466` (pass list, `_run_python_pass`),
`codegen/kernelbuilder/` (emitter to crib), `mlir/.../NkipyOps.td` (`Nkipy_UseOp`).

**nki-autotune:** `nkigym/synthesis/numpy_to_nkigym.py:339`, `nkigym/ir/ir.py:119`,
`nkigym/environment/mdp.py`, `nkigym/transforms/`, `nkigym/codegen/render.py:17`,
`autotune/runner/api.py:24`.
