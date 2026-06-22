# Architecture Review & Tech-Debt Cleanup Plan

**Date**: 2026-06-10
**Scope**: full `nkigen` checkout on branch `kb_codegen` (post-Phase-6 of the
textual kernelbuilder codegen plan)
**Status**: partially fixed (see ✅ markers below)

Baseline at review time: `python -m pytest tests/ -q -n auto` → **307 passed,
5 failed** (~42 s). All 5 failures are in the new kernelbuilder backend and are
root-caused in §1.

After fixes: **309 passed, 3 failed**. Fixed: head_deconcat, attention
(rank-reducing subview + index_cast). Remaining: custom-op tests (need kb
`func.call` handler), qwen3 (DMA shape mismatch in deep reshape chain).

This review covers: the kb codegen bugs, the compiler-pass organization
(the stated concern), cross-cutting single-source-of-truth violations, dead
code, and a recommended cleanup sequence. §7 lists things inspected and
judged *fine as-is*, so they don't get re-litigated later.

---

## 1. The five failing tests — root causes (P0)

All five failures share two design causes (silent fallbacks + an IR construct
the indexing layer doesn't know) and one structural oversight (declaration-only
functions). None require redesign.

### 1.1 ✅ `IndexError: attempt to access out of bounds block`
*Tests: `test_custom_op.py::test_kernel_builder_silu`,
`test_custom_op.py::test_matmul_custom_activation_string_check`*

`_ModuleEmitter.run()` iterates every `func.func` in the module
(`codegen/kernelbuilder/__init__.py:150-155`) and `_emit_func` immediately does
`func.regions[0].blocks[0]` (`__init__.py:166`). Custom ops emit **body-less
private declarations** (`frontend/builder.py:186-208`,
`emit_custom_op_declarations`), which have a region with zero blocks → crash.

**Fix (two stages):**
1. *De-crash*: skip declaration-only funcs in `irutils.func_ops()` (or in
   `run()`): `if not func.regions[0].blocks: continue`. The `func.call` at the
   call site then surfaces as a `# TODO unhandled op: func.call` comment —
   visible, not crashing.
2. *Real support*: a custom op's NISA body is stashed as MLIR text in
   `nkipy.custom_op_bodies` (consumed by `codegen/nisa/custom_ops.py`), which
   kb cannot translate. But `CustomOp.from_kernel_builder` users originally
   *wrote* kernel_builder Python — keep that original source on the `CustomOp`
   object and splice it into the generated module as a nested function +
   call. Until then, raise `NotImplementedError("custom ops not yet supported
   by the kernelbuilder backend")` so the gap is loud (see §2.1).

### 1.2 ✅ `0  # TODO: unresolved index` corrupting emitted slices + rank-reducing subview bug
*Test: `test_attention.py::test_attention_scores_loop[2-4-256-256-tile_size0]`*

Emitted code contains
`output_0[0  # TODO: unresolved index:0  # TODO: unresolved index + 1, ...]` —
a comment spliced *inside* a subscript.

Root cause: traced `fori_loop` IVs are `i32` and reach subview offsets through
`arith.index_cast` (`%0 = arith.index_cast %arg2 : i32 to index`).
`index_expr()` only understands constants, named values, and the
`_ARITH_BINOP` set (`emit_indexing.py:30-38`); `index_cast` matches nothing
and falls to the sentinel (`emit_indexing.py:90`). C++-generated `scf.for`
loops have `index`-typed IVs, which is why everything else works.

Same root cause, second-order bug: `_depends_on_loop_reg()`
(`emit_indexing.py:50-61`) also stops at `index_cast`, so a Reg-dependent
slice behind a cast would render as a Python slice instead of `nb.ds(...)`.

**Fix (~10 lines):** treat `arith.index_cast` / `arith.index_castui` as
transparent: in `index_expr` and `_depends_on_loop_reg`, recurse into
operand 0; add both names to `_SILENT_SKIP` in
`codegen/kernelbuilder/__init__.py:361`.

### 1.3 ⚠️ `NameError: UNSUPPORTED_RESHAPE` (head_deconcat ✅, qwen3 remaining)
*Tests: `test_qwen3_layer.py::test_qwen3_layer`,
`test_head_deconcat.py::test_head_deconcat`*

`_compose_chain()` punts on true multi-dim split/merge reshape chains
(`emit_indexing.py:352-358` handles only the single-non-unit-block collapse
case; `:373-374` rejects expand groups with >1 non-unit dim) and
`memref_expr()` then emits the `"UNSUPPORTED_RESHAPE"` sentinel
(`emit_indexing.py:419`).

**Applied fixes:**
- Wired `_cross_collapse` / `_cross_expand` into `_compose_chain` for
  multi-dim groups.
- Added `_fold_subview_through_collapse` for `subview(collapse_shape(...))`
  chains (aligned-trailing-dim decomposition via div/mod).
- Handled rank-reducing subviews: mark dropped dims (size=1) as squeeze.
- Added rank-mismatch tolerance when inner chain already resolved to base.

**Result:** `test_head_deconcat` passes. `test_qwen3_layer` progresses past
UNSUPPORTED_RESHAPE but hits a DMA shape mismatch (16384 vs 8192 elements)
in a deeper nested reshape chain — needs further generalization of the
subview-through-collapse decomposition for non-aligned splits.

**Remaining fix:** targeted unit tests for the `_cross_*` / `_fold_*` helpers
first — the function currently has no direct tests, only e2e coverage.

---

## 2. Fail-loud: the silent-fallback philosophy (P1)

The kb backend's policy is "emit a comment/sentinel and keep going". That is
the right *authoring* posture for a backend being built phase-by-phase, but it
is now the direct cause of the worst debugging experiences:

- sentinels surface as `NameError` inside `nb.fori_loop` body callbacks at
  simulate time, several frames away from codegen
  (`emit_indexing.py:90`, `:419`);
- unknown dtypes/memspaces emit `nb.float32  # TODO dtype ...`
  (`api.py:138,141`) — *silently wrong numerics*, worse than a crash;
- unsupported reduction/generic shapes emit TODO comments
  (`emit_compute.py:173,208,222,237`);
- unhandled ops emit `# TODO unhandled op: <name>`
  (`__init__.py:296`) — fine as a visible marker, but nothing ever checks it.

The NISA backend has the mirror-image problem: every `@pattern` rewriter
returns silently when preconditions fail (e.g. `elementwise.py:21,26`,
`copy.py:35`, `matmul.py:34`), leaving the `linalg.*` op in place; the failure
then appears as an unrelated parse/verify error in the downstream NKI parser.
Pattern registration itself relies on side-effect imports in
`codegen/nisa/__init__.py:22-31` — dropping one import line silently
unregisters a whole rewriter family.

**Proposed fix — one mechanism per backend, both cheap:**

1. **kb**: add a post-emission validation in `linalg_to_kernelbuilder()`:
   scan the generated source for `TODO` / `UNSUPPORTED` markers and raise
   `KernelBuilderEmitError` listing them (with `strict=False` escape hatch for
   interactive use). Convert `api.py` dtype/memspace fallbacks to raises —
   silently changing numeric types is never acceptable.
2. **nisa**: after `_walk_and_rewrite()` (`walk.py:42-59`), walk once more and
   collect any surviving `linalg.*` / unrewritten candidate ops; raise with op
   name + location. This single check catches both "pattern skipped on
   precondition" and "module never imported / pattern never registered".

---

## 3. Pass-pipeline organization (P1 — the stated concern)

### 3.1 Diagnosis: the pipeline is sound; its *representation* is the debt

The actual pass *content and ordering* looks deliberate and is well-commented.
The problems are all "pipeline described in N places, all drifting":

| Source of truth | Location | State |
|---|---|---|
| The real pass list | `driver/pipeline.py:282-360` | authoritative |
| Docstring enumeration "1..24" | `pipeline.py:218-256` | numbering diverges after the group expands (1 entry → 3 passes shifts every index by 2); prose still says NISA lowering "currently stripped" while `py:linalg-to-nisa` exists |
| Inline `# 6`, `# 7`, `# 7b`… comments | `pipeline.py:305-353` | same off-by-N problem |
| README pass tables "1–26" | `README.md:155+` | references `nkigen/transforms/nkipy_opt.py` — a path that **no longer exists** post-restructure (`README.md:87,159,264,315`) |
| C++ registration | `Passes.td` + 7 `PassWrapper` files | names duplicated, see §3.3 |

Additionally, pass-spec strings are re-parsed with the same
`p.split('=')[0].split('"')[0].strip()` idiom in four places
(`_expand_pass_groups:39`, `_resolve_pass_index:72`, `flush_batch:434`,
plus `_pass_to_arg`'s quote handling), and `PASS_GROUPS` (`pipeline.py:24-32`)
is infrastructure built for N groups currently carrying exactly one.

### 3.2 Proposed fix: pipeline-as-data

Replace the string list with a declarative module-level table; derive
everything else from it. Sketch:

```python
@dataclass(frozen=True)
class PassSpec:
    name: str                       # registered pass name
    kind: str = "cpp"               # "cpp" | "py"
    options: dict = field(default_factory=dict)   # {"target": ...} filled at run time
    phase: str = ""                 # "Layout & Tiling", "Bufferization", ...
    note: str = ""                  # one-liner; docstring/README generated from these
    group: str = ""                 # optional group label, replaces PASS_GROUPS

PIPELINE: list[PassSpec] = [
    PassSpec("remove-redundant-zero-fill", group="canonicalize-linalg-for-nisa",
             phase="Linalg prep", note="drop linalg.fill(0) feeding matmul"),
    ...
    PassSpec("linalg-to-nisa", kind="py", phase="NISA lowering"),
]
```

Wins, in order of value:
- `stop_after` / `stop_before` resolve against `PassSpec.name` — the
  `name:N` / `py:` string-parsing in `_resolve_pass_index` (`pipeline.py:47-86`)
  mostly disappears.
- One serializer (`spec_to_cli_arg`) replaces the four parse sites; options
  become a typed dict instead of embedded quoted strings
  (`'infer-layout="target={target}"'` → `options={"target": target}`).
- Index comments are deleted; phases live in data; the docstring shrinks to a
  description of *behavior*, and the README table can be generated (or simply
  replaced by "see `driver/pipeline.py:PIPELINE`").
- Numbering can never drift again because there is no numbering.

This is a mechanical refactor; `apply_complete_knob_pipeline`'s public
signature and behavior stay identical, so all `stop_after=` tests keep
passing unmodified.

### 3.3 The Python↔C++ name contract is convention-only

Nothing verifies that every name in the pipeline is a registered nkipy-opt
pass; a rename in `Passes.td` surfaces as a mid-pipeline subprocess failure.

**Fix (cheap, high leverage):** one unit test that runs `nkipy-opt --help`
once and asserts every `PIPELINE` entry with `kind="cpp"` appears in the
output. ~15 lines, kills the whole drift class.

### 3.4 C++ side: dual pass declaration for 7 passes

All 22 passes are declared in `Passes.td`, and `Passes.cpp` registers them via
`GEN_PASS_REGISTRATION`. But 7 implementations predate the tablegen base
classes and use `PassWrapper` with **hand-written duplicate**
`getArgument()`/`getDescription()` strings:

- `CanonicalizeReshape.cpp:92-102`, `SimplifyLinalg.cpp:727-733`,
  `ApplyAndStripTransforms.cpp:31`, `EliminateUninitializedCopies.cpp:79`,
  `InsertMemRefDealloc.cpp:57`,
  `CanonicalizeLinalgForNisa/MatmulPrep.cpp:99,324`,
  `CanonicalizeLinalgForNisa/Arithmetic.cpp`.

The other ~12 already use the generated `…Base` classes (via `PassGen.h`
`GEN_PASS_CLASSES`). Migrating the 7 stragglers deletes the duplicated
name/description strings (the drift §3.3 tests for) and the
`MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID` boilerplate. Mechanical,
since the .td entries already exist.

Minor, same area:
- `PassGen.h:19` closes with `#endif // Allo_MLIR_PASSDETAIL_H` — a leftover
  from the Allo codebase it was adapted from; cosmetic but confusing.
- `CanonicalizeLinalgForNisa/` has no README linking it to the Python
  pass-group of the same name; one paragraph fixes discoverability.

### 3.5 Not worth changing

- Subprocess + tempfile per batch (`run_nkipy_opt_passes`): necessitated by
  the NISA-dialect global-ctor conflict documented in the module docstring;
  per-pass subprocesses in `dump_dir` mode are an acceptable debug cost.
- Per-pass `target` option redeclaration in `Passes.td` (4×): idiomatic MLIR;
  the lookup is already centralized in `HardwareConstants.h::lookupTarget`.
- `LegalizeLayout.cpp` (2352 lines) and `InferLayout.cpp` (973): big but
  internally sectioned with clear helper structure (`LayoutInfo`,
  `BlockLoopNest`, …). Splitting is optional polish, not debt; revisit only
  when one of them next needs a substantial feature.

---

## 4. Single-source-of-truth violations (P1/P2)

### 4.1 ✅ Memory-space encodings — five Python copies, two of them wrong

Authority: `mlir/include/nkipy/Dialect/NkipyAttrs.td:19-23` —
`{Hbm:1, Psum:2, Sbuf:3, SharedHbm:4, Constant:5}`, **zero deliberately
reserved** (an `IntegerAttr(0)` memspace is dropped during memref uniquing —
the .td comment explains this).

| Site | Encoding | Verdict |
|---|---|---|
| `frontend/knob.py:41` `_MEM_SPACE_MAP` | 1..4 | correct, documented |
| `codegen/kernelbuilder/irutils.py:19-22` | 1..4 | correct |
| `frontend/builder.py:1612` (in `annotate()`) | **0..3** | **off-by-one** — `Hbm→0` would be silently dropped, `Psum→1` would mean *Hbm* |
| `frontend/builder.py:33` `_MEM_SPACE_CONSTANT = 4` | 4 | **mislabeled** — 4 is SharedHbm; the documented intent (`traced_array.py:342`: "CONSTANT memory space (mem_space=5)") is `Constant=5` |
| `codegen/nisa` predicates (`patterns.py` `_is_hbm/_sbuf/_psum`) | parse attrs | fine (string-level, derived) |

Mitigating context, verified:
- `builder.annotate()` (`builder.py:1586-1643`) has **zero callers** — all
  annotation goes through `knob.py`. The off-by-one is latent, not live.
- `constant_tensor()` is reachable only via `lift_constant`
  (`traced_array.py:339-350`), whose only call site is the unary-ufunc
  scalar-lift branch (`traced_array.py:104-108`) that appears practically
  unreachable (a unary ufunc dispatched to `__array_ufunc__` always has the
  TracedArray itself as the input). Binary scalar ops take
  `_scalar_binary` instead and never build a Constant tensor. The C++
  consumer (`AnnotateMemorySpace.cpp:122` `MemSpaceEnum::Constant`
  verification) is therefore dead today.

**Fix:** create one module — suggest `nkigen/memspace.py` —

```python
class MemSpace(enum.IntEnum):
    # Values MUST match mlir/include/nkipy/Dialect/NkipyAttrs.td.
    # Zero reserved: IntegerAttr(0) memspace is dropped by MemRefType.
    HBM = 1; PSUM = 2; SBUF = 3; SHARED_HBM = 4; CONSTANT = 5
```

import it from `knob.py` and `kernelbuilder/irutils.py`; **delete**
`builder.annotate()` outright (dead, buggy); and make a decision on the
Constant-marker half-feature: either fix `_MEM_SPACE_CONSTANT` to 5 and add a
test that exercises `lift_constant` → `AnnotateMemorySpace` verification, or
delete `constant_tensor`/`lift_constant` and the C++ `Constant` case together.
Half-implemented on both sides is the worst state.

### 4.2 Op-classification tables — kb and nisa each have one

- kb: `codegen/kernelbuilder/ops.py` (`LINALG_OPS`, `ARITH_BODY_OPS`) — the
  good one: dataclass per op, role names, explicit "single source of truth"
  docstring.
- nisa: `codegen/nisa/patterns.py:23-38` (`_LINALG_TO_ARITH_OP`,
  `_REDUCE_BODY_OP_TO_ARITH`) plus activation tables in `activation.py`.

Adding one linalg op today means editing both backends in different shapes.
The tables differ only in value type (kb: enum-member *names*; nisa: real
`nisa.ArithOp` enums — and nisa must stay importable without the kb modules
and vice versa).

**Fix:** promote kb's `ops.py` to `codegen/ops.py` (it already has the right
shape and no dependencies); nisa resolves members at use site via
`getattr(nisa.ArithOp, info.member)`. Keep backend-specific quirks (e.g.
nisa's cross-lane reduce mapping) local. While there, the duplicated
`DYN_SENTINEL = -(1 << 63)` (`kernelbuilder/emit_indexing.py:25`,
`nisa/access.py:10`) moves to the same shared module.

**Anti-fix, deliberately:** do *not* try to share the subview-chain walkers
(`emit_indexing._compose_chain` vs `access._get_base_and_offsets`). They look
similar but operate on different binding stacks (upstream `mlir.ir` vs the NKI
wheel's `nk_ir`) and materialize into different targets (Python strings vs
arith IR). A premature common abstraction would couple the backends through
their most fragile logic. Cross-reference them in docstrings instead.

### 4.3 Dtype mappings — three frontend copies

`builder.py:39-59` (`_MLIR_TO_NP`/`_np_to_mlir`), `mlir_utils.py:12-77`
(`to_mlir_type`, the canonical one), `utils.py:21-53`
(`np_supported_types`/`ctype_map`, vendored from Allo for the LLVM runner).
Plus kb's `api.py:110-121` `_DTYPES` (justified — it maps to *kb source
strings*, a different codomain).

**Fix (low priority):** fold `builder.py`'s maps into `mlir_utils.py`;
leave `utils.py` alone (its ctypes tables serve the execution engine and the
file is clearly marked as vendored).

---

## 5. Dead code & stale artifacts (P2 — quick deletions)

All verified by grep to have zero call sites:

| Item | Location | Action |
|---|---|---|
| ✅ `apply_custom_op()` | `frontend/builder.py:1779-1813` | ~~delete~~ deleted |
| ✅ `builder.annotate()` | `frontend/builder.py:1586-1643` | ~~delete~~ deleted |
| ✅ `passmanager` import | `frontend/builder.py:17` | ~~remove~~ removed |
| `apply_passes()` | `driver/pass_manager.py` (whole file) | delete or quarantine — never *called* anywhere; `tests/passes/pass_utils.py:15` imports it and doesn't use it; exported from `nkigen/__init__.py:12`. It's also the only in-process PassManager user, a path the project abandoned for the subprocess driver. If kept for notebooks, move the doc-example into README and stop exporting it top-level |
| `verify_against_numpy()` | `execution/execution.py` | decide: exported at top level but unused by the test suite (tests use `pass_utils.verify_tiled_mlir_with_numpy`). Either adopt it in tests or stop exporting |
| `_const_int()` | `codegen/nisa/access.py:44-58` | delete (kb's `irutils.const_int` is the live twin) |
| `push_indent()/pop_indent()` | `codegen/kernelbuilder/emitter.py:76-81` | delete — only the context manager is used |
| `tests/e2e/outputs/**` committed | 7 dirs incl. `kb_code.py` with TODO/UNSUPPORTED markers | `git rm -r --cached tests/e2e/outputs` — `.gitignore:27` already excludes the path; the tracked copies are stale generated artifacts (two of them are literally the broken outputs of the failing tests) |
| README pipeline section | `README.md:87,155-264,315` | rewrite after §3.2: stale module path (`nkigen/transforms/nkipy_opt.py`), stale pass numbering, stale repo-layout tree |

---

## 6. Frontend structure (P3 — optional, do last)

`frontend/builder.py` (1813 lines) is a working monolith with clean section
markers. A split is *worthwhile but not urgent* — it has no correctness
implications and the section comments already give it navigability. If/when
splitting, the natural seams (verified against the section map) are:

- `builder/core.py` — `IRBuilder`, `TensorHandle`, `LoopIndexHandle`, `_loc`
  (`builder.py:75-227`)
- `builder/generic.py` — broadcast/cast/`_scalar_binary`/`_binary_dispatch`/
  unary wrappers (`:230-533`)
- `builder/ops.py` — the ~60 public ops (`:538-1378`) — could stay one file;
  it's a flat dictionary of small functions
- `builder/indexing.py` — slicing/gather/concat (`:1380-1578`)
- control flow + scalar lifting (`:1646-1776`) → merge into existing
  `frontend/control_flow.py`

Two genuine (small) issues to fix regardless of the split:

1. **`IRBuilder` resource handling** (`builder.py:125-138, 221-226`): manual
   `__enter__` on context/location with no `try/finally` in `__init__`, and
   cleanup depends on callers remembering `finally: b.cleanup()` (trace.py
   does; future callers may not). Make `IRBuilder` a context manager and use
   `with IRBuilder(...) as b:` in `trace.py`.
2. **`_to_handle` defined twice** with different scalar policies
   (`op_vtable.py:23-33` passes scalars through; `control_flow.py:71-80`
   lifts them via `lift_scalar_to_tensor`). Rename the control-flow one
   (`_to_loop_carried_handle`) or fold the scalar-lift policy into one
   function with an explicit flag — the silent behavioral difference under
   one name is the trap.

Known wart, explicitly *deferred*: the `knob` module-vs-function shadowing
dance (`nkigen/__init__.py:15-20`, `frontend/__init__.py` comment,
`apis.py:8`). `knob.knob(x)` is awkward, but every test and README example
uses it; renaming is API churn best decided alongside the open-sourcing API
review, not during this cleanup.

Layering note: the only frontend→driver dependency is
`IRBuilder.run_canonicalize` → `run_nkipy_opt_passes`
(`builder.py:210-216`), forced by the `nkipy.yield` verification issue
(documented in `docs/2026-06-05-nkipy-block-no-terminator-error.md`). It is
acyclic and commented; leave it.

---

## 7. Reviewed and fine — do not "fix"

- **The Phase-0 restructure paid off.** `frontend / driver / codegen{nisa,kb}
  / execution` layering is real and acyclic; lazy imports at the
  `driver→codegen` boundary are intentional (NKI wheel optional for
  pre-Phase-5 tests).
- **`codegen/nisa/_vendor.py`** — single import point for NKI internals is
  honored everywhere (one inline `scf` import in `gather.py:346`, harmless).
- **`walk.py` collect-then-rewrite** — safe with current patterns (each
  rewriter erases its own op; no cross-pattern result deps). Worth a 3-line
  comment stating that invariant, nothing more.
- **kb `api.py` versioning** — the Protocol-based seam is genuine (no
  `nisa.*` strings outside it). Converting to an ABC is optional polish.
- **kb emitter state** (`names` / `loop_regs` / `_tile_roles` rebuilt per
  function) — clean.
- **Test architecture** — `Mode` flags incl. `Mode.CODEGEN` round-trip via
  `nb.simulate_kernel` is a strong harness design; per-pass test dirs mirror
  pass names; `stop_after`/`stop_before`-by-name keeps tests robust to
  pipeline insertions (37+ uses).
- **The text-level regex strip** of `dst_indirect_max_index`
  (`codegen/nisa/__init__.py:92-96`) — fragile by nature but thoroughly
  commented and root-caused upstream (NKI builder injects an attribute its
  own verifier rejects); keep until the wheel fixes it.
- **finalize.py post-passes ordering** — the fold passes legitimately need
  all per-op rewrites complete; the double-DCE in `walk.py:56-59` is
  deliberate. A short "why these are post-passes" comment would help.
- **Hardware constants** — C++-only today (`HardwareConstants.h`); Python
  codegen has no duplicated magic sizes (checked: only comments mention 128).
  No action until a Python pass needs them; then generate, don't copy.

---

## 8. Recommended sequence

Small PRs, each leaving the suite green (312 after PR-1/2):

| # | Content | Effort | Risk |
|---|---|---|---|
| 1 | ✅ **kb bugfixes**: `index_cast` transparency (§1.2), skip declaration funcs (§1.1), rank-reducing subview squeeze, dead code cleanup (§4.1, §5) | done | — |
| 2 | ⚠️ **kb reshape + rank-reducing** (§1.2, §1.3): multi-dim collapse/expand, `_fold_subview_through_collapse`, rank-reducing squeeze — head_deconcat + attention fixed; qwen3 DMA mismatch remains | mostly done | — |
| 3 | **Fail-loud** (§2): kb post-emission validation + raise on unknown dtype/memspace; nisa post-walk leftover check | ~½ day | may surface latent gaps in other tests — that's the point |
| 4 | **Single sources of truth** (§4): `memspace.py`, shared `codegen/ops.py`, dtype-map fold; delete dead code list (§5) | ~1 day | mechanical |
| 5 | **Pipeline-as-data** (§3.2) + pass-name contract test (§3.3); regenerate docstring; fix README | ~1 day | mechanical, high readability payoff |
| 6 | **C++ hygiene** (§3.4): PassWrapper→tablegen-Base ×7, `PassGen.h` guard comment, `CanonicalizeLinalgForNisa/README` | ~½ day | mechanical, needs rebuild |
| 7 | *Optional*: `IRBuilder` context manager, `_to_handle` rename, `builder.py` split (§6) | as desired | cosmetic |

PRs 1–3 unblock "moving on" (the autotune-backend plan); 4–5 are the
debt-clearing the pause was for; 6–7 are opportunistic.
