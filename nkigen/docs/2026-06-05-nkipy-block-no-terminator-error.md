# "block with no terminator" on nkipy Ops (Terminator TypeID Split)

**Date**: 2026-06-05
**Status**: Known constraint + adopted workaround

## TL;DR

Do **not** parse or run passes on IR that still contains an nkipy op with a
region (today only `nkipy.gather`, whose region is terminated by `nkipy.yield`)
using the in-process `mlir` Python bindings. It fails with:

```
error: block with no terminator, has "nkipy.yield"(...) : (...) -> ()
```

or, from a pass pipeline:

```
error: empty block: expect at least a terminator
```

Lower nkipy ops away first with the `nkipy-opt` **subprocess** (e.g.
`inline-nkipy-reference`, or just `canonicalize`), then parse the resulting
pure linalg/tensor IR in-process. This is already the pattern the rest of the
pipeline uses.

## Symptom

Only the `np.take` / gather tests fail; everything else passes. The pytest
summary shows a misleading wrapper:

```
mlir._mlir_libs._site_initialize.<locals>.MLIRError: Failure while executing pass pipeline:
```

The real diagnostic (truncated in the summary) is the "block with no
terminator" error above. Confusingly, the same IR run through the `nkipy-opt`
**command-line tool** canonicalizes fine — and it "works on a colleague's
machine."

## Root cause: trait TypeID / ODR split across two MLIR copies

In MLIR an op trait (e.g. `Terminator`) is identified by a `TypeID`, whose
identity is *the address of a function-local `static` variable* the compiler
emits for that trait. Two pieces of code agree that an op "is a Terminator"
only if they observe the **same** static (same address). Across shared
libraries that requires the symbol to be **exported** so the dynamic linker
coalesces the copies into one.

This process loads MLIR **twice**, as two independently-linked libraries:

| Role | Library |
|------|---------|
| Runs the parser / verifier / PassManager | upstream `libMLIRPythonCAPI.so` (the `mlir` pip/install package) |
| Registers the nkipy dialect (incl. `nkipy.yield`'s `Terminator` trait) | `libNkipyMLIRAggregateCAPI.so` (built from this repo, loaded by `nkigen/_mlir`) |

Both were compiled with `-fvisibility=hidden` and **each bundles its own
hidden copy** of the `IsTerminator` TypeID static (verified: 0 exported copies
in each, vs. the shared `libMLIRIR.so` which exports it). Because both copies
are hidden, they cannot coalesce. So:

- The nkipy library stamps `nkipy.yield`'s `Terminator` trait under **its**
  TypeID.
- The upstream verifier checks `hasTrait<Terminator>()` under a **different**
  TypeID.
- The trait is therefore invisible to the verifier → "block with no
  terminator".

`nkipy-opt` is a single self-consistent binary with one copy of the static, so
the trait is honored and the same IR passes. A colleague's from-source LLVM
likely coalesces the copies (e.g. `-DLLVM_LINK_LLVM_DYLIB=ON`, a single shared
`libMLIR.so`, or non-hidden Python-CAPI visibility); our prebuilt
`/opt/llvm-mlir` ships split component libraries with hidden TypeIDs, so it
does not. **Same source, different LLVM build → different outcome.**

### Why only gather

`nkipy.gather` is currently the only nkipy op that carries a region terminated
by `nkipy.yield` and is constructed during tracing (`builder.py` `take()`).
Every other yield in traced IR is an upstream `linalg.yield` / `scf.yield`,
whose traits live in the shared MLIR and work fine. Any future region-bearing
nkipy op will hit the same wall.

## Workaround (adopted)

Keep nkipy ops out of the in-process verifier; let `nkipy-opt` lower them
first.

- `IRBuilder.run_canonicalize()` (in `builder.py`) canonicalizes via the
  `nkipy-opt` subprocess and returns **text**. `trace.to_mlir()` returns that
  text. (Every caller already stringifies the result; nothing used the
  `Module` object's methods.)
- `extract_and_clean_func_from_module()` (in `llvm.py`) runs
  `inline-nkipy-reference` + `canonicalize` via `nkipy-opt` **before** its first
  `Module.parse`, so the in-process parser only ever sees pure linalg/tensor
  IR.

Both go through `nkigen.transforms.nkipy_opt.run_nkipy_opt_passes`, the same
subprocess wrapper the rest of the pipeline uses.

## Proper fix (not adopted; for future reference)

Make `nkigen._mlir` a self-contained MLIR Python runtime — re-add
`MLIRPythonSources` and `MLIRPythonExtension.RegisterEverything` to
`_source_components` in `mlir/include/nkipy/Bindings/CMakeLists.txt` and
repoint the ~13 `from mlir import ...` sites to `from nkigen._mlir import ...`
so registration and verification happen in one universe.

This was deliberately removed previously (see the comment in that
CMakeLists.txt) to avoid nanobind / dialect-namespace conflicts with the
system `mlir` and `nki.compiler._internal` packages, which also load MLIR.
Mixing the contexts aborts with
`Trying to register different dialects for the same namespace: builtin`.
Reintroducing the bundle is the cleaner long-term answer but needs that
conflict resolved first, so it is out of scope for the gather fix.

## Testing note

`Mode.HW` e2e tests share a single Neuron device. Running them under
`pytest -n auto` causes flaky `NRT:nrt_load` failures from device contention
(not a logic error). Run `tests/e2e` serially; `tests/unit` and `tests/passes`
are safe in parallel.
