"""
Python wrapper for nkipy-opt passes that can't run from Python.

These passes use NISA dialect which has global constructors that conflict with
Python bindings. They must be run via the nkipy-opt C++ tool instead.
"""

import os
import subprocess
import tempfile
from pathlib import Path


def _resolve_pass_index(passes: list[str], spec: str) -> int:
    """Return the index of the pass named by ``spec`` in the flat pass list.

    ``spec`` accepts a bare pass name or a ``py:`` prefix (matched the same as
    the bare name). Raises ValueError if not found.
    """
    req_name = spec[len('py:'):] if spec.startswith('py:') else spec

    for i, p in enumerate(passes):
        raw = p[len('py:'):] if p.startswith('py:') else p
        base_name = raw.split('=')[0].split('"')[0].strip()
        if base_name == req_name:
            return i

    available = [
        (p[len('py:'):] if p.startswith('py:') else p)
        .split('=')[0].split('"')[0].strip()
        for p in passes
    ]
    raise ValueError(
        f"Pass '{spec}' not found in pipeline. Available passes: {available}"
    )


def _pass_to_arg(pass_name: str) -> str:
    """Convert a pass spec to a CLI argument.

    Examples:
        'canonicalize-compute' -> '--canonicalize-compute'
        'insert-spill-reload="target=trn2"' -> '--insert-spill-reload=target=trn2'
    """
    if '=' in pass_name:
        name, opts = pass_name.split('=', 1)
        opts = opts.strip('"').strip("'")
        return f'--{name}={opts}'
    return f'--{pass_name}'


def get_nkipy_opt_path():
    """Get the path to the nkipy-opt executable."""
    # Assumes we're in the NKIPyKernelGen package
    package_dir = Path(__file__).parent.parent.parent
    nkipy_opt = package_dir / "build" / "bin" / "nkipy-opt"

    if not nkipy_opt.exists():
        raise FileNotFoundError(
            f"nkipy-opt not found at {nkipy_opt}. "
            "Please build the project first."
        )

    return str(nkipy_opt)


def run_nkipy_opt_passes(
    mlir_module,
    passes: list[str],
    print_ir_after_all: bool = False,
    print_stderr: bool = False,
    print_debuginfo: bool = False,
    print_generic: bool = False,
) -> str:
    """
    Run nkipy-opt passes on an MLIR module.

    Args:
        mlir_module: MLIR module (string or Module object)
        passes: List of pass names (e.g., ['cleanup-bufferization-artifacts'])
        print_ir_after_all: If True, print IR after each pass (adds --mlir-print-ir-after-all)
        print_stderr: If True, print stderr output (useful for debugging pass diagnostics)
        print_debuginfo: If True, include source locations in output (adds --mlir-print-debuginfo)

    Returns:
        Transformed MLIR module text (when print_ir_after_all=False)
        Or IR dumps from all passes followed by final module (when print_ir_after_all=True)

    Raises:
        RuntimeError: If nkipy-opt fails
    """
    nkipy_opt = get_nkipy_opt_path()

    # Convert Module object to string if needed
    mlir_text = str(mlir_module) if not isinstance(mlir_module, str) else mlir_module

    # Create temporary files for input and output
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f_in:
        f_in.write(mlir_text)
        input_file = f_in.name

    try:
        # Build command as a list to avoid shell injection
        cmd = [nkipy_opt]
        if print_ir_after_all:
            cmd.append('--mlir-print-ir-after-all')
        if print_debuginfo:
            cmd.append('--mlir-print-debuginfo')
        if print_generic:
            cmd.append('--mlir-print-op-generic')
        cmd.extend(_pass_to_arg(p) for p in passes)
        cmd.append(input_file)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            cmd_str = ' '.join(cmd)
            error_msg = f"nkipy-opt failed with return code {result.returncode}\n"
            error_msg += f"Command: {cmd_str}\n"
            error_msg += f"Stdout:\n{result.stdout}\n"
            error_msg += f"Stderr:\n{result.stderr}"
            raise RuntimeError(error_msg)

        # Print stderr output if requested or when print_ir_after_all is enabled.
        # MLIR outputs IR dumps and pass diagnostics to stderr.
        # Always return just the clean module (stdout) so output can be parsed.
        if result.stderr and (print_stderr or print_ir_after_all):
            if print_ir_after_all:
                print("// === IR After Each Pass ===")
            else:
                print("// === stderr output ===")
            print(result.stderr)
            if print_ir_after_all:
                print("// === End IR Dumps ===\n")
            else:
                print("// === end stderr ===\n")

        return result.stdout

    finally:
        # Clean up temporary file
        if os.path.exists(input_file):
            os.unlink(input_file)


def apply_complete_knob_pipeline(
    mlir_module: str,
    target: str = "trn2",
    print_ir_after_all: bool = False,
    dump_dir: str = None,
    stop_after=None,
    stop_before=None,
    print_debuginfo: bool = False,
    print_generic: bool = False,
) -> str:
    """
    Apply the complete knob-driven compilation pipeline in a single pass.

    This avoids switching between Python bindings and nkipy-opt, running all
    passes through nkipy-opt in sequence:

    The pipeline is memref-native end to end (tracing emits linalg-on-memref),
    so there is no tensor->memref bufferization step and no post-bufferize copy
    cleanup. The phase structure follows
    docs/2026-06-22-memref-native-pipeline-refactor.md.

    Phase 1: Canonicalization
     1. canonicalize-compute: div→recip*mul, decompose batch_matmul, remove fill(0)
     2. infer-layout: Infer tiling, placement, and partition_dim for unannotated ops
     3. canonicalize-partition-dim: Insert transposes to ensure partition_dim=0
     4. assign-linalg-op-ids: Assign unique IDs to linalg ops

    Phase 2: Loop Tiling
     5. knob-driven-tiling: Tile + promote + apply transforms + strip

    Phase 3: Fusion
     6. knob-driven-fusion: Fuse sibling loops + canonicalize-loop-step + canonicalize

    Phase 4: Layout Legalization
     9. annotate-memory-space: Apply HBM / SBUF / PSUM memory space attributes
    10. canonicalize-reshape: Classify expand/collapse_shape + canonicalize
    11. legalize-layout: Attach #sbuf_map, tile HBM↔SBUF copies + canonicalize

    Phase 5: Scheduling
    12. simplify-linalg: Decompose high-rank transposes, canonicalize trivial-broadcast generics
    13. insert-spill-reload: Insert spill/reload for SBUF overflow
    14. insert-memref-dealloc: Insert deallocs + canonicalize

    Phase 6: Codegen
    15. py:linalg-to-nisa: Lower to NISA instructions (Python backend)

    Note: nkipy.annotate ops are removed in annotate-memory-space (pass 9).
    Note: The prior NISA-lowering steps (linalg-to-nisa, resolve-custom-ops,
    prepare-for-nki) are currently stripped. They will be reimplemented in
    Python using the public nki wheel as part of open-sourcing.

    Args:
        mlir_module: MLIR module text with tensor operations and knob annotations
        target: Hardware target (default "trn2")
        print_ir_after_all: If True, print IR after each pass
        dump_dir: If provided, save intermediate MLIR files after each pass to this directory
        stop_after: Controls how many passes to run. Can be:
            - None: run all passes (default)
            - str: stop after the named pass.
        stop_before: Stop just *before* the named pass (str), i.e. run every
            pass up to but excluding it. Mutually exclusive with stop_after. Useful to
            obtain the IR a downstream consumer expects (e.g. the linalg-level
            IR just before "linalg-to-nisa" that the kernelbuilder backend
            walks).
        print_debuginfo: If True, include source locations in output (--mlir-print-debuginfo)
        print_generic: If True, print ops in generic form (--mlir-print-op-generic)

    Returns:
        Fully transformed MLIR module with NISA operations
    """
    passes = [
        # Phase 1: Canonicalization
        # Rewrite linalg ops for NISA: div→recip*mul, decompose batch_matmul,
        # remove fill(0) before matmul.
        'canonicalize-compute',                                                  # 1
        # InferLayout infers tiling, placement (mem_space), and partition_dim for
        # elementwise ops that lack explicit annotations, by propagating from
        # annotated neighbors
        f'infer-layout="target={target}"',                                      # 3
        # CanonicalizePartitionDim inserts transposes to ensure partition_dim=0
        # everywhere. Must run after infer-layout (so partition_dim is propagated)
        # and before assign-linalg-op-ids (so new transposes get op IDs)
        f'canonicalize-partition-dim="target={target}"',                        # 4
        # AssignLinalgOpIds assigns unique nkipy.op_id to each linalg op
        # (including transposes inserted above)
        'assign-linalg-op-ids',                                                 # 5

        # Phase 2: Loop Tiling
        # KnobDrivenTiling generates Transform dialect IR, applies it, and
        # strips the transform module. Includes SBUF/PSUM promotion.
        'knob-driven-tiling',                                                   # 5

        # Phase 3: Fusion
        # KnobDrivenFusion fuses sibling scf.for loops sharing a knob.fuse()
        # annotation (after tiling has produced the per-op loops).
        # Internally runs canonicalize-loop-step + canonicalize as epilogue.
        'knob-driven-fusion',                                                   # 8

        # Phase 4: Layout Legalization
        'annotate-memory-space',                                                 # 9
        # CanonicalizeReshape: classify expand/collapse_shape by mem_space and
        # partition_dim. Internally canonicalizes dead allocs/subviews.
        'canonicalize-reshape',                                                  # 10
        # LegalizeLayout attaches #nkipy.sbuf_map to multi-block SBUF allocs
        # and tiles HBM↔SBUF copies/transposes into block loops.
        # Internally canonicalizes after tiling.
        f'legalize-layout="target={target}"',                                    # 11

        # Phase 5: Scheduling
        'simplify-linalg',                                                       # 12
        f'insert-spill-reload="target={target}"',                                # 13
        # InsertMemRefDealloc inserts memref.dealloc at allocation scope end.
        # Internally runs CSE + canonicalize as epilogue.
        'insert-memref-dealloc',                                                 # 14

        # Phase 6: Codegen (Python) — reimplementation of the deleted C++
        # linalg-to-nisa / resolve-custom-ops / prepare-for-nki passes using
        # the `nki` wheel's Python bindings. Marked as Python-phase so the
        # driver below dispatches to `linalg_to_nisa_py` instead of nkipy-opt.
        'py:linalg-to-nisa',                                                     # 15
    ]

    # Expand pass groups first so stop_after / slicing operates on the
    # flat pass list that the driver actually runs.

    if stop_after is not None and stop_before is not None:
        raise ValueError("stop_after and stop_before are mutually exclusive")

    # Slice passes if stop_after is provided
    if stop_after is not None:
        idx = _resolve_pass_index(passes, stop_after)
        passes = passes[:idx + 1]

    # Slice passes if stop_before is provided (exclude the named pass).
    if stop_before is not None:
        idx = _resolve_pass_index(passes, stop_before)
        passes = passes[:idx]

    return _run_passes_with_python_dispatch(
        mlir_module,
        passes,
        target=target,
        print_ir_after_all=print_ir_after_all,
        dump_dir=dump_dir,
        print_debuginfo=print_debuginfo,
        print_generic=print_generic,
    )


def _run_passes_with_python_dispatch(
    mlir_module: str,
    passes: list[str],
    target: str,
    print_ir_after_all: bool,
    dump_dir: str | None,
    print_debuginfo: bool,
    print_generic: bool,
) -> str:
    """Run a pass list, batching consecutive nkipy-opt passes and dispatching
    any `py:<name>` entries to their Python implementation.

    Having a single driver keeps `dump_dir` numbering coherent across the
    C++/Python boundary: every pass — whether it runs in nkipy-opt or in
    Python — writes the same `NN_<pass>.mlir` artifact.
    """
    current = mlir_module

    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        with open(os.path.join(dump_dir, "00_input.mlir"), 'w') as f:
            f.write(str(current))

    batch: list[str] = []
    batch_start_idx = 1

    def flush_batch(next_idx: int) -> None:
        nonlocal current, batch
        if not batch:
            return
        if dump_dir:
            # Run each pass separately when dumping so we save per-pass IR.
            for j, p in enumerate(batch):
                current = run_nkipy_opt_passes(
                    current, [p], print_ir_after_all,
                    print_debuginfo=print_debuginfo, print_generic=print_generic,
                )
                simple_name = p.split('=')[0].split('"')[0].strip()
                filename = f"{batch_start_idx + j:02d}_{simple_name}.mlir"
                with open(os.path.join(dump_dir, filename), 'w') as f:
                    f.write(current)
        else:
            current = run_nkipy_opt_passes(
                current, batch, print_ir_after_all,
                print_debuginfo=print_debuginfo, print_generic=print_generic,
            )
        batch = []

    for i, pass_name in enumerate(passes, start=1):
        if pass_name.startswith('py:'):
            flush_batch(i)
            py_name = pass_name[len('py:'):]
            current = _run_python_pass(
                py_name, current, target=target, print_generic=print_generic,
            )
            if dump_dir:
                filename = f"{i:02d}_{py_name}.mlir"
                with open(os.path.join(dump_dir, filename), 'w') as f:
                    f.write(current)
            batch_start_idx = i + 1
        else:
            if not batch:
                batch_start_idx = i
            batch.append(pass_name)

    flush_batch(len(passes) + 1)
    return current


def _run_python_pass(
    name: str, mlir_text: str, target: str, print_generic: bool = False,
) -> str:
    """Dispatch a `py:<name>` pass to its Python implementation."""
    if name == 'linalg-to-nisa':
        # Imported lazily because the NKI wheel and upstream `mlir` are only
        # required for this pass; tests that stop before phase 5 do not need
        # either installed.
        from ..codegen.nisa import linalg_to_nisa
        return linalg_to_nisa(mlir_text, target=target, print_generic=print_generic)
    raise ValueError(f"Unknown Python pass: {name!r}")


# Export the main interface
__all__ = [
    'get_nkipy_opt_path',
    'run_nkipy_opt_passes',
    'apply_complete_knob_pipeline',
]
