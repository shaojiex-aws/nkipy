"""
Python wrapper for nkipy-opt passes that can't run from Python.

These passes use NISA dialect which has global constructors that conflict with
Python bindings. They must be run via the nkipy-opt C++ tool instead.
"""

import os
import subprocess
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Pass groups
# ---------------------------------------------------------------------------
# A pass group is a single name in `passes = [...]` that expands to a list
# of underlying passes when the pipeline runs.  Groups keep the top-level
# pipeline readable (one entry per "phase") without giving up per-pass
# debugging — each member is still callable individually via nkipy-opt.
#
# `stop_after='<group-name>'` stops *after* the last member of the group
# has run.
PASS_GROUPS: dict[str, list[str]] = {
    # linalg-level rewrites that adapt programs to NISA's hardware
    # constraints.  Operate on tensor or memref IR before infer-layout / tiling.
    'canonicalize-linalg-for-nisa': [
        'prepare-arithmetic',
        'prepare-matmul',
    ],
}


def _expand_pass_groups(passes: list[str]) -> list[str]:
    """Expand any pass group entries in-place into their member passes."""
    out: list[str] = []
    for p in passes:
        base = p.split('=')[0].split('"')[0].strip()
        if base in PASS_GROUPS:
            out.extend(PASS_GROUPS[base])
        else:
            out.append(p)
    return out


def _resolve_pass_index(passes: list[str], spec: str) -> int:
    """Return the index of the pass named by ``spec`` in the flat pass list.

    ``spec`` accepts a bare pass name, a ``py:`` prefix (matched the same as
    the bare name), a pass-group name (resolves to its last member), and a
    ``name:N`` suffix selecting the Nth (1-indexed) occurrence. Raises
    ValueError if not found.
    """
    name = spec
    nth = 1
    if ':' in spec:
        head, tail = spec.rsplit(':', 1)
        if tail.isdigit():
            name, nth = head, int(tail)

    req_name = name[len('py:'):] if name.startswith('py:') else name
    if req_name in PASS_GROUPS:
        members = PASS_GROUPS[req_name]
        if not members:
            raise ValueError(f"Pass group '{req_name}' is empty")
        req_name = members[-1]

    occurrence = 0
    for i, p in enumerate(passes):
        raw = p[len('py:'):] if p.startswith('py:') else p
        base_name = raw.split('=')[0].split('"')[0].strip()
        if base_name == req_name:
            occurrence += 1
            if occurrence == nth:
                return i

    available = [
        (p[len('py:'):] if p.startswith('py:') else p)
        .split('=')[0].split('"')[0].strip()
        for p in passes
    ] + list(PASS_GROUPS)
    raise ValueError(
        f"Pass '{spec}' not found in pipeline. Available passes: {available}"
    )


def _pass_to_arg(pass_name: str) -> str:
    """Convert a pass spec to a CLI argument.

    Examples:
        'prepare-arithmetic' -> '--prepare-arithmetic'
        'one-shot-bufferize="opt1 opt2"' -> '--one-shot-bufferize=opt1 opt2'
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
     1. prepare-arithmetic: Convert div to mul+reciprocal (NISA has no divide)
     2. prepare-matmul: Decompose batch_matmul, transpose LHS, remove fill(0)
     3. infer-layout: Infer tiling, placement, and partition_dim for unannotated ops
     4. canonicalize-partition-dim: Insert transposes to ensure partition_dim=0 everywhere
     5. assign-linalg-op-ids: Assign unique IDs to linalg ops (incl. new transposes)

    Phase 2: Loop Tiling
     6. knob-driven-tiling: Rewrite linalg ops to tiled loops using transform dialect
     7. apply-and-strip-transforms: Apply the generated transforms, then erase
        the transform module (so downstream passes — including the Python
        linalg->NISA phase — see no transform-dialect ops).

    Phase 3: Fusion
     8. knob-driven-fusion: Fuse sibling scf.for loops sharing a knob.fuse()
     9. canonicalize-loop-step: Normalize loop steps to 1 (then canonicalize
        cleans up memref operations)

    Phase 4: Layout Legalization
    10. annotate-memory-space: Apply HBM / SBUF / PSUM memory space attributes
    11. canonicalize-reshape: Classify expand/collapse_shape by mem_space and partition_dim
    12. canonicalize: Clean up dead allocs
    13. legalize-layout: Attach #sbuf_map to multi-block SBUF allocs, tile HBM↔SBUF copies
    14. canonicalize: Clean up after layout legalization

    Phase 5: Scheduling
    15. simplify-linalg: Decompose high-rank transposes, canonicalize trivial-broadcast generics
    16. insert-spill-reload: Insert spill/reload for SBUF overflow
    17. insert-memref-dealloc: Insert memref.dealloc at allocation scope end
    18. cse: Common subexpression elimination
    19. canonicalize: DCE for unused subviews and cleanup

    Phase 6: Codegen
    20. py:linalg-to-nisa: Lower to NISA instructions (Python backend)

    Note: nkipy.annotate ops are removed in annotate-memory-space (pass 10).
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
            - int: stop after pass N (1-indexed)
            - str: stop after the first occurrence of the named pass.
              For passes that appear multiple times (e.g. "canonicalize"),
              use "name:N" to stop at the Nth occurrence (1-indexed).
        stop_before: Stop just *before* the named pass (str), i.e. run every
            pass up to but excluding it. Resolved by pass name so it is robust
            to pipeline reordering. Accepts the same "py:" prefix / "name:N"
            forms as stop_after. Mutually exclusive with stop_after. Useful to
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
        # Linalg-level rewrites for NISA hardware constraints (pre-tiling).
        # Expands to prepare-arithmetic + prepare-matmul. See PASS_GROUPS above.
        'canonicalize-linalg-for-nisa',                                         # 1-2
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
        # KnobDrivenTiling generates Transform dialect IR; the fused pass
        # applies it and then erases the transform module so downstream
        # (including the Python linalg->NISA phase) sees no transform-dialect
        # ops in the IR.
        'knob-driven-tiling',                                                   # 6
        'apply-and-strip-transforms',                                           # 7

        # Phase 3: Fusion
        # KnobDrivenFusion fuses sibling scf.for loops sharing a knob.fuse()
        # annotation (after tiling has produced the per-op loops).  Must run
        # before canonicalize-loop-step, which it matches on loop bounds.
        'knob-driven-fusion',                                                   # 8
        # CanonicalizeLoopStep normalizes loop steps to 1 (e.g., for %i = 0 to 512 step 128)
        # This simplifies index expressions from %i*128/128 to just %i
        'canonicalize-loop-step',                                               # 9
        'canonicalize',                                                         # 9b

        # Phase 4: Layout Legalization
        # The IR is already memref-native (no bufferization needed). Allocation
        # and promotion are explicit, so the post-bufferize cleanup passes
        # (eliminate-uninitialized-copies, eliminate-same-memspace-copy) are gone.
        'annotate-memory-space',                                                 # 10
        # CanonicalizeReshape: classify expand/collapse_shape by mem_space and
        # partition_dim. HBM reshapes and SBUF non-pdim reshapes stay as views.
        # SBUF partition dim splits get alloc+copy (NISA has no modulo).
        # Returned expand_shape views of func args and direct returns of func
        # args get alloc+copy (NISA needs separate output allocations).
        'canonicalize-reshape',                                                  # 11
        'canonicalize',  # Clean up dead allocs and subviews                     # 12
        # LegalizeLayout attaches #nkipy.sbuf_map to multi-block SBUF allocs
        # and tiles HBM↔SBUF copies/transposes into block loops
        f'legalize-layout="target={target}"',                                    # 13
        'canonicalize',                                                          # 14

        # Phase 5: Scheduling
        # Simplify linalg ops before NISA lowering: decompose high-rank
        # transposes to loops of 2D, collapse >2D SBUF transpose to 2D,
        # canonicalize trivial-broadcast generics to named ops.
        # Runs before insert-spill-reload so any SBUF temps it creates
        # are accounted for in spill/reload memory budgeting.
        'simplify-linalg',                                                       # 15
        # Insert spill/reload for SBUF memory pressure.  Runs after legalize-layout
        # so SBUF allocs are already in physical per-partition layout and their
        # total byte size equals the per-partition SBUF consumption.
        f'insert-spill-reload="target={target}"',                                # 16
        'insert-memref-dealloc',  # Insert memref.dealloc ops at allocation scope end  # 17
        'cse',  # Common subexpression elimination                               # 18
        'canonicalize',  # DCE for unused subviews and cleanup                   # 19

        # Phase 6: Codegen (Python) — reimplementation of the deleted C++
        # linalg-to-nisa / resolve-custom-ops / prepare-for-nki passes using
        # the `nki` wheel's Python bindings. Marked as Python-phase so the
        # driver below dispatches to `linalg_to_nisa_py` instead of nkipy-opt.
        'py:linalg-to-nisa',                                                     # 20
    ]

    # Expand pass groups first so stop_after / slicing operates on the
    # flat pass list that the driver actually runs.
    passes = _expand_pass_groups(passes)

    if stop_after is not None and stop_before is not None:
        raise ValueError("stop_after and stop_before are mutually exclusive")

    # Slice passes if stop_after is provided
    if stop_after is not None:
        if isinstance(stop_after, int):
            passes = passes[:stop_after]
        elif isinstance(stop_after, str):
            idx = _resolve_pass_index(passes, stop_after)
            passes = passes[:idx + 1]
        else:
            raise TypeError(f"stop_after must be int, str, or None, got {type(stop_after)}")

    # Slice passes if stop_before is provided (exclude the named pass).
    if stop_before is not None:
        if not isinstance(stop_before, str):
            raise TypeError(f"stop_before must be str or None, got {type(stop_before)}")
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
