"""KernelBuilder Python codegen backend.

Walks the post-Phase-4 IR (memref + scf + linalg + arith, with nkipy
annotations) and emits equivalent ``kernel_builder`` Python source code as
text — the same IR the NISA backend consumes (after pass 23, before the
``py:linalg-to-nisa`` pass).

Unlike the NISA backend, this one only *reads* the IR, so it uses the upstream
``mlir.ir`` bindings directly (see ``irutils``) and has no dependency on the
NKI wheel's builders.

Public entry point: :func:`linalg_to_kernelbuilder`.
"""

from __future__ import annotations

from mlir import ir as up_ir  # type: ignore[import-not-found]

from .api import get_api
from .emitter import Emitter
from . import irutils


def linalg_to_kernelbuilder(
    mlir_text: str,
    kernel_name: str,
    target: str = "trn2",
    api_version: str = "v1",
) -> str:
    """Translate post-Phase-4 MLIR to kernel_builder Python source (text -> text).

    Args:
        mlir_text: post-Phase-4 MLIR (memref+scf+linalg+arith+func).
        kernel_name: name for the generated Python function. The MLIR func's
            own ``sym_name`` is used if this is falsy.
        target: hardware target tag (recorded in a header comment for now).
        api_version: which :class:`~.api.KernelBuilderAPI` version to render.

    Returns:
        A string of valid Python source defining one function per ``func.func``
        in the module.
    """
    api = get_api(api_version)

    ctx = up_ir.Context()
    ctx.load_all_available_dialects()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = up_ir.Module.parse(mlir_text)
        gen = _ModuleEmitter(module, api, kernel_name, target)
        return gen.run()


# The pass that lowers linalg -> NISA. The IR the kernelbuilder backend walks
# is the pipeline state just before it (memref + scf + linalg + arith).
_NISA_LOWERING_PASS = "linalg-to-nisa"


def compile_to_tiled_ir(traced_func, target: str = "trn2", dump_dir: str | None = None) -> str:
    """Compile a traced function to the tiled linalg-level IR the backend walks.

    Runs the full knob pipeline but stops just *before* the NISA lowering pass,
    yielding tiled/bufferized memref+scf+linalg+arith IR — the same input the
    NISA backend consumes. The stop point is resolved by pass name, so it is
    robust to pipeline reordering.
    """
    from ...driver.pipeline import apply_complete_knob_pipeline

    mlir_text = traced_func.to_mlir()
    return apply_complete_knob_pipeline(
        mlir_text, target=target, stop_before=_NISA_LOWERING_PASS, dump_dir=dump_dir
    )


def trace_to_kernelbuilder(
    traced_func,
    target: str = "trn2",
    api_version: str = "v1",
    dump_dir: str | None = None,
) -> str:
    """Trace -> tiled IR -> kernel_builder Python source (end to end).

    When ``dump_dir`` is set, the generated source is also written to
    ``<dump_dir>/kb_code.py`` alongside the pipeline's MLIR dumps, so the kb
    output sits next to the IR stages that produced it.
    """
    ir = compile_to_tiled_ir(traced_func, target=target, dump_dir=dump_dir)
    name = traced_func.__wrapped__.__name__
    code = linalg_to_kernelbuilder(
        ir, kernel_name=name, target=target, api_version=api_version
    )
    if dump_dir:
        import os
        os.makedirs(dump_dir, exist_ok=True)
        with open(os.path.join(dump_dir, "kb_code.py"), "w") as f:
            f.write(code)
    return code


class _ModuleEmitter:
    """Drives emission for one module: header, imports, one fn per func.func."""

    def __init__(self, module, api, kernel_name: str, target: str) -> None:
        self.module = module
        self.api = api
        self.kernel_name = kernel_name
        self.target = target
        self.em = Emitter()
        # SSA value -> generated Python variable name, rebuilt per function.
        self.names: dict = {}
        # Loop induction variables that are fori_loop Regs at runtime. Slices
        # whose offset depends on one of these must render as nb.ds(...) rather
        # than a Python slice (kb rejects Reg-valued Python slices).
        self.loop_regs: set = set()
        # Written-value -> role-name hint (e.g. exp_out), built per function.
        self._tile_roles: dict = {}

    def run(self) -> str:
        for stmt in self.api.imports():
            self.em.add_import(stmt)

        funcs = irutils.func_ops(self.module)
        for i, func in enumerate(funcs):
            if i:
                self.em.blank()
                self.em.blank()
            self._emit_func(func)
        return self.em.getvalue()

    # -- function ----------------------------------------------------------

    def _emit_func(self, func) -> None:
        self.names = {}
        self.loop_regs = set()
        self._tile_roles = self._build_tile_roles(func)
        name = self.kernel_name or irutils.func_name(func)

        block = func.regions[0].blocks[0]
        # Kernel I/O parameters are annotated ``: nb.Tensor`` — this is how
        # nb.simulate_kernel / build_kernel identify tensor arguments (matched
        # by name against the inputs/outputs dicts); unannotated params would
        # be treated as hyperparameters.
        params = []
        # Inputs: HBM memref block arguments.
        for idx, arg in enumerate(block.arguments):
            pname = self.em.reserve_name(f"input_{idx}")
            self.names[arg] = pname
            params.append(f"{pname}: nb.Tensor")

        # Outputs: the func returns HBM allocs. Surface them as output
        # parameters (matching the kernel_builder convention) and pre-bind the
        # returning SSA values to those names so their defining `memref.alloc`
        # is skipped and uses render as the parameter.
        for idx, retval in enumerate(self._returned_values(block)):
            if retval in self.names:
                # Aliases an input (output == input): already in the signature.
                continue
            oname = self.em.reserve_name(f"output_{idx}")
            self.names[retval] = oname
            params.append(f"{oname}: nb.Tensor")

        self.em.comment(f"Generated by nkigen kernelbuilder backend (target={self.target}).")
        self.em.line(f"def {name}({', '.join(params)}):")
        with self.em.indent():
            self.emit_block(block)

    @staticmethod
    def _returned_values(block) -> list:
        """SSA values returned by the block's ``func.return`` terminator."""
        for op in block.operations:
            if op.operation.name == "func.return":
                return list(op.operation.operands)
        return []

    # -- naming hints ------------------------------------------------------

    def tile_hint(self, op, memspace) -> str:
        """A readable variable-name hint for the tile allocated by ``op``.

        Prefers a role derived from the op that *writes* this tile (its
        producer), e.g. ``exp_out``, ``add_out``, ``mm_psum`` — falling back to
        the memory space (``sbuf``/``psum``/``hbm``) when no producer role is
        known. The per-function writer map is built lazily in :meth:`_emit_func`.
        """
        from .api import MEMSPACE_PSUM, MEMSPACE_SBUF
        role = self._tile_roles.get(op.operation.results[0])
        if role is not None:
            return role
        if memspace == MEMSPACE_SBUF:
            return "sbuf"
        if memspace == MEMSPACE_PSUM:
            return "psum"
        return "hbm"

    def _build_tile_roles(self, func) -> dict:
        """Map each written memref value -> a role name from its writer op.

        The writer is the op whose *destination* operand is that value. We name
        the tile after the op's semantics so the variable reads like what it
        holds (the result of an exp, an add, a matmul, ...).
        """
        roles: dict = {}

        def visit(op_handle):
            role = _writer_role(op_handle.opview)
            if role is not None:
                dst = _dst_operand(op_handle.opview)
                if dst is not None:
                    roles.setdefault(dst, role)
            return up_ir.WalkResult.ADVANCE

        func.operation.walk(visit)
        return roles

    # -- block / op dispatch ----------------------------------------------

    def emit_block(self, block) -> None:
        """Emit every op in a block, falling back to ``pass`` if none produced
        an executable statement. Public so control-flow handlers can recurse
        into nested regions."""
        emitted_stmt = False
        for op in block.operations:
            if self._emit_op(op):
                emitted_stmt = True
        if not emitted_stmt:
            self.em.line("pass")

    def _emit_op(self, op) -> bool:
        """Emit one op. Returns True if it produced an executable statement.

        Phase 1 establishes the dispatch skeleton; later phases fill in the
        handlers (memory, compute, control flow). Unhandled ops are surfaced
        as a TODO comment rather than silently dropped, so gaps are visible in
        the generated source — but a comment is not a statement, so it does not
        by itself satisfy a block.
        """
        name = op.operation.name
        handler = _DISPATCH.get(name)
        if handler is not None:
            return handler(self, op)
        # Terminators / structural ops we intentionally skip without noise.
        if name in _SILENT_SKIP:
            return False
        self.em.comment(f"TODO unhandled op: {name}")
        return False


# Op name -> role stem for the tile it writes (its destination). The tile is
# named ``<role>_out`` (e.g. ``exp_out``), or ``<role>`` directly when the
# suffix would be redundant. linalg.generic is resolved from its body below.
_OP_ROLE = {
    "linalg.add": "add",
    "linalg.sub": "sub",
    "linalg.mul": "mul",
    "linalg.max": "max",
    "linalg.min": "min",
    "linalg.exp": "exp",
    "linalg.tanh": "tanh",
    "linalg.log": "log",
    "linalg.sqrt": "sqrt",
    "linalg.abs": "abs",
    "linalg.square": "square",
    "linalg.reciprocal": "recip",
    "linalg.rsqrt": "rsqrt",
    "linalg.sigmoid": "sigmoid",
    "linalg.matmul_transpose_a": "matmul",
    "linalg.transpose": "transpose",
    "linalg.fill": "fill",
}


# Single-op linalg.generic bodies -> role stem (matches emit_compute's maps).
_GENERIC_BODY_ROLE = {
    "arith.addf": "add", "arith.addi": "add",
    "arith.subf": "sub", "arith.subi": "sub",
    "arith.mulf": "mul", "arith.muli": "mul",
    "arith.divf": "div",
    "arith.maximumf": "max", "arith.minimumf": "min",
}


def _writer_role(op) -> str | None:
    """Role-name stem for the tile written by ``op`` (None if not a writer)."""
    name = op.operation.name
    role = _OP_ROLE.get(name)
    if role is not None:
        return f"{role}_out"
    if name == "linalg.generic":
        # Classify by the single arith op in the body, if any (reduction vs
        # elementwise both read like ``<op>_out`` for naming purposes).
        try:
            body = [o for o in op.regions[0].blocks[0].operations
                    if o.name != "linalg.yield"]
        except (IndexError, AttributeError):
            body = []
        if len(body) == 1:
            stem = _GENERIC_BODY_ROLE.get(body[0].name)
            if stem is not None:
                return f"{stem}_out"
        return "tile"
    return None


def _dst_operand(op):
    """The destination (written) operand of a destination-passing op.

    For the linalg/memref ops we name, the output buffer is the trailing
    operand (``outs`` for linalg named/generic ops; the target for memref.copy
    is operand 1). Returns the SSA value, or None if it can't be determined.
    """
    operands = list(op.operation.operands)
    if not operands:
        return None
    name = op.operation.name
    if name == "memref.copy":
        return operands[1] if len(operands) > 1 else None
    # linalg destination-passing ops: outs is the last operand.
    return operands[-1]


def _build_dispatch() -> dict:
    """Assemble the op-name -> handler table from the emit_* modules."""
    from . import emit_memory
    from . import emit_compute
    from . import emit_control_flow

    dispatch: dict = {}
    emit_memory.register(dispatch)
    emit_compute.register(dispatch)
    emit_control_flow.register(dispatch)
    return dispatch


# Op handlers contributed by the emit_* modules (Phases 2-4). Built once at
# import time. Each handler has signature ``(gen, op) -> bool``.
_DISPATCH: dict = _build_dispatch()

# Structural ops that produce no standalone statement in the generated code.
# arith.* index math and memref view ops (subview/collapse/expand/cast) are
# materialized lazily by emit_indexing at the point they're referenced, so the
# defining ops themselves emit nothing.
_SILENT_SKIP = {
    "func.return",
    "scf.yield",
    "linalg.yield",
    "arith.constant",
    "arith.muli",
    "arith.addi",
    "arith.subi",
    "arith.divui",
    "arith.divsi",
    "arith.remui",
    "arith.remsi",
    "memref.subview",
    "memref.collapse_shape",
    "memref.expand_shape",
    "memref.reinterpret_cast",
}


__all__ = [
    "linalg_to_kernelbuilder",
    "compile_to_tiled_ir",
    "trace_to_kernelbuilder",
]
