"""Emit compute operations: arithmetic, activation, matmul, reduction, fill.

The post-Phase-4 linalg ops are in *memref* (destination-passing) form: the
result tile is the trailing ``outs`` operand and the op has no SSA result.

All op classification (which linalg op -> which NISA builder + enum member)
lives in :mod:`ops`; this module just reads operands and calls the matching
``gen.api`` render method. Each handler emits via ``gen.em`` and returns
``True`` when it produced a statement.
"""

from __future__ import annotations

from . import irutils
from . import ops
from .emit_indexing import index_expr, memref_expr


def _emit_binary(gen, op) -> bool:
    """``linalg.add/sub/... ins(lhs, rhs) outs(dst)`` -> tensor_tensor_arith."""
    lhs, rhs, dst = op.operands[0], op.operands[1], op.operands[2]
    info = ops.LINALG_OPS[op.operation.name]
    gen.em.line(gen.api.tensor_tensor_arith(
        memref_expr(gen, dst), memref_expr(gen, lhs), memref_expr(gen, rhs),
        info.member,
    ))
    return True


def _emit_unary(gen, op) -> bool:
    """``linalg.exp/... ins(src) outs(dst)`` -> nisa.activation(dst, src, fn)."""
    src, dst = op.operands[0], op.operands[1]
    info = ops.LINALG_OPS[op.operation.name]
    gen.em.line(gen.api.activation(
        memref_expr(gen, dst), memref_expr(gen, src), info.member,
    ))
    return True


def _emit_matmul(gen, op) -> bool:
    """``linalg.matmul_transpose_a ins(A, B) outs(C)`` -> nisa.matmul.

    Computes ``C = Aᵀ · B`` — A is the stationary operand (already transposed
    into the NISA contraction layout), B the moving operand.

    When the matmul sits in a K-reduction loop (its PSUM accumulator is
    allocated *outside* the enclosing loop), each iteration must accumulate:
    the first (k==0) zeroes PSUM (accum=False), the rest add (accum=True). We
    emit ``accum=(<k> != 0)`` keyed on the enclosing loop IV. A standalone
    (single-block) matmul has its accumulator in the same scope -> accum=False.
    """
    a, b, c = op.operands[0], op.operands[1], op.operands[2]
    gen.em.line(gen.api.matmul(
        memref_expr(gen, c), memref_expr(gen, a), memref_expr(gen, b),
        accum=_accum_expr(gen, op, c),
    ))
    return True


def _accum_expr(gen, op, psum_dst):
    """The ``accum=`` value for a matmul: ``"<iv> != 0"`` if it accumulates
    across its enclosing K-loop, else ``False``.

    Accumulation is detected structurally: the matmul accumulates iff its PSUM
    destination's backing alloc lives *outside* the matmul's enclosing
    ``scf.for`` (so the same PSUM tile persists across loop iterations).
    """
    parent = op.operation.parent
    if parent is None or parent.name != "scf.for":
        return False
    alloc = irutils.backing_alloc(psum_dst)
    if alloc is None:
        return False
    alloc_parent = getattr(alloc.owner.operation, "parent", None)
    # Accumulator allocated in the loop body -> fresh each iteration -> no
    # accumulation. Allocated outside -> persists across iterations.
    if alloc_parent == parent:
        return False
    iv_value = parent.opview.regions[0].blocks[0].arguments[0]
    iv_name = gen.names.get(iv_value)
    if iv_name is None:
        return False
    return f"{iv_name} != 0"


def _emit_transpose(gen, op) -> bool:
    """``linalg.transpose ins(src) outs(dst) permutation=[...]`` -> dma_transpose."""
    src, dst = op.operands[0], op.operands[1]
    permutation = [int(x) for x in op.operation.attributes["permutation"]]
    gen.em.line(gen.api.dma_transpose(
        memref_expr(gen, dst), memref_expr(gen, src), permutation,
    ))
    return True


def _emit_fill(gen, op) -> bool:
    """``linalg.fill ins(scalar) outs(dst)`` -> ``nisa.memset(dst, value)``.

    Skips fills targeting HBM: memset is an on-chip (SBUF/PSUM) op, and an
    HBM fill is redundant here anyway — the buffer is staged into SBUF and
    overwritten before use. This mirrors the NISA backend, which only lowers
    SBUF/PSUM fills.
    """
    scalar, dst = op.operands[0], op.operands[1]
    if not irutils.is_on_chip(dst.type):
        return False
    val = irutils.const_scalar(scalar)
    value_expr = repr(val) if val is not None else index_expr(gen, scalar)
    gen.em.line(gen.api.memset(memref_expr(gen, dst), value_expr))
    return True


# ---------------------------------------------------------------------------
# linalg.generic dispatch
# ---------------------------------------------------------------------------


def _num_reduction_dims(op) -> int:
    return sum("reduction" in str(t) for t in op.operation.attributes["iterator_types"])


def _body_ops(op) -> list:
    """Non-yield ops in a linalg.generic body."""
    block = op.regions[0].blocks[0]
    return [o for o in block.operations if o.name != "linalg.yield"]


def _emit_generic(gen, op) -> bool:
    """Dispatch a ``linalg.generic`` by iterator types and body shape."""
    return (_emit_reduction if _num_reduction_dims(op) else _emit_elementwise)(gen, op)


def _emit_reduction(gen, op) -> bool:
    """A reduction generic -> nisa.tensor_reduce_arith, op from the body."""
    src, dst = op.operands[0], op.operands[1]
    info = next((ops.ARITH_BODY_OPS[o.name] for o in _body_ops(op)
                 if o.name in ops.ARITH_BODY_OPS), None)
    if info is None:
        gen.em.comment(f"TODO reduction body={[o.name for o in _body_ops(op)]}")
        return False
    gen.em.line(gen.api.tensor_reduce_arith(
        memref_expr(gen, dst), memref_expr(gen, src), info.member,
        num_r_dim=_num_reduction_dims(op),
    ))
    return True


def _emit_elementwise(gen, op) -> bool:
    """A parallel (elementwise) generic with a single binop body.

    - ``out = in op cst`` (1 tensor input + scalar const) -> tensor_scalar_arith
    - ``out = in0 op in1`` (2 tensor inputs, maybe broadcasting) -> tensor_tensor_arith
    """
    body = _body_ops(op)
    if len(body) != 1 or body[0].name not in ops.ARITH_BODY_OPS:
        names = [o.name for o in body]
        gen.em.comment(f"TODO linalg.generic (parallel) body={names}")
        return False
    info = ops.ARITH_BODY_OPS[body[0].name]

    # operandSegmentSizes = [num_ins, num_outs]; outs is the trailing operand.
    num_ins = int(op.operation.attributes["operandSegmentSizes"][0])
    ins = list(op.operands[:num_ins])
    dst = op.operands[num_ins]

    if num_ins == 1:
        scalar = next((irutils.const_scalar(o) for o in body[0].operands
                       if irutils.const_scalar(o) is not None), None)
        if scalar is None:
            gen.em.comment(f"TODO unary generic without scalar: {body[0].name}")
            return False
        gen.em.line(gen.api.tensor_scalar_arith(
            memref_expr(gen, dst), memref_expr(gen, ins[0]), repr(scalar),
            info.member,
        ))
        return True

    if num_ins == 2:
        gen.em.line(gen.api.tensor_tensor_arith(
            memref_expr(gen, dst), memref_expr(gen, ins[0]), memref_expr(gen, ins[1]),
            info.member,
        ))
        return True

    gen.em.comment(f"TODO linalg.generic ins={num_ins} body={body[0].name}")
    return False


# Map each linalg op kind to its handler. Named ops dispatch by their entry in
# ops.LINALG_OPS; linalg.generic is classified at emit time from its body.
_KIND_HANDLERS = {
    ops.ARITH: _emit_binary,
    ops.ACTIVATION: _emit_unary,
    ops.MATMUL: _emit_matmul,
    ops.TRANSPOSE: _emit_transpose,
    ops.FILL: _emit_fill,
}


def register(dispatch: dict) -> None:
    for name, info in ops.LINALG_OPS.items():
        dispatch[name] = _KIND_HANDLERS[info.kind]
    dispatch["linalg.generic"] = _emit_generic
