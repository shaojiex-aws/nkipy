"""Emit compute operations: arithmetic, activation, matmul, reduction.

The post-Phase-4 linalg ops are in *memref* (destination-passing) form: the
result tile is the trailing ``outs`` operand and the op has no SSA result.

- Binary named ops (``linalg.add/sub/mul/max/min``) -> ``nisa.tensor_tensor_arith``
- Unary named ops (``linalg.exp/sqrt/...``)         -> ``nisa.activation``
- ``linalg.reciprocal``                              -> ``nisa.activation`` (reciprocal)
- ``linalg.matmul_transpose_a``                      -> ``nisa.matmul``
- ``linalg.fill``                                    -> scalar broadcast copy
- ``linalg.generic`` (reduction / broadcast / cast)  -> dispatched below

Each handler emits via ``gen.em`` and returns ``True`` when it produced a
statement.
"""

from __future__ import annotations

from mlir import ir as up_ir  # type: ignore[import-not-found]

from . import irutils
from .emit_indexing import index_expr, memref_expr


# linalg named binary op -> the arith_op key the api layer understands.
_BINARY_OPS = {
    "linalg.add",
    "linalg.sub",
    "linalg.mul",
    "linalg.max",
    "linalg.min",
    "linalg.div",
}

# linalg named unary op -> activation rendering (key recognized by api layer).
_UNARY_ACTIVATIONS = {
    "linalg.exp",
    "linalg.tanh",
    "linalg.log",
    "linalg.sqrt",
    "linalg.abs",
    "linalg.square",
    "linalg.reciprocal",
    "linalg.rsqrt",
    "linalg.sigmoid",
}


def _emit_binary(gen, op) -> bool:
    """``linalg.add/sub/... ins(lhs, rhs) outs(dst)`` -> tensor_tensor_arith."""
    lhs, rhs, dst = op.operands[0], op.operands[1], op.operands[2]
    arith_op = gen.api.ARITH_OPS.get(op.operation.name)
    if arith_op is None:
        gen.em.comment(f"TODO unsupported binary op: {op.operation.name}")
        return False
    dst_e = memref_expr(gen, dst)
    lhs_e = memref_expr(gen, lhs)
    rhs_e = memref_expr(gen, rhs)
    gen.em.line(gen.api.tensor_tensor_arith(dst_e, lhs_e, rhs_e, arith_op))
    return True


def _emit_unary(gen, op) -> bool:
    """``linalg.exp/... ins(src) outs(dst)`` -> nisa.activation(dst, src, func)."""
    src, dst = op.operands[0], op.operands[1]
    func = gen.api.ACTIVATION_FUNCS.get(op.operation.name)
    if func is None:
        gen.em.comment(f"TODO unsupported unary op: {op.operation.name}")
        return False
    gen.em.line(gen.api.activation(memref_expr(gen, dst), memref_expr(gen, src), func))
    return True


def _emit_matmul(gen, op) -> bool:
    """``linalg.matmul_transpose_a ins(A, B) outs(C)`` -> nisa.matmul.

    ``matmul_transpose_a`` computes ``C = Aᵀ · B`` — A is the stationary
    operand (already transposed in the NISA contraction layout) and B the
    moving operand. accum=False (PSUM is zeroed by the matmul).
    """
    a, b, c = op.operands[0], op.operands[1], op.operands[2]
    dst = memref_expr(gen, c)
    stationary = memref_expr(gen, a)
    moving = memref_expr(gen, b)
    gen.em.line(gen.api.matmul(dst, stationary, moving, accum=False))
    return True


def _emit_fill(gen, op) -> bool:
    """``linalg.fill ins(%scalar) outs(%dst)`` -> ``nisa.memset(dst, value)``."""
    scalar, dst = op.operands[0], op.operands[1]
    val = irutils.const_scalar(scalar)
    value_expr = repr(val) if val is not None else index_expr(gen, scalar)
    gen.em.line(gen.api.memset(memref_expr(gen, dst), value_expr))
    return True


# ---------------------------------------------------------------------------
# linalg.generic dispatch
# ---------------------------------------------------------------------------


def _generic_iterator_types(op) -> list[str]:
    types = op.operation.attributes["iterator_types"]
    out = []
    for t in types:
        # Each entry stringifies as e.g. #linalg.iterator_type<reduction>.
        s = str(t)
        out.append("reduction" if "reduction" in s else "parallel")
    return out


def _generic_body_ops(op) -> list[str]:
    """Names of the ops in a linalg.generic body (excluding the yield)."""
    block = op.regions[0].blocks[0]
    return [o.name for o in block.operations if o.name != "linalg.yield"]


# arith body op (in a reduction generic) -> arith_op key.
_REDUCE_BODY_TO_OP = {
    "arith.addf": "linalg.add",
    "arith.addi": "linalg.add",
    "arith.maximumf": "linalg.max",
    "arith.minimumf": "linalg.min",
    "arith.mulf": "linalg.mul",
}


# arith binary body op -> arith_op key, for parallel (elementwise) generics.
_BODY_BINOP_TO_OP = {
    "arith.addf": "linalg.add",
    "arith.subf": "linalg.sub",
    "arith.mulf": "linalg.mul",
    "arith.divf": "linalg.div",
    "arith.maximumf": "linalg.max",
    "arith.minimumf": "linalg.min",
    "arith.addi": "linalg.add",
    "arith.subi": "linalg.sub",
    "arith.muli": "linalg.mul",
}


def _emit_generic(gen, op) -> bool:
    """Dispatch a ``linalg.generic`` by its iterator types and body shape."""
    iters = _generic_iterator_types(op)
    if "reduction" in iters:
        return _emit_reduction(gen, op)
    return _emit_elementwise_generic(gen, op)


def _emit_elementwise_generic(gen, op) -> bool:
    """A parallel (elementwise) ``linalg.generic`` with a single binop body.

    Two shapes are recognized (both common in nkigen output):

    - ``out = in <op> cst`` (one tensor input + a scalar constant) ->
      ``nisa.tensor_scalar_arith`` — scale/bias patterns.
    - ``out = in0 <op> in1`` (two tensor inputs, possibly broadcasting) ->
      ``nisa.tensor_tensor_arith``.
    """
    block = op.regions[0].blocks[0]
    body = [o for o in block.operations if o.name != "linalg.yield"]
    if len(body) != 1:
        gen.em.comment(f"TODO linalg.generic (parallel) body={[o.name for o in body]}")
        return False

    inner = body[0]
    arith_key = _BODY_BINOP_TO_OP.get(inner.name)
    if arith_key is None:
        gen.em.comment(f"TODO linalg.generic (parallel) body=['{inner.name}']")
        return False
    arith_op = gen.api.ARITH_OPS[arith_key]

    # operandSegmentSizes = [num_ins, num_outs]; the out tile is the trailing
    # operand, the ins precede it.
    seg = [int(x) for x in op.operation.attributes["operandSegmentSizes"]]
    num_ins = seg[0]
    ins = list(op.operands[:num_ins])
    dst = op.operands[num_ins]

    scalar = next(
        (irutils.const_scalar(o) for o in inner.operands
         if irutils.const_scalar(o) is not None),
        None,
    )

    if num_ins == 1 and scalar is not None:
        gen.em.line(
            gen.api.tensor_scalar_arith(
                memref_expr(gen, dst), memref_expr(gen, ins[0]),
                repr(scalar), arith_op,
            )
        )
        return True

    if num_ins == 2:
        gen.em.line(
            gen.api.tensor_tensor_arith(
                memref_expr(gen, dst),
                memref_expr(gen, ins[0]), memref_expr(gen, ins[1]),
                arith_op,
            )
        )
        return True

    gen.em.comment(
        f"TODO linalg.generic (parallel) ins={num_ins} body={inner.name}"
    )
    return False


def _emit_transpose(gen, op) -> bool:
    """``linalg.transpose ins(%src) outs(%dst) permutation=[...]`` -> dma_transpose."""
    src, dst = op.operands[0], op.operands[1]
    permutation = [int(x) for x in op.operation.attributes["permutation"]]
    gen.em.line(
        gen.api.dma_transpose(
            memref_expr(gen, dst), memref_expr(gen, src), permutation
        )
    )
    return True


def _emit_reduction(gen, op) -> bool:
    """A reduction ``linalg.generic`` -> nisa.tensor_reduce_arith.

    Reads the single arith op in the body to pick the reduce operator.
    """
    body = _generic_body_ops(op)
    reduce_op = None
    for bname in body:
        if bname in _REDUCE_BODY_TO_OP:
            reduce_op = gen.api.ARITH_OPS[_REDUCE_BODY_TO_OP[bname]]
            break
    src = op.operands[0]
    dst = op.operands[1]
    if reduce_op is None:
        gen.em.comment(f"TODO reduction with body={body}")
        return False
    gen.em.line(
        gen.api.tensor_reduce_arith(
            memref_expr(gen, dst), memref_expr(gen, src), reduce_op
        )
    )
    return True


def register(dispatch: dict) -> None:
    for name in _BINARY_OPS:
        dispatch[name] = _emit_binary
    for name in _UNARY_ACTIVATIONS:
        dispatch[name] = _emit_unary
    dispatch["linalg.matmul_transpose_a"] = _emit_matmul
    dispatch["linalg.fill"] = _emit_fill
    dispatch["linalg.transpose"] = _emit_transpose
    dispatch["linalg.generic"] = _emit_generic