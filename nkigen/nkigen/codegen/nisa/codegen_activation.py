"""linalg.reciprocal and unary activation linalg ops -> nisa.activation /
nisa.reciprocal."""

from __future__ import annotations

from nki.compiler._internal import ir as nk_ir  # type: ignore[import-not-found]
from nki.compiler._internal.dialects import nisa  # type: ignore[import-not-found]

from .access import _get_base_and_offsets
from .affine_map import (
    _build_nisa_map,
    _empty_operand_kwargs,
    _operand_kwargs,
    _scalar_operand_kwargs,
)
from .patterns import (
    _RewriteContext,
    _enclosing_block,
    _is_psum,
    _is_sbuf,
    _static_shape,
    pattern,
)

@pattern("linalg.reciprocal")
def _rewrite_reciprocal(rctx: _RewriteContext, op: nk_ir.OpView) -> None:
    operands = list(op.operation.operands)
    if len(operands) < 2:
        return
    src, dst = operands[0], operands[1]
    if not (
        (_is_sbuf(src.type) or _is_psum(src.type))
        and (_is_sbuf(dst.type) or _is_psum(dst.type))
    ):
        return
    shape = _static_shape(dst.type)
    if shape is None:
        return
    with nk_ir.InsertionPoint(op), rctx.loc:
        src_acc = _get_base_and_offsets(rctx.ctx, src, rctx.loc)
        dst_acc = _get_base_and_offsets(rctx.ctx, dst, rctx.loc)
        rank = len(shape)
        src_map = _build_nisa_map(rctx.ctx, rank, src_acc)
        dst_map = _build_nisa_map(rctx.ctx, rank, dst_acc)
        nisa.reciprocal(
            **_operand_kwargs("dst", dst_acc, dst_map, shape),
            **_operand_kwargs("src", src_acc, src_map, shape),
            engine=nisa.Engine.Vector,
        )
    op.operation.erase()


# ---------------------------------------------------------------------------
# linalg.{exp, square, sqrt, abs, log, tanh} -> nisa.activation
# ---------------------------------------------------------------------------


_LINALG_TO_ACTIVATION = {
    "linalg.exp": nisa.ActivationFunction.exp,
    "linalg.square": nisa.ActivationFunction.square,
    "linalg.sqrt": nisa.ActivationFunction.sqrt,
    "linalg.abs": nisa.ActivationFunction.abs,
    "linalg.log": nisa.ActivationFunction.log,
    "linalg.tanh": nisa.ActivationFunction.tanh,
}


def _emit_activation(
    rctx: _RewriteContext,
    op: nk_ir.OpView,
    src: nk_ir.Value,
    dst: nk_ir.Value,
    act_kind,
) -> bool:
    if not (_is_sbuf(src.type) and _is_sbuf(dst.type)):
        return False
    shape = _static_shape(dst.type)
    if shape is None:
        return False

    block = _enclosing_block(op)
    bias = rctx.f32_const(block, 0.0)
    scale = rctx.f32_const(block, 1.0)

    with nk_ir.InsertionPoint(op), rctx.loc:
        src_acc = _get_base_and_offsets(rctx.ctx, src, rctx.loc)
        dst_acc = _get_base_and_offsets(rctx.ctx, dst, rctx.loc)
        rank = len(shape)
        src_map = _build_nisa_map(rctx.ctx, rank, src_acc)
        dst_map = _build_nisa_map(rctx.ctx, rank, dst_acc)
        nisa.activation(
            **_operand_kwargs("dst", dst_acc, dst_map, shape),
            **_empty_operand_kwargs("reduce_res"),
            **_operand_kwargs("src", src_acc, src_map, shape),
            **_scalar_operand_kwargs("bias", bias),
            **_scalar_operand_kwargs("scale", scale),
            **_empty_operand_kwargs("alpha"),
            op=act_kind,
            engine=nisa.Engine.Scalar,
        )
    return True


@pattern(*_LINALG_TO_ACTIVATION.keys())
def _rewrite_linalg_activation(rctx: _RewriteContext, op: nk_ir.OpView) -> None:
    act_kind = _LINALG_TO_ACTIVATION[op.operation.name]
    operands = list(op.operation.operands)
    if len(operands) < 2:
        return
    if _emit_activation(rctx, op, operands[0], operands[1], act_kind):
        op.operation.erase()


# ---------------------------------------------------------------------------
# linalg.fill -> nisa.memset
# ---------------------------------------------------------------------------
