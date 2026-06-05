"""linalg.fill -> nisa.memset."""

from __future__ import annotations

from ._vendor import nk_ir, nisa

from .access import _get_base_and_offsets
from .affine_map import _build_nisa_map, _operand_kwargs
from .patterns import (
    _RewriteContext,
    _is_psum,
    _is_sbuf,
    _static_shape,
    pattern,
)

@pattern("linalg.fill")
def _rewrite_linalg_fill(rctx: _RewriteContext, op: nk_ir.OpView) -> None:
    operands = list(op.operation.operands)
    if len(operands) < 2:
        return
    scalar, dst = operands[0], operands[1]
    dst_ty = dst.type
    if not (_is_sbuf(dst_ty) or _is_psum(dst_ty)):
        return
    shape = _static_shape(dst_ty)
    if shape is None:
        return
    with nk_ir.InsertionPoint(op), rctx.loc:
        dst_acc = _get_base_and_offsets(rctx.ctx, dst, rctx.loc)
        rank = len(shape)
        dst_map = _build_nisa_map(rctx.ctx, rank, dst_acc)
        nisa.memset(
            **_operand_kwargs("dst", dst_acc, dst_map, shape),
            value=scalar,
            engine=nisa.Engine.Vector,
        )
    op.operation.erase()


# ---------------------------------------------------------------------------
# linalg.generic (scalar / broadcast / same-shape / type cast / reduction /
# unary math / powf)
# ---------------------------------------------------------------------------
