"""linalg.add/sub/mul/max/min -> nisa.tensor_tensor_arith."""

from __future__ import annotations

from nki.compiler._internal import ir as nk_ir  # type: ignore[import-not-found]
from nki.compiler._internal.dialects import nisa  # type: ignore[import-not-found]

from .access import _get_base_and_offsets
from .affine_map import _build_nisa_map, _operand_kwargs
from .patterns import (
    _LINALG_TO_ARITH_OP,
    _RewriteContext,
    _static_shape,
    pattern,
)

@pattern("linalg.add", "linalg.sub", "linalg.mul", "linalg.max", "linalg.min")
def _rewrite_elementwise(rctx: _RewriteContext, op: nk_ir.OpView) -> None:
    arith_kind = _LINALG_TO_ARITH_OP[op.operation.name]
    operands = list(op.operation.operands)
    if len(operands) < 3:
        return
    lhs, rhs, dst = operands[0], operands[1], operands[2]

    shape = _static_shape(dst.type)
    if shape is None:
        return

    with nk_ir.InsertionPoint(op), rctx.loc:
        dst_acc = _get_base_and_offsets(rctx.ctx, dst, rctx.loc)
        lhs_acc = _get_base_and_offsets(rctx.ctx, lhs, rctx.loc)
        rhs_acc = _get_base_and_offsets(rctx.ctx, rhs, rctx.loc)
        rank = len(shape)
        dst_map = _build_nisa_map(rctx.ctx, rank, dst_acc)
        lhs_map = _build_nisa_map(rctx.ctx, rank, lhs_acc)
        rhs_map = _build_nisa_map(rctx.ctx, rank, rhs_acc)

        kwargs: dict = {}
        kwargs.update(_operand_kwargs("dst", dst_acc, dst_map, shape))
        kwargs.update(_operand_kwargs("lhs", lhs_acc, lhs_map, shape))
        kwargs.update(_operand_kwargs("rhs", rhs_acc, rhs_map, shape))
        nisa.tensor_tensor_arith(
            op=arith_kind, engine=nisa.Engine.Vector, **kwargs
        )
    op.operation.erase()


# ---------------------------------------------------------------------------
# memref.copy + linalg.copy
# ---------------------------------------------------------------------------


def _pad_shape_to_2d(shape: list[int]) -> list[int]:
    if len(shape) < 2:
        return shape + [1] * (2 - len(shape))
    return shape
