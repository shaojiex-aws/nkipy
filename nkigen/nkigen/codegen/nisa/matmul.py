"""linalg.matmul_transpose_a -> nisa.matmul."""

from __future__ import annotations

from ._vendor import nk_ir, nisa

from .access import _emit_const_index, _get_base_and_offsets
from .affine_map import _build_nisa_map, _operand_kwargs
from .patterns import (
    _RewriteContext,
    _is_psum,
    _is_sbuf,
    _static_shape,
    pattern,
)

@pattern("linalg.matmul_transpose_a")
def _rewrite_matmul_transpose_a(rctx: _RewriteContext, op: nk_ir.OpView) -> None:
    operands = list(op.operation.operands)
    if len(operands) != 3:
        return
    mat_a, mat_b, mat_c = operands
    a_ty, b_ty, c_ty = mat_a.type, mat_b.type, mat_c.type

    a_shape = _static_shape(a_ty)
    b_shape = _static_shape(b_ty)
    c_shape = _static_shape(c_ty)
    if a_shape is None or b_shape is None or c_shape is None:
        return
    if len(a_shape) != 2 or len(b_shape) != 2 or len(c_shape) != 2:
        return
    K, M = a_shape
    if b_shape[0] != K or c_shape[0] != M or c_shape[1] != b_shape[1]:
        return

    if not (_is_sbuf(a_ty) and _is_sbuf(b_ty) and _is_psum(c_ty)):
        return

    N = b_shape[1]

    with nk_ir.InsertionPoint(op), rctx.loc:
        a_acc = _get_base_and_offsets(rctx.ctx, mat_a, rctx.loc)
        b_acc = _get_base_and_offsets(rctx.ctx, mat_b, rctx.loc)
        c_acc = _get_base_and_offsets(rctx.ctx, mat_c, rctx.loc)
        a_map = _build_nisa_map(rctx.ctx, 2, a_acc)
        b_map = _build_nisa_map(rctx.ctx, 2, b_acc)
        c_map = _build_nisa_map(rctx.ctx, 2, c_acc)

        row_pos = _emit_const_index(rctx.ctx, 0, rctx.loc)
        col_pos = _emit_const_index(rctx.ctx, 0, rctx.loc)
        nisa.matmul(
            **_operand_kwargs("dst", c_acc, c_map, [M, N]),
            **_operand_kwargs("stationary", a_acc, a_map, [K, M]),
            **_operand_kwargs("moving", b_acc, b_map, [K, N]),
            row_pos=row_pos,
            col_pos=col_pos,
            psum_accumulate_flags=None,
            is_transpose=False,
            perf_opt=nisa.PerfOptMode.None_,
            psum_zero_region=nisa.MatmulZeroRegion.Size2048,
            engine=nisa.Engine.Tensor,
        )
    op.operation.erase()


# ---------------------------------------------------------------------------
# linalg.reciprocal
# ---------------------------------------------------------------------------
