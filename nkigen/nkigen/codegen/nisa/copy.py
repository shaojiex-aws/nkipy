"""memref.copy and linalg.copy -> nisa.dma_copy / SBUF copy."""

from __future__ import annotations

from ._vendor import nk_ir, nisa

from .access import _get_base_and_offsets
from .affine_map import _build_nisa_map, _operand_kwargs
from .patterns import (
    _RewriteContext,
    _is_hbm,
    _is_psum,
    _is_sbuf,
    _pad_shape_to_2d,
    _static_shape,
    pattern,
)

@pattern("memref.copy")
def _rewrite_memref_copy(rctx: _RewriteContext, op: nk_ir.OpView) -> None:
    src = op.operation.operands[0]
    dst = op.operation.operands[1]
    src_ty = src.type
    dst_ty = dst.type

    src_hbm, dst_hbm = _is_hbm(src_ty), _is_hbm(dst_ty)
    src_sbuf, dst_sbuf = _is_sbuf(src_ty), _is_sbuf(dst_ty)
    src_psum, dst_psum = _is_psum(src_ty), _is_psum(dst_ty)

    needs_dma = src_hbm or dst_hbm
    on_tpb = (
        (src_sbuf and dst_sbuf) or (src_sbuf and dst_psum) or (src_psum and dst_sbuf)
    )
    if not (needs_dma or on_tpb):
        return

    shape = _static_shape(dst_ty)
    if shape is None:
        return
    shape = _pad_shape_to_2d(shape)

    with nk_ir.InsertionPoint(op), rctx.loc:
        src_acc = _get_base_and_offsets(rctx.ctx, src, rctx.loc)
        dst_acc = _get_base_and_offsets(rctx.ctx, dst, rctx.loc)

        num_iter = len(shape)
        src_map = _build_nisa_map(rctx.ctx, num_iter, src_acc)
        dst_map = _build_nisa_map(rctx.ctx, num_iter, dst_acc)

        needs_psum_hop = needs_dma and (
            (src_hbm and dst_psum) or (src_psum and dst_hbm)
        )

        if needs_psum_hop:
            sbuf_attr = nk_ir.Attribute.parse("#nisa.mem<sbuf>")
            inter_ty = nk_ir.MemRefType.get(
                shape, dst_ty.element_type, memory_space=sbuf_attr  # type: ignore[attr-defined]
            )
            inter_val = nisa.alloc(memref_type=inter_ty, alignment=64)
            inter_acc = _get_base_and_offsets(rctx.ctx, inter_val, rctx.loc)
            inter_map = _build_nisa_map(rctx.ctx, num_iter, inter_acc)
            if src_hbm and dst_psum:
                nisa.dma_copy(
                    **_operand_kwargs("dst", inter_acc, inter_map, shape),
                    **_operand_kwargs("src", src_acc, src_map, shape),
                )
                nisa.tensor_copy(
                    **_operand_kwargs("dst", dst_acc, dst_map, shape),
                    **_operand_kwargs("src", inter_acc, inter_map, shape),
                    engine=nisa.Engine.Vector,
                )
            else:
                nisa.tensor_copy(
                    **_operand_kwargs("dst", inter_acc, inter_map, shape),
                    **_operand_kwargs("src", src_acc, src_map, shape),
                    engine=nisa.Engine.Vector,
                )
                nisa.dma_copy(
                    **_operand_kwargs("dst", dst_acc, dst_map, shape),
                    **_operand_kwargs("src", inter_acc, inter_map, shape),
                )
        elif needs_dma:
            nisa.dma_copy(
                **_operand_kwargs("dst", dst_acc, dst_map, shape),
                **_operand_kwargs("src", src_acc, src_map, shape),
            )
        else:
            nisa.tensor_copy(
                **_operand_kwargs("dst", dst_acc, dst_map, shape),
                **_operand_kwargs("src", src_acc, src_map, shape),
                engine=nisa.Engine.Vector,
            )
    op.operation.erase()


@pattern("linalg.copy")
def _rewrite_linalg_copy(rctx: _RewriteContext, op: nk_ir.OpView) -> None:
    operands = list(op.operation.operands)
    if len(operands) < 2:
        return
    src, dst = operands[0], operands[1]
    src_ty, dst_ty = src.type, dst.type
    src_sbuf, dst_sbuf = _is_sbuf(src_ty), _is_sbuf(dst_ty)
    src_psum, dst_psum = _is_psum(src_ty), _is_psum(dst_ty)
    if not ((src_sbuf and dst_sbuf) or (src_sbuf and dst_psum) or (src_psum and dst_sbuf)):
        return
    shape = _static_shape(dst_ty)
    if shape is None:
        return
    with nk_ir.InsertionPoint(op), rctx.loc:
        src_acc = _get_base_and_offsets(rctx.ctx, src, rctx.loc)
        dst_acc = _get_base_and_offsets(rctx.ctx, dst, rctx.loc)
        rank = len(shape)
        src_map = _build_nisa_map(rctx.ctx, rank, src_acc)
        dst_map = _build_nisa_map(rctx.ctx, rank, dst_acc)
        nisa.tensor_copy(
            **_operand_kwargs("dst", dst_acc, dst_map, shape),
            **_operand_kwargs("src", src_acc, src_map, shape),
            engine=nisa.Engine.Vector,
        )
    op.operation.erase()


# ---------------------------------------------------------------------------
# memref.alloc / memref.dealloc
# ---------------------------------------------------------------------------
