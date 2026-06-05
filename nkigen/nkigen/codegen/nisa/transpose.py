"""linalg.transpose -> nisa.dma_transpose / copy, plus unit-dim drop
helpers and a small index-constant emitter."""

from __future__ import annotations

from ._vendor import nk_ir, nisa

from .access import _Access, _get_base_and_offsets
from .affine_map import _build_nisa_map, _operand_kwargs
from .patterns import (
    _RewriteContext,
    _is_hbm,
    _is_sbuf,
    _static_shape,
    pattern,
)

def _non_unit_dims(shape: list[int]) -> list[int]:
    return [i for i, s in enumerate(shape) if s != 1]


def _drop_unit_dims_from_access(
    access: _Access, operand_shape: list[int]
) -> _Access:
    """Extend access.dropped_dims to also drop dims whose operand extent is 1.

    When an operand is a subview like ``[1, 128, 128]`` over a 3-D base,
    the leading ``1`` is a single-element slice, not an iterated range.
    The nisa affine map must treat that dim as a fixed symbol offset
    (effectively "dropped"), otherwise iter dims get mis-placed on it
    and the resulting DMA stride reads past the end of the base.

    This only applies when operand rank equals base rank (no rank-
    reducing in the chain); otherwise the mapping between operand and
    base dims is ambiguous and we leave dropped_dims alone.
    """
    if len(operand_shape) != access.base_rank:
        return access
    new_dropped = list(access.dropped_dims)
    while len(new_dropped) < access.base_rank:
        new_dropped.append(False)
    for i, s in enumerate(operand_shape):
        if s == 1:
            new_dropped[i] = True
    return _Access(
        base=access.base,
        indices=access.indices,
        base_type=access.base_type,
        dropped_dims=new_dropped,
    )


@pattern("linalg.transpose")
def _rewrite_linalg_transpose(rctx: _RewriteContext, op: nk_ir.OpView) -> None:
    operands = list(op.operation.operands)
    if len(operands) < 2:
        return
    src, dst = operands[0], operands[1]
    src_ty, dst_ty = src.type, dst.type
    src_shape = _static_shape(src_ty)
    dst_shape = _static_shape(dst_ty)
    if src_shape is None or dst_shape is None:
        return

    attrs = op.operation.attributes
    if "permutation" not in attrs:
        return
    perm_str = str(attrs["permutation"])
    try:
        inside = perm_str.split(":", 1)[1].rstrip(">").strip()
        perm = [int(x.strip()) for x in inside.split(",") if x.strip()]
    except (IndexError, ValueError):
        return

    non_unit_src = _non_unit_dims(src_shape)
    if len(non_unit_src) > 2:
        return

    needs_transpose = False
    if len(non_unit_src) == 2:
        s0, s1 = non_unit_src[0], non_unit_src[1]
        d0 = perm.index(s0)
        d1 = perm.index(s1)
        needs_transpose = d0 > d1

    src_hbm = _is_hbm(src_ty)
    src_sbuf = _is_sbuf(src_ty)
    dst_sbuf = _is_sbuf(dst_ty)
    dst_hbm = _is_hbm(dst_ty)

    # 2D tile shapes derived from non-unit dims.
    src_tile = [src_shape[d] for d in non_unit_src]
    dst_tile = [dst_shape[d] for d in _non_unit_dims(dst_shape)]
    while len(src_tile) < 2:
        src_tile.append(1)
    while len(dst_tile) < 2:
        dst_tile.append(1)

    with nk_ir.InsertionPoint(op), rctx.loc:
        src_acc = _get_base_and_offsets(rctx.ctx, src, rctx.loc)
        dst_acc = _get_base_and_offsets(rctx.ctx, dst, rctx.loc)
        # When the transpose operand is a subview with size-1 dims (e.g.
        # `[1, 128, 128]` carved out of a 3-D HBM alloc whose batch dim
        # isn't rank-reduced), the iter-dim count from non-unit dims is
        # smaller than the base rank. Mark those size-1 dims as
        # dropped in the nisa map so the access pattern skips them
        # instead of striding across them at the base-level stride.
        src_acc = _drop_unit_dims_from_access(src_acc, src_shape)
        dst_acc = _drop_unit_dims_from_access(dst_acc, dst_shape)
        num_iter = 2
        src_map = _build_nisa_map(rctx.ctx, num_iter, src_acc)
        dst_map = _build_nisa_map(rctx.ctx, num_iter, dst_acc)

        if needs_transpose:
            if not ((src_hbm or src_sbuf) and dst_sbuf):
                return
            nisa.dma_transpose(
                **_operand_kwargs("dst", dst_acc, dst_map, dst_tile),
                **_operand_kwargs("src", src_acc, src_map, src_tile),
                permutation=[1, 0],
                dge_mode=nisa.DGEType.NoDGE,
                oob_is_err=True,
                engine=nisa.Engine.DMA,
            )
            op.operation.erase()
            return

        cross = (src_hbm and dst_sbuf) or (src_sbuf and dst_hbm)
        if cross:
            nisa.dma_copy(
                **_operand_kwargs("dst", dst_acc, dst_map, dst_tile),
                **_operand_kwargs("src", src_acc, src_map, src_tile),
            )
            op.operation.erase()
            return
