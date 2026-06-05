"""nkipy.gather -> nisa.dma_copy_indirect iteration."""

from __future__ import annotations

from nki.compiler._internal import ir as nk_ir  # type: ignore[import-not-found]
from nki.compiler._internal.dialects import nisa  # type: ignore[import-not-found]

from .access import (
    _Access,
    _emit_addi,
    _emit_const_index,
    _get_base_and_offsets,
)
from .affine_map import _build_nisa_map
from .patterns import (
    _RewriteContext,
    _static_shape,
    pattern,
)

def _build_dma_copy_indirect_op(
    *,
    ctx: nk_ir.Context,
    loc: nk_ir.Location,
    dst_memloc: nk_ir.Value,
    dst_indices: list[nk_ir.Value],
    dst_ap: nk_ir.Attribute,
    dst_tile_shape: list[int],
    src_memloc: nk_ir.Value,
    src_indices: list[nk_ir.Value],
    src_ap: nk_ir.Attribute,
    src_tile_shape: list[int],
    src_index_memloc: nk_ir.Value,
    src_index_ap: nk_ir.Attribute,
    src_index_tile_shape: list[int],
    src_indirect_max_index: int,
    tile_par_dims: int = 1,
) -> None:
    """Raw builder for nisa.dma_copy_indirect.

    We bypass the generated Python builder because it translates
    ``dst_indirect_max_index=None`` into ``[0 : i32]``, which the
    verifier then rejects since there's no matching ``dst_index``
    operand.  Here we set the attribute to an empty ArrayAttr so the
    verifier sees both ``dst_index`` and ``dst_indirect_max_index`` as
    absent.
    """
    i32_ty = nk_ir.IntegerType.get_signless(32, ctx)
    bool_true = nk_ir.BoolAttr.get(True, ctx)

    operands = [dst_memloc, *dst_indices, src_memloc, *src_indices,
                src_index_memloc]
    # operandSegmentSizes: 13 segments in declaration order.
    seg_sizes = [
        1, len(dst_indices), 0,        # dst
        1, len(src_indices), 0,        # src
        1, 0, 0,                       # src_index
        0, 0, 0,                       # dst_index (absent)
        0,                             # dma_qos (absent)
    ]
    attrs = {
        "dst_ap": dst_ap,
        "dst_static_tile_shape": nk_ir.DenseI64ArrayAttr.get(
            dst_tile_shape, ctx
        ),
        "dst_tile_par_dims": nk_ir.IntegerAttr.get(i32_ty, tile_par_dims),
        "src_ap": src_ap,
        "src_static_tile_shape": nk_ir.DenseI64ArrayAttr.get(
            src_tile_shape, ctx
        ),
        "src_tile_par_dims": nk_ir.IntegerAttr.get(i32_ty, tile_par_dims),
        "src_index_ap": src_index_ap,
        "src_index_static_tile_shape": nk_ir.DenseI64ArrayAttr.get(
            src_index_tile_shape, ctx
        ),
        "src_index_tile_par_dims": nk_ir.IntegerAttr.get(i32_ty, tile_par_dims),
        "dst_index_static_tile_shape": nk_ir.DenseI64ArrayAttr.get([], ctx),
        "dst_index_tile_par_dims": nk_ir.IntegerAttr.get(i32_ty, 0),
        "src_indirect_max_index": nk_ir.ArrayAttr.get(
            [nk_ir.IntegerAttr.get(i32_ty, src_indirect_max_index)], ctx,
        ),
        "dst_indirect_max_index": nk_ir.ArrayAttr.get([], ctx),
        "dst_rmw_op": nk_ir.ArrayAttr.get([], ctx),
        "oob_is_err": bool_true,
        "unique_indices": bool_true,
        "engine": nk_ir.IntegerAttr.get(i32_ty, nisa.Engine.DMA.value),
        "operandSegmentSizes": nk_ir.DenseI32ArrayAttr.get(seg_sizes, ctx),
    }
    nk_ir.Operation.create(
        "nisa.dma_copy_indirect",
        results=[],
        operands=operands,
        attributes=attrs,
        loc=loc,
    )


# ---------------------------------------------------------------------------
# nkipy.gather -> nisa.dma_copy_indirect
# ---------------------------------------------------------------------------
#
# Port of the deleted C++ NkipyGatherToNisaPattern.  The source table stays
# in HBM and each iteration gathers one row (or one partition's worth of
# rows) into SBUF via dma_copy_indirect, then DMAs the result back to the
# output HBM tensor.
#
# Layouts:
#   2D output [N, H]:  single gather of N rows (one index per partition).
#   3D output [N, I, H]: wrap in scf.for over I, gathering one row/partition
#                        per iteration; output[:, i, :] = source[indices[:, i]].
#
# The indirect DMA uses three specially-shaped affine maps:
#   dst     : standard [d0, d1]        — fill the SBUF output tile
#   src     : [s0, d1 + s1]            — look up one row; d0 is unused, the
#                                         row index comes from the index buffer
#   index   : standard [d0, d1]        — read N indices, one per partition


def _gather_src_indirect_map(ctx: nk_ir.Context) -> nk_ir.Attribute:
    # (d0, d1)[s0, s1] -> [s0, d1 + s1]
    s0 = nk_ir.AffineSymbolExpr.get(0)
    d1 = nk_ir.AffineDimExpr.get(1)
    s1 = nk_ir.AffineSymbolExpr.get(1)
    amap = nk_ir.AffineMap.get(2, 2, [s0, d1 + s1])
    return nisa.flatten_affine_map(amap, ctx)


def _gather_standard_2d_map(ctx: nk_ir.Context, num_symbols: int) -> nk_ir.Attribute:
    # Standard 2D map with d0/d1 on each dim, plus per-dim symbol offsets.
    d0 = nk_ir.AffineDimExpr.get(0)
    d1 = nk_ir.AffineDimExpr.get(1)
    exprs: list[nk_ir.AffineExpr] = []
    if num_symbols >= 1:
        exprs.append(d0 + nk_ir.AffineSymbolExpr.get(0))
    else:
        exprs.append(d0)
    if num_symbols >= 2:
        exprs.append(d1 + nk_ir.AffineSymbolExpr.get(1))
    else:
        exprs.append(d1)
    return nisa.flatten_affine_map(
        nk_ir.AffineMap.get(2, num_symbols, exprs), ctx
    )


def _emit_gather_iteration(
    rctx: _RewriteContext,
    indices_sbuf: nk_ir.Value,
    sbuf_output: nk_ir.Value,
    idx_access: _Access,
    src_access: _Access,
    out_access: _Access,
    zero_idx: nk_ir.Value,
    idx_offset: nk_ir.Value,
    out_indices: list[nk_ir.Value],
    N: int,
    H: int,
    base_v: int,
    tile_par_dims: int = 1,
) -> None:
    ctx = rctx.ctx
    loc = rctx.loc

    # --- 1. DMA one "column" of indices from HBM/SBUF into the indices SBUF ---
    dst_map_std = _gather_standard_2d_map(ctx, num_symbols=2)
    if not idx_access.indices:
        idx_src_indices = [zero_idx, idx_offset]
    else:
        idx_src_indices = list(idx_access.indices)
        # Add idx_offset to the innermost index
        idx_src_indices[-1] = _emit_addi(
            idx_src_indices[-1], idx_offset, ctx, loc
        )
    idx_src_map = _build_nisa_map(
        ctx, 2,
        _Access(
            base=idx_access.base,
            indices=idx_src_indices,
            base_type=idx_access.base_type,
            dropped_dims=idx_access.dropped_dims,
        ),
    )
    nisa.dma_copy(
        dst_memloc=indices_sbuf,
        dst_indices=[zero_idx, zero_idx],
        dst_ap=dst_map_std,
        dst_static_tile_shape=[N, 1],
        dst_tile_par_dims=tile_par_dims,
        src_memloc=idx_access.base,
        src_indices=idx_src_indices,
        src_ap=idx_src_map,
        src_static_tile_shape=[N, 1],
        src_tile_par_dims=tile_par_dims,
        oob_is_err=True,
        engine=nisa.Engine.DMA,
    )

    # --- 2. dma_copy_indirect: gather H elements from HBM using SBUF indices ---
    # dst: standard [d0, d1] 2D map (no symbols — direct to SBUF alloc base).
    gather_dst_map = nisa.flatten_affine_map(
        nk_ir.AffineMap.get(
            2, 0,
            [nk_ir.AffineDimExpr.get(0), nk_ir.AffineDimExpr.get(1)],
        ),
        ctx,
    )
    # src: (d0, d1)[s0, s1] -> [s0, d1 + s1] — s0 is the indirect row index
    # supplied by the index buffer; d1+s1 covers the gather's free dim.
    gather_src_map = _gather_src_indirect_map(ctx)
    col_offset = (
        src_access.indices[1]
        if len(src_access.indices) > 1 else zero_idx
    )
    src_indirect_indices = [col_offset]
    # index: standard [d0, d1] 2D map, no symbol offsets
    index_map = nisa.flatten_affine_map(
        nk_ir.AffineMap.get(
            2, 0,
            [nk_ir.AffineDimExpr.get(0), nk_ir.AffineDimExpr.get(1)],
        ),
        ctx,
    )

    # Emit the op via builder and then repair its inherent properties.
    # The Python builder sets `dst_indirect_max_index = [0 : i32]` when
    # no dst_index is provided, which the verifier rejects.  MLIR's
    # `dma_op.attributes[...]` assignment adds a *discardable* attribute
    # but leaves the op's inherent `Properties` struct untouched, so we
    # rebuild the op by printing it, textually clearing the bogus attr,
    # and parsing back in place. Same trick the C++ side used implicitly
    # by constructing the op in one shot.
    dma_op = nisa.dma_copy_indirect(
        dst_memloc=sbuf_output,
        dst_indices=[],
        dst_ap=gather_dst_map,
        dst_static_tile_shape=[N, H],
        dst_tile_par_dims=tile_par_dims,
        src_memloc=src_access.base,
        src_indices=src_indirect_indices,
        src_ap=gather_src_map,
        src_static_tile_shape=[N, H],
        src_tile_par_dims=tile_par_dims,
        src_index_memloc=indices_sbuf,
        src_index_indices=[],
        src_index_ap=index_map,
        src_index_static_tile_shape=[N, 1],
        src_index_tile_par_dims=tile_par_dims,
        dst_index_memloc=None,
        dst_index_indices=[],
        dst_index_ap=None,
        dst_index_static_tile_shape=[],
        dst_index_tile_par_dims=0,
        oob_is_err=True,
        src_indirect_max_index=base_v,
        unique_indices=True,
        engine=nisa.Engine.DMA,
    )

    # --- 3. DMA the gathered SBUF tile back out to the output HBM buffer ---
    src_copy_map = _gather_standard_2d_map(ctx, num_symbols=2)
    dst_copy_map = _build_nisa_map(
        ctx, 2,
        _Access(
            base=out_access.base,
            indices=out_indices,
            base_type=out_access.base_type,
            dropped_dims=out_access.dropped_dims,
        ),
    )
    nisa.dma_copy(
        dst_memloc=out_access.base,
        dst_indices=out_indices,
        dst_ap=dst_copy_map,
        dst_static_tile_shape=[N, H],
        dst_tile_par_dims=tile_par_dims,
        src_memloc=sbuf_output,
        src_indices=[zero_idx, zero_idx],
        src_ap=src_copy_map,
        src_static_tile_shape=[N, H],
        src_tile_par_dims=tile_par_dims,
        oob_is_err=True,
        engine=nisa.Engine.DMA,
    )


@pattern("nkipy.gather")
def _rewrite_nkipy_gather(rctx: _RewriteContext, op: nk_ir.OpView) -> None:
    operands = list(op.operation.operands)
    if len(operands) < 3:
        return
    source = operands[0]
    indices = operands[1]
    output = operands[2]

    src_ty = source.type
    idx_ty = indices.type
    out_ty = output.type
    if not (isinstance(src_ty, nk_ir.MemRefType)
            and isinstance(idx_ty, nk_ir.MemRefType)
            and isinstance(out_ty, nk_ir.MemRefType)):
        return

    src_shape = _static_shape(src_ty)
    out_shape = _static_shape(out_ty)
    if src_shape is None or out_shape is None:
        return
    if len(src_shape) != 2:
        return
    out_rank = len(out_shape)
    if out_rank < 2 or out_rank > 3:
        return

    H = src_shape[1]
    N = out_shape[0]
    base_v = src_shape[0]
    ctx = rctx.ctx

    sbuf_attr = nk_ir.Attribute.parse("#nisa.mem<sbuf>")
    i32_ty = nk_ir.IntegerType.get_signless(32, ctx)
    idx_elt_ty = idx_ty.element_type  # type: ignore[attr-defined]
    out_elt_ty = out_ty.element_type  # type: ignore[attr-defined]

    with nk_ir.InsertionPoint(op), rctx.loc:
        src_acc = _get_base_and_offsets(ctx, source, rctx.loc)
        idx_acc = _get_base_and_offsets(ctx, indices, rctx.loc)
        out_acc = _get_base_and_offsets(ctx, output, rctx.loc)

        zero_idx = _emit_const_index(ctx, 0, rctx.loc)

        indices_sbuf_ty = nk_ir.MemRefType.get(
            [N, 1], idx_elt_ty, memory_space=sbuf_attr,
        )
        indices_sbuf = nisa.alloc(memref_type=indices_sbuf_ty, alignment=0)
        sbuf_output_ty = nk_ir.MemRefType.get(
            [N, H], out_elt_ty, memory_space=sbuf_attr,
        )
        sbuf_output = nisa.alloc(memref_type=sbuf_output_ty, alignment=0)

        if out_rank == 3:
            I_size = out_shape[1]
            upper = _emit_const_index(ctx, I_size, rctx.loc)
            one = _emit_const_index(ctx, 1, rctx.loc)

            # scf.for_ is a generator that yields the induction variable
            # inside its body's insertion point.  With no iter_args the
            # single yield is the IV; the body terminates with scf.yield.
            from nki.compiler._internal.dialects import scf as _scf
            for iv in _scf.for_(zero_idx, upper, one, iter_args=[]):
                out_indices = list(out_acc.indices) if out_acc.indices \
                    else [zero_idx, zero_idx, zero_idx]
                # Insert the loop IV at the second dim (I) of the output.
                if len(out_indices) == 3:
                    out_indices[1] = _emit_addi(
                        out_indices[1], iv, ctx, rctx.loc
                    )
                _emit_gather_iteration(
                    rctx, indices_sbuf, sbuf_output,
                    idx_acc, src_acc, out_acc,
                    zero_idx, iv, out_indices,
                    N, H, base_v,
                )
                _scf.yield_([])
        else:
            out_indices = list(out_acc.indices) if out_acc.indices \
                else [zero_idx, zero_idx]
            _emit_gather_iteration(
                rctx, indices_sbuf, sbuf_output,
                idx_acc, src_acc, out_acc,
                zero_idx, zero_idx, out_indices,
                N, H, base_v,
            )

        nisa.release(memref=indices_sbuf)
        nisa.release(memref=sbuf_output)

    op.operation.erase()
