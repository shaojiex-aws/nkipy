"""memref.alloc -> nisa.alloc, reinterpret_cast folding, memref.dealloc
-> nisa.release."""

from __future__ import annotations

from nki.compiler._internal import ir as nk_ir  # type: ignore[import-not-found]
from nki.compiler._internal.dialects import nisa  # type: ignore[import-not-found]

from .patterns import (
    _RewriteContext,
    _is_hbm,
    _is_psum,
    _is_sbuf,
    pattern,
)

@pattern("memref.alloc")
def _rewrite_memref_alloc(rctx: _RewriteContext, op: nk_ir.OpView) -> None:
    result_ty = op.operation.results[0].type
    if not (_is_sbuf(result_ty) or _is_psum(result_ty) or _is_hbm(result_ty)):
        return

    alignment = 0
    attrs = op.operation.attributes
    if "alignment" in attrs:
        alignment = nk_ir.IntegerAttr(attrs["alignment"]).value

    with nk_ir.InsertionPoint(op), rctx.loc:
        new_val = nisa.alloc(memref_type=result_ty, alignment=alignment)

    op.operation.results[0].replace_all_uses_with(new_val)
    op.operation.erase()


def _fold_reinterpret_casts(rctx: _RewriteContext) -> None:
    casts: list[nk_ir.OpView] = []

    def visit(op_handle: nk_ir.Operation) -> nk_ir.WalkResult:
        if op_handle.name == "memref.reinterpret_cast":
            casts.append(op_handle.opview)
        return nk_ir.WalkResult.ADVANCE

    rctx.module.operation.walk(visit)

    for cast_op in casts:
        src = cast_op.operation.operands[0]
        src_owner = getattr(src, "owner", None)
        if src_owner is None:
            continue
        src_op = src_owner.opview if hasattr(src_owner, "opview") else src_owner
        if getattr(src_op, "name", None) != "nisa.alloc":
            continue
        try:
            st_off = [int(x) for x in cast_op.operation.attributes["static_offsets"]]
            if any(x != 0 for x in st_off):
                continue
        except (KeyError, ValueError):
            continue

        new_ty = cast_op.operation.results[0].type

        alignment = 0
        if "alignment" in src_op.attributes:
            alignment = nk_ir.IntegerAttr(src_op.attributes["alignment"]).value

        with nk_ir.InsertionPoint(src_op), rctx.loc:
            new_alloc = nisa.alloc(memref_type=new_ty, alignment=alignment)

        cast_op.operation.results[0].replace_all_uses_with(new_alloc)
        cast_op.operation.erase()
        if list(src.uses):
            src.replace_all_uses_with(new_alloc)
        src_op.erase()


@pattern("memref.dealloc")
def _rewrite_memref_dealloc(rctx: _RewriteContext, op: nk_ir.OpView) -> None:
    target = op.operation.operands[0]
    target_ty = target.type
    if not (_is_sbuf(target_ty) or _is_psum(target_ty)):
        return
    with nk_ir.InsertionPoint(op), rctx.loc:
        nisa.release(memref=target)
    op.operation.erase()


# ---------------------------------------------------------------------------
# linalg.transpose
# ---------------------------------------------------------------------------
