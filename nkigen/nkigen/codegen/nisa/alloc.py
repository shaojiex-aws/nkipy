"""memref.alloc -> nisa.alloc, reinterpret_cast folding, memref.dealloc
-> nisa.release."""

from __future__ import annotations

from ._vendor import nk_ir, nisa

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


@pattern("memref.dealloc")
def _rewrite_memref_dealloc(rctx: _RewriteContext, op: nk_ir.OpView) -> None:
    target = op.operation.operands[0]
    target_ty = target.type
    if not (_is_sbuf(target_ty) or _is_psum(target_ty)):
        return
    with nk_ir.InsertionPoint(op), rctx.loc:
        nisa.release(memref=target)
    op.operation.erase()
