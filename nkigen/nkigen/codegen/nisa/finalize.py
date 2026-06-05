"""Post-pass cleanups run after the per-op rewrites: fold reinterpret_cast on
fresh allocs and fold HBM collapse_shape/expand_shape into the nisa.alloc."""

from __future__ import annotations

from ._vendor import nk_ir, nisa

from .patterns import _RewriteContext, _is_hbm


# ---------------------------------------------------------------------------
# Fold reinterpret_cast(nisa.alloc) -> a single nisa.alloc of the cast type.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Fold HBM collapse_shape/expand_shape into the nisa.alloc.
# ---------------------------------------------------------------------------


def _alloc_defining_op(v: nk_ir.Value):
    owner = getattr(v, "owner", None)
    if owner is None:
        return None
    op = owner.opview if hasattr(owner, "opview") else owner
    if getattr(op, "name", None) != "nisa.alloc":
        return None
    return op


def _is_block_arg(v: nk_ir.Value) -> bool:
    owner = getattr(v, "owner", None)
    return isinstance(owner, nk_ir.Block)


def _try_fold_hbm_reshape_alloc(rctx: _RewriteContext, op: nk_ir.OpView) -> bool:
    src = op.operation.operands[0]
    src_ty = src.type
    if not _is_hbm(src_ty):
        return False
    alloc_op = _alloc_defining_op(src)
    if alloc_op is None:
        return False
    dst_ty = op.operation.results[0].type
    if not isinstance(dst_ty, nk_ir.MemRefType):
        return False

    alignment = 0
    if "alignment" in alloc_op.attributes:
        alignment = nk_ir.IntegerAttr(alloc_op.attributes["alignment"]).value

    src_shape = list(getattr(src_ty, "shape", ()))
    src_elt = src_ty.element_type  # type: ignore[attr-defined]

    with nk_ir.InsertionPoint(alloc_op), rctx.loc:
        new_alloc = nisa.alloc(memref_type=dst_ty, alignment=alignment)

    for user in list(alloc_op.result.uses):
        user_op = user.owner
        if user_op.name != "nisa.dma_copy":
            continue
        if user_op.operands[0] != alloc_op.result:
            continue
        user_op.operands[0] = new_alloc
        existing = (
            user_op.attributes["dst_shape"]
            if "dst_shape" in user_op.attributes else None
        )
        if existing is None or str(existing) in ("", "array<i64>"):
            user_op.attributes["dst_shape"] = nk_ir.DenseI64ArrayAttr.get(src_shape)
        user_op.attributes["dst_elt_ty"] = nk_ir.TypeAttr.get(src_elt)

    op.operation.results[0].replace_all_uses_with(new_alloc)
    op.operation.erase()
    if not list(alloc_op.result.uses):
        alloc_op.erase()
    return True


def _try_fold_hbm_reshape_arg(rctx: _RewriteContext, op: nk_ir.OpView) -> bool:
    src = op.operation.operands[0]
    src_ty = src.type
    if not _is_hbm(src_ty):
        return False
    if not _is_block_arg(src):
        return False
    src_shape = list(getattr(src_ty, "shape", ()))
    src_elt = src_ty.element_type  # type: ignore[attr-defined]
    users = list(op.operation.results[0].uses)
    if not users:
        op.operation.erase()
        return True
    for user in users:
        user_op = user.owner
        if user_op.name != "nisa.dma_copy":
            return False
        if user_op.operands[1] != op.operation.results[0]:
            return False

    for user in users:
        user_op = user.owner
        user_op.operands[1] = src
        existing = (
            user_op.attributes["src_shape"]
            if "src_shape" in user_op.attributes else None
        )
        if existing is None or str(existing) in ("", "array<i64>"):
            user_op.attributes["src_shape"] = nk_ir.DenseI64ArrayAttr.get(src_shape)
        user_op.attributes["src_elt_ty"] = nk_ir.TypeAttr.get(src_elt)

    op.operation.erase()
    return True


def _fold_hbm_reshapes(rctx: _RewriteContext) -> None:
    while True:
        candidates: list[nk_ir.OpView] = []

        def visit(op_handle: nk_ir.Operation) -> nk_ir.WalkResult:
            if op_handle.name in ("memref.collapse_shape", "memref.expand_shape"):
                candidates.append(op_handle.opview)
            return nk_ir.WalkResult.ADVANCE

        rctx.module.operation.walk(visit)
        progressed = False
        for op in candidates:
            if _try_fold_hbm_reshape_alloc(rctx, op):
                progressed = True
                continue
            if _try_fold_hbm_reshape_arg(rctx, op):
                progressed = True
        if not progressed:
            return
