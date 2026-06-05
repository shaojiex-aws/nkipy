"""Top-level walk + dead-view DCE driving the pattern registry."""

from __future__ import annotations

from nki.compiler._internal import ir as nk_ir  # type: ignore[import-not-found]

from .codegen_alloc import _fold_reinterpret_casts
from .finalize import _fold_hbm_reshapes
from .patterns import _PATTERNS, _RewriteContext

# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


_DEAD_VIEW_OPS = (
    "memref.subview",
    "memref.collapse_shape",
    "memref.expand_shape",
    "memref.reinterpret_cast",
)


def _dce_dead_view_ops(rctx: _RewriteContext) -> None:
    while True:
        dead: list[nk_ir.OpView] = []

        def visit(op_handle: nk_ir.Operation) -> nk_ir.WalkResult:
            if (
                op_handle.name in _DEAD_VIEW_OPS
                and not list(op_handle.results[0].uses)
            ):
                dead.append(op_handle.opview)
            return nk_ir.WalkResult.ADVANCE

        rctx.module.operation.walk(visit)
        if not dead:
            return
        for op in dead:
            op.operation.erase()


def _walk_and_rewrite(rctx: _RewriteContext) -> None:
    candidates: list[nk_ir.OpView] = []

    def visit(op_handle: nk_ir.Operation) -> nk_ir.WalkResult:
        name = op_handle.name
        if name in _PATTERNS:
            candidates.append(op_handle.opview)
        return nk_ir.WalkResult.ADVANCE

    rctx.module.operation.walk(visit)

    for op in candidates:
        _PATTERNS[op.operation.name](rctx, op)

    _dce_dead_view_ops(rctx)
    _fold_reinterpret_casts(rctx)
    _dce_dead_view_ops(rctx)
    _fold_hbm_reshapes(rctx)
