"""NISA MLIR codegen backend.

Walks tiled linalg IR (memref+scf+linalg+arith+func with integer-encoded
nkipy memory spaces) and emits NISA MLIR assembly as text.
"""

from __future__ import annotations

from .custom_ops import _resolve_custom_ops


def linalg_to_nisa(
    mlir_text: str, target: str = "trn2", print_generic: bool = True,
) -> str:
    """Translate tiled linalg IR to NISA MLIR (text -> text).

    If the module carries custom-op bodies, the emitted NISA text keeps the
    func.call sites, the body-less declarations, and the ``nkipy.custom_op_bodies``
    module attribute. A final resolve step re-parses that text and inlines each
    stashed body at its call site, mirroring the deleted C++ ResolveCustomOps
    pass. This replaces the C++ ``resolve-custom-ops`` + ``prepare-for-nki``
    stages the plan folds into this one Python phase.
    """
    from mlir import ir as up_ir  # type: ignore[import-not-found]
    from nkigen._mlir.dialects import nkipy as nkipy_d
    from .emit import NisaEmitter

    ctx = up_ir.Context()
    ctx.load_all_available_dialects()
    nkipy_d.register_dialect(ctx)
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = up_ir.Module.parse(mlir_text)
        emitter = NisaEmitter()
        nisa_text = emitter.emit_module(module, target=target)

    if "nkipy.custom_op_bodies" in nisa_text:
        nisa_text = _resolve_custom_ops_text(nisa_text)
    return nisa_text


def _resolve_custom_ops_text(nisa_text: str) -> str:
    """Re-parse emitted NISA text in an NKI-wheel context and inline stashed
    custom-op bodies. Kept separate because ``_resolve_custom_ops`` operates on
    the wheel's private ``nk_ir`` bindings, not the upstream ``mlir`` package
    the emitter reads with."""
    from nki.compiler._internal import ir as nk_ir  # type: ignore[import-not-found]
    from nki.compiler._internal._mlir_libs import _nki  # type: ignore[import-not-found]

    ctx = nk_ir.Context()
    _nki.register_all_dialects(ctx)
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = nk_ir.Module.parse(nisa_text, ctx)
        _resolve_custom_ops(module, ctx)
        return str(module)


__all__ = [
    "linalg_to_nisa",
    "_resolve_custom_ops",
]
