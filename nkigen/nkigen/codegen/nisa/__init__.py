"""NISA MLIR codegen backend.

Walks tiled linalg IR (memref+scf+linalg+arith+func with integer-encoded
nkipy memory spaces) and emits NISA MLIR assembly as text.
"""

from __future__ import annotations

from .custom_ops import _resolve_custom_ops


def linalg_to_nisa(
    mlir_text: str, target: str = "trn2", print_generic: bool = True,
) -> str:
    """Translate tiled linalg IR to NISA MLIR (text -> text)."""
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
        return emitter.emit_module(module, target=target)


__all__ = [
    "linalg_to_nisa",
    "_resolve_custom_ops",
]
