"""NISA MLIR codegen backend (split from the former monolithic
transforms/linalg_to_nisa_py.py).

Reads post-Phase-4 MLIR (linalg+memref+scf+arith+func, integer-encoded
nkipy memory spaces) and emits NISA MLIR via the nki wheel's Python
bindings. See the per-module docstrings for the section breakdown.

IMPORTANT: every codegen_* module is imported below so that its
@pattern(...) decorators run and populate the shared _PATTERNS registry
before _walk_and_rewrite dispatches on it."""

from __future__ import annotations

import re

from nki.compiler._internal import ir as nk_ir  # type: ignore[import-not-found]

from .context import _to_nki_module
from .patterns import _RewriteContext, pattern
from .walk import _walk_and_rewrite
from .custom_ops import _resolve_custom_ops

# Import for @pattern side effects: populate _PATTERNS.
from . import codegen_elementwise  # noqa: F401
from . import codegen_copy  # noqa: F401
from . import codegen_alloc  # noqa: F401
from . import codegen_transpose  # noqa: F401
from . import codegen_matmul  # noqa: F401
from . import codegen_activation  # noqa: F401
from . import codegen_fill  # noqa: F401
from . import codegen_reduction  # noqa: F401
from . import codegen_gather  # noqa: F401

def _finalize_for_nki(module: nk_ir.Module, ctx: nk_ir.Context, target: str) -> None:
    def strip_nkipy_attrs(op: nk_ir.Operation) -> nk_ir.WalkResult:
        to_remove = [
            named.name for named in op.attributes
            if named.name.startswith("nkipy.")
        ]
        for name in to_remove:
            del op.attributes[name]
        return nk_ir.WalkResult.ADVANCE

    module.operation.walk(strip_nkipy_attrs)

    for op in module.body.operations:
        if op.operation.name != "func.func":
            continue
        if "nki.output_names" in op.attributes:
            continue
        func_ty_attr = op.attributes["function_type"]
        func_ty = nk_ir.TypeAttr(func_ty_attr).value
        num_results = len(list(func_ty.results))  # type: ignore[attr-defined]
        if num_results == 0:
            continue
        names = [
            f"output_{i}" if num_results > 1 else "output"
            for i in range(num_results)
        ]
        op.attributes["nki.output_names"] = nk_ir.ArrayAttr.get(
            [nk_ir.StringAttr.get(n) for n in names]
        )

    target_attr = nk_ir.Attribute.parse(f"#nisa.target<{target}>")
    module.operation.attributes["nisa.target"] = target_attr


def linalg_to_nisa(
    mlir_text: str, target: str = "trn2", print_generic: bool = True,
) -> str:
    """Translate post-Phase-4 MLIR to NISA MLIR (text -> text).

    Defaults to generic form because the NISA pretty-printer omits the element
    type for ``nisa.dma_copy``'s ``view(...)`` syntax, which its own parser
    then rejects. Generic form roundtrips cleanly through the downstream NKI
    parser. Callers that consume the output as text (STRING_CHECK/FILECHECK)
    should pass ``print_generic=False``.
    """
    ctx, module = _to_nki_module(mlir_text)
    with ctx:
        rctx = _RewriteContext(ctx, module)
        _walk_and_rewrite(rctx)
        _resolve_custom_ops(module, ctx)
        _finalize_for_nki(module, ctx, target)
        out = module.operation.get_asm(
            print_generic_op_form=print_generic, assume_verified=True
        )
    # Strip the bogus `dst_indirect_max_index = [0 : i32]` the Python
    # builder for nisa.dma_copy_indirect injects on gather-only ops.
    # The verifier requires it to be absent when dst_index is absent,
    # but we can't clear the inherent property from Python — just edit
    # the text on the way out.
    out = re.sub(
        r",\s*dst_indirect_max_index\s*=\s*\[0\s*:\s*i32\]",
        "",
        out,
    )
    return out


__all__ = [
    "linalg_to_nisa",
    "pattern",
]
