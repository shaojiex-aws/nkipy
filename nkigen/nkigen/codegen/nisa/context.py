"""Module parse/serialize bridge: upstream MLIR ctx -> NKI ctx, with
integer memspace markers rewritten to #nisa.mem<...> attribute syntax."""

from __future__ import annotations

import re

from mlir import ir as up_ir  # type: ignore[import-not-found]

from nki.compiler._internal import ir as nk_ir  # type: ignore[import-not-found]
from nki.compiler._internal._mlir_libs import _nki  # type: ignore[import-not-found]

# NKIPy emits memref memory-space annotations as `N : i32` integers (matching
# the `MemSpaceEnum` in `NkipyAttrs.td`). NKI's parser needs them as
# `#nisa.mem<...>` attribute syntax. The enum values start at 1 (not 0) — see
# the comment in NkipyAttrs.td for why 0 cannot be used.
_NKIPY_TO_NISA_MEMSPACE = {
    1: "hbm",
    2: "psum",
    3: "sbuf",
    4: "shared_hbm",
}

_INT_MEMSPACE_RE = re.compile(r", (\d+) : i32>")


def _rewrite_memspace_text(generic: str) -> str:
    def repl(m: re.Match[str]) -> str:
        n = int(m.group(1))
        name = _NKIPY_TO_NISA_MEMSPACE.get(n)
        if name is None:
            return m.group(0)
        return f", #nisa.mem<{name}>>"

    return _INT_MEMSPACE_RE.sub(repl, generic)


def _to_nki_module(src: str) -> tuple[nk_ir.Context, nk_ir.Module]:
    # Re-serialize through nkipy-opt with `--mlir-print-op-generic` so any
    # nkipy-dialect ops that survive into this phase (currently just
    # `nkipy.gather`, which we lower below) arrive in generic form
    # `"nkipy.gather"(...)`. The upstream MLIR Python bindings don't know
    # about the nkipy dialect; pretty-form `nkipy.gather(...)` would fail
    # to parse (`allow_unregistered_dialects` only covers generic form).
    from ...driver.pipeline import run_nkipy_opt_passes  # avoid circular import
    src = run_nkipy_opt_passes(src, passes=[], print_generic=True)

    up_ctx = up_ir.Context()
    up_ctx.load_all_available_dialects()
    up_ctx.allow_unregistered_dialects = True
    with up_ctx:
        up_mod = up_ir.Module.parse(src)
        generic = up_mod.operation.get_asm(
            print_generic_op_form=True, assume_verified=True
        )
    generic = _rewrite_memspace_text(generic)

    nk_ctx = nk_ir.Context()
    _nki.register_all_dialects(nk_ctx)
    nk_ctx.allow_unregistered_dialects = True
    with nk_ctx:
        nk_mod = nk_ir.Module.parse(generic)
    return nk_ctx, nk_mod
