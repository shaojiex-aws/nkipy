"""Pattern registry (@pattern decorator + _PATTERNS), the _RewriteContext,
the linalg/arith op maps, and memory-space predicates."""

from __future__ import annotations

from typing import Callable

from nki.compiler._internal import ir as nk_ir  # type: ignore[import-not-found]
from nki.compiler._internal.dialects import nisa  # type: ignore[import-not-found]

_Pattern = Callable[["_RewriteContext", nk_ir.OpView], None]
_PATTERNS: dict[str, _Pattern] = {}


def pattern(*op_names: str) -> Callable[[_Pattern], _Pattern]:
    def register(fn: _Pattern) -> _Pattern:
        for name in op_names:
            _PATTERNS[name] = fn
        return fn

    return register


_LINALG_TO_ARITH_OP = {
    "linalg.add": nisa.ArithOp.Add,
    "linalg.sub": nisa.ArithOp.Subtract,
    "linalg.mul": nisa.ArithOp.Multiply,
    "linalg.max": nisa.ArithOp.Max,
    "linalg.min": nisa.ArithOp.Min,
}

_REDUCE_BODY_OP_TO_ARITH = {
    "arith.addf": nisa.ArithOp.Add,
    "arith.addi": nisa.ArithOp.Add,
    "arith.mulf": nisa.ArithOp.Multiply,
    "arith.muli": nisa.ArithOp.Multiply,
    "arith.maximumf": nisa.ArithOp.Max,
    "arith.minimumf": nisa.ArithOp.Min,
}

_ARITH_TO_CROSS_LANE = {
    nisa.ArithOp.Add: nisa.CrossLaneReduceArithOp.Add,
    nisa.ArithOp.Max: nisa.CrossLaneReduceArithOp.Max,
}


class _RewriteContext:
    def __init__(self, ctx: nk_ir.Context, module: nk_ir.Module):
        self.ctx = ctx
        self.module = module
        self.loc = nk_ir.Location.unknown(ctx)
        self._f32_const_cache: dict[tuple[int, float], nk_ir.Value] = {}

    def f32_const(self, block: nk_ir.Block, value: float) -> nk_ir.Value:
        # Cache by (id(block), value) with a liveness check — when a pass
        # later splices or erases blocks, id() can be recycled and the
        # cached SSA Value may belong to a different, unrelated block.
        # Verify the cache entry still lives in the expected block before
        # returning it; otherwise create a fresh constant.
        key = (id(block), value)
        cached = self._f32_const_cache.get(key)
        if cached is not None:
            owner = getattr(cached, "owner", None)
            cached_block = getattr(owner, "block", None) if owner else None
            if cached_block is block:
                return cached
        with self.loc:
            f32 = nk_ir.F32Type.get(self.ctx)
            attr = nk_ir.FloatAttr.get(f32, value)
            with nk_ir.InsertionPoint.at_block_begin(block):
                const_op = nk_ir.Operation.create(
                    "arith.constant",
                    results=[f32],
                    attributes={"value": attr},
                    loc=self.loc,
                )
        self._f32_const_cache[key] = const_op.result
        return const_op.result


def _enclosing_block(op: nk_ir.OpView) -> nk_ir.Block:
    return op.operation.block  # type: ignore[attr-defined]


def _is_memspace(ty: nk_ir.Type, name: str) -> bool:
    ms = getattr(ty, "memory_space", None)
    if ms is None:
        return False
    return f"<{name}>" in str(ms)


def _is_hbm(ty: nk_ir.Type) -> bool:
    return _is_memspace(ty, "hbm") or _is_memspace(ty, "shared_hbm")


def _is_sbuf(ty: nk_ir.Type) -> bool:
    return _is_memspace(ty, "sbuf")


def _is_psum(ty: nk_ir.Type) -> bool:
    return _is_memspace(ty, "psum")


def _static_shape(ty: nk_ir.Type) -> list[int] | None:
    shape = list(getattr(ty, "shape", ()))
    if any(s < 0 for s in shape):
        return None
    return shape
