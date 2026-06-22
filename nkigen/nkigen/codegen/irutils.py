"""Small helpers over the *upstream* MLIR bindings (``mlir.ir``).

The kernel_builder backend emits text, so it walks the post-Phase-4 IR with the
standard upstream bindings — it never needs the NKI wheel. These helpers wrap
the few attribute accesses the walkers need (memref shape / element type /
memory space, constant folding, op operands) so the emit_* modules stay
readable.
"""

from __future__ import annotations

import re

from mlir import ir as up_ir  # type: ignore[import-not-found]


# Integer memory-space markers in post-Phase-4 memref types (matching
# MemSpaceEnum in NkipyAttrs.td; the enum starts at 1).
MEMSPACE_HBM = 1
MEMSPACE_PSUM = 2
MEMSPACE_SBUF = 3
MEMSPACE_SHARED_HBM = 4

# "<n> : i32" memory-space marker -> int.
_MEMSPACE_RE = re.compile(r"(\d+)\s*:\s*i32")


def memref_shape(ty: up_ir.Type) -> list[int]:
    """Static shape of a memref type. Dynamic dims appear as negative values."""
    return list(up_ir.MemRefType(ty).shape)


def memref_elem_type(ty: up_ir.Type) -> str:
    """MLIR element-type spelling of a memref, e.g. ``"f32"``."""
    return str(up_ir.MemRefType(ty).element_type)


def memref_memspace(ty: up_ir.Type) -> int | None:
    """Integer memory-space marker of a memref, or ``None`` if unset.

    Post-Phase-4 memrefs carry the space as an ``i32`` attribute (1=hbm,
    2=psum, 3=sbuf, 4=shared_hbm).
    """
    ms = up_ir.MemRefType(ty).memory_space
    if ms is None:
        return None
    m = _MEMSPACE_RE.search(str(ms))
    return int(m.group(1)) if m else None


def is_on_chip(ty: up_ir.Type) -> bool:
    """True if a memref lives in on-chip memory (SBUF or PSUM).

    Compute ops (memset, tensor_*) require on-chip operands — they cannot read
    or write HBM directly.
    """
    return memref_memspace(ty) in (MEMSPACE_SBUF, MEMSPACE_PSUM)


def is_memref(ty: up_ir.Type) -> bool:
    """True if ``ty`` is a memref type."""
    try:
        up_ir.MemRefType(ty)
        return True
    except (ValueError, TypeError):
        return False


def op_id(op: up_ir.OpView) -> int | None:
    """The ``nkipy.op_id`` integer stamped on an op, if present."""
    attrs = op.operation.attributes
    if "nkipy.op_id" not in attrs:
        return None
    try:
        return up_ir.IntegerAttr(attrs["nkipy.op_id"]).value
    except (ValueError, TypeError):
        return None


def is_hbm(ty: up_ir.Type) -> bool:
    """True if a memref lives in HBM (private or shared)."""
    return memref_memspace(ty) in (MEMSPACE_HBM, MEMSPACE_SHARED_HBM)


_VIEW_OPS = (
    "memref.subview", "memref.collapse_shape",
    "memref.expand_shape", "memref.reinterpret_cast",
)


def backing_alloc(value: up_ir.Value):
    """Trace ``value`` through view ops to the ``memref.alloc`` result backing
    it, or None (e.g. a function argument or a value with no alloc)."""
    owner = getattr(value, "owner", None)
    op = owner.opview if hasattr(owner, "opview") else owner
    if op is None:
        return None
    name = getattr(op, "name", None)
    if name == "memref.alloc":
        return op.operation.results[0]
    if name in _VIEW_OPS:
        return backing_alloc(op.operation.operands[0])
    return None


def func_ops(module: up_ir.Module) -> list[up_ir.OpView]:
    """All ``func.func`` ops at the top of ``module``."""
    return [
        op for op in module.body.operations
        if op.operation.name == "func.func"
    ]


def func_name(func: up_ir.OpView) -> str:
    return up_ir.StringAttr(func.attributes["sym_name"]).value


def const_int(value: up_ir.Value) -> int | None:
    """If ``value`` is defined by ``arith.constant <int>``, return the int.

    Index math is integer-only, so this rejects float constants (unlike
    :func:`const_scalar`).
    """
    v = const_scalar(value)
    return v if isinstance(v, int) else None


def const_scalar(value: up_ir.Value) -> int | float | None:
    """If ``value`` is an ``arith.constant`` (int or float), return its value."""
    owner = getattr(value, "owner", None)
    if owner is None:
        return None
    op = owner.opview if hasattr(owner, "opview") else owner
    if getattr(op, "name", None) != "arith.constant":
        return None
    attr = op.attributes["value"]
    try:
        return up_ir.IntegerAttr(attr).value
    except (ValueError, TypeError):
        pass
    try:
        return up_ir.FloatAttr(attr).value
    except (ValueError, TypeError):
        return None
