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


# "<n> : i32" memory-space marker -> int (matches MemSpaceEnum in NkipyAttrs.td).
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


def is_memref(ty: up_ir.Type) -> bool:
    try:
        up_ir.MemRefType(ty)
        return True
    except (ValueError, TypeError):
        return False


def func_ops(module: up_ir.Module) -> list[up_ir.OpView]:
    """All ``func.func`` ops at the top of ``module``."""
    return [
        op for op in module.body.operations
        if op.operation.name == "func.func"
    ]


def func_signature(func: up_ir.OpView) -> tuple[list[up_ir.Type], list[up_ir.Type]]:
    """``(input_types, result_types)`` of a ``func.func`` op."""
    ft = up_ir.TypeAttr(func.attributes["function_type"]).value
    return list(ft.inputs), list(ft.results)


def func_name(func: up_ir.OpView) -> str:
    return up_ir.StringAttr(func.attributes["sym_name"]).value


def const_int(value: up_ir.Value) -> int | None:
    """If ``value`` is defined by ``arith.constant <int>``, return the int."""
    owner = getattr(value, "owner", None)
    if owner is None:
        return None
    op = owner.opview if hasattr(owner, "opview") else owner
    if getattr(op, "name", None) != "arith.constant":
        return None
    try:
        return up_ir.IntegerAttr(op.attributes["value"]).value
    except (KeyError, ValueError, TypeError):
        return None


def op_id(op: up_ir.OpView) -> int | None:
    """The ``nkipy.op_id`` integer stamped on an op, if present."""
    attrs = op.operation.attributes
    if "nkipy.op_id" not in attrs:
        return None
    try:
        return up_ir.IntegerAttr(attrs["nkipy.op_id"]).value
    except (ValueError, TypeError):
        return None
