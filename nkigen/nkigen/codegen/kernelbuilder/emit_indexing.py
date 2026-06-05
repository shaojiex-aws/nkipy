"""Render memref values and subviews as Python expressions.

This is the indexing core the memory/compute emitters build on:

- :func:`index_expr` turns an integer/index SSA value (loop IVs, arith.muli/
  addi chains, constants) into a readable Python arithmetic expression.
- :func:`subview_slice` turns a ``memref.subview`` into a ``[par, free]`` tile
  slice string.
- :func:`memref_expr` turns any memref SSA value into the Python expression
  that names it — a variable for allocs/args, or ``base[slice]`` for a
  subview, tracing through rank-preserving view ops.

All functions take ``gen`` (the ``_ModuleEmitter``) for its ``names`` map (SSA
value -> Python variable name).
"""

from __future__ import annotations

from . import irutils


# Dynamic-offset sentinel used by memref.subview's static_offsets attribute
# (INT64_MIN). A sentinel in slot i means "the offset for dim i is the next
# dynamic offset operand", not a literal.
DYN_SENTINEL = -(1 << 63)

# Operator precedence for parenthesizing arithmetic expressions only where
# needed (so we emit `i * 128 + 1`, not `((i * 128) + 1)`).
_PREC = {"+": 10, "-": 10, "*": 20, "//": 20, "%": 20}
_ARITH_BINOP = {
    "arith.addi": "+",
    "arith.subi": "-",
    "arith.muli": "*",
    "arith.divui": "//",
    "arith.divsi": "//",
    "arith.remui": "%",
    "arith.remsi": "%",
}


def _defining_op(value):
    owner = getattr(value, "owner", None)
    if owner is None:
        return None
    return owner.opview if hasattr(owner, "opview") else owner


def index_expr(gen, value, parent_prec: int = 0) -> str:
    """Render an index/integer SSA value as a Python expression.

    ``parent_prec`` is the precedence of the enclosing operator; a subexpression
    binding more loosely than its parent is wrapped in parentheses.
    """
    # A constant folds to its literal.
    c = irutils.const_int(value)
    if c is not None:
        return str(c)

    # A named value (loop IV, or anything we've already bound) uses its name.
    if value in gen.names:
        return gen.names[value]

    op = _defining_op(value)
    if op is not None:
        binop = _ARITH_BINOP.get(op.operation.name)
        if binop is not None:
            prec = _PREC[binop]
            lhs = index_expr(gen, op.operands[0], prec)
            rhs = index_expr(gen, op.operands[1], prec + 1)
            expr = f"{lhs} {binop} {rhs}"
            return f"({expr})" if prec < parent_prec else expr

    # Fallback: an index we can't trace (shouldn't happen for well-formed IR).
    return "0  # TODO: unresolved index"


def _subview_components(op):
    """Return (static_offsets, static_sizes, static_strides, dyn_offset_values).

    ``dyn_offset_values`` are the SSA operands that fill the sentinel slots in
    static_offsets, in order.
    """
    import mlir.ir as up_ir  # local import: keep module import-light

    def ints(attr_name):
        return [int(x) for x in op.operation.attributes[attr_name]]

    static_offsets = ints("static_offsets")
    static_sizes = ints("static_sizes")
    static_strides = ints("static_strides")
    # operands[0] is the source memref; the rest are dynamic offsets, then
    # dynamic sizes, then dynamic strides (per operandSegmentSizes). We only
    # consume dynamic offsets here; sizes/strides are static in practice.
    dyn_offsets = list(op.operation.operands)[1:]
    return static_offsets, static_sizes, static_strides, dyn_offsets


def subview_slice(gen, op) -> str:
    """Render a ``memref.subview`` as a ``[d0, d1, ...]`` slice string.

    A full-extent dim collapses to ``:``; a unit static offset emits
    ``off:off+size``; a dynamic offset emits ``<expr>:<expr> + size``.
    """
    static_offsets, static_sizes, static_strides, dyn_offsets = _subview_components(op)
    src_ty = op.operation.operands[0].type
    src_shape = irutils.memref_shape(src_ty)

    dyn_iter = iter(dyn_offsets)
    dims: list[str] = []
    for i, (off, size) in enumerate(zip(static_offsets, static_sizes)):
        if off == DYN_SENTINEL:
            off_expr = index_expr(gen, next(dyn_iter), _PREC["+"])
            dims.append(f"{off_expr}:{off_expr} + {size}")
            continue
        # Static offset.
        full = i < len(src_shape) and off == 0 and size == src_shape[i]
        if full:
            dims.append(":")
        elif off == 0:
            dims.append(f"0:{size}")
        else:
            dims.append(f"{off}:{off + size}")
    return "[" + ", ".join(dims) + "]"


# View ops that change shape but pass through the same storage. For now we
# render them as the underlying value (a TODO for proper reshape handling).
_PASSTHROUGH_VIEW_OPS = (
    "memref.collapse_shape",
    "memref.expand_shape",
    "memref.reinterpret_cast",
)


def memref_expr(gen, value) -> str:
    """Render a memref SSA value as the Python expression that refers to it."""
    if value in gen.names:
        return gen.names[value]

    op = _defining_op(value)
    if op is not None:
        name = op.operation.name
        if name == "memref.subview":
            base = memref_expr(gen, op.operation.operands[0])
            return f"{base}{subview_slice(gen, op)}"
        if name in _PASSTHROUGH_VIEW_OPS:
            # TODO: emit an explicit reshape when kb gains one; for now defer to
            # the source storage.
            return memref_expr(gen, op.operation.operands[0])

    return "None  # TODO: unresolved memref"
