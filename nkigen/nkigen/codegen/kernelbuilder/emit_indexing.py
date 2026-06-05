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


def _dim_slices(gen, op, squeeze: bool) -> list[str]:
    """Per-dim index expressions for a ``memref.subview``.

    Each dim renders as one of:
      - ``":"``              full extent (offset 0, size == src dim)
      - ``"<expr>"``         a squeezed unit dim (size 1) -> integer index,
                             so kb drops the dim (only when ``squeeze``)
      - ``"<expr>:<expr>+n"``a sliced range otherwise

    ``squeeze`` collapses size-1 dims to integer indices. This is how a 4-D
    physical SBUF block ``[128, 1, 1, 128]`` becomes the 2-D ``[128, 128]`` tile
    that the compute ops consume (see the 4D-layout doc): the unit block dims
    are squeezed, the partition/free dims stay full.
    """
    static_offsets, static_sizes, _strides, dyn_offsets = _subview_components(op)
    src_shape = irutils.memref_shape(op.operation.operands[0].type)
    dyn_iter = iter(dyn_offsets)

    dims: list[str] = []
    for i, (off, size) in enumerate(zip(static_offsets, static_sizes)):
        off_expr = (
            index_expr(gen, next(dyn_iter), _PREC["+"])
            if off == DYN_SENTINEL else str(off)
        )
        full = off != DYN_SENTINEL and off == 0 and i < len(src_shape) and size == src_shape[i]
        if squeeze and size == 1:
            dims.append(off_expr)            # integer index -> kb squeezes the dim
        elif full:
            dims.append(":")
        elif off_expr == "0":
            dims.append(f"0:{size}")
        else:
            dims.append(f"{off_expr}:{off_expr} + {size}")
    return dims


def subview_slice(gen, op, squeeze: bool = False) -> str:
    """Render a ``memref.subview`` as a ``[d0, d1, ...]`` slice string."""
    return "[" + ", ".join(_dim_slices(gen, op, squeeze)) + "]"


# View ops that reinterpret storage without reindexing — passed through to the
# underlying value (the kb tile model has no equivalent op to emit).
_PASSTHROUGH_VIEW_OPS = (
    "memref.expand_shape",
    "memref.reinterpret_cast",
)


def _compose_subview_chain(gen, value):
    """Resolve a (possibly nested) subview chain to ``(base_value, offsets)``.

    legalize-layout can feed a compute operand through *two* stacked subviews
    (an outer one that picks one block dim and keeps another full, then an
    inner one that picks the remaining block dim). Chaining ``__getitem__`` in
    the generated Python does not compose correctly, so we instead walk the
    chain and accumulate, per base-memref dim, the index expression selecting
    that dim — yielding a single index list against the base alloc.

    Returns ``(base_value, offsets)`` where ``offsets[d]`` is the index
    expression for base dim ``d`` (``None`` means "full extent / not narrowed").
    """
    op = _defining_op(value)
    if op is None or op.operation.name != "memref.subview":
        return value, None

    base = op.operation.operands[0]
    base_offsets = None
    base_op = _defining_op(base)
    if base_op is not None and base_op.operation.name == "memref.subview":
        base, base_offsets = _compose_subview_chain(gen, base)

    static_offsets, static_sizes, _strides, dyn_offsets = _subview_components(op)
    dyn_iter = iter(dyn_offsets)

    # This subview's offsets are expressed in its *source*'s dim space. When the
    # source was itself a subview, its non-full dims line up with the same base
    # dims (the chain only ever narrows), so we merge index-wise: a dim narrowed
    # here overrides one left full by the outer subview.
    n = len(static_offsets)
    offsets = list(base_offsets) if base_offsets is not None else [None] * n
    if len(offsets) < n:
        offsets += [None] * (n - len(offsets))

    for i, (off, size) in enumerate(zip(static_offsets, static_sizes)):
        if off == DYN_SENTINEL:
            offsets[i] = index_expr(gen, next(dyn_iter), _PREC["+"])
        elif off != 0 or size == 1:
            # A non-zero static offset, or a size-1 selector, narrows this dim.
            if offsets[i] is None:
                offsets[i] = str(off)
    return base, offsets


def _collapse_to_2d_expr(gen, op) -> str:
    """Render ``memref.collapse_shape`` of a physical SBUF block as a 2-D tile.

    legalize-layout wraps each compute operand as
    ``collapse_shape(subview*(4D alloc))`` where the (possibly stacked)
    subviews select one block per block-dim and the collapse folds the unit
    block dims away to a 2-D ``[partTile, freeTile]`` tile. We reproduce that by
    composing the subview chain into one index list against the base alloc and
    squeezing the size-1 block dims to integer indices, which kb collapses for
    us — no reshape op needed.
    """
    base, offsets = _compose_subview_chain(gen, op.operation.operands[0])
    if offsets is None:
        # Collapse not fed by a subview (e.g. directly on an alloc).
        return memref_expr(gen, op.operation.operands[0])

    base_expr = memref_expr(gen, base)
    # Squeeze: a narrowed block dim -> integer index (kb drops it); a full dim
    # (partition / free) -> ``:``. When every block dim is squeezed, the
    # remaining ``:`` dims are the 2-D tile the compute op consumes.
    dims = [off if off is not None else ":" for off in offsets]
    return f"{base_expr}[{', '.join(dims)}]"


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
        if name == "memref.collapse_shape":
            return _collapse_to_2d_expr(gen, op)
        if name in _PASSTHROUGH_VIEW_OPS:
            # TODO: emit an explicit reshape when kb gains one; for now defer to
            # the source storage.
            return memref_expr(gen, op.operation.operands[0])

    return "None  # TODO: unresolved memref"
