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
    """The op that defines ``value``, or None if it is a block argument."""
    owner = getattr(value, "owner", None)
    # A block argument's owner is a Block (no defining op).
    if owner is None or not hasattr(owner, "opview"):
        return None
    return owner.opview


def _depends_on_loop_reg(gen, value) -> bool:
    """True if ``value`` transitively derives from a fori_loop induction Reg.

    Slices with such an offset must use ``nb.ds(...)`` (kb forbids Python
    slices indexed by a runtime Reg).
    """
    if value in gen.loop_regs:
        return True
    op = _defining_op(value)
    if op is None or op.operation.name not in _ARITH_BINOP:
        return False
    return any(_depends_on_loop_reg(gen, o) for o in op.operation.operands)


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
    """Return ``(static_offsets, static_sizes, dyn_offset_values)``.

    ``dyn_offset_values`` are the SSA operands that fill the sentinel slots in
    ``static_offsets``, in order. Strides are static (1) in the IR we consume,
    so they are not returned.
    """
    def ints(attr_name):
        return [int(x) for x in op.operation.attributes[attr_name]]

    static_offsets = ints("static_offsets")
    static_sizes = ints("static_sizes")
    # operands[0] is the source memref; the rest are dynamic offsets, then
    # dynamic sizes, then dynamic strides (per operandSegmentSizes). We only
    # consume dynamic offsets here; sizes/strides are static in practice.
    dyn_offsets = list(op.operation.operands)[1:]
    return static_offsets, static_sizes, dyn_offsets


# View ops that reinterpret storage without reindexing — passed through to the
# underlying value (the kb tile model has no equivalent op to emit).
_PASSTHROUGH_VIEW_OPS = (
    "memref.expand_shape",
    "memref.reinterpret_cast",
)


class _Dim:
    """One composed base-memref dim: an offset expression, its size, and
    whether the offset depends on a runtime fori_loop Reg.

    ``offset`` is None for a full-extent (offset-0) dim. Offsets compose
    *additively* down a subview chain (each inner subview's offset is relative
    to its source), so two stacked subviews on dim d sum to one expression.
    """
    __slots__ = ("offset", "size", "on_reg")

    def __init__(self, offset, size, on_reg):
        self.offset = offset
        self.size = size
        self.on_reg = on_reg


def _add_offsets(a: str | None, b: str | None) -> str | None:
    if a is None:
        return b
    if b is None:
        return a
    return f"{a} + {b}"


def _compose_subview_chain(gen, value):
    """Resolve a (possibly nested) ``memref.subview`` chain to ``(base, dims)``.

    Stacked subviews are composed into a *single* index list against the base
    memref — offsets summed per dim — rather than chained ``[...][...]`` (kb
    does not compose relative offsets across chained slices the way MLIR does).

    Returns ``(base_value, list[_Dim])``, or ``(value, None)`` if ``value`` is
    not produced by a subview.
    """
    op = _defining_op(value)
    if op is None or op.operation.name != "memref.subview":
        return value, None

    source = op.operation.operands[0]
    base, base_dims = _compose_subview_chain(gen, source)

    static_offsets, static_sizes, dyn_offsets = _subview_components(op)
    dyn_iter = iter(dyn_offsets)

    dims: list[_Dim] = []
    for i, (off, size) in enumerate(zip(static_offsets, static_sizes)):
        if off == DYN_SENTINEL:
            dyn_val = next(dyn_iter)
            this_off = index_expr(gen, dyn_val, _PREC["+"])
            this_reg = _depends_on_loop_reg(gen, dyn_val)
        elif off != 0:
            this_off, this_reg = str(off), False
        else:
            this_off, this_reg = None, False

        if base_dims is not None and i < len(base_dims):
            outer = base_dims[i]
            dims.append(_Dim(
                _add_offsets(outer.offset, this_off),
                size,                       # innermost size wins
                outer.on_reg or this_reg,
            ))
        else:
            dims.append(_Dim(this_off, size, this_reg))
    return base, dims


def _render_dim(dim: _Dim, full_size: int, squeeze: bool) -> str:
    """Render one composed dim as a slice token.

    ``squeeze`` drops a size-1 dim to a bare integer index (kb squeezes it) —
    used for the 4-D physical SBUF -> 2-D tile collapse.
    """
    off = dim.offset
    if squeeze and dim.size == 1:
        return off if off is not None else "0"
    if off is None and dim.size == full_size:
        return ":"
    if dim.on_reg:
        return f"nb.ds({off}, {dim.size})"
    base = off if off is not None else "0"
    return f"{base}:{base} + {dim.size}"


def _render_subview(gen, base, dims, squeeze: bool) -> str:
    base_expr = memref_expr(gen, base)
    base_shape = irutils.memref_shape(base.type)
    tokens = [
        _render_dim(d, base_shape[i] if i < len(base_shape) else -1, squeeze)
        for i, d in enumerate(dims)
    ]
    return f"{base_expr}[{', '.join(tokens)}]"


def subview_slice(gen, op) -> str:
    """Render a ``memref.subview`` (and any outer subviews) as one slice.

    Each dim is ``":"`` (full extent), ``"nb.ds(off, n)"`` (runtime-Reg offset),
    or ``"off:off + n"``.
    """
    base, dims = _compose_subview_chain(gen, op.operation.results[0])
    return _render_subview(gen, base, dims, squeeze=False)


def _collapse_to_2d_expr(gen, op) -> str:
    """Render ``memref.collapse_shape`` of a physical SBUF block as a 2-D tile.

    legalize-layout wraps each compute operand as
    ``collapse_shape(subview*(4D alloc))``; the subviews select one block per
    block dim and the collapse folds the unit block dims to a 2-D
    ``[partTile, freeTile]`` tile. We compose the subview chain and squeeze the
    size-1 block dims to integer indices, which kb collapses for us — no
    reshape op needed.
    """
    base, dims = _compose_subview_chain(gen, op.operation.operands[0])
    if dims is None:
        return memref_expr(gen, op.operation.operands[0])
    return _render_subview(gen, base, dims, squeeze=True)


def memref_expr(gen, value) -> str:
    """Render a memref SSA value as the Python expression that refers to it."""
    if value in gen.names:
        return gen.names[value]

    op = _defining_op(value)
    if op is not None:
        name = op.operation.name
        if name == "memref.subview":
            return subview_slice(gen, op)
        if name == "memref.collapse_shape":
            return _collapse_to_2d_expr(gen, op)
        if name in _PASSTHROUGH_VIEW_OPS:
            # TODO: emit an explicit reshape when kb gains one; for now defer to
            # the source storage.
            return memref_expr(gen, op.operation.operands[0])

    return "None  # TODO: unresolved memref"
