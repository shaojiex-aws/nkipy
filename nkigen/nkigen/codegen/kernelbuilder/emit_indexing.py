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
    """One composed dim of the base memref's index space.

    - ``offset``: index expression for the dim's start (None == 0 / full).
    - ``size``:   sliced extent.
    - ``on_reg``: offset depends on a fori_loop Reg (-> render as ``nb.ds``).
    - ``squeeze``: this is a unit dim to drop to a bare integer index so kb
      collapses it (the 4-D physical SBUF -> 2-D tile case).

    Offsets compose *additively* down a subview chain (each inner subview's
    offset is relative to its source).
    """
    __slots__ = ("offset", "size", "on_reg", "squeeze")

    def __init__(self, offset, size, on_reg=False, squeeze=False):
        self.offset = offset
        self.size = size
        self.on_reg = on_reg
        self.squeeze = squeeze


def _full_dim(size: int) -> _Dim:
    return _Dim(None, size)


def _add_offsets(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return f"{a} + {b}"


def _subview_dim(gen, off, size, dyn_iter) -> _Dim:
    """Build a _Dim from one (static_offset, size) entry of a subview."""
    if off == DYN_SENTINEL:
        dyn_val = next(dyn_iter)
        return _Dim(index_expr(gen, dyn_val, _PREC["+"]), size,
                    on_reg=_depends_on_loop_reg(gen, dyn_val))
    if off != 0:
        return _Dim(str(off), size)
    return _Dim(None, size)


def _reassoc_groups(op) -> list[list[int]]:
    """Reassociation index groups of an expand/collapse_shape op."""
    return [[int(x) for x in g] for g in op.operation.attributes["reassociation"]]


def _merge_dims(outer_dims, inner_dims):
    """Compose two same-rank _Dim lists: sum offsets, inner size wins."""
    if outer_dims is None:
        return inner_dims
    merged = []
    for i, d in enumerate(inner_dims):
        outer = outer_dims[i] if i < len(outer_dims) else None
        if outer is None:
            merged.append(d)
        else:
            merged.append(_Dim(_add_offsets(outer.offset, d.offset), d.size,
                               on_reg=outer.on_reg or d.on_reg,
                               squeeze=outer.squeeze))
    return merged


def _fold_subview_through_expand(expand_op, result_dims):
    """Fold a subview's result-space dims back to the expand's source space.

    ``expand_shape`` splits each source dim into a result group; a subview on
    the result selects one block per result dim. Merge a group's selections
    back to the source index: for group ``[r0, r1, ...]`` with result sizes
    ``[n0, n1, ...]`` the source offset is ``off_r0*(n1*n2..) + off_r1*(n2..) +
    ..`` and the source size is the product of the selected sizes. Returns one
    _Dim per source dim, or None if a selection is too complex to fold.
    """
    res_shape = irutils.memref_shape(expand_op.operation.results[0].type)
    src_shape = irutils.memref_shape(expand_op.operation.operands[0].type)
    out: list[_Dim] = []
    for sdim, group in enumerate(_reassoc_groups(expand_op)):
        sel = [result_dims[r] for r in group]
        # Full source dim: every result dim in the group is full.
        if all(d.offset is None and d.size == res_shape[r] for d, r in zip(sel, group)):
            out.append(_full_dim(src_shape[sdim]))
            continue
        terms, on_reg, size = [], False, 1
        for pos, (d, r) in enumerate(zip(sel, group)):
            on_reg = on_reg or d.on_reg
            size *= d.size
            if d.offset is None:
                continue
            stride = 1
            for later in group[pos + 1:]:
                stride *= res_shape[later]
            terms.append(f"{d.offset} * {stride}" if stride != 1 else d.offset)
        offset = " + ".join(terms) if terms else None
        out.append(_Dim(offset, size, on_reg=on_reg))
    return out


def _group_sizes(shape, group):
    return [shape[i] for i in group]


def _cross_collapse(op, result_dims):
    """Map result-space dims of a ``collapse_shape`` back to its source dims.

    Each source group merges to one result dim. With ≤1 non-unit dim per group
    the result index carries through to that dim (others are unit). For a true
    multi-dim merge it splits via div/mod over the group's sizes.
    """
    src_shape = irutils.memref_shape(op.operation.operands[0].type)
    src_dims = [None] * len(src_shape)
    for rdim, group in enumerate(_reassoc_groups(op)):
        d = result_dims[rdim] if rdim < len(result_dims) else _full_dim(0)
        sizes = _group_sizes(src_shape, group)
        non_unit = [g for g in group if src_shape[g] != 1]
        if len(non_unit) <= 1:
            carrier = non_unit[0] if non_unit else group[-1]
            for g in group:
                if g == carrier:
                    src_dims[g] = d
                else:
                    # A unit source dim merged into the result -> drop it (kb
                    # squeezes an integer index), so the carrier alone forms the
                    # 2-D tile the compute op consumes.
                    src_dims[g] = _Dim(None, 1, squeeze=True)
            continue
        # True merge: split d.offset across the group via div/mod. Only a full
        # (offset None) or pure-expression offset is splittable; sliced sizes
        # other than the whole result dim are not handled.
        if d.offset is None:
            for g in group:
                src_dims[g] = _full_dim(src_shape[g])
            continue
        idx = f"({d.offset})"
        for pos, g in enumerate(group):
            stride = 1
            for s in sizes[pos + 1:]:
                stride *= s
            comp = f"{idx} // {stride}" if stride != 1 else idx
            if pos != 0:
                comp = f"({comp}) % {src_shape[g]}"
            src_dims[g] = _Dim(comp, 1, on_reg=d.on_reg, squeeze=(src_shape[g] != 1))
    return src_dims


def _cross_expand(op, result_dims):
    """Map result-space dims of an ``expand_shape`` back to its source dims.

    Each source dim splits into a result group. With ≤1 non-unit result dim per
    group the non-unit result index carries through. For a true multi-dim split
    the source index is ``i0*s1*.. + i1*s2*.. + ..`` over the group.
    """
    res_shape = irutils.memref_shape(op.operation.results[0].type)
    src_shape = irutils.memref_shape(op.operation.operands[0].type)
    src_dims = [None] * len(src_shape)
    for sdim, group in enumerate(_reassoc_groups(op)):
        non_unit = [r for r in group if res_shape[r] != 1]
        if len(non_unit) <= 1:
            carrier = non_unit[0] if non_unit else group[-1]
            src_dims[sdim] = result_dims[carrier] if carrier < len(result_dims) else _full_dim(src_shape[sdim])
            continue
        terms, on_reg = [], False
        for pos, r in enumerate(group):
            d = result_dims[r]
            on_reg = on_reg or d.on_reg
            if d.offset is None:
                continue
            stride = 1
            for later in group[pos + 1:]:
                stride *= res_shape[later]
            terms.append(f"{d.offset} * {stride}" if stride != 1 else d.offset)
        merged = " + ".join(terms) if terms else None
        src_dims[sdim] = _Dim(merged, src_shape[sdim], on_reg=on_reg)
    return src_dims


def _compose_chain(gen, value):
    """Resolve a subview / reshape chain to ``(base, dims)``.

    Walks subview, collapse_shape and expand_shape ops, composing them into a
    single index list (``list[_Dim]``) against ``base`` — a value that is *not*
    itself a view op (an alloc, block arg, or anything ``memref_expr`` renders
    directly). This keeps ``memref_expr`` from re-entering the chain, so there
    is no mutual recursion.

    Returns ``(base, dims)``; ``dims`` is None if ``value`` isn't a view op or a
    reshape we can't remap (caller then renders ``value`` plainly).
    """
    op = _defining_op(value)
    if op is None:
        return value, None
    name = op.operation.name

    if name == "memref.subview":
        source = op.operation.operands[0]
        static_offsets, static_sizes, dyn_offsets = _subview_components(op)
        dyn_iter = iter(dyn_offsets)
        here = [_subview_dim(gen, off, size, dyn_iter)
                for off, size in zip(static_offsets, static_sizes)]

        # If the subview indexes a reshape *result*, fold its indices back to
        # the reshape's source space (merging split dims with i*size+j) before
        # composing further — the subview and reshape don't share a rank.
        src_op = _defining_op(source)
        if src_op is not None and src_op.operation.name == "memref.expand_shape":
            folded = _fold_subview_through_expand(src_op, here)
            if folded is None:
                return value, None
            base, base_dims = _compose_chain(gen, src_op.operation.operands[0])
            return base, _merge_dims(base_dims, folded)

        base, base_dims = _compose_chain(gen, source)
        if base_dims is None:
            return base, here
        return base, _merge_dims(base_dims, here)

    if name == "memref.collapse_shape":
        # Data flows source -> result. Compose the source; the source_dims
        # already carry any subview block offsets. The collapse only tells us
        # which *unit* source dims are merged away -> squeeze them. (This is the
        # common case: the collapse result feeds a compute op directly. A true
        # multi-dim merge that is then re-sliced is not handled here.)
        base, src_dims = _compose_chain(gen, op.operation.operands[0])
        src_shape = irutils.memref_shape(op.operation.operands[0].type)
        if src_dims is None:
            src_dims = [_full_dim(s) for s in src_shape]
        if len(src_dims) != len(src_shape):
            return value, None  # composed rank != collapse source rank — bail
        out = list(src_dims)
        for group in _reassoc_groups(op):
            non_unit = [g for g in group if src_shape[g] != 1]
            if len(non_unit) > 1:
                return value, None  # true merge feeding a slice — unsupported
            carrier = non_unit[0] if non_unit else group[-1]
            for g in group:
                if g != carrier and src_shape[g] == 1:
                    out[g] = _Dim(out[g].offset, 1, on_reg=out[g].on_reg, squeeze=True)
        return base, out

    if name == "memref.expand_shape":
        # Inverse: result has more dims. Compose the source; map each source
        # dim's index onto its result group's single non-unit dim.
        base, src_dims = _compose_chain(gen, op.operation.operands[0])
        src_shape = irutils.memref_shape(op.operation.operands[0].type)
        res_shape = irutils.memref_shape(op.operation.results[0].type)
        if src_dims is None:
            src_dims = [_full_dim(s) for s in src_shape]
        if len(src_dims) != len(src_shape):
            return value, None  # composed rank != expand source rank — bail
        out: list[_Dim] = []
        for sdim, group in enumerate(_reassoc_groups(op)):
            non_unit = [r for r in group if res_shape[r] != 1]
            if len(non_unit) > 1:
                return value, None  # true split — unsupported
            for r in group:
                out.append(src_dims[sdim] if res_shape[r] != 1
                           else _Dim(None, 1, squeeze=True))
        return base, out

    if name == "memref.reinterpret_cast":
        return _compose_chain(gen, op.operation.operands[0])

    return value, None


def _render_dim(dim: _Dim, full_size: int) -> str:
    off = dim.offset
    if dim.squeeze:
        return off if off is not None else "0"
    if off is None and dim.size == full_size:
        return ":"
    if dim.on_reg:
        return f"nb.ds({off}, {dim.size})"
    base = off if off is not None else "0"
    return f"{base}:{base} + {dim.size}"


def subview_slice(gen, op) -> str:
    """Render a ``memref.subview`` (composed with any outer views) as a slice."""
    return memref_expr(gen, op.operation.results[0])


def memref_expr(gen, value) -> str:
    """Render a memref SSA value as the Python expression that refers to it."""
    if value in gen.names:
        return gen.names[value]

    base, dims = _compose_chain(gen, value)
    if dims is None:
        # Not a view op (or an unsupported reshape): render the base directly.
        op = _defining_op(value)
        if op is not None and op.operation.name in _PASSTHROUGH_VIEW_OPS:
            return memref_expr(gen, op.operation.operands[0])
        if base is not value:
            return memref_expr(gen, base)
        # A reshape we can't remap (e.g. a true multi-dim split/merge then
        # sliced). Emit a parseable sentinel so the rest of the module still
        # parses; it surfaces as a NameError at exec, pinpointing the gap.
        return "UNSUPPORTED_RESHAPE"

    base_expr = memref_expr(gen, base)
    base_shape = irutils.memref_shape(base.type)
    tokens = [_render_dim(d, base_shape[i] if i < len(base_shape) else -1)
              for i, d in enumerate(dims)]
    return f"{base_expr}[{', '.join(tokens)}]"
