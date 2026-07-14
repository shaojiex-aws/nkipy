"""``knob(...).use(kernel)``: replace a traced subgraph with a kernel_builder kernel.

A ``knob(inputs..., outputs...).use(kernel)`` call names the *boundary* tensors of
a region. The role of each boundary is classified from the graph:

* a boundary that is a block arg, or is produced *outside* the region → **input**
* a boundary produced by ops *between* the inputs → **output**

Extraction cannot run at ``.use()`` call-time: escape analysis needs the whole
traced function, and ops after the ``.use()`` line aren't emitted yet. So ``.use()``
only records the boundary values in a module-level registry; ``extract_use_regions``
runs post-trace (before canonicalize/DCE, while the recorded ``ir.Value`` handles are
still valid) and does the real work: classify → find region → validate → bridge the
kernel to NISA text (via :class:`CustomOp`) → splice a ``func.call`` in.

The resulting ``func.call`` + body-less decl + stashed ``nkipy.custom_op_bodies`` attr
are exactly what ``_resolve_custom_ops`` inlines inside ``linalg_to_nisa`` — so this
front-end reuses the Phase-1 custom-op machinery with zero new pipeline passes.
"""

from __future__ import annotations

import inspect
from typing import Callable, List, Tuple

from mlir import ir
from mlir.dialects import func as func_d

from .custom_op import CustomOp, _get_registry
from ..mlir_utils import to_mlir_type


# Module-level registry of pending .use() regions. Single-threaded tracing, so
# no locking. Each entry: (boundary ir.Values in call order, kernel fn, verify).
_use_registry: List[Tuple[List[ir.Value], Callable, bool]] = []


def _get_use_registry() -> list:
    return _use_registry


def _clear_use_registry() -> None:
    _use_registry.clear()


def record_use(boundary_values: List[ir.Value], kernel: Callable, verify: bool) -> None:
    """Record a pending .use() region for post-trace extraction."""
    _use_registry.append((list(boundary_values), kernel, verify))


# ----------------------------------------------------------------------
# View / DPS helpers (Python ports of IRHelpers.cpp + CanonicalizePartitionDim)
# ----------------------------------------------------------------------

_VIEW_OPS = (
    "memref.subview",
    "memref.cast",
    "memref.reinterpret_cast",
    "memref.collapse_shape",
    "memref.expand_shape",
)


def _is_block_arg(value: ir.Value) -> bool:
    """True if ``value`` is a block argument.

    A value read back through ``op.operands`` is a plain ``ir.Value`` (not an
    ``ir.BlockArgument``) even when it names a block arg, so ``isinstance`` is
    unreliable. The owner distinguishes them: a block arg's owner is a Block,
    an op result's owner is an Operation.
    """
    owner = getattr(value, "owner", None)
    if owner is None:
        return False
    return not hasattr(owner, "operation")


def _defining_op(value: ir.Value):
    """Return the OpView that defines ``value``, or None for a block arg."""
    owner = getattr(value, "owner", None)
    if owner is None or not hasattr(owner, "operation"):
        return None
    return owner.opview if hasattr(owner, "opview") else owner


def _is_view(op) -> bool:
    return op is not None and op.operation.name in _VIEW_OPS


def _view_source(op) -> ir.Value:
    """The source memref a view op reads (operand 0)."""
    return op.operation.operands[0]


def _base_memref(value: ir.Value) -> ir.Value:
    """Walk a view chain (subview/cast/reshape) to the underlying base memref."""
    cur = value
    while True:
        op = _defining_op(cur)
        if not _is_view(op):
            return cur
        cur = _view_source(op)


def _value_in(value: ir.Value, values) -> bool:
    """Membership by MLIR value identity (== ), not Python ``id``.

    A value read through ``op.operands`` is a fresh Python handle for the same
    SSA value, so ``id()``/``is`` fail; MLIR ``==`` is the identity test.
    Regions are small, so a linear scan is fine.
    """
    for v in values:
        if v == value:
            return True
    return False


def _dps_split(op) -> Tuple[int, int]:
    """Return (num_ins, num_inits) for a linalg DPS op from operandSegmentSizes.

    Returns (0, 0) for a non-DPS op (no operandSegmentSizes attribute).
    """
    operation = op.operation
    try:
        oss = operation.attributes["operandSegmentSizes"]
    except KeyError:
        return (0, 0)
    vals = list(ir.DenseI32ArrayAttr(oss))
    if len(vals) != 2:
        return (0, 0)
    return (vals[0], vals[1])


def _is_linalg(op) -> bool:
    return op is not None and op.operation.name.startswith("linalg.")


def _dps_inits(op) -> List[ir.Value]:
    """The buffers a linalg op writes (its DPS init operands)."""
    num_ins, num_inits = _dps_split(op)
    if num_inits == 0:
        return []
    operands = list(op.operation.operands)
    return operands[num_ins:num_ins + num_inits]


def _dps_inputs(op) -> List[ir.Value]:
    """The buffers/scalars a linalg op reads (its DPS input operands)."""
    num_ins, _ = _dps_split(op)
    operands = list(op.operation.operands)
    return operands[:num_ins]


def _writers_of(value: ir.Value, func_op) -> List:
    """All linalg ops that write ``value`` (or an alias of it) as a DPS init.

    Alias-aware: a subview/reshape of ``value`` written by a linalg op counts.
    Unlike a single-writer lookup this returns BOTH the zero-fill and the matmul
    that share an accumulator buffer.
    """
    base = _base_memref(value)
    writers = []
    seen = set()

    def visit(op):
        opv = op.opview if hasattr(op, "opview") else op
        if _is_linalg(opv):
            for init in _dps_inits(opv):
                if _base_memref(init) == base:
                    key = opv.operation
                    if key not in seen:
                        seen.add(key)
                        writers.append(opv)
        return ir.WalkResult.ADVANCE

    func_op.operation.walk(visit)
    return writers


def _cone_bases(value: ir.Value, func_op) -> list:
    """Base memrefs strictly upstream of ``value`` (its backward dependency
    cone, excluding ``value`` itself).

    Used to classify boundaries: if boundary A's base appears in boundary B's
    cone, A feeds B, so A is an input and B is downstream. Walks DPS-input
    writers transitively; stops at block args.
    """
    bases: list = []
    visited_ops: set = set()

    def visit(v):
        for w in _writers_of(v, func_op):
            if w.operation in visited_ops:
                continue
            visited_ops.add(w.operation)
            for inp in _dps_inputs(w):
                if not ir.MemRefType.isinstance(inp.type):
                    continue  # skip scalar operands (e.g. fill value)
                b = _base_memref(inp)
                if not _value_in(b, bases):
                    bases.append(b)
                visit(inp)

    visit(value)
    return bases


def _readers_of(value: ir.Value):
    """All ops that read ``value`` or any alias of it (direct users of the
    value and of every view derived from it).

    Dedup is by *operation* (stable, hashable) and by MLIR value identity
    (``==``) for the view frontier — ``id()`` on ``ir.Value`` handles is
    unstable/reused, so it silently drops readers behind deep view chains.
    """
    readers = []
    reader_ops: set = set()
    frontier = [value]
    visited_vals: list = []
    while frontier:
        v = frontier.pop()
        if _value_in(v, visited_vals):
            continue
        visited_vals.append(v)
        for use in v.uses:
            user = use.owner
            userv = user.opview if hasattr(user, "opview") else user
            if userv.operation not in reader_ops:
                reader_ops.add(userv.operation)
                readers.append(userv)
            if _is_view(userv):
                frontier.append(userv.operation.results[0])
    return readers


def _mlir_elem_str(value: ir.Value) -> str:
    """Element-type string of a memref value (e.g. 'f32')."""
    return str(ir.MemRefType(value.type).element_type)


def _mlir_shape(value: ir.Value) -> tuple:
    return tuple(ir.MemRefType(value.type).shape)


def _nb_tensor_spec(shape: tuple, elem_str: str):
    """Build an ``nb.Tensor(shape, dtype, shared_hbm)`` spec for the bridge."""
    import nki.compiler.kernel_builder as nb
    dtype_map = {"f32": nb.float32, "f16": nb.float16, "bf16": nb.bfloat16}
    if elem_str not in dtype_map:
        raise ValueError(
            f".use(): unsupported element type '{elem_str}' at a region "
            f"boundary (supported: {sorted(dtype_map)})."
        )
    return nb.Tensor(tuple(shape), dtype_map[elem_str], nb.shared_hbm)


# ----------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------


def extract_use_regions(module: ir.Module, func_op) -> None:
    """Post-trace pass: replace each recorded ``.use()`` region with a
    ``func.call`` to a bridged custom op. Mutates ``module`` in place and
    registers the derived :class:`CustomOp`\\ s for declaration + resolution.

    Runs before ``run_canonicalize`` so recorded boundary ``ir.Value`` handles
    are still valid and per-op constants/fills are still private.
    """
    registry = _get_use_registry()
    if not registry:
        return

    claimed_ops: set = set()  # operation identities already in some region

    # Mutable copies of each marker's boundary list. Extracting one region
    # replaces its outputs' uses with a call result and erases the producing
    # ops, so a later marker that named such an output holds a stale handle —
    # remap those boundaries to the call result after each extraction.
    pending = [list(boundaries) for boundaries, _, _ in registry]

    for site_id, (_, kernel, verify) in enumerate(registry):
        if verify:
            raise NotImplementedError(
                ".use(verify=True): numeric region-vs-kernel verification is a "
                "planned follow-up; call .use(...) without verify for now."
            )
        replacements = _extract_one(
            module, func_op, pending[site_id], kernel, site_id, claimed_ops,
        )
        for later in range(site_id + 1, len(pending)):
            pending[later] = [
                _remap(v, replacements) for v in pending[later]
            ]


def _remap(value, replacements):
    """Return the replacement for ``value`` if it was rewritten, else itself."""
    for old, new in replacements:
        if old == value:
            return new
    return value


def _extract_one(module, func_op, boundaries, kernel, site_id, claimed_ops):
    ctx = module.context

    # 1. Classify boundaries into inputs vs outputs by graph position.
    #    A boundary is an INPUT if it is a block arg, or if it is upstream of
    #    another boundary (i.e. it feeds into another boundary's producing ops).
    #    It is an OUTPUT if nothing else in the boundary set depends on it.
    #    Example: knob(mm, y) with y = silu(mm) → mm feeds y so mm is the input
    #    and y the output, even though mm is itself produced by a (retained)
    #    matmul earlier in the trace.
    cones = [_cone_bases(v, func_op) for v in boundaries]
    inputs: List[ir.Value] = []
    outputs: List[ir.Value] = []
    for i, v in enumerate(boundaries):
        base_v = _base_memref(v)
        feeds_other = any(
            j != i and _value_in(base_v, cones[j]) for j in range(len(boundaries))
        )
        if _is_block_arg(v) or feeds_other:
            inputs.append(v)
        else:
            outputs.append(v)

    if not inputs:
        raise ValueError(
            ".use(): no inputs found — at least one knob() tensor must be a "
            "trace input or feed another boundary tensor."
        )
    if not outputs:
        raise ValueError(
            ".use(): no outputs found — at least one knob() tensor must be "
            "produced by ops between the inputs."
        )

    input_bases = [_base_memref(v) for v in inputs]

    # 2. Backward walk from outputs via DPS-init writers, through view ops,
    #    stopping at inputs / block args. Collect the region's ops.
    region_ops: List = []            # ordered, deduped by operation identity
    region_op_ids: set = set()
    internal_allocs: List = []

    def add_op(opv):
        key = opv.operation
        if key in region_op_ids:
            return False
        region_op_ids.add(key)
        region_ops.append(opv)
        return True

    def walk_back(value):
        # Stop at a declared input boundary (compare by base memref identity).
        if _value_in(_base_memref(value), input_bases):
            return
        if _is_block_arg(value):
            return
        # Pull in every writer of this buffer (fill + matmul, etc.).
        for w in _writers_of(value, func_op):
            if add_op(w):
                for inp in _dps_inputs(w):
                    walk_back(inp)
        # Follow the value's own defining view/alloc chain.
        defop = _defining_op(value)
        if _is_view(defop):
            if add_op(defop):
                walk_back(_view_source(defop))
        elif defop is not None and defop.operation.name == "memref.alloc":
            if defop.operation not in region_op_ids:
                region_op_ids.add(defop.operation)
                internal_allocs.append(defop)

    for out in outputs:
        walk_back(out)

    if not region_ops:
        raise ValueError(
            ".use(): the region is empty — no ops connect the declared inputs "
            f"to the outputs. An output that is a pure view of an input, or "
            f"produced only by a non-linalg op, isn't a replaceable region."
        )

    # Also capture the alloc + producing view chain of each internal buffer a
    # region op writes (so the buffers we erase include their allocs).
    for opv in list(region_ops):
        if _is_linalg(opv):
            for init in _dps_inits(opv):
                _collect_alloc(init, region_op_ids, internal_allocs)

    output_bases = [_base_memref(v) for v in outputs]

    # 3. Validate (returns interior annotation ops to erase with the region).
    hint_ops = _validate_region(
        region_ops, inputs, outputs, input_bases, output_bases,
        func_op, claimed_ops,
    )

    # 4. Bridge kernel -> CustomOp (NISA text). Specs from classified
    #    boundaries zipped with kernel param names (inputs first, then outputs).
    param_names = list(inspect.signature(kernel).parameters.keys())
    n_in, n_out = len(inputs), len(outputs)
    if len(param_names) != n_in + n_out:
        raise ValueError(
            f".use(): kernel '{kernel.__name__}' takes {len(param_names)} "
            f"parameters but the region has {n_in} input(s) + {n_out} "
            f"output(s) = {n_in + n_out} boundary tensors."
        )
    in_names = param_names[:n_in]
    out_names = param_names[n_in:]
    input_specs = {
        name: _nb_tensor_spec(_mlir_shape(v), _mlir_elem_str(v))
        for name, v in zip(in_names, inputs)
    }
    output_specs = {
        name: _nb_tensor_spec(_mlir_shape(v), _mlir_elem_str(v))
        for name, v in zip(out_names, outputs)
    }

    custom = CustomOp.from_kernel_builder(
        kernel_func=kernel,
        input_specs=input_specs,
        output_specs=output_specs,
    )

    # Register for declaration + body stashing (Phase-1 machinery).
    reg = _get_registry()
    if not any(op.func_name == custom.func_name for op in reg):
        reg.append(custom)

    # Mark ops claimed (disjointness across markers) before erasing.
    for opv in region_ops:
        claimed_ops.add(opv.operation)

    # 5. Rewrite: insert func.call, replace external uses of outputs, erase.
    #    Returns (output_value, call_result) pairs so later markers that named
    #    one of these outputs can be remapped to the call result.
    return _rewrite_region(
        ctx, func_op, region_ops, internal_allocs, hint_ops,
        inputs, outputs, custom,
    )


def _collect_alloc(value, region_op_ids, internal_allocs):
    """Add the alloc (and any view chain) backing ``value`` to internal_allocs."""
    cur = value
    while True:
        defop = _defining_op(cur)
        if defop is None:
            return
        name = defop.operation.name
        if name == "memref.alloc":
            if defop.operation not in region_op_ids:
                region_op_ids.add(defop.operation)
                internal_allocs.append(defop)
            return
        if _is_view(defop):
            if defop.operation not in region_op_ids:
                region_op_ids.add(defop.operation)
                internal_allocs.append(defop)
            cur = _view_source(defop)
            continue
        return


def _validate_region(region_ops, inputs, outputs, input_bases, output_bases,
                     func_op, claimed_ops):
    region_set = {opv.operation for opv in region_ops}

    # Disjointness across markers.
    for opv in region_ops:
        if opv.operation in claimed_ops:
            raise ValueError(
                ".use(): overlapping regions — an op is claimed by two "
                ".use() markers. Regions must be disjoint."
            )

    # Shared-base output check: if an output is a *view* (subview/reshape) of a
    # larger buffer, replacing only the view's uses leaves the base alive while
    # its producing ops get erased — a dangling base. Reject rather than miscompile.
    for out in outputs:
        base = _base_memref(out)
        if base == out:
            continue  # output is its own buffer, not a view
        for reader in _readers_of(base):
            if reader.operation in region_set:
                continue
            raise ValueError(
                f".use(): output {out} is a view of a buffer that is used "
                f"elsewhere; pass a whole tensor as the output, not a slice."
            )

    # Undeclared-input check: every non-constant memref operand a region op
    # reads must be produced inside the region or be a declared input.
    for opv in region_ops:
        if not _is_linalg(opv):
            continue
        for inp in _dps_inputs(opv):
            if not ir.MemRefType.isinstance(inp.type):
                continue  # scalar operand (e.g. fill value)
            base = _base_memref(inp)
            if _value_in(base, input_bases):
                continue
            producers = _writers_of(inp, func_op)
            if any(p.operation in region_set for p in producers):
                continue
            raise ValueError(
                f".use(): value {inp} feeds the region but is neither a "
                f"declared input nor produced inside it. Add it to knob()."
            )

    # Escape check: an internal buffer (not a declared output) read/written
    # outside the region escapes. nkipy.layout/tile_op are pure annotations on
    # the buffer — they're collected into the region and erased with it (see
    # _hint_ops_on), so they don't count as escapes here, but any *other*
    # external reader does.
    hint_ops = _hint_ops_on(region_ops, output_bases)
    for opv in region_ops:
        if not _is_linalg(opv):
            continue
        for init in _dps_inits(opv):
            if _value_in(_base_memref(init), output_bases):
                continue
            for reader in _readers_of(init):
                if reader.operation in region_set:
                    continue
                if reader.operation in hint_ops:
                    continue
                raise ValueError(
                    f".use(): value {init} is produced inside the region but "
                    f"used outside it. Add it to knob() as an output."
                )

    return hint_ops


def _hint_ops_on(region_ops, output_bases) -> set:
    """nkipy.layout / nkipy.tile_op ops annotating an *internal* buffer of the
    region (not a declared output). These are erased with the region — an
    annotation on a buffer that no longer exists would dangle."""
    hints: set = set()
    for opv in region_ops:
        if not _is_linalg(opv):
            continue
        for init in _dps_inits(opv):
            if _value_in(_base_memref(init), output_bases):
                continue
            for reader in _readers_of(init):
                if reader.operation.name in ("nkipy.layout", "nkipy.tile_op"):
                    hints.add(reader.operation)
    return hints


def _rewrite_region(ctx, func_op, region_ops, internal_allocs, hint_ops,
                   inputs, outputs, custom):
    region_set = {opv.operation for opv in region_ops}
    region_set.update(a.operation for a in internal_allocs)

    # Insertion point: right after the last region op in program order. The
    # last op writing an output dominates all external uses of that output.
    last_op = _last_in_block(region_ops, func_op)

    result_types = [v.type for v in outputs]
    loc = last_op.operation.location

    with _after(last_op.operation):
        call = func_d.CallOp(result_types, custom.func_name,
                             list(inputs), loc=loc)

    # Replace only EXTERNAL uses of each output buffer with the call result.
    replacements = list(zip(list(outputs), list(call.results)))
    for out_val, call_res in replacements:
        _replace_external_uses(out_val, call_res, region_set)

    # Erase interior annotation ops first (no results, so no dependents), then
    # region ops + internal allocs in reverse program order.
    for hint in hint_ops:
        hint.erase()
    to_erase = list(region_ops) + list(internal_allocs)
    for opv in _reverse_program_order(to_erase, func_op):
        opv.operation.erase()

    return replacements


def _after(operation):
    """InsertionPoint just after ``operation`` in its block."""
    block = operation.block
    ops = list(block.operations)
    idx = next(i for i, o in enumerate(ops) if o.operation == operation)
    if idx + 1 < len(ops):
        return ir.InsertionPoint(ops[idx + 1].operation)
    return ir.InsertionPoint(block)


def _last_in_block(region_ops, func_op):
    """Return the region op that appears last in program order."""
    order = _program_index(func_op)
    return max(region_ops, key=lambda o: order[o.operation])


def _reverse_program_order(ops, func_op):
    order = _program_index(func_op)
    return sorted(ops, key=lambda o: order.get(o.operation, -1), reverse=True)


def _program_index(func_op) -> dict:
    """Map each operation to a global program-order index via a pre-order walk."""
    index = {}
    counter = [0]

    def visit(op):
        index[op] = counter[0]
        counter[0] += 1
        return ir.WalkResult.ADVANCE

    func_op.operation.walk(visit)
    return index


def _replace_external_uses(old_val, new_val, region_set):
    """Replace uses of ``old_val`` that are NOT in ``region_set``."""
    for use in list(old_val.uses):
        user = use.owner
        op = user if not hasattr(user, "opview") else user.opview
        if op.operation in region_set:
            continue
        use.owner.operation.operands[use.operand_number] = new_val
