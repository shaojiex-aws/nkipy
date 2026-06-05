"""Python port of ResolveCustomOpsPass: inline stashed NISA custom-op
bodies at each call site."""

from __future__ import annotations

from ._vendor import nk_ir

def _resolve_custom_ops(module: nk_ir.Module, ctx: nk_ir.Context) -> None:
    """Python port of the deleted C++ ResolveCustomOpsPass.

    Replaces calls to custom-op declarations with the inlined NISA body
    stashed in the module attribute ``nkipy.custom_op_bodies`` (a dict of
    funcname → MLIR-text NISA body).  Matches both conventions:

    - *Output-as-argument* (`func @f(%in, %out) { return }`): allocate
      buffers for the trailing arguments and replace call results with
      the allocated buffers.
    - *Return-value* (kernel_builder, `func @f(%in) -> %out`): splice the
      body and rewire `func.return` operands to call results.

    Must run before ``_finalize_for_nki`` which strips ``nkipy.*`` attrs.
    """
    module_op = module.operation
    if "nkipy.custom_op_bodies" not in module_op.attributes:
        return
    bodies_attr = nk_ir.DictAttr(module_op.attributes["nkipy.custom_op_bodies"])

    # Collect custom-op declarations (body-less func.func with
    # `nkipy.custom_op` attr).
    decls: list[tuple[str, nk_ir.OpView]] = []
    for op in module.body.operations:
        if op.operation.name != "func.func":
            continue
        if "nkipy.custom_op" not in op.attributes:
            continue
        # A declaration has an empty region body.
        regions = list(op.operation.regions)
        if regions and list(regions[0].blocks):
            # Not a pure declaration; skip (body already present).
            continue
        sym = nk_ir.StringAttr(op.attributes["sym_name"]).value
        decls.append((sym, op))

    if not decls:
        del module_op.attributes["nkipy.custom_op_bodies"]
        return

    for func_name, decl_op in decls:
        # Look up the stashed NISA body string.
        if func_name not in bodies_attr:
            raise RuntimeError(
                f"no stashed NISA body for custom op '{func_name}'"
            )
        body_text = nk_ir.StringAttr(bodies_attr[func_name]).value
        body_module = nk_ir.Module.parse(body_text, ctx)

        # Find the (single) non-declaration func in the parsed body.
        nisa_func = None
        for bop in body_module.body.operations:
            if bop.operation.name != "func.func":
                continue
            regions = list(bop.operation.regions)
            if regions and list(regions[0].blocks):
                nisa_func = bop
                break
        if nisa_func is None:
            raise RuntimeError(
                f"no function body in stashed NISA module for '{func_name}'"
            )

        func_ty = nk_ir.TypeAttr(nisa_func.attributes["function_type"]).value
        num_results = len(list(func_ty.results))  # type: ignore[attr-defined]
        num_args = len(list(func_ty.inputs))  # type: ignore[attr-defined]
        # Output-names drives the output-as-argument split.
        if "nki.output_names" in nisa_func.attributes:
            num_outputs = len(list(
                nk_ir.ArrayAttr(nisa_func.attributes["nki.output_names"])
            ))
        else:
            num_outputs = 0

        is_return_value_style = num_results > 0
        if is_return_value_style:
            num_inputs = num_args
            num_outputs = num_results
        else:
            num_inputs = num_args - num_outputs

        # Collect all call sites for this custom op across the module.
        call_sites: list[nk_ir.OpView] = []

        def collect_calls(op: nk_ir.Operation) -> nk_ir.WalkResult:
            if op.name == "func.call":
                callee = nk_ir.FlatSymbolRefAttr(
                    op.attributes["callee"]
                ).value
                if callee == func_name:
                    call_sites.append(op.opview)
            return nk_ir.WalkResult.ADVANCE

        module.operation.walk(collect_calls)

        nisa_block = list(nisa_func.operation.regions[0].blocks)[0]
        nisa_args = list(nisa_block.arguments)

        for call_op in call_sites:
            call_operation = call_op.operation
            call_operands = list(call_operation.operands)

            with nk_ir.InsertionPoint(call_operation), call_operation.location:

                def maybe_cast(arg: nk_ir.Value, expected: nk_ir.Type) -> nk_ir.Value:
                    if str(arg.type) == str(expected):
                        return arg
                    v = arg
                    while True:
                        owner = getattr(v, "owner", None)
                        if owner is None:
                            break
                        owner_op = owner.opview if hasattr(owner, "opview") else owner
                        if getattr(owner_op, "name", None) != "memref.cast":
                            break
                        src = owner_op.operands[0]
                        if str(src.type) == str(expected):
                            return src
                        v = src
                    cast_op = nk_ir.Operation.create(
                        "memref.cast",
                        results=[expected],
                        operands=[arg],
                        loc=call_operation.location,
                    )
                    return cast_op.result

                # `pairs` maps every NISA-body Value that must be
                # rewritten (block args + cloned-op results) to its
                # replacement in the caller. Lookups are O(len(pairs))
                # which is fine for the short bodies we inline.
                pairs: list[tuple[nk_ir.Value, nk_ir.Value]] = []

                if is_return_value_style:
                    for nisa_arg, call_arg in zip(nisa_args, call_operands):
                        pairs.append(
                            (nisa_arg, maybe_cast(call_arg, nisa_arg.type))
                        )
                    return_operands: list[nk_ir.Value] = []
                    for body_op in list(nisa_block.operations):
                        if body_op.operation.name == "func.return":
                            for op_v in body_op.operation.operands:
                                replacement = op_v
                                for old, new in pairs:
                                    if op_v == old:
                                        replacement = new
                                        break
                                return_operands.append(replacement)
                            continue
                        _clone_op_with_map(body_op, pairs, pairs, ctx)
                    for i, retv in enumerate(return_operands):
                        call_op.results[i].replace_all_uses_with(retv)
                else:
                    # Input args come first.
                    for i in range(num_inputs):
                        pairs.append(
                            (nisa_args[i],
                             maybe_cast(call_operands[i], nisa_args[i].type))
                        )
                    # Allocate outputs for trailing NISA args.
                    out_allocs: list[nk_ir.Value] = []
                    for i in range(num_outputs):
                        nisa_out_ty = nisa_args[num_inputs + i].type
                        alloc_op = nk_ir.Operation.create(
                            "memref.alloc",
                            results=[nisa_out_ty],
                            operands=[],
                            attributes={
                                "operandSegmentSizes":
                                    nk_ir.DenseI32ArrayAttr.get([0, 0], ctx),
                            },
                            loc=call_operation.location,
                        )
                        pairs.append(
                            (nisa_args[num_inputs + i], alloc_op.result)
                        )
                        out_allocs.append(alloc_op.result)
                    for body_op in list(nisa_block.operations):
                        if body_op.operation.name == "func.return":
                            continue
                        _clone_op_with_map(body_op, pairs, pairs, ctx)
                    for i in range(num_outputs):
                        call_op.results[i].replace_all_uses_with(out_allocs[i])

            call_operation.erase()

        # Fix enclosing function return types after inlining — NISA-body
        # result types may differ from the caller's declared return type
        # (e.g. non-strided vs strided memrefs).
        for op in module.body.operations:
            if op.operation.name != "func.func":
                continue
            if "nkipy.custom_op" in op.attributes:
                continue
            regions = list(op.operation.regions)
            if not regions or not list(regions[0].blocks):
                continue
            block = list(regions[0].blocks)[0]
            term = block.operations[len(list(block.operations)) - 1]
            if term.operation.name != "func.return":
                continue
            ret_types = [v.type for v in term.operation.operands]
            func_ty_attr = op.attributes["function_type"]
            cur_ty = nk_ir.TypeAttr(func_ty_attr).value
            cur_results = list(cur_ty.results)  # type: ignore[attr-defined]
            if len(ret_types) == len(cur_results) and all(
                str(a) == str(b) for a, b in zip(ret_types, cur_results)
            ):
                continue
            new_inputs = list(cur_ty.inputs)  # type: ignore[attr-defined]
            new_ty = nk_ir.FunctionType.get(new_inputs, ret_types)
            op.attributes["function_type"] = nk_ir.TypeAttr.get(new_ty)

        decl_op.operation.erase()

    del module_op.attributes["nkipy.custom_op_bodies"]


def _clone_op_with_map(
    src_op: nk_ir.OpView,
    pairs: list[tuple[nk_ir.Value, nk_ir.Value]],
    results_pairs: list[tuple[nk_ir.Value, nk_ir.Value]],
    ctx: nk_ir.Context,
) -> None:
    """Clone ``src_op`` at the current insertion point, rebinding any
    operand appearing in ``pairs`` to its partner.  Each entry in
    ``pairs`` is (original_value, replacement_value); we do a linear
    scan on the clone's operands comparing by MLIR Value equality.
    Newly-produced results are appended to ``results_pairs`` so later
    operations in the same body can find them.
    """
    operation = src_op.operation
    # Clone inserts at the currently-active InsertionPoint.
    cloned = operation.clone()

    def find_replacement(v: nk_ir.Value) -> nk_ir.Value | None:
        for old, new in pairs:
            if v == old:
                return new
        return None

    def patch(op: nk_ir.Operation) -> nk_ir.WalkResult:
        for i in range(len(op.operands)):
            repl = find_replacement(op.operands[i])
            if repl is not None:
                op.operands[i] = repl
        return nk_ir.WalkResult.ADVANCE

    cloned.walk(patch)
    for i, r in enumerate(operation.results):
        results_pairs.append((r, cloned.results[i]))
