"""linalg.generic dispatch: reductions, broadcasts, unary activations,
identity copies, type casts, powf -> NISA ops."""

from __future__ import annotations

from ._vendor import nk_ir, nisa

from .access import _get_base_and_offsets
from .affine_map import (
    _build_nisa_map,
    _empty_operand_kwargs,
    _operand_kwargs,
    _scalar_operand_kwargs,
)
from .activation import _emit_activation
from .patterns import (
    _ARITH_TO_CROSS_LANE,
    _REDUCE_BODY_OP_TO_ARITH,
    _RewriteContext,
    _enclosing_block,
    _is_hbm,
    _is_psum,
    _is_sbuf,
    _static_shape,
    pattern,
)

_BODY_OP_TO_ARITH = {
    "arith.addf": nisa.ArithOp.Add,
    "arith.addi": nisa.ArithOp.Add,
    "arith.subf": nisa.ArithOp.Subtract,
    "arith.subi": nisa.ArithOp.Subtract,
    "arith.mulf": nisa.ArithOp.Multiply,
    "arith.muli": nisa.ArithOp.Multiply,
    "arith.divf": nisa.ArithOp.Divide,
    "arith.divsi": nisa.ArithOp.DivideInt,
    "arith.divui": nisa.ArithOp.DivideInt,
    "arith.remf": nisa.ArithOp.Mod,
    "arith.remsi": nisa.ArithOp.ModInt,
}

_CMPF_PRED_TO_ARITH = {
    1: nisa.ArithOp.IsEQ,
    2: nisa.ArithOp.IsGT,
    3: nisa.ArithOp.IsGE,
    4: nisa.ArithOp.IsLT,
    5: nisa.ArithOp.IsLE,
    6: nisa.ArithOp.IsNE,
}

_CMPI_PRED_TO_ARITH = {
    0: nisa.ArithOp.IsEQ,
    1: nisa.ArithOp.IsNE,
    2: nisa.ArithOp.IsLT,
    3: nisa.ArithOp.IsLE,
    4: nisa.ArithOp.IsGT,
    5: nisa.ArithOp.IsGE,
}

_BODY_MATH_TO_ACTIVATION = {
    "math.sin": nisa.ActivationFunction.sin,
    "math.copysign": nisa.ActivationFunction.sign,
}


def _defining_op(v: nk_ir.Value):
    owner = getattr(v, "owner", None)
    if owner is None:
        return None
    return owner.opview if hasattr(owner, "opview") else owner


def _predicate_int(op):
    attrs = op.attributes
    if "predicate" not in attrs:
        return None
    return nk_ir.IntegerAttr(attrs["predicate"]).value


def _is_constant_value(v: nk_ir.Value) -> bool:
    owner = getattr(v, "owner", None)
    if owner is None:
        return False
    op = owner.opview if hasattr(owner, "opview") else owner
    return getattr(op, "name", None) == "arith.constant"


def _shape_match_broadcast(in_shape, out_shape):
    if len(in_shape) != len(out_shape):
        return False
    had_broadcast = False
    for i, o in zip(in_shape, out_shape):
        if i == 1 and o > 1:
            had_broadcast = True
        elif i != o:
            return False
    return had_broadcast


def _analyze_generic_body(op: nk_ir.OpView):
    region = op.regions[0]
    block = region.blocks[0]
    ops = list(block.operations)
    if not ops:
        return None
    yield_op = ops[-1]
    if yield_op.name != "linalg.yield":
        return None
    yielded = list(yield_op.operands)
    if len(yielded) != 1:
        return None
    root = _defining_op(yielded[0])
    if root is None or not hasattr(root, "name"):
        return None
    name = root.name
    kind = _BODY_OP_TO_ARITH.get(name)
    if kind is not None:
        return kind, root.operands[0], root.operands[1]
    if name == "arith.uitofp":
        inner = _defining_op(root.operands[0])
        if inner is None:
            return None
        if inner.name == "arith.cmpf":
            pred = _predicate_int(inner)
            k = _CMPF_PRED_TO_ARITH.get(pred) if pred is not None else None
            if k is not None:
                return k, inner.operands[0], inner.operands[1]
        if inner.name in ("arith.andi", "arith.ori"):
            logical_kind = (
                nisa.ArithOp.LogicalAnd if inner.name == "arith.andi"
                else nisa.ArithOp.LogicalOr
            )
            lhs_cast = _defining_op(inner.operands[0])
            rhs_cast = _defining_op(inner.operands[1])
            if (lhs_cast and rhs_cast and
                lhs_cast.name == "arith.fptoui" and
                rhs_cast.name == "arith.fptoui"):
                return logical_kind, lhs_cast.operands[0], rhs_cast.operands[0]
        return None
    if name == "arith.extui":
        inner = _defining_op(root.operands[0])
        if inner is None or inner.name != "arith.cmpi":
            return None
        pred = _predicate_int(inner)
        k = _CMPI_PRED_TO_ARITH.get(pred) if pred is not None else None
        if k is not None:
            return k, inner.operands[0], inner.operands[1]
    return None


def _match_generic_powf(op: nk_ir.OpView):
    region = op.regions[0]
    block = region.blocks[0]
    ops = list(block.operations)
    if len(ops) != 2:
        return None
    inner, yield_op = ops[0], ops[1]
    if inner.name != "math.powf" or yield_op.name != "linalg.yield":
        return None
    if list(yield_op.operands) != [inner.results[0]]:
        return None
    return inner.operands[0], inner.operands[1]


def _match_reduction_body(op: nk_ir.OpView):
    region = op.regions[0]
    block = region.blocks[0]
    ops = list(block.operations)
    if not ops or ops[-1].name != "linalg.yield":
        return None
    body_ops = ops[:-1]
    found = None
    for bo in body_ops:
        k = _REDUCE_BODY_OP_TO_ARITH.get(bo.name)
        if k is not None:
            if found is not None:
                return None
            found = (k, bo)
    return found


def _classify_reduction(op: nk_ir.OpView):
    attrs = op.operation.attributes
    it_str = str(attrs["iterator_types"])
    kinds = []
    for token in it_str.split("#linalg.iterator_type<"):
        close = token.find(">")
        if close < 0:
            continue
        k = token[:close]
        if k in ("parallel", "reduction"):
            kinds.append(k)
    if not kinds:
        return None
    num_red = sum(1 for k in kinds if k == "reduction")
    if num_red == 0:
        return None
    is_left = all(kinds[i] == "reduction" for i in range(num_red))
    is_right = all(kinds[-1 - i] == "reduction" for i in range(num_red))
    if not (is_left or is_right):
        return None
    return num_red, is_left, is_right


def _rewrite_linalg_generic_reduction(
    rctx: _RewriteContext, op: nk_ir.OpView, num_ins: int
) -> bool:
    if num_ins != 1:
        return False
    operands = list(op.operation.operands)
    src = operands[0]
    dst = operands[1]

    match = _match_reduction_body(op)
    if match is None:
        return False
    arith_kind, inner = match

    block = op.regions[0].blocks[0]
    block_args = list(block.arguments)
    out_block_arg = block_args[-1]
    in0 = inner.operands[0]
    in1 = inner.operands[1]
    if not (str(in0) == str(out_block_arg) or str(in1) == str(out_block_arg)):
        return False

    classified = _classify_reduction(op)
    if classified is None:
        return False
    num_red_dims, is_left, is_right = classified

    src_ty = src.type
    dst_ty = dst.type
    if not _is_sbuf(src_ty):
        return False
    if not (_is_sbuf(dst_ty) or _is_psum(dst_ty)):
        return False
    dst_shape = _static_shape(dst_ty)
    src_shape = _static_shape(src_ty)
    if dst_shape is None or src_shape is None:
        return False

    with nk_ir.InsertionPoint(op), rctx.loc:
        src_acc = _get_base_and_offsets(rctx.ctx, src, rctx.loc)
        dst_acc = _get_base_and_offsets(rctx.ctx, dst, rctx.loc)

        if is_left:
            cross_op = _ARITH_TO_CROSS_LANE.get(arith_kind)
            if cross_op is None:
                return False
            # cross_lane_reduce_arith uses each operand's data shape as its
            # iteration/tile domain. src spans the full input (parallel +
            # partition reduction dim), dst spans the output. Using dst_shape
            # for src tile (as before) made the hardware reduce a 1-wide
            # slice, returning 0 for axis=0 sums.
            src_map = _build_nisa_map(rctx.ctx, len(src_shape), src_acc)
            dst_map = _build_nisa_map(rctx.ctx, len(dst_shape), dst_acc)
            nisa.cross_lane_reduce_arith(
                **_operand_kwargs("dst", dst_acc, dst_map, dst_shape),
                **_operand_kwargs("src", src_acc, src_map, src_shape),
                reduce_op=cross_op,
                num_r_dim=0,
                engine=nisa.Engine.Gpsimd,
            )
            op.operation.erase()
            return True

        # Rightmost reduction: tensor_reduce_arith into temp, then
        # tensor_tensor_arith to accumulate into dst.
        #
        # tensor_reduce_arith's iteration domain spans the SOURCE shape
        # (parallel dims + reduction dims), so affine maps for both src and
        # temp_dst have `numSrcIterDims = len(src_shape)` dimensions. Tile
        # shapes reflect each operand's own shape: src_shape for src,
        # dst_shape for the dst/temp. Using dst_shape for src (as before)
        # made the hardware only reduce a 1-wide slice, giving wrong sums.
        num_src_iter_dims = len(src_shape)

        src_map = _build_nisa_map(rctx.ctx, num_src_iter_dims, src_acc)
        dst_reduce_map = _build_nisa_map(rctx.ctx, num_src_iter_dims, dst_acc)

        temp_ty = nk_ir.MemRefType.get(
            dst_shape,
            dst_ty.element_type,  # type: ignore[attr-defined]
            memory_space=dst_ty.memory_space,  # type: ignore[attr-defined]
        )
        temp_val = nisa.alloc(memref_type=temp_ty, alignment=0)
        temp_acc = _get_base_and_offsets(rctx.ctx, temp_val, rctx.loc)
        temp_reduce_map = _build_nisa_map(
            rctx.ctx, num_src_iter_dims, temp_acc,
        )

        nisa.tensor_reduce_arith(
            **_operand_kwargs("dst", temp_acc, temp_reduce_map, dst_shape),
            **_operand_kwargs("src", src_acc, src_map, src_shape),
            op=arith_kind,
            negated=False,
            num_r_dim=num_red_dims,
            engine=nisa.Engine.Vector,
        )

        # Accumulation uses the dst iteration domain only (parallel dims).
        num_dst_iter_dims = len(dst_shape)
        dst_accum_map = _build_nisa_map(rctx.ctx, num_dst_iter_dims, dst_acc)
        temp_accum_map = _build_nisa_map(
            rctx.ctx, num_dst_iter_dims, temp_acc,
        )
        nisa.tensor_tensor_arith(
            **_operand_kwargs("dst", dst_acc, dst_accum_map, dst_shape),
            **_operand_kwargs("lhs", dst_acc, dst_accum_map, dst_shape),
            **_operand_kwargs("rhs", temp_acc, temp_accum_map, dst_shape),
            op=arith_kind,
            engine=nisa.Engine.Vector,
        )
        nisa.release(memref=temp_val)

    op.operation.erase()
    return True


def _match_generic_unary_activation(op: nk_ir.OpView):
    region = op.regions[0]
    block = region.blocks[0]
    ops = list(block.operations)
    if len(ops) != 2:
        return None
    inner, yield_op = ops[0], ops[1]
    if yield_op.name != "linalg.yield":
        return None
    if list(yield_op.operands) != [inner.results[0]]:
        return None
    return _BODY_MATH_TO_ACTIVATION.get(inner.name)


def _match_generic_identity_body(op: nk_ir.OpView) -> bool:
    """True if the generic's body yields the first block argument directly.

    Mirrors the C++ LinalgGenericIdentityCopyPattern. Uses `walk()` to find
    the yield op — touching `block.operations` (by iteration or index)
    corrupts the NKI Python binding's iterator state and breaks later
    matchers on the same generic.
    """
    block = op.regions[0].blocks[0]
    yield_op: list[nk_ir.Operation] = []

    def visit(o: nk_ir.Operation) -> nk_ir.WalkResult:
        if o.name == "linalg.yield":
            yield_op.append(o)
            return nk_ir.WalkResult.INTERRUPT
        return nk_ir.WalkResult.ADVANCE

    op.operation.walk(visit)
    if not yield_op:
        return False
    terminator = yield_op[0]
    operands = list(terminator.operands)
    if len(operands) != 1:
        return False
    args = list(block.arguments)
    if not args:
        return False
    return operands[0] == args[0]


def _emit_copy_from_identity_generic(
    rctx: _RewriteContext,
    op: nk_ir.OpView,
    src: nk_ir.Value,
    dst: nk_ir.Value,
) -> None:
    """Lower an identity-body linalg.generic to nisa.tensor_copy / dma_copy.

    Same lowering strategy as _rewrite_memref_copy / _rewrite_linalg_copy.
    """
    src_ty, dst_ty = src.type, dst.type
    src_hbm, dst_hbm = _is_hbm(src_ty), _is_hbm(dst_ty)
    src_sbuf, dst_sbuf = _is_sbuf(src_ty), _is_sbuf(dst_ty)
    src_psum, dst_psum = _is_psum(src_ty), _is_psum(dst_ty)

    needs_dma = src_hbm or dst_hbm
    on_tpb = (
        (src_sbuf and dst_sbuf) or (src_sbuf and dst_psum) or (src_psum and dst_sbuf)
    )
    if not (needs_dma or on_tpb):
        return

    shape = _static_shape(dst_ty)
    if shape is None:
        return

    with nk_ir.InsertionPoint(op), rctx.loc:
        src_acc = _get_base_and_offsets(rctx.ctx, src, rctx.loc)
        dst_acc = _get_base_and_offsets(rctx.ctx, dst, rctx.loc)
        rank = len(shape)
        src_map = _build_nisa_map(rctx.ctx, rank, src_acc)
        dst_map = _build_nisa_map(rctx.ctx, rank, dst_acc)
        if needs_dma:
            nisa.dma_copy(
                **_operand_kwargs("dst", dst_acc, dst_map, shape),
                **_operand_kwargs("src", src_acc, src_map, shape),
            )
        else:
            nisa.tensor_copy(
                **_operand_kwargs("dst", dst_acc, dst_map, shape),
                **_operand_kwargs("src", src_acc, src_map, shape),
                engine=nisa.Engine.Vector,
            )
    op.operation.erase()


def _match_generic_type_cast(op: nk_ir.OpView) -> bool:
    region = op.regions[0]
    block = region.blocks[0]
    ops = list(block.operations)
    if len(ops) != 2:
        return False
    inner, yield_op = ops[0], ops[1]
    if inner.name not in ("arith.sitofp", "arith.fptosi"):
        return False
    if yield_op.name != "linalg.yield":
        return False
    if list(yield_op.operands) != [inner.results[0]]:
        return False
    return True


@pattern("linalg.generic")
def _rewrite_linalg_generic(rctx: _RewriteContext, op: nk_ir.OpView) -> None:
    attrs = op.operation.attributes
    if "operandSegmentSizes" not in attrs:
        return
    seg_str = str(attrs["operandSegmentSizes"])
    try:
        nums = [int(x.strip()) for x in seg_str.split(":")[1].strip(" >").split(",")]
        num_ins, num_outs = nums[0], nums[1]
    except (ValueError, IndexError):
        return

    if num_outs != 1:
        return

    if "iterator_types" not in attrs:
        return
    if "reduction" in str(attrs["iterator_types"]):
        _rewrite_linalg_generic_reduction(rctx, op, num_ins)
        return

    operands = list(op.operation.operands)
    inputs = operands[:num_ins]
    output = operands[num_ins]
    out_ty = output.type
    out_shape = _static_shape(out_ty)

    # Identity-body generic (body is just `linalg.yield %arg0`) — lower to
    # a copy. Ported from the pre-open-source LinalgGenericIdentityCopyPattern
    # in LinalgToNisa.cpp. Arises from broadcast_to in the tracer, and from
    # trivial transposes reconstructed by legalize-layout.
    if (num_ins == 1
            and "reduction" not in str(attrs["iterator_types"])
            and _match_generic_identity_body(op)):
        _emit_copy_from_identity_generic(rctx, op, inputs[0], output)
        return

    if num_ins == 1:
        act_kind = _match_generic_unary_activation(op)
        if act_kind is not None:
            if _emit_activation(rctx, op, inputs[0], output, act_kind):
                op.operation.erase()
                return

    if num_ins == 1 and _match_generic_type_cast(op):
        if (
            "parallel" in str(attrs["iterator_types"])
            and "reduction" not in str(attrs["iterator_types"])
            and out_shape is not None
            and _is_sbuf(inputs[0].type)
            and _is_sbuf(out_ty)
        ):
            zero = rctx.f32_const(_enclosing_block(op), 0.0)
            with nk_ir.InsertionPoint(op), rctx.loc:
                src_acc = _get_base_and_offsets(rctx.ctx, inputs[0], rctx.loc)
                dst_acc = _get_base_and_offsets(rctx.ctx, output, rctx.loc)
                rank = len(out_shape)
                src_map = _build_nisa_map(rctx.ctx, rank, src_acc)
                dst_map = _build_nisa_map(rctx.ctx, rank, dst_acc)
                nisa.tensor_scalar_arith(
                    **_operand_kwargs("dst", dst_acc, dst_map, out_shape),
                    **_operand_kwargs("src", src_acc, src_map, out_shape),
                    **_scalar_operand_kwargs("operand0", zero),
                    **_empty_operand_kwargs("operand1"),
                    op0=nisa.ArithOp.Add,
                    op1=None,
                    reverse_operands=nisa.TensScalarRevOps.None_,
                    engine=nisa.Engine.Vector,
                )
            op.operation.erase()
            return

    pow_match = _match_generic_powf(op) if num_ins == 2 else None
    if pow_match is not None and out_shape is not None:
        base, exp_v = pow_match
        block_arg0 = op.regions[0].blocks[0].arguments[0]
        swap = str(base) != str(block_arg0)
        base_v = inputs[1] if swap else inputs[0]
        exp_v = inputs[0] if swap else inputs[1]
        if (_is_sbuf(base_v.type) and _is_sbuf(exp_v.type)
                and _is_sbuf(out_ty)):
            with nk_ir.InsertionPoint(op), rctx.loc:
                base_acc = _get_base_and_offsets(rctx.ctx, base_v, rctx.loc)
                exp_acc = _get_base_and_offsets(rctx.ctx, exp_v, rctx.loc)
                dst_acc = _get_base_and_offsets(rctx.ctx, output, rctx.loc)
                rank = len(out_shape)
                nisa.tensor_tensor_power(
                    **_operand_kwargs(
                        "dst", dst_acc, _build_nisa_map(rctx.ctx, rank, dst_acc),
                        out_shape,
                    ),
                    **_operand_kwargs(
                        "lhs", base_acc, _build_nisa_map(rctx.ctx, rank, base_acc),
                        out_shape,
                    ),
                    **_operand_kwargs(
                        "rhs", exp_acc, _build_nisa_map(rctx.ctx, rank, exp_acc),
                        out_shape,
                    ),
                    engine=nisa.Engine.Gpsimd,
                )
            op.operation.erase()
            return

    analysis = _analyze_generic_body(op)
    if analysis is None or out_shape is None:
        return
    arith_kind, body_lhs, body_rhs = analysis

    if not _is_sbuf(out_ty):
        return

    if num_ins == 1:
        input_v = inputs[0]
        if not _is_sbuf(input_v.type):
            return
        lhs_const = _is_constant_value(body_lhs)
        rhs_const = _is_constant_value(body_rhs)
        if lhs_const == rhs_const:
            return
        scalar_v = body_lhs if lhs_const else body_rhs
        reverse = (
            nisa.TensScalarRevOps.First if lhs_const else nisa.TensScalarRevOps.None_
        )
        with nk_ir.InsertionPoint(op), rctx.loc:
            src_acc = _get_base_and_offsets(rctx.ctx, input_v, rctx.loc)
            dst_acc = _get_base_and_offsets(rctx.ctx, output, rctx.loc)
            rank = len(out_shape)
            src_map = _build_nisa_map(rctx.ctx, rank, src_acc)
            dst_map = _build_nisa_map(rctx.ctx, rank, dst_acc)
            nisa.tensor_scalar_arith(
                **_operand_kwargs("dst", dst_acc, dst_map, out_shape),
                **_operand_kwargs("src", src_acc, src_map, out_shape),
                **_scalar_operand_kwargs("operand0", scalar_v),
                **_empty_operand_kwargs("operand1"),
                op0=arith_kind,
                op1=None,
                reverse_operands=reverse,
                engine=nisa.Engine.Vector,
            )
        op.operation.erase()
        return

    # num_ins == 2
    in0, in1 = inputs[0], inputs[1]
    in0_shape = _static_shape(in0.type)
    in1_shape = _static_shape(in1.type)
    if in0_shape is None or in1_shape is None:
        return
    if not (_is_sbuf(in0.type) and _is_sbuf(in1.type)):
        return
    if _is_constant_value(body_lhs) or _is_constant_value(body_rhs):
        return

    in0_bcast = _shape_match_broadcast(in0_shape, out_shape)
    in1_bcast = _shape_match_broadcast(in1_shape, out_shape)

    if in0_shape == out_shape and in1_shape == out_shape:
        block_arg0 = op.regions[0].blocks[0].arguments[0]
        swap = str(body_lhs) != str(block_arg0)
        lhs_v = in1 if swap else in0
        rhs_v = in0 if swap else in1
        with nk_ir.InsertionPoint(op), rctx.loc:
            lhs_acc = _get_base_and_offsets(rctx.ctx, lhs_v, rctx.loc)
            rhs_acc = _get_base_and_offsets(rctx.ctx, rhs_v, rctx.loc)
            dst_acc = _get_base_and_offsets(rctx.ctx, output, rctx.loc)
            rank = len(out_shape)
            lhs_map = _build_nisa_map(rctx.ctx, rank, lhs_acc)
            rhs_map = _build_nisa_map(rctx.ctx, rank, rhs_acc)
            dst_map = _build_nisa_map(rctx.ctx, rank, dst_acc)
            nisa.tensor_tensor_arith(
                **_operand_kwargs("dst", dst_acc, dst_map, out_shape),
                **_operand_kwargs("lhs", lhs_acc, lhs_map, out_shape),
                **_operand_kwargs("rhs", rhs_acc, rhs_map, out_shape),
                op=arith_kind,
                engine=nisa.Engine.Vector,
            )
        op.operation.erase()
        return

    if in0_bcast != in1_bcast:
        tensor_v = in1 if in0_bcast else in0
        vec_v = in0 if in0_bcast else in1
        vec_shape = in0_shape if in0_bcast else in1_shape
        block_arg0 = op.regions[0].blocks[0].arguments[0]
        block_arg1 = op.regions[0].blocks[0].arguments[1]
        vec_arg = block_arg0 if in0_bcast else block_arg1
        vec_is_lhs = str(body_lhs) == str(vec_arg)
        reverse = (
            nisa.TensScalarRevOps.First if vec_is_lhs else nisa.TensScalarRevOps.None_
        )
        with nk_ir.InsertionPoint(op), rctx.loc:
            src_acc = _get_base_and_offsets(rctx.ctx, tensor_v, rctx.loc)
            vec_acc = _get_base_and_offsets(rctx.ctx, vec_v, rctx.loc)
            dst_acc = _get_base_and_offsets(rctx.ctx, output, rctx.loc)
            rank = len(out_shape)
            src_map = _build_nisa_map(rctx.ctx, rank, src_acc)
            vec_map = _build_nisa_map(rctx.ctx, rank, vec_acc)
            dst_map = _build_nisa_map(rctx.ctx, rank, dst_acc)
            # Match the deleted C++ pattern: all operands use
            # tile_par_dims = rank - 1 so the broadcast operand's
            # free-dim product = 1 (the broadcast dimension), which
            # NISA's tensor_scalar_arith verifier requires. The vec
            # operand carries its own shape, not the full out_shape.
            par_dims = rank - 1
            nisa.tensor_scalar_arith(
                **_operand_kwargs("dst", dst_acc, dst_map, out_shape, par_dims),
                **_operand_kwargs("src", src_acc, src_map, out_shape, par_dims),
                **_operand_kwargs("operand0", vec_acc, vec_map, vec_shape, par_dims),
                **_empty_operand_kwargs("operand1"),
                op0=arith_kind,
                op1=None,
                reverse_operands=reverse,
                engine=nisa.Engine.Vector,
            )
        op.operation.erase()
        return
