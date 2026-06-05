"""MemRef access trace: _Access (mirror of C++ MemRefAccess) plus the
arith index-math emit helpers and _get_base_and_offsets."""

from __future__ import annotations

from dataclasses import dataclass

from nki.compiler._internal import ir as nk_ir  # type: ignore[import-not-found]

DYN_SENTINEL = -(1 << 63)


@dataclass
class _Access:
    """Mirror of C++ ``MemRefAccess``.

    ``indices`` contains one ``arith`` SSA value per base-rank dim (zero for
    dims with no subview offset). ``dropped_dims`` marks rank-reducing or
    unit-collapsed dims that should carry only a symbol offset in the NISA
    affine map (no iteration dim).

    The flat-affine map is built from this triple via ``_build_nisa_map``.
    """

    base: nk_ir.Value
    indices: list[nk_ir.Value]
    base_type: nk_ir.MemRefType
    dropped_dims: list[bool]  # len == base_rank

    @property
    def base_rank(self) -> int:
        return self.base_type.rank  # type: ignore[attr-defined]


def _reassoc_groups(attr: nk_ir.Attribute) -> list[list[int]]:
    outer = nk_ir.ArrayAttr(attr)
    groups: list[list[int]] = []
    for g in outer:
        inner = nk_ir.ArrayAttr(g)
        groups.append([int(nk_ir.IntegerAttr(x).value) for x in inner])
    return groups


def _const_int(v: nk_ir.Value) -> int | None:
    owner = getattr(v, "owner", None)
    if owner is None:
        return None
    op = owner.opview if hasattr(owner, "opview") else owner
    if getattr(op, "name", None) != "arith.constant":
        return None
    try:
        attr = op.attributes["value"]
    except KeyError:
        return None
    try:
        return nk_ir.IntegerAttr(attr).value
    except Exception:
        return None


def _emit_const_index(ctx: nk_ir.Context, value: int, loc: nk_ir.Location) -> nk_ir.Value:
    idx_ty = nk_ir.IndexType.get(ctx)
    attr = nk_ir.IntegerAttr.get(idx_ty, value)
    op = nk_ir.Operation.create(
        "arith.constant", results=[idx_ty], attributes={"value": attr}, loc=loc
    )
    return op.result


def _emit_addi(a: nk_ir.Value, b: nk_ir.Value, ctx: nk_ir.Context,
               loc: nk_ir.Location) -> nk_ir.Value:
    idx_ty = nk_ir.IndexType.get(ctx)
    op = nk_ir.Operation.create(
        "arith.addi", results=[idx_ty], operands=[a, b], loc=loc
    )
    return op.result


def _emit_muli(a: nk_ir.Value, b: nk_ir.Value, ctx: nk_ir.Context,
               loc: nk_ir.Location) -> nk_ir.Value:
    idx_ty = nk_ir.IndexType.get(ctx)
    op = nk_ir.Operation.create(
        "arith.muli", results=[idx_ty], operands=[a, b], loc=loc
    )
    return op.result


def _emit_divui(a: nk_ir.Value, b: nk_ir.Value, ctx: nk_ir.Context,
                loc: nk_ir.Location) -> nk_ir.Value:
    idx_ty = nk_ir.IndexType.get(ctx)
    op = nk_ir.Operation.create(
        "arith.divui", results=[idx_ty], operands=[a, b], loc=loc
    )
    return op.result


def _get_base_and_offsets(ctx: nk_ir.Context, operand: nk_ir.Value,
                          loc: nk_ir.Location) -> _Access:
    """Port of C++ ``getBaseAndOffsets``. Walks subview/collapse/expand chains
    back to the base alloc or block arg, materialising arith ops as needed.

    The current insertion point must be positioned where new arith ops can be
    safely emitted (typically the op being rewritten).
    """
    base = operand
    base_type = operand.type
    indices: list[nk_ir.Value] = []
    dropped_dims: list[bool] = []

    changed = True
    while changed:
        changed = False
        owner = getattr(base, "owner", None)
        if owner is None:
            break
        op = owner.opview if hasattr(owner, "opview") else owner
        name = getattr(op, "name", None)

        if name == "memref.subview":
            source = op.operands[0]
            source_ty = source.type
            if not isinstance(source_ty, nk_ir.MemRefType):
                break
            src_rank = source_ty.rank
            try:
                static_offsets = [int(x) for x in op.attributes["static_offsets"]]
            except (KeyError, ValueError):
                break
            if len(static_offsets) != src_rank:
                break
            dyn_ops = list(op.operands)[1:]
            dyn_idx = 0
            subview_offsets: list[nk_ir.Value] = []
            for i in range(src_rank):
                if static_offsets[i] == DYN_SENTINEL:
                    subview_offsets.append(dyn_ops[dyn_idx])
                    dyn_idx += 1
                else:
                    subview_offsets.append(
                        _emit_const_index(ctx, static_offsets[i], loc)
                    )

            # Rank-reducing detection via static_sizes==1 vs result shape.
            try:
                static_sizes = [int(x) for x in op.attributes["static_sizes"]]
            except (KeyError, ValueError):
                break
            result_shape = list(getattr(op.results[0].type, "shape", ()))
            # Determine dropped dims: rank-reducing means result_rank < src_rank.
            # Heuristic: dims with static_sizes[i]==1 that are NOT present in
            # the result shape are dropped. We align by order: walk source dims,
            # match to result in order, preferring size equality.
            dropped = [False] * src_rank
            if len(result_shape) < src_rank:
                ri = 0
                for si in range(src_rank):
                    if ri < len(result_shape) and static_sizes[si] == result_shape[ri]:
                        ri += 1
                    else:
                        # If size == 1 and we haven't matched yet, drop it.
                        if static_sizes[si] == 1:
                            dropped[si] = True
                        elif ri < len(result_shape) and static_sizes[si] == 1:
                            dropped[si] = True
                        else:
                            # fall back: mark as dropped if no remaining result dim
                            # matches
                            dropped[si] = True
                # If we ended up dropping too many or too few, bail
                non_dropped = sum(1 for d in dropped if not d)
                if non_dropped != len(result_shape):
                    break

            # Accumulate offsets:
            if not indices:
                # No carried indices yet. Usually this is the first subview in
                # the chain, but it can also follow a collapse_shape that
                # reduced a multi-dim HBM operand to a lower-rank view — in
                # which case `dropped_dims` is already populated and must be
                # preserved. The subview is same-rank here (src_rank ==
                # result_rank) for a collapse→subview chain, so just seed
                # indices with the subview offsets and merge `dropped` with
                # any preserved `dropped_dims` element-wise.
                indices = subview_offsets
                if dropped_dims and len(dropped_dims) == src_rank:
                    dropped_dims = [a or b for a, b in zip(dropped_dims, dropped)]
                else:
                    dropped_dims = dropped
            else:
                # Nested subview. Expand current indices (in result-rank space
                # after prior ops) to source rank.
                if any(dropped):
                    # Rank-reducing: dropped dims get pure subview offset;
                    # kept dims get accumulated + subview offset.
                    expanded: list[nk_ir.Value] = []
                    kept_idx = 0
                    for si in range(src_rank):
                        if dropped[si]:
                            expanded.append(subview_offsets[si])
                        else:
                            assert kept_idx < len(indices)
                            expanded.append(
                                _emit_addi(indices[kept_idx], subview_offsets[si],
                                           ctx, loc)
                            )
                            kept_idx += 1
                    indices = expanded
                    merged_dropped = [False] * src_rank
                    kept_idx = 0
                    for si in range(src_rank):
                        if dropped[si]:
                            merged_dropped[si] = True
                        else:
                            if kept_idx < len(dropped_dims) and dropped_dims[kept_idx]:
                                merged_dropped[si] = True
                            kept_idx += 1
                    dropped_dims = merged_dropped
                else:
                    # Same-rank subview: add element-wise.
                    if len(indices) != src_rank:
                        break
                    indices = [
                        _emit_addi(indices[i], subview_offsets[i], ctx, loc)
                        for i in range(src_rank)
                    ]

            base = source
            base_type = source_ty
            changed = True
            continue

        if name == "memref.collapse_shape":
            source = op.operands[0]
            source_ty = source.type
            if not isinstance(source_ty, nk_ir.MemRefType):
                break
            try:
                groups = _reassoc_groups(op.attributes["reassociation"])
            except KeyError:
                break
            src_shape = list(getattr(source_ty, "shape", ()))
            if any(s < 0 for s in src_shape):
                break
            src_rank = len(src_shape)

            # Determine whether any group has multiple non-unit dims.
            has_multi_non_unit = any(
                sum(1 for d in grp if src_shape[d] != 1) > 1 for grp in groups
            )

            ms = getattr(source_ty, "memory_space", None)
            is_hbm = ms is not None and (
                "<hbm>" in str(ms) or "<shared_hbm>" in str(ms)
            )

            if has_multi_non_unit and is_hbm:
                # Stop tracing — HBM collapse is fine as-is; NCC handles it.
                break

            # Find primary dim (largest size) per group — for multi-non-unit.
            def primary_dim(group: list[int]) -> int:
                best_idx, best_size = -1, 0
                for i, d in enumerate(group):
                    if src_shape[d] > best_size:
                        best_size = src_shape[d]
                        best_idx = i
                return best_idx

            # Expand droppedDims from collapsed to source rank.
            expanded_dropped = [False] * src_rank
            if is_hbm:
                for gi, grp in enumerate(groups):
                    collapsed_dropped = (
                        gi < len(dropped_dims) and dropped_dims[gi] if dropped_dims
                        else False
                    )
                    if collapsed_dropped:
                        for d in grp:
                            expanded_dropped[d] = True
                    elif len(grp) > 1:
                        for d in grp:
                            if src_shape[d] == 1:
                                expanded_dropped[d] = True
            else:
                for gi, grp in enumerate(groups):
                    collapsed_dropped = (
                        gi < len(dropped_dims) and dropped_dims[gi] if dropped_dims
                        else False
                    )
                    if collapsed_dropped:
                        for d in grp:
                            expanded_dropped[d] = True
                    elif len(grp) > 1 and has_multi_non_unit:
                        p = primary_dim(grp)
                        for i, d in enumerate(grp):
                            if i != p:
                                expanded_dropped[d] = True
                    elif len(grp) > 1:
                        # Drop only the unit dims, but keep one per group so
                        # the group still has a home for its iteration dim.
                        # For single-non-unit groups that dim is obviously
                        # the keeper; for all-unit groups we keep the first
                        # position so downstream affine-map construction has
                        # somewhere to place d_gi.
                        non_unit = [d for d in grp if src_shape[d] != 1]
                        keeper = non_unit[0] if non_unit else grp[0]
                        for d in grp:
                            if d != keeper:
                                expanded_dropped[d] = True

            # Expand indices from collapsed rank to source rank.
            if indices:
                expanded_indices: list[nk_ir.Value] = []
                for gi, grp in enumerate(groups):
                    if len(grp) == 1:
                        expanded_indices.append(indices[gi])
                    else:
                        non_unit_count = sum(1 for d in grp if src_shape[d] != 1)
                        if non_unit_count <= 1:
                            zero = _emit_const_index(ctx, 0, loc)
                            for d in grp:
                                if src_shape[d] != 1:
                                    expanded_indices.append(indices[gi])
                                else:
                                    expanded_indices.append(zero)
                        else:
                            p = primary_dim(grp)
                            primary_size = src_shape[grp[p]]
                            zero = _emit_const_index(ctx, 0, loc)
                            size_val = _emit_const_index(ctx, primary_size, loc)
                            batch = _emit_divui(indices[gi], size_val, ctx, loc)
                            for i, d in enumerate(grp):
                                if i == p:
                                    expanded_indices.append(zero)
                                elif src_shape[d] == 1:
                                    expanded_indices.append(zero)
                                else:
                                    expanded_indices.append(batch)
                indices = expanded_indices

            dropped_dims = expanded_dropped
            base = source
            base_type = source_ty
            changed = True
            continue

        if name == "memref.expand_shape":
            source = op.operands[0]
            source_ty = source.type
            if not isinstance(source_ty, nk_ir.MemRefType):
                break
            try:
                groups = _reassoc_groups(op.attributes["reassociation"])
            except KeyError:
                break
            dst_shape = list(getattr(op.results[0].type, "shape", ()))
            src_rank = source_ty.rank

            if indices:
                src_indices: list[nk_ir.Value] = []
                for grp in groups:
                    combined = indices[grp[0]]
                    for k in range(1, len(grp)):
                        inner_size = dst_shape[grp[k]]
                        scale = _emit_const_index(ctx, inner_size, loc)
                        combined = _emit_muli(combined, scale, ctx, loc)
                        combined = _emit_addi(combined, indices[grp[k]], ctx, loc)
                    src_indices.append(combined)
                indices = src_indices

            if dropped_dims:
                new_dropped = [False] * src_rank
                for gi, grp in enumerate(groups):
                    all_dropped = all(
                        d < len(dropped_dims) and dropped_dims[d] for d in grp
                    )
                    if all_dropped:
                        new_dropped[gi] = True
                dropped_dims = new_dropped

            base = source
            base_type = source_ty
            changed = True
            continue

        if name == "memref.reinterpret_cast":
            # Pass-through when it preserves rank and has zero static offsets.
            source = op.operands[0]
            source_ty = source.type
            if not isinstance(source_ty, nk_ir.MemRefType):
                break
            src_shape = list(getattr(source_ty, "shape", ()))
            dst_shape = list(getattr(op.results[0].type, "shape", ()))
            if src_shape != dst_shape:
                break
            try:
                st_off = [int(x) for x in op.attributes["static_offsets"]]
            except (KeyError, ValueError):
                st_off = []
            if any(x != 0 for x in st_off):
                break
            base = source
            base_type = source_ty
            changed = True
            continue

        break

    # Ensure we have rank-many indices (fresh allocs / block args need zeros).
    if not indices:
        rank = base_type.rank  # type: ignore[attr-defined]
        indices = [_emit_const_index(ctx, 0, loc) for _ in range(rank)]

    # Normalise dropped_dims length to base rank.
    rank = base_type.rank  # type: ignore[attr-defined]
    if len(dropped_dims) < rank:
        dropped_dims = dropped_dims + [False] * (rank - len(dropped_dims))
    elif len(dropped_dims) > rank:
        dropped_dims = dropped_dims[:rank]

    return _Access(
        base=base,
        indices=indices,
        base_type=base_type,  # type: ignore[arg-type]
        dropped_dims=dropped_dims,
    )
