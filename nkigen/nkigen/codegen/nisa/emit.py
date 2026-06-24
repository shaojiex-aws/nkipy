"""Textual NISA codegen: walk nkipy IR, emit NISA MLIR assembly as text.

Single nkipy context for reading. No NKI wheel dependency during codegen.
The output is plain NISA MLIR text that the NKI compiler parses downstream.
"""

from __future__ import annotations

from mlir import ir as up_ir  # type: ignore[import-not-found]

from nkigen._mlir._mlir_libs._nkipy import nkipy as _nkipy_native

from .. import irutils

_MEMSPACE_STR = {
    irutils.MEMSPACE_HBM: "#nisa.mem<hbm>",
    irutils.MEMSPACE_PSUM: "#nisa.mem<psum>",
    irutils.MEMSPACE_SBUF: "#nisa.mem<sbuf>",
    irutils.MEMSPACE_SHARED_HBM: "#nisa.mem<shared_hbm>",
}

_LINALG_TO_ARITH_OP = {
    "linalg.add": "add",
    "linalg.sub": "subtract",
    "linalg.mul": "multiply",
    "linalg.max": "max",
    "linalg.min": "min",
}

_LINALG_TO_ACTIVATION = {
    "linalg.exp": "exp",
    "linalg.sqrt": "sqrt",
    "linalg.square": "square",
    "linalg.abs": "abs",
    "linalg.log": "log",
    "linalg.tanh": "tanh",
}

_ARITH_BODY_TO_OP = {
    "arith.addf": "add",
    "arith.addi": "add",
    "arith.mulf": "multiply",
    "arith.muli": "multiply",
    "arith.subf": "subtract",
    "arith.subi": "subtract",
    "arith.maximumf": "max",
    "arith.minimumf": "min",
}


class NisaEmitter:
    """Walks nkipy IR top-down and emits NISA MLIR text."""

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._indent = 0
        self._counter = 0
        self._names: dict = {}  # nkipy Value -> str (e.g. "%0", "%arg0")
        self._used: set[str] = set()

    # -- text output helpers --

    def _line(self, text: str) -> None:
        self._lines.append("  " * self._indent + text)

    def _fresh(self, hint: str = "") -> str:
        """Generate a unique SSA name."""
        if hint:
            candidate = f"%{hint}"
            if candidate not in self._used:
                self._used.add(candidate)
                return candidate
            i = 0
            while f"%{hint}_{i}" in self._used:
                i += 1
            name = f"%{hint}_{i}"
        else:
            name = f"%{self._counter}"
            self._counter += 1
        self._used.add(name)
        return name

    def _name(self, val: up_ir.Value) -> str:
        return self._names[val]

    def _set_name(self, val: up_ir.Value, name: str) -> None:
        self._names[val] = name

    def _memref_type_str_nisa(self, ty: up_ir.Type) -> str:
        """Render memref type for NISA emission.

        HBM keeps its original rank. SBUF/PSUM are projected to 2D
        (partition x free) since tile allocs are always 2D.
        For SBUF with sbuf_map, uses physical dimensions from the map.
        """
        mrt = up_ir.MemRefType(ty)
        dims = list(mrt.shape)
        elem = str(mrt.element_type)
        ms = irutils.memref_memspace(ty)
        ms_str = _MEMSPACE_STR.get(ms, "")
        if ms in (irutils.MEMSPACE_SBUF, irutils.MEMSPACE_PSUM):
            layout = mrt.layout
            if _nkipy_native.SbufMapAttr.isinstance(layout):
                # Physical alloc = full logical shape, folded if par > 128
                logical_par = dims[0]
                logical_free = 1
                for d in dims[1:]:
                    logical_free *= d
                if logical_par > 128:
                    par = 128
                    free = (logical_par // 128) * logical_free
                else:
                    par = logical_par
                    free = logical_free
                shape = f"{par}x{free}"
            elif len(dims) > 2:
                # Strip leading unit dims (temp SBUF from loop tiling)
                while len(dims) > 2 and dims[0] == 1:
                    dims = dims[1:]
                par = dims[0]
                free = 1
                for d in dims[1:]:
                    free *= d
                shape = f"{par}x{free}"
            else:
                shape = "x".join(str(d) for d in dims)
        else:
            shape = "x".join(str(d) for d in dims)
        if ms_str:
            return f"memref<{shape}x{elem}, {ms_str}>"
        return f"memref<{shape}x{elem}>"

    def _has_sbuf_map(self, ty: up_ir.Type) -> bool:
        layout = up_ir.MemRefType(ty).layout
        return _nkipy_native.SbufMapAttr.isinstance(layout)

    def _get_sbuf_map(self, ty: up_ir.Type):
        layout = up_ir.MemRefType(ty).layout
        return _nkipy_native.SbufMapAttr(layout)

    # -- top-level --

    def emit_module(self, module: up_ir.Module, target: str = "trn2") -> str:
        self._line(f'module attributes {{nisa.target = #nisa.target<{target}>}} {{')
        self._indent += 1
        for op in module.body.operations:
            if op.operation.name == "func.func":
                self._emit_func(op)
        self._indent -= 1
        self._line("}")
        return "\n".join(self._lines) + "\n"

    # -- structural ops --

    def _emit_func(self, func_op) -> None:
        op = func_op.operation
        sym_name = up_ir.StringAttr(op.attributes["sym_name"]).value
        func_ty = up_ir.FunctionType(up_ir.TypeAttr(op.attributes["function_type"]).value)

        block = list(op.regions[0].blocks)[0]
        params = []
        for i, arg in enumerate(block.arguments):
            name = f"%arg{i}"
            self._set_name(arg, name)
            params.append(f"{name}: {self._memref_type_str_nisa(arg.type)}")

        results = [self._memref_type_str_nisa(t) for t in func_ty.results]
        ret_str = f' -> {results[0]}' if len(results) == 1 else ""
        if len(results) > 1:
            ret_str = f' -> ({", ".join(results)})'

        attrs = ' attributes {nki.output_names = ["output"]}' if results else ""
        self._line(f'func.func @{sym_name}({", ".join(params)}){ret_str}{attrs} {{')
        self._indent += 1
        self._emit_block(block)
        self._indent -= 1
        self._line("}")

    def _emit_block(self, block) -> None:
        for op in block.operations:
            self._emit_op(op.operation)

    def _emit_op(self, op: up_ir.Operation) -> None:
        name = op.name
        if name == "arith.constant":
            self._emit_arith_constant(op)
        elif name in ("arith.muli", "arith.addi", "arith.subi",
                      "arith.divui", "arith.remui"):
            self._emit_arith_binop(op)
        elif name == "scf.for":
            self._emit_scf_for(op)
        elif name == "scf.yield":
            self._emit_scf_yield(op)
        elif name == "func.return":
            self._emit_return(op)
        elif name == "memref.alloc":
            self._emit_alloc(op)
        elif name == "memref.dealloc":
            self._emit_dealloc(op)
        elif name in ("memref.subview", "memref.collapse_shape",
                      "memref.expand_shape", "memref.reinterpret_cast"):
            for r in op.results:
                if r not in self._names:
                    self._names[r] = None
        elif name == "memref.copy":
            self._emit_copy(op)
        elif name in _LINALG_TO_ARITH_OP:
            self._emit_elementwise(op)
        elif name in _LINALG_TO_ACTIVATION:
            self._emit_activation(op)
        elif name == "linalg.reciprocal":
            self._emit_reciprocal(op)
        elif name == "linalg.fill":
            self._emit_fill(op)
        elif name == "linalg.transpose":
            self._emit_transpose(op)
        elif name == "linalg.matmul_transpose_a":
            self._emit_matmul(op)
        elif name == "linalg.generic":
            self._emit_linalg_generic(op)

    # -- arith --

    def _emit_arith_constant(self, op: up_ir.Operation) -> None:
        val = irutils.const_int(op.results[0])
        if val is not None:
            name = self._fresh(f"c{val}")
            self._set_name(op.results[0], name)
            self._line(f"{name} = arith.constant {val} : index")
        else:
            attr = op.attributes["value"]
            name = self._fresh("cst")
            self._set_name(op.results[0], name)
            # attr already includes "value : type", just emit directly
            self._line(f"{name} = arith.constant {attr}")

    def _emit_arith_binop(self, op: up_ir.Operation) -> None:
        op_name = op.name.split(".")[-1]
        lhs = self._name(op.operands[0])
        rhs = self._name(op.operands[1])
        result_name = self._fresh()
        self._set_name(op.results[0], result_name)
        self._line(f"{result_name} = arith.{op_name} {lhs}, {rhs} : index")

    # -- scf --

    def _emit_scf_for(self, op: up_ir.Operation) -> None:
        lb = self._name(op.operands[0])
        ub = self._name(op.operands[1])
        step = self._name(op.operands[2])

        body_block = list(list(op.regions)[0].blocks)[0]
        iv = list(body_block.arguments)[0]
        iv_name = self._fresh("iv")
        self._set_name(iv, iv_name)

        self._line(f"scf.for {iv_name} = {lb} to {ub} step {step} {{")
        self._indent += 1
        self._emit_block(body_block)
        self._indent -= 1
        self._line("}")

    def _emit_scf_yield(self, op: up_ir.Operation) -> None:
        if list(op.operands):
            operands = ", ".join(self._name(o) for o in op.operands)
            types = ", ".join(str(o.type) for o in op.operands)
            self._line(f"scf.yield {operands} : {types}")

    def _emit_return(self, op: up_ir.Operation) -> None:
        operands = list(op.operands)
        if operands:
            vals = ", ".join(self._name(o) for o in operands)
            types = ", ".join(self._memref_type_str_nisa(o.type) for o in operands)
            self._line(f"return {vals} : {types}")
        else:
            self._line("return")

    # -- memory --

    def _emit_alloc(self, op: up_ir.Operation) -> None:
        result = op.results[0]
        ty_str = self._memref_type_str_nisa(result.type)
        name = self._fresh("mem")
        self._set_name(result, name)
        self._line(f"{name} = nisa.alloc alignment=64 : {ty_str}")

    def _emit_dealloc(self, op: up_ir.Operation) -> None:
        target = op.operands[0]
        ms = irutils.memref_memspace(target.type)
        if ms not in (irutils.MEMSPACE_SBUF, irutils.MEMSPACE_PSUM):
            return
        name = self._name(target)
        ty_str = self._memref_type_str_nisa(target.type)
        self._line(f"nisa.release {name} : {ty_str}")

    # -- access tracing --

    def _trace_access(self, val: up_ir.Value) -> tuple[str, list[str], list[int], up_ir.Type]:
        """Trace a memref value back to its base, collecting offsets.

        Returns (base_name, offset_exprs, tile_shape, base_type).
        """
        base = val
        offsets: list[str | None] = None

        while True:
            owner = getattr(base, "owner", None)
            if owner is None:
                break
            op = owner.opview if hasattr(owner, "opview") else owner
            op_name = getattr(op, "name", None)

            if op_name == "memref.reinterpret_cast":
                base = op.operation.operands[0]
                continue

            if op_name in ("memref.collapse_shape", "memref.expand_shape"):
                break

            if op_name == "memref.subview":
                source = op.operation.operands[0]
                src_rank = up_ir.MemRefType(source.type).rank
                result_rank = up_ir.MemRefType(op.operation.results[0].type).rank
                static_offsets_attr = op.operation.attributes["static_offsets"]
                static_offsets = [int(x) for x in static_offsets_attr]
                static_sizes_attr = op.operation.attributes["static_sizes"]
                static_sizes = [int(x) for x in static_sizes_attr]
                dyn_ops = list(op.operation.operands)[1:]
                dyn_idx = 0

                sv_offsets: list[str] = []
                for i in range(src_rank):
                    if static_offsets[i] == -(1 << 63):
                        sv_offsets.append(self._name(dyn_ops[dyn_idx]))
                        dyn_idx += 1
                    else:
                        if static_offsets[i] == 0:
                            sv_offsets.append(self._emit_const_index(0))
                        else:
                            sv_offsets.append(self._emit_const_index(static_offsets[i]))

                if offsets is None:
                    offsets = sv_offsets
                else:
                    # Rank-reducing subview: align inner offsets to kept dims
                    if result_rank < src_rank:
                        kept_dims = [i for i in range(src_rank)
                                     if static_sizes[i] != 1]
                        new_offsets = list(sv_offsets)
                        for j, src_dim in enumerate(kept_dims):
                            if j < len(offsets):
                                new_offsets[src_dim] = self._emit_addi(
                                    offsets[j], sv_offsets[src_dim])
                    else:
                        new_offsets = []
                        for i in range(min(len(offsets), len(sv_offsets))):
                            new_offsets.append(
                                self._emit_addi(offsets[i], sv_offsets[i]))
                    offsets = new_offsets

                base = source
                continue

            break

        base_name = self._name(base)
        base_type = base.type
        tile_shape = list(up_ir.MemRefType(val.type).shape)

        if offsets is None:
            base_ty = up_ir.MemRefType(base_type)
            offsets = [self._emit_const_index(0) for _ in range(base_ty.rank)]

        return base_name, offsets, tile_shape, base_type

    def _emit_const_index(self, val: int) -> str:
        name = self._fresh(f"c{val}")
        self._line(f"{name} = arith.constant {val} : index")
        return name

    def _emit_addi(self, a: str, b: str) -> str:
        result = self._fresh()
        self._line(f"{result} = arith.addi {a}, {b} : index")
        return result

    def _emit_muli(self, a: str, b: str) -> str:
        result = self._fresh()
        self._line(f"{result} = arith.muli {a}, {b} : index")
        return result

    def _emit_divui(self, a: str, b: str) -> str:
        result = self._fresh()
        self._line(f"{result} = arith.divui {a}, {b} : index")
        return result

    def _operand_str(self, val: up_ir.Value, prefix: str) -> str:
        """Build operand string: prefix<tile_shape>=memloc_ref[subscripts]

        BIR requires dma_copy src/dst to have matching rank. Since SBUF is
        always 2D, >2D operands are projected to 2D:
        - SBUF/PSUM: emitted type is already 2D, no view() needed
        - HBM: uses view() to reinterpret the >2D memref as 2D
        """
        base_name, offsets, tile_shape, base_type = self._trace_access(val)
        ms = irutils.memref_memspace(base_type)
        is_onchip = ms in (irutils.MEMSPACE_SBUF, irutils.MEMSPACE_PSUM)
        base_shape = list(up_ir.MemRefType(base_type).shape)
        base_rank = len(base_shape)

        # HBM with >2D base memref needs view() regardless of tile rank.
        # A rank-reducing subview gives a 2D tile but the memref is still >2D.
        needs_view = not is_onchip and base_rank > 2

        if needs_view:
            # Find the first accessed dim (tile size > 1) — this splits
            # the base into [batch... | accessed_row | accessed_col...]
            first_accessed = 0
            for i, t in enumerate(tile_shape):
                if t > 1:
                    first_accessed = i
                    break

            par = tile_shape[first_accessed]
            free = 1
            for d in tile_shape[first_accessed + 1:]:
                free *= d
            tile_str = f"{par}| {free}"

            view_c = 1
            for d in base_shape[first_accessed + 1:]:
                view_c *= d
            view_r = 1
            for d in base_shape[:first_accessed + 1]:
                view_r *= d

            row_offset = self._linearize_offsets(
                offsets[:first_accessed + 1], base_shape[:first_accessed + 1])
            if first_accessed + 1 < len(offsets):
                col_offset = self._linearize_offsets(
                    offsets[first_accessed + 1:], base_shape[first_accessed + 1:])
            else:
                col_offset = self._emit_const_index(0)
            dims = [f"{row_offset} + d0", f"{col_offset} + d1"]

            orig_ty = self._memref_type_str_nisa(base_type)
            elem = str(up_ir.MemRefType(base_type).element_type)
            memloc_ref = (
                f"view({orig_ty} {base_name}, {elem}, [{view_r}, {view_c}])"
            )
        elif is_onchip and self._has_sbuf_map(base_type):
            # Multi-block SBUF: remap logical offsets to physical 2D
            sbuf_map = self._get_sbuf_map(base_type)
            par = tile_shape[0]
            free = 1
            for d in tile_shape[1:]:
                free *= d
            tile_str = f"{par}| {free}"

            par_offset, free_offset = self._remap_sbuf_offsets(offsets, sbuf_map)
            dims = [f"{par_offset} + d0", f"{free_offset} + d1"]
            memloc_ref = f"{self._memref_type_str_nisa(base_type)} {base_name}"
        elif is_onchip and len(tile_shape) > 2:
            # On-chip >2D tile without sbuf_map: strip leading 1s, flatten to 2D
            skip = 0
            while skip < len(tile_shape) - 2 and tile_shape[skip] == 1:
                skip += 1
            par = tile_shape[skip]
            free = 1
            for d in tile_shape[skip + 1:]:
                free *= d
            tile_str = f"{par}| {free}"

            if skip > 0:
                par_offset = offsets[skip] if skip < len(offsets) else self._emit_const_index(0)
            else:
                par_offset = offsets[0] if offsets else self._emit_const_index(0)
            remaining_offsets = offsets[skip + 1:]
            remaining_shape = base_shape[skip + 1:]
            if remaining_offsets:
                free_offset = self._linearize_offsets(remaining_offsets, remaining_shape)
            else:
                free_offset = self._emit_const_index(0)
            dims = [f"{par_offset} + d0", f"{free_offset} + d1"]
            memloc_ref = f"{self._memref_type_str_nisa(base_type)} {base_name}"
        elif len(tile_shape) == 2:
            tile_str = f"{tile_shape[0]}| {tile_shape[1]}"
            dims = []
            for i in range(2):
                if i < len(offsets):
                    dims.append(f"{offsets[i]} + d{i}")
                else:
                    dims.append(f"d{i}")
            memloc_ref = f"{self._memref_type_str_nisa(base_type)} {base_name}"
        else:
            tile_str = f"{tile_shape[0]}"
            dims = [f"{offsets[0]} + d0"] if offsets else ["d0"]
            memloc_ref = f"{self._memref_type_str_nisa(base_type)} {base_name}"

        return f"{prefix}<{tile_str}>={memloc_ref}[{', '.join(dims)}]"

    def _operand_str_multidim(self, val: up_ir.Value, prefix: str) -> str:
        """Build operand string with full rank — no view(), no flattening.

        Used for HBM↔HBM copies where both sides keep their native rank.
        NISA computes correct strides from the memref type directly.
        """
        base_name, offsets, tile_shape, base_type = self._trace_access(val)
        par = tile_shape[0]
        free_dims = ", ".join(str(d) for d in tile_shape[1:])
        tile_str = f"{par}| {free_dims}"

        dims = []
        for i in range(len(tile_shape)):
            if i < len(offsets):
                dims.append(f"{offsets[i]} + d{i}")
            else:
                dims.append(f"d{i}")

        memloc_ref = f"{self._memref_type_str_nisa(base_type)} {base_name}"
        return f"{prefix}<{tile_str}>={memloc_ref}[{', '.join(dims)}]"

    def _linearize_offsets(self, offsets: list[str],
                           dim_sizes: list[int]) -> str:
        """Linearize N free-dim offsets into one: off[0]*stride[0] + ... + off[N-1].

        stride[i] = product(dim_sizes[i+1:])

        Optimization: if all offsets are the same zero constant (common for
        SBUF tiles where the alloc starts at offset 0), skip the arithmetic.
        """
        # Fast path: all offsets are the same value (typically %c0 for SBUF tiles)
        if len(set(offsets)) == 1 and offsets[0].startswith("%c0"):
            return offsets[0]

        n = len(offsets)
        strides = [1] * n
        for i in range(n - 2, -1, -1):
            strides[i] = strides[i + 1] * dim_sizes[i + 1]

        terms = []
        for off, stride in zip(offsets, strides):
            if stride == 1:
                terms.append(off)
            else:
                stride_name = self._emit_const_index(stride)
                mul_result = self._fresh()
                self._line(f"{mul_result} = arith.muli {off}, {stride_name} : index")
                terms.append(mul_result)

        if not terms:
            return self._emit_const_index(0)
        result = terms[0]
        for t in terms[1:]:
            add_result = self._fresh()
            self._line(f"{add_result} = arith.addi {result}, {t} : index")
            result = add_result
        return result

    def _remap_sbuf_offsets(self, offsets: list[str], sbuf_map) -> tuple[str, str]:
        """Remap logical N-D offsets to physical 2D [partition, free] for SBUF with sbuf_map.

        Physical shape: [tile[0], blocks[0], ..., blocks[R-1], tile[R-1]]
        Flattened 2D:   [tile[0], blocks[0] * ... * blocks[R-1] * tile[R-1]]

        When blocks[0] > 1, the partition offset folds into free as a block index.
        Accesses are always tile-aligned so within-block partition offset is 0.
        """
        tile_par = sbuf_map.tile_size(0)
        num_par_blocks = sbuf_map.num_blocks(0)
        rank = sbuf_map.rank

        # Free offset: logical column offsets map directly into the free dimension.
        # Use full logical extents (tile * blocks) as dim sizes for linearization.
        free_dims = [sbuf_map.tile_size(i) * sbuf_map.num_blocks(i)
                     for i in range(1, rank)]
        free_offsets = offsets[1:]
        free_offset = self._linearize_offsets(free_offsets, free_dims)

        if num_par_blocks == 1:
            return offsets[0], free_offset

        # Fold partition block index into free.
        # Stride = number of free elements per partition-block.
        stride = 1
        for d in free_dims:
            stride *= d
        par_offset = self._emit_const_index(0)
        c_tile_par = self._emit_const_index(tile_par)
        block_idx = self._emit_divui(offsets[0], c_tile_par)
        c_stride = self._emit_const_index(stride)
        fold_contrib = self._emit_muli(block_idx, c_stride)
        free_offset = self._emit_addi(fold_contrib, free_offset)

        return par_offset, free_offset


    # -- data movement --

    def _emit_copy(self, op: up_ir.Operation) -> None:
        src = op.operands[0]
        dst = op.operands[1]
        src_ms = irutils.memref_memspace(src.type)
        dst_ms = irutils.memref_memspace(dst.type)

        src_is_hbm = src_ms in (irutils.MEMSPACE_HBM, irutils.MEMSPACE_SHARED_HBM)
        dst_is_hbm = dst_ms in (irutils.MEMSPACE_HBM, irutils.MEMSPACE_SHARED_HBM)

        if dst_ms == irutils.MEMSPACE_PSUM and src_is_hbm:
            # HBM -> psum: stage through sbuf
            self._emit_staged_copy(src, dst, "hbm_to_psum")
            return
        if src_ms == irutils.MEMSPACE_PSUM and dst_is_hbm:
            # psum -> HBM: stage through sbuf
            self._emit_staged_copy(src, dst, "psum_to_hbm")
            return

        both_hbm = src_is_hbm and dst_is_hbm
        if both_hbm:
            dst_str = self._operand_str_multidim(dst, "dst")
            src_str = self._operand_str_multidim(src, "src")
        else:
            dst_str = self._operand_str(dst, "dst")
            src_str = self._operand_str(src, "src")

        if src_is_hbm or dst_is_hbm:
            self._line(
                f"nisa.dma_copy({dst_str}, {src_str}, "
                f"dge_mode=unassigned, oob_is_err=true) engine=dma"
            )
        else:
            self._line(
                f"nisa.tensor_copy({dst_str}, {src_str}) engine=vector"
            )

    def _emit_staged_copy(self, src, dst, direction: str) -> None:
        """Stage a copy through an sbuf intermediate (psum<->HBM)."""
        # SBUF intermediate is always 2D (partition x free)
        ref_val = src if direction == "psum_to_hbm" else dst
        tile_shape = list(up_ir.MemRefType(ref_val.type).shape)
        elem = irutils.memref_elem_type(ref_val.type)
        par = tile_shape[0]
        free = 1
        for d in tile_shape[1:]:
            free *= d
        shape_str = f"{par}x{free}x{elem}"
        sbuf_ty = f"memref<{shape_str}, {_MEMSPACE_STR[irutils.MEMSPACE_SBUF]}>"

        tmp = self._fresh("mem")
        self._line(f"{tmp} = nisa.alloc alignment=64 : {sbuf_ty}")

        # Build tmp operand string (always 2D)
        tmp_offsets = [self._emit_const_index(0) for _ in range(2)]
        tile_str = f"{par}| {free}"
        tmp_dims = ", ".join(f"{tmp_offsets[i]} + d{i}" for i in range(2))

        if direction == "psum_to_hbm":
            # psum -> sbuf (tensor_copy), then sbuf -> HBM (dma_copy)
            src_str = self._operand_str(src, "src")
            self._line(
                f"nisa.tensor_copy(dst<{tile_str}>={sbuf_ty} {tmp}[{tmp_dims}], "
                f"{src_str}) engine=vector"
            )
            dst_str = self._operand_str(dst, "dst")
            self._line(
                f"nisa.dma_copy({dst_str}, "
                f"src<{tile_str}>={sbuf_ty} {tmp}[{tmp_dims}], "
                f"dge_mode=unassigned, oob_is_err=true) engine=dma"
            )
        else:
            # HBM -> sbuf (dma_copy), then sbuf -> psum (tensor_copy)
            src_str = self._operand_str(src, "src")
            self._line(
                f"nisa.dma_copy(dst<{tile_str}>={sbuf_ty} {tmp}[{tmp_dims}], "
                f"{src_str}, dge_mode=unassigned, oob_is_err=true) engine=dma"
            )
            dst_str = self._operand_str(dst, "dst")
            self._line(
                f"nisa.tensor_copy({dst_str}, "
                f"src<{tile_str}>={sbuf_ty} {tmp}[{tmp_dims}]) engine=vector"
            )
        self._line(f"nisa.release {tmp} : {sbuf_ty}")

    def _emit_transpose(self, op: up_ir.Operation) -> None:
        src = op.operands[0]
        dst = op.operands[1]
        permutation = [int(x) for x in op.attributes["permutation"]]

        dst_str = self._operand_str(dst, "dst")
        src_str = self._operand_str(src, "src")

        perm_str = ", ".join(str(p) for p in permutation)
        self._line(
            f"nisa.dma_transpose({dst_str}, {src_str}, "
            f"permutation=[{perm_str}], dge_mode=no_dge, oob_is_err=true) engine=dma"
        )

    # -- compute: elementwise --

    def _emit_elementwise(self, op: up_ir.Operation) -> None:
        arith_op = _LINALG_TO_ARITH_OP[op.name]
        operands = list(op.operands)
        lhs, rhs, dst = operands[0], operands[1], operands[2]

        dst_str = self._operand_str(dst, "dst")
        lhs_str = self._operand_str(lhs, "lhs")
        rhs_str = self._operand_str(rhs, "rhs")

        self._line(
            f"nisa.tensor_tensor_arith({dst_str}, {lhs_str}, {rhs_str}, "
            f"op={arith_op}) engine=vector"
        )

    # -- compute: activation --

    def _emit_activation(self, op: up_ir.Operation) -> None:
        """linalg.exp/sqrt/log/... -> nisa.activation."""
        act_fn = _LINALG_TO_ACTIVATION[op.name]
        src = op.operands[0]
        dst = op.operands[1]

        dst_str = self._operand_str(dst, "dst")
        src_str = self._operand_str(src, "src")

        scale = self._fresh("cst")
        self._line(f"{scale} = arith.constant 1.000000e+00 : f32")
        bias = self._fresh("cst")
        self._line(f"{bias} = arith.constant 0.000000e+00 : f32")

        self._line(
            f"nisa.activation({dst_str}, {src_str}, "
            f"bias=f32 {bias}, scale=f32 {scale}, op={act_fn}) engine=scalar"
        )

    def _emit_reciprocal(self, op: up_ir.Operation) -> None:
        """linalg.reciprocal -> nisa.reciprocal."""
        src = op.operands[0]
        dst = op.operands[1]

        dst_str = self._operand_str(dst, "dst")
        src_str = self._operand_str(src, "src")

        self._line(f"nisa.reciprocal({dst_str}, {src_str}) engine=vector")

    # -- compute: fill --

    def _emit_fill(self, op: up_ir.Operation) -> None:
        """linalg.fill -> nisa.memset."""
        scalar = op.operands[0]
        dst = op.operands[1]

        if not irutils.is_on_chip(dst.type):
            return

        scalar_name = self._name(scalar)
        dst_str = self._operand_str(dst, "dst")
        elem_ty = irutils.memref_elem_type(dst.type)

        self._line(
            f"nisa.memset({dst_str}, value={elem_ty} {scalar_name}) engine=vector"
        )

    # -- compute: matmul --

    def _emit_matmul(self, op: up_ir.Operation) -> None:
        """linalg.matmul_transpose_a -> nisa.matmul."""
        mat_a = op.operands[0]  # stationary [K, M] (already transposed)
        mat_b = op.operands[1]  # moving [K, N]
        mat_c = op.operands[2]  # dst [M, N] in psum

        dst_str = self._operand_str(mat_c, "dst")
        stat_str = self._operand_str(mat_a, "stationary")
        mov_str = self._operand_str(mat_b, "moving")

        row_pos = self._emit_const_index(0)
        col_pos = self._emit_const_index(0)

        self._line(
            f"nisa.matmul({dst_str}, {stat_str}, {mov_str}, "
            f"row_pos=index {row_pos}, col_pos=index {col_pos}, "
            f"is_transpose=false, perf_opt=none_, psum_zero_region=size2048) engine=tensor"
        )

    # -- linalg.generic dispatch --

    def _emit_linalg_generic(self, op: up_ir.Operation) -> None:
        """Dispatch linalg.generic by iterator types and body."""
        iterator_types = [str(t) for t in op.attributes["iterator_types"]]
        has_reduction = any("reduction" in t for t in iterator_types)

        if has_reduction:
            self._emit_reduction_generic(op)
        else:
            self._emit_elementwise_generic(op)

    def _emit_elementwise_generic(self, op: up_ir.Operation) -> None:
        """Parallel linalg.generic -> tensor_tensor_arith or tensor_scalar_arith."""
        body = list(list(op.regions)[0].blocks)[0]
        body_ops = [o for o in body.operations if o.operation.name != "linalg.yield"]
        if len(body_ops) != 1:
            return

        body_op_name = body_ops[0].operation.name
        arith_op = _ARITH_BODY_TO_OP.get(body_op_name)
        if arith_op is None:
            return

        num_ins = int(op.attributes["operandSegmentSizes"][0])
        ins = list(op.operands[:num_ins])
        dst = op.operands[num_ins]

        if num_ins == 1:
            self._emit_unary_generic(op, ins[0], dst, body_ops[0], arith_op)
            return

        if num_ins != 2:
            return

        dst_shape = list(up_ir.MemRefType(dst.type).shape)
        dst_free = _free_elems(dst_shape)

        in0_shape = list(up_ir.MemRefType(ins[0].type).shape)
        in1_shape = list(up_ir.MemRefType(ins[1].type).shape)
        in0_free = _free_elems(in0_shape)
        in1_free = _free_elems(in1_shape)

        bcast0 = in0_free == 1 and dst_free != 1
        bcast1 = in1_free == 1 and dst_free != 1

        if not bcast0 and not bcast1:
            dst_str = self._operand_str(dst, "dst")
            lhs_str = self._operand_str(ins[0], "lhs")
            rhs_str = self._operand_str(ins[1], "rhs")
            self._line(
                f"nisa.tensor_tensor_arith({dst_str}, {lhs_str}, {rhs_str}, "
                f"op={arith_op}) engine=vector"
            )
        else:
            # One operand broadcasts: use tensor_scalar_arith
            if bcast1:
                tensor_v, vec_v = ins[0], ins[1]
                # Check if the broadcast operand was the left body operand
                block_args = list(body.arguments)
                vec_is_lhs = str(body_ops[0].operation.operands[0]) == str(block_args[1])
            else:
                tensor_v, vec_v = ins[1], ins[0]
                vec_is_lhs = str(body_ops[0].operation.operands[0]) == str(block_args[0])

            reverse = "first" if vec_is_lhs else "none_"
            dst_str = self._operand_str(dst, "dst")
            src_str = self._operand_str(tensor_v, "src")
            op0_str = self._operand_str(vec_v, "operand0")
            self._line(
                f"nisa.tensor_scalar_arith({dst_str}, {src_str}, {op0_str}, "
                f"op0={arith_op}, reverse_operands={reverse}) engine=vector"
            )

    def _emit_unary_generic(self, op, src, dst, body_op, arith_op: str) -> None:
        """Single-input elementwise generic with a scalar constant in body."""
        scalar = None
        for operand in body_op.operation.operands:
            v = irutils.const_scalar(operand)
            if v is not None:
                scalar = v
                break

        if scalar is None:
            return

        scalar_name = self._fresh("cst")
        self._line(f"{scalar_name} = arith.constant {_format_float(scalar)} : f32")

        dst_str = self._operand_str(dst, "dst")
        src_str = self._operand_str(src, "src")
        self._line(
            f"nisa.tensor_scalar_arith({dst_str}, {src_str}, operand0=f32 {scalar_name}, "
            f"op0={arith_op}, reverse_operands=none_) engine=vector"
        )

    def _emit_reduction_generic(self, op: up_ir.Operation) -> None:
        """Reduction generic -> tensor_reduce_arith + accumulation."""
        body = list(list(op.regions)[0].blocks)[0]
        body_ops = [o for o in body.operations if o.operation.name != "linalg.yield"]
        if len(body_ops) != 1:
            return

        body_op_name = body_ops[0].operation.name
        arith_op = _ARITH_BODY_TO_OP.get(body_op_name)
        if arith_op is None:
            return

        num_ins = int(op.attributes["operandSegmentSizes"][0])
        src = op.operands[0]
        dst = op.operands[num_ins]

        iterator_types = [str(t) for t in op.attributes["iterator_types"]]
        num_r_dim = sum("reduction" in t for t in iterator_types)

        # Check if body reads the output accumulator (accumulating reduction)
        out_arg = list(body.arguments)[-1]
        accumulates = any(
            o == out_arg for o in body_ops[0].operation.operands
        )

        dst_str = self._operand_str(dst, "dst")
        src_str = self._operand_str(src, "src")

        if not accumulates:
            self._line(
                f"nisa.tensor_reduce_arith({dst_str}, {src_str}, "
                f"op={arith_op}, negated=false, num_r_dim={num_r_dim}) engine=vector"
            )
        else:
            # Alloc temp, reduce into temp, accumulate into dst
            dst_shape = list(up_ir.MemRefType(dst.type).shape)
            par = dst_shape[0]
            free = 1
            for d in dst_shape[1:]:
                free *= d
            temp_ty = self._memref_type_str_nisa(dst.type)
            temp_name = self._fresh("mem")
            self._line(f"{temp_name} = nisa.alloc : {temp_ty}")
            self._set_name(None, temp_name)  # no SSA value to bind

            # Emit reduce into temp (2D SBUF)
            temp_offsets = [self._emit_const_index(0) for _ in range(2)]
            base_ty = temp_ty
            tile_str = f"{par}| {free}"
            temp_dims = ", ".join(f"{temp_offsets[i]} + d{i}" for i in range(2))
            temp_operand = f"dst<{tile_str}>={base_ty} {temp_name}[{temp_dims}]"

            self._line(
                f"nisa.tensor_reduce_arith({temp_operand}, {src_str}, "
                f"op={arith_op}, negated=false, num_r_dim={num_r_dim}) engine=vector"
            )

            # Accumulate: dst = dst op temp
            rhs_operand = f"rhs<{tile_str}>={base_ty} {temp_name}[{temp_dims}]"
            self._line(
                f"nisa.tensor_tensor_arith({dst_str}, {dst_str.replace('dst<', 'lhs<')}, "
                f"{rhs_operand}, op={arith_op}) engine=vector"
            )

            # Release temp
            self._line(f"nisa.release {temp_name} : {temp_ty}")


def _free_elems(shape: list[int]) -> int:
    """Product of free (non-partition) dims — everything after dim 0."""
    n = 1
    for s in shape[1:]:
        n *= s
    return n


def _format_float(val: float) -> str:
    if val != val:
        return "0x7FC00000"
    if val == float("inf"):
        return "0x7F800000"
    if val == float("-inf"):
        return "0xFF800000"
    return f"{val:e}"
