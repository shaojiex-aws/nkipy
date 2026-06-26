"""
MLIR utility functions for building IR constructs.
"""

from typing import Callable, Dict, Tuple, Union
from mlir import ir
from mlir.dialects import arith, memref, linalg
import ml_dtypes
import numpy as np

from nkigen._mlir._mlir_libs._nkipy import nkipy as _nkipy_native

# Canonical memory-space mapping — values MUST match NkipyAttrs.td.
# Zero is intentionally reserved: MemRefType::get drops an IntegerAttr(0).
MEM_SPACE_MAP: Dict[str, int] = {
    "Hbm": 1,
    "Psum": 2,
    "Sbuf": 3,
    "SharedHbm": 4,
    "Constant": 5,
}


def mem_space_attr(name: str) -> ir.Attribute:
    """Create a #nkipy.mem<name> memory space attribute from a string name."""
    i32_attr = ir.IntegerAttr.get(
        ir.IntegerType.get_signless(32), MEM_SPACE_MAP[name]
    )
    return _nkipy_native.MemSpaceEnum.get(i32_attr)

# ---------- Type Mapping ----------
_BASE_TYPE_MAP: Dict[str, Callable[[], ir.Type]] = {
    "float16": lambda: ir.F16Type.get(),
    "bfloat16": lambda: ir.BF16Type.get(),
    "float32": lambda: ir.F32Type.get(),
    "float64": lambda: ir.F64Type.get(),
    "int8": lambda: ir.IntegerType.get_signless(8),
    "int16": lambda: ir.IntegerType.get_signless(16),
    "int32": lambda: ir.IntegerType.get_signless(32),
    "int64": lambda: ir.IntegerType.get_signless(64),
    "uint8": lambda: ir.IntegerType.get_signless(8),
    "uint16": lambda: ir.IntegerType.get_signless(16),
}

_MLIR_ALIASES: Dict[str, str] = {
    "f16": "float16",
    "bf16": "bfloat16",
    "f32": "float32",
    "f64": "float64",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
}

NP_TO_MLIR_TYPE_MAP: Dict[str, Callable[[], ir.Type]] = {
    **_BASE_TYPE_MAP,
    **{alias: _BASE_TYPE_MAP[canonical] for alias, canonical in _MLIR_ALIASES.items()},
}

NUMPY_TO_STR_MAP = {
    np.float16: "float16",
    ml_dtypes.bfloat16: "bfloat16",
    np.float32: "float32",
    np.float64: "float64",
    np.int8: "int8",
    np.int16: "int16",
    np.int32: "int32",
    np.int64: "int64",
    np.uint8: "uint8",
    np.uint16: "uint16",
}

def to_mlir_type(dtype: Union[str, np.dtype, type]) -> ir.Type:
    """
    Convert either:
      - a string (e.g. "float32", "i32")
      - a numpy dtype object (np.float32, np.dtype("float32"), etc.)
    into the corresponding MLIR type.
    """
    # Case 1: string
    if isinstance(dtype, str):
        key = dtype.lower()
        if key not in NP_TO_MLIR_TYPE_MAP:
            raise KeyError(f"Unsupported dtype string: {dtype}")
        return NP_TO_MLIR_TYPE_MAP[key]()

    # Case 2: NumPy dtype
    if isinstance(dtype, (np.dtype, type)):
        # Normalize to a numpy.dtype
        dtype = np.dtype(dtype)
        if dtype.type not in NUMPY_TO_STR_MAP:
            raise KeyError(f"Unsupported numpy dtype: {dtype}")
        key = NUMPY_TO_STR_MAP[dtype.type]
        return NP_TO_MLIR_TYPE_MAP[key]()

    raise TypeError(f"Unsupported dtype input type: {type(dtype)}")


def memref_of(shape: Tuple[int, ...], elem_ty: ir.Type,
              memory_space: ir.Attribute = None) -> ir.MemRefType:
    """Create a memref type, optionally with a memory space attribute."""
    return ir.MemRefType.get(shape, elem_ty, memory_space=memory_space)


def make_alloc(loc: ir.Location, shape: Tuple[int, ...], elem_ty: ir.Type,
               memory_space: ir.Attribute = None) -> ir.Value:
    """Create a memref.alloc with given shape and element type (uninitialized)."""
    memref_type = memref_of(shape, elem_ty, memory_space=memory_space)
    return memref.AllocOp(memref_type, [], [], loc=loc).result


def make_alloc_filled(loc: ir.Location, shape: Tuple[int, ...], elem_ty: ir.Type,
                      fill_value: Union[int, float],
                      memory_space: ir.Attribute = None) -> ir.Value:
    """Create a memref.alloc filled with a scalar via linalg.fill."""
    buf = make_alloc(loc, shape, elem_ty, memory_space=memory_space)
    cst = const_scalar(fill_value, elem_ty, loc)
    filled = linalg.FillOp([], [cst], [buf], loc=loc)
    region = filled.regions[0]
    if len(region.blocks) == 0:
        block = region.blocks.append(elem_ty, elem_ty)
        with ir.InsertionPoint(block):
            linalg.YieldOp([block.arguments[0]], loc=loc)
    return buf


def make_alloc_zeros(loc: ir.Location, shape: Tuple[int, ...], elem_ty: ir.Type,
                     memory_space: ir.Attribute = None) -> ir.Value:
    """Create a memref.alloc initialized to zero via linalg.fill."""
    return make_alloc_filled(loc, shape, elem_ty, 0.0, memory_space=memory_space)


def const_scalar(val: Union[int, float], elem_ty: ir.Type, loc: ir.Location) -> ir.Value:
    """Create a scalar constant."""
    if isinstance(elem_ty, ir.FloatType):
        attr = ir.FloatAttr.get(elem_ty, float(val))
        return arith.ConstantOp(elem_ty, attr, loc=loc).result
    elif isinstance(elem_ty, ir.IntegerType):
        attr = ir.IntegerAttr.get(elem_ty, int(val))
        return arith.ConstantOp(elem_ty, attr, loc=loc).result
    else:
        raise TypeError(f"Unsupported element type: {elem_ty}")
