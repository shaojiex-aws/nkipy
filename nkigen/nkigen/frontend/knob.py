"""
Knob API for annotating tensors with transformation hints.

Usage (builder-style):

    # Memory placement hint.
    knob(result).layout(mem_space="Sbuf", partition_dim=0)

    # Loop-tiling hint.
    # tile_size has one entry per iterator of the producing op, in the
    # same order linalg emits — tile_size[i] applies to iterator i.
    # Examples:
    #   - elementwise on [M, N]:                     tile_size=[M_t, N_t]
    #   - np.sum(x[M, N], axis=-1):                  tile_size=[M_t, N_t]
    #     (iterators [parallel, reduction])
    #   - np.sum(x[M, N, K], axis=1):                tile_size=[M_t, N_t, K_t]
    #     (iterators [parallel, reduction, parallel])
    #   - matmul A[M,K] @ B[K,N] -> C[M,N]:          tile_size=[M_t, N_t, K_t]
    #     (iterators [parallel, parallel, reduction])
    knob(result).tile_op(tile_size=[128, 128, 128])

    # Chain both.
    knob(result).tile_op(tile_size=[64, 64]).layout(mem_space="Sbuf")

The physical factorization tile for SBUF is always auto-derived from the
consuming tile_op via indexing maps (InferLayout pass). No tile_size
parameter on .layout().
"""

from typing import Union, Any, Optional, List
from mlir import ir
from .traced_array import TracedArray
from nkigen._mlir.dialects import nkipy as nkipy_d


# Values MUST match mlir/include/nkipy/Dialect/NkipyAttrs.td.
# Zero is intentionally reserved: MemRefType::get drops an IntegerAttr(0)
# memorySpace, so zero-valued enum cases cannot be attached to a memref.
_MEM_SPACE_MAP = {"Hbm": 1, "Psum": 2, "Sbuf": 3, "SharedHbm": 4}


class _KnobBuilder:
    """Builder returned by `knob(tensor)`.  Methods are side-effecting and
    emit nkipy.layout / nkipy.tile_op ops; they return ``self`` for
    chaining."""

    __slots__ = ("_tensor", "_value", "_loc", "_tile_op_tile_size")

    def __init__(self, tensor: Union[TracedArray, Any]):
        self._tensor = tensor
        self._value = None
        self._loc = None
        self._tile_op_tile_size: Optional[ir.DenseI64ArrayAttr] = None
        if not isinstance(tensor, TracedArray):
            return
        value = tensor.value
        if value.owner is None:
            return  # block argument; cannot annotate
        self._value = value
        self._loc = value.owner.location

    # ------------------------------------------------------------------
    # Builder methods
    # ------------------------------------------------------------------

    def layout(
        self,
        *,
        partition_dim: Optional[int] = None,
        mem_space: Optional[str] = None,
    ) -> "_KnobBuilder":
        """Declare the tensor's memory placement.

        Physical factorization tile for SBUF is auto-derived from the
        consuming tile_op via indexing maps (InferLayout pass).
        """
        if self._value is None:
            return self

        if mem_space is not None:
            self._validate_mem_space(mem_space)
        if partition_dim is not None:
            self._validate_partition_dim(partition_dim)

        mem_space_attr = _mem_space_attr(mem_space)
        partition_dim_attr = _partition_dim_attr(partition_dim)

        if mem_space_attr is None and partition_dim_attr is None:
            return self

        nkipy_d.LayoutOp(
            target=self._value,
            mem_space=mem_space_attr,
            partition_dim=partition_dim_attr,
            tile_size=None,
            loc=self._loc,
        )
        return self

    def tile_op(
        self,
        *,
        tile_size: Optional[List[int]] = None,
    ) -> "_KnobBuilder":
        """Declare the loop tile for the op producing this tensor.

        ``tile_size`` has one entry per iterator of the producing op,
        in the linalg iterator order — ``tile_size[i]`` applies to
        iterator ``i`` (no reordering).

        - Elementwise: matches output rank.
        - Reduction (e.g. ``np.sum(x[M,N], axis=-1)``): matches input
          rank ([M_t, N_t]); the compiler knows which axis reduces.
        - Matmul (``A[M,K] @ B[K,N] -> C[M,N]``): three iter dims
          ([M_t, N_t, K_t]).
        """
        if self._value is None:
            return self

        if tile_size is not None:
            self._validate_tile_size(tile_size)

        tile_size_attr = _dense_i64_attr(tile_size)

        if tile_size_attr is None:
            return self

        nkipy_d.TileOp(
            target=self._value,
            loop_tile_size=tile_size_attr,
            loc=self._loc,
        )
        self._tile_op_tile_size = tile_size_attr
        return self

    def cache(
        self,
        input_tensor: Union["TracedArray", Any],
        *,
        axis: Optional[List[int]] = None,
        prefetch: bool = False,
    ) -> "_KnobBuilder":
        """Declare SBUF caching for an input when computing this op.

        ``axis`` lists post-tiling loop levels where a cache buffer for
        ``input_tensor`` will exist.  axis=[-1] means innermost (minimal
        staging, no reuse).  The buffer shape at each level is auto-derived
        from loop bounds and indexing maps.
        """
        if self._value is None:
            return self
        if not isinstance(input_tensor, TracedArray):
            return self
        input_value = input_tensor.value
        if input_value is None:
            return self

        if self._tile_op_tile_size is None:
            raise ValueError(
                ".cache() requires a preceding .tile_op(). "
                "Chain as: knob(x).tile_op(...).cache(...)"
            )

        if axis is None:
            axis = [-1]

        num_levels = 2 * len(self._tile_op_tile_size)
        for a in axis:
            if a < -num_levels or a >= num_levels:
                raise ValueError(
                    f"axis={a} is out of bounds. "
                    f"tile_op produces {num_levels} post-tiling loop levels "
                    f"(valid range [{-num_levels}, {num_levels - 1}])"
                )

        axes_attr = ir.DenseI64ArrayAttr.get(axis)
        prefetch_attr = ir.BoolAttr.get(prefetch)

        nkipy_d.CacheOp(
            target=self._value,
            input=input_value,
            axes=axes_attr,
            prefetch=prefetch_attr,
            loc=self._loc,
        )
        return self

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_mem_space(self, mem_space: str) -> None:
        if mem_space not in _MEM_SPACE_MAP:
            raise ValueError(
                f"Invalid mem_space '{mem_space}'. "
                f"Must be one of: {set(_MEM_SPACE_MAP)}"
            )

    def _validate_partition_dim(self, partition_dim: int) -> None:
        if partition_dim < 0:
            raise ValueError(
                f"partition_dim must be non-negative, got {partition_dim}"
            )
        tensor_type = self._value.type
        if hasattr(tensor_type, "shape"):
            rank = len(tensor_type.shape)
            if partition_dim >= rank:
                raise ValueError(
                    f"partition_dim {partition_dim} must be less than "
                    f"tensor rank {rank}"
                )

    def _validate_tile_size(self, tile_size: List[int]) -> None:
        if any(t <= 0 for t in tile_size):
            raise ValueError(
                f"tile_size values must be positive, got {tile_size}"
            )
        # Length is op-specific (output rank for elementwise, input rank
        # for reductions, iter-space rank for matmul). The compiler
        # validates against the iteration domain in KnobDrivenTiling.


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def knob(tensor: Union[TracedArray, Any]) -> _KnobBuilder:
    """Return a builder for annotating ``tensor``.  Usage:

        knob(x).layout(mem_space="Sbuf")
        knob(x).tile_op(tile_size=[64, 64]).layout(mem_space="Sbuf")

    If ``tensor`` is not a TracedArray (e.g. a plain numpy array during
    eager execution), the returned builder is a no-op.
    """
    return _KnobBuilder(tensor)


def fuse(*tensors: Union[TracedArray, Any]) -> None:
    """Hint the compiler to fuse the scf.for loops producing the given
    tensors into a single loop.  Each tensor must already carry a
    matching ``.tile_op(tile_size=...)`` annotation; fusion runs after
    tiling and only succeeds when the resulting loops have identical
    bounds.

    Usage:
        c = a + b
        d = c * 2
        knob.knob(c).tile_op(tile_size=[128, 128])
        knob.knob(d).tile_op(tile_size=[128, 128])
        knob.fuse(c, d)   # one loop instead of two

    Non-TracedArray inputs (eager mode) are skipped, matching the
    no-op behaviour of ``knob()``.
    """
    if len(tensors) < 2:
        raise ValueError(f"fuse() requires at least 2 tensors, got {len(tensors)}")
    values = []
    loc = None
    for t in tensors:
        if not isinstance(t, TracedArray):
            return
        v = t.value
        if v.owner is None:
            return  # block argument; cannot fuse
        values.append(v)
        if loc is None:
            loc = v.owner.location
    nkipy_d.FuseOp(targets=values, loc=loc)


# ----------------------------------------------------------------------
# Attribute helpers
# ----------------------------------------------------------------------


def _mem_space_attr(mem_space: Optional[str]) -> Optional[ir.Attribute]:
    if mem_space is None:
        return None
    return ir.IntegerAttr.get(
        ir.IntegerType.get_signless(32), _MEM_SPACE_MAP[mem_space]
    )


def _partition_dim_attr(partition_dim: Optional[int]) -> Optional[ir.Attribute]:
    if partition_dim is None:
        return None
    return ir.IntegerAttr.get(ir.IntegerType.get_unsigned(32), partition_dim)


def _dense_i64_attr(values: Optional[List[int]]) -> Optional[ir.Attribute]:
    if values is None:
        return None
    return ir.DenseI64ArrayAttr.get(values)
