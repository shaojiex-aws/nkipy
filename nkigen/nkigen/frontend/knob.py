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
from nkigen.mlir_utils import MEM_SPACE_MAP, mem_space_attr


class _KnobBuilder:
    """Builder returned by `knob(t1, ...)`.

    Methods are side-effecting (emit nkipy ops) and return ``self``
    for chaining. In eager mode (no TracedArrays), `knob()` returns
    a _NoOpKnobBuilder instead.
    """

    __slots__ = ("_values", "_locs", "_tile_op_tile_size")

    def __init__(self, *tensors: Union[TracedArray, Any]):
        self._values: List = []
        self._locs: List = []
        self._tile_op_tile_size: Optional[ir.DenseI64ArrayAttr] = None

        has_traced = False
        for i, t in enumerate(tensors):
            if not isinstance(t, TracedArray):
                continue
            has_traced = True
            v = t.value
            if v.owner is None:
                continue
            self._values.append(v)
            self._locs.append(v.owner.location)

        if has_traced:
            for i, t in enumerate(tensors):
                if not isinstance(t, TracedArray):
                    raise TypeError(
                        f"knob() argument {i} is {type(t).__name__}, "
                        f"expected TracedArray."
                    )

    # ------------------------------------------------------------------
    # Builder methods
    # ------------------------------------------------------------------

    def layout(
        self,
        *,
        partition_dim: Optional[int] = None,
        mem_space: Optional[str] = None,
    ) -> "_KnobBuilder":
        """Declare the tensor's memory placement."""
        if not self._values:
            return self

        if mem_space is not None:
            self._validate_mem_space(mem_space)
        if partition_dim is not None:
            if mem_space is not None and mem_space != "Sbuf":
                raise ValueError(
                    f"partition_dim is only valid with mem_space='Sbuf', "
                    f"got mem_space='{mem_space}'. "
                    f"HBM has no partition/free dimension concept."
                )

        ms_attr = _mem_space_attr(mem_space)
        pdim_attr = _partition_dim_attr(partition_dim)
        if ms_attr is None and pdim_attr is None:
            return self

        for value, loc in zip(self._values, self._locs):
            if partition_dim is not None:
                self._validate_partition_dim(value, partition_dim)
            nkipy_d.LayoutOp(
                target=value, mem_space=ms_attr,
                partition_dim=pdim_attr, tile_size=None, loc=loc,
            )
        return self

    def tile_op(
        self,
        *,
        tile_size: Optional[List[int]] = None,
    ) -> "_KnobBuilder":
        """Declare the loop tile for the op producing this tensor.

        ``tile_size`` has one entry per iterator of the producing op,
        in the linalg iterator order.
        """
        if not self._values:
            return self

        if tile_size is not None:
            self._validate_tile_size(tile_size)

        tile_size_attr = _dense_i64_attr(tile_size)
        if tile_size_attr is None:
            return self

        for value, loc in zip(self._values, self._locs):
            nkipy_d.TileOp(target=value, loop_tile_size=tile_size_attr, loc=loc)
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
        staging, no reuse).
        """
        if not self._values:
            return self
        if not isinstance(input_tensor, TracedArray):
            raise TypeError(
                f".cache() input_tensor must be TracedArray, "
                f"got {type(input_tensor).__name__}"
            )
        input_value = input_tensor.value

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
        prefetch_attr = ir.BoolAttr.get(True) if prefetch else None

        for value, loc in zip(self._values, self._locs):
            nkipy_d.CacheOp(
                target=value, input=input_value,
                axes=axes_attr, prefetch=prefetch_attr, loc=loc,
            )
        return self

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_mem_space(self, mem_space: str) -> None:
        if mem_space not in MEM_SPACE_MAP:
            raise ValueError(
                f"Invalid mem_space '{mem_space}'. "
                f"Must be one of: {set(MEM_SPACE_MAP)}"
            )

    def _validate_partition_dim(self, value, partition_dim: int) -> None:
        if partition_dim < 0:
            raise ValueError(
                f"partition_dim must be non-negative, got {partition_dim}"
            )
        tensor_type = value.type
        if hasattr(tensor_type, "shape"):
            rank = len(tensor_type.shape)
            if partition_dim >= rank:
                raise ValueError(
                    f"partition_dim {partition_dim} must be less than "
                    f"tensor rank {rank}"
                )

    def fuse(self) -> "_KnobBuilder":
        """Fuse the scf.for loops producing these tensors into one loop.

        Requires 2+ tensors, each with a matching .tile_op() annotation.
        """
        if not self._values:
            return self
        if len(self._values) < 2:
            raise ValueError(
                f"fuse() requires at least 2 tensors, got {len(self._values)}"
            )
        nkipy_d.FuseOp(targets=self._values, loc=self._locs[0])
        return self

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


def knob(*tensors: Union[TracedArray, Any]) -> _KnobBuilder:
    """Return a builder for annotating tensors. Usage:

        knob(x).tile_op(tile_size=[64, 64]).layout(mem_space="Sbuf")
        knob(a, b).fuse()
    """
    return _KnobBuilder(*tensors)


# ----------------------------------------------------------------------
# Attribute helpers
# ----------------------------------------------------------------------


def _mem_space_attr(mem_space: Optional[str]) -> Optional[ir.Attribute]:
    if mem_space is None:
        return None
    return mem_space_attr(mem_space)


def _partition_dim_attr(partition_dim: Optional[int]) -> Optional[ir.Attribute]:
    if partition_dim is None:
        return None
    return ir.IntegerAttr.get(ir.IntegerType.get_unsigned(32), partition_dim)


def _dense_i64_attr(values: Optional[List[int]]) -> Optional[ir.Attribute]:
    if values is None:
        return None
    return ir.DenseI64ArrayAttr.get(values)
