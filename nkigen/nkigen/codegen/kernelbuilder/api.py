"""kernel_builder API surface abstraction.

All knowledge of the concrete ``kernel_builder`` API — module aliases, function
names, argument order, enum paths, dtype spellings — lives here, behind a small
interface. The IR-walking ``emit_*`` layer expresses *intent* ("allocate an
SBUF tile of this shape", "DMA-copy src to dst") and this layer renders the
exact Python call text.

Why the indirection: the kernel_builder API evolves. When it does, we add or
branch a :class:`KernelBuilderAPI` implementation here — a ``KernelBuilderV2``
— without touching the IR walkers in ``__init__``/``emit_*`` or the formatting
in ``emitter``. The entry point selects a version by name (``api_version=``).

This is intentionally a *string-rendering* layer: every method returns a
snippet of Python source, never a live object. The generated text is what the
user reads, edits, and later runs through the standard NKI flow.
"""

from __future__ import annotations

from typing import Protocol


# Integer memory-space markers as they appear in post-Phase-4 memref types
# (matching MemSpaceEnum in NkipyAttrs.td; enum starts at 1).
MEMSPACE_HBM = 1
MEMSPACE_PSUM = 2
MEMSPACE_SBUF = 3
MEMSPACE_SHARED_HBM = 4


class KernelBuilderAPI(Protocol):
    """Renders kernel_builder API calls as Python source strings.

    Implementations are stateless; an :class:`Emitter` collects the imports
    reported by :meth:`imports` and the call strings returned by each method.
    """

    def imports(self) -> list[str]:
        """Import statements the generated module needs (order-independent)."""
        ...

    # -- dtypes / spaces ---------------------------------------------------

    def dtype(self, mlir_elem_type: str) -> str:
        """Render an MLIR element type (e.g. ``f32``) as a kb dtype expr."""
        ...

    def memory_space(self, memspace_int: int) -> str:
        """Render an integer memspace marker as a kb space expr."""
        ...

    # -- memory ------------------------------------------------------------

    def alloc(self, shape: tuple[int, ...], dtype: str, space: str) -> str:
        ...

    def release(self, tile: str) -> str:
        ...

    # -- data movement -----------------------------------------------------

    def dma_copy(self, dst: str, src: str) -> str:
        ...

    def tensor_copy(self, dst: str, src: str) -> str:
        ...

    # -- compute -----------------------------------------------------------

    def tensor_tensor_arith(self, dst: str, lhs: str, rhs: str, arith_op: str) -> str:
        ...

    def tensor_scalar_arith(self, dst: str, src: str, scalar: str, arith_op: str) -> str:
        ...

    def tensor_reduce_arith(self, dst: str, src: str, arith_op: str) -> str:
        ...

    def activation(self, dst: str, src: str, activation: str,
                   bias: str = "0.0", scale: str = "1.0") -> str:
        ...

    def matmul(self, dst: str, stationary: str, moving: str, accum: bool) -> str:
        ...

    def dma_transpose(self, dst: str, src: str, permutation: list[int]) -> str:
        ...

    def memset(self, dst: str, value: str) -> str:
        ...


# ---------------------------------------------------------------------------
# v1 implementation — current kernel_builder API (nki wheel ~0.4)
# ---------------------------------------------------------------------------


class KernelBuilderV1:
    """Renders the current kernel_builder API.

    Module aliases used in generated code::

        import nki.compiler.kernel_builder as nb
        from nki.compiler.kernel_builder import isa as nisa
    """

    NB = "nb"
    NISA = "nisa"

    # MLIR element type -> kb dtype expression.
    _DTYPES = {
        "f32": "nb.float32",
        "f16": "nb.float16",
        "bf16": "nb.bfloat16",
        "f8E4M3": "nb.float8_e4m3",
        "f8E5M2": "nb.float8_e5m2",
        "i32": "nb.int32",
        "i16": "nb.int16",
        "i8": "nb.int8",
        "tf32": "nb.tfloat32",
    }

    _SPACES = {
        MEMSPACE_HBM: "nb.hbm",
        MEMSPACE_PSUM: "nb.psum",
        MEMSPACE_SBUF: "nb.sbuf",
        MEMSPACE_SHARED_HBM: "nb.shared_hbm",
    }

    def imports(self) -> list[str]:
        return [
            "import nki.compiler.kernel_builder as nb",
            "from nki.compiler.kernel_builder import isa as nisa",
        ]

    # -- dtypes / spaces ---------------------------------------------------

    def dtype(self, mlir_elem_type: str) -> str:
        return self._DTYPES.get(mlir_elem_type, f"nb.float32  # TODO dtype {mlir_elem_type!r}")

    def memory_space(self, memspace_int: int) -> str:
        return self._SPACES.get(memspace_int, f"None  # TODO memspace {memspace_int}")

    # -- memory ------------------------------------------------------------

    def alloc(self, shape: tuple[int, ...], dtype: str, space: str) -> str:
        shape_txt = "(" + ", ".join(str(s) for s in shape) + ")"
        return f"{self.NB}.compiler.alloc({shape_txt}, dtype={dtype}, space={space})"

    def release(self, tile: str) -> str:
        return f"{self.NB}.compiler.release({tile})"

    # -- data movement -----------------------------------------------------

    def dma_copy(self, dst: str, src: str) -> str:
        return f"{self.NISA}.dma_copy({dst}, {src})"

    def tensor_copy(self, dst: str, src: str) -> str:
        return f"{self.NISA}.tensor_copy({dst}, {src})"

    # -- compute -----------------------------------------------------------
    #
    # ``arith_op`` / ``activation`` args are bare enum member names (e.g. "Add",
    # "exp") from ops.OpInfo.member; this layer adds the nisa.arith_op. /
    # nisa.activation_function. prefix so the emit layer never spells nisa.*.

    def _arith_op(self, member: str) -> str:
        return f"{self.NISA}.arith_op.{member}"

    def _activation_fn(self, member: str) -> str:
        return f"{self.NISA}.activation_function.{member}"

    def tensor_tensor_arith(self, dst: str, lhs: str, rhs: str, arith_op: str) -> str:
        return (
            f"{self.NISA}.tensor_tensor_arith("
            f"{dst}, {lhs}, {rhs}, op={self._arith_op(arith_op)})"
        )

    def tensor_scalar_arith(self, dst: str, src: str, scalar: str, arith_op: str) -> str:
        return (
            f"{self.NISA}.tensor_scalar_arith("
            f"{dst}, {src}, {scalar}, op0={self._arith_op(arith_op)})"
        )

    def tensor_reduce_arith(self, dst: str, src: str, arith_op: str) -> str:
        return f"{self.NISA}.tensor_reduce_arith({dst}, {src}, op={self._arith_op(arith_op)})"

    def activation(self, dst: str, src: str, activation: str,
                   bias: str = "0.0", scale: str = "1.0") -> str:
        return (
            f"{self.NISA}.activation({dst}, {src}, "
            f"bias={bias}, scale={scale}, op={self._activation_fn(activation)})"
        )

    def matmul(self, dst: str, stationary: str, moving: str, accum: bool) -> str:
        return (
            f"{self.NISA}.matmul({dst}, {stationary}, {moving}, accum={accum})"
        )

    def dma_transpose(self, dst: str, src: str, permutation: list[int]) -> str:
        return f"{self.NISA}.dma_transpose({dst}, {src}, permutation={list(permutation)})"

    def memset(self, dst: str, value: str) -> str:
        return f"{self.NISA}.memset({dst}, {value})"


_API_VERSIONS: dict[str, type] = {
    "v1": KernelBuilderV1,
}


def get_api(api_version: str = "v1") -> KernelBuilderV1:
    """Return a kernel_builder API renderer for ``api_version``."""
    try:
        return _API_VERSIONS[api_version]()
    except KeyError:
        raise ValueError(
            f"unknown kernel_builder api_version {api_version!r}; "
            f"known: {sorted(_API_VERSIONS)}"
        )
