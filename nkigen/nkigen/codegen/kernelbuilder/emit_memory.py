"""Emit memory operations: alloc, release (dealloc), and DMA copy.

Each handler takes ``(gen, op)`` where ``gen`` is the ``_ModuleEmitter`` and
``op`` is the upstream-MLIR op, emits zero or more lines via ``gen.em``, and
returns ``True`` if it produced an executable statement.

Handlers are wired into the walker's dispatch table by :func:`register`.
"""

from __future__ import annotations

from .. import irutils
from ..irutils import MEMSPACE_PSUM, MEMSPACE_SBUF
from .emit_indexing import memref_expr


def _emit_alloc(gen, op) -> bool:
    """``memref.alloc`` -> ``tile = nb.compiler.alloc(shape, dtype, space)``.

    A returned alloc (HBM output) is bound to its output parameter name instead
    of being allocated, so it never appears as a statement.
    """
    result = op.operation.results[0]
    # Outputs are surfaced as function params; the walker pre-binds them in
    # gen.names, so skip emitting an alloc for an already-named result.
    if result in gen.names:
        return False

    ty = result.type
    shape = tuple(irutils.memref_shape(ty))
    # On-chip tiles must be >= 2-D (SBUF/PSUM are partition x free); a 1-D
    # reduction result becomes (N, 1). HBM tiles keep their rank. A dma_copy
    # between the padded 2-D tile and a 1-D HBM buffer is handled by kb.
    if irutils.is_on_chip(ty) and len(shape) == 1:
        shape = (shape[0], 1)
    dtype = gen.api.dtype(irutils.memref_elem_type(ty))
    space = gen.api.memory_space(irutils.memref_memspace(ty))

    name = gen.em.fresh_name(gen.tile_name(op))
    gen.names[result] = name
    gen.em.line(f"{name} = {gen.api.alloc(shape, dtype, space)}")
    return True


def _emit_dealloc(gen, op) -> bool:
    """``memref.dealloc`` -> ``nb.compiler.release(tile)``."""
    target = op.operation.operands[0]
    gen.em.line(gen.api.release(memref_expr(gen, target)))
    return True


def _emit_copy(gen, op) -> bool:
    """``linalg.copy ins(src) outs(dst)`` -> a DMA or on-chip tensor copy.

    Operands are ``(source, target)`` positionally; the kb calls take
    ``(dst, src)``, so they are swapped. The engine is chosen by
    memory space, mirroring the NISA backend:

    - HBM on either side  -> ``nisa.dma_copy`` (DMA engine).
    - on-chip (SBUF<->SBUF, SBUF<->PSUM) -> ``nisa.tensor_copy`` (DMA can't
      read/write PSUM, and SBUF<->SBUF is cheaper on a compute engine).

    The rare HBM<->PSUM copy needs a two-hop through SBUF (DMA can't touch
    PSUM); that path is emitted explicitly with an intermediate tile.
    """
    src = op.operation.operands[0]
    dst = op.operation.operands[1]
    src_ms = irutils.memref_memspace(src.type)
    dst_ms = irutils.memref_memspace(dst.type)
    src_expr = memref_expr(gen, src)
    dst_expr = memref_expr(gen, dst)

    src_hbm, dst_hbm = irutils.is_hbm(src.type), irutils.is_hbm(dst.type)
    needs_psum_hop = (
        (src_hbm and dst_ms == MEMSPACE_PSUM)
        or (src_ms == MEMSPACE_PSUM and dst_hbm)
    )

    if needs_psum_hop:
        shape = tuple(irutils.memref_shape(dst.type))
        dtype = gen.api.dtype(irutils.memref_elem_type(dst.type))
        inter = gen.em.fresh_name("sbuf")
        gen.em.line(f"{inter} = {gen.api.alloc(shape, dtype, gen.api.memory_space(MEMSPACE_SBUF))}")
        if src_hbm:  # HBM -> SBUF (dma) -> PSUM (tensor)
            gen.em.line(gen.api.dma_copy(inter, src_expr))
            gen.em.line(gen.api.tensor_copy(dst_expr, inter))
        else:        # PSUM -> SBUF (tensor) -> HBM (dma)
            gen.em.line(gen.api.tensor_copy(inter, src_expr))
            gen.em.line(gen.api.dma_copy(dst_expr, inter))
        gen.em.line(gen.api.release(inter))
        return True

    if src_hbm or dst_hbm:
        gen.em.line(gen.api.dma_copy(dst_expr, src_expr))
    else:
        gen.em.line(gen.api.tensor_copy(dst_expr, src_expr))
    return True


def register(dispatch: dict) -> None:
    dispatch["memref.alloc"] = _emit_alloc
    dispatch["memref.dealloc"] = _emit_dealloc
    # linalg.copy is buffer-semantic here (ins=source, outs=target).
    dispatch["linalg.copy"] = _emit_copy
