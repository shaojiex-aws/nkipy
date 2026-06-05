"""Emit memory operations: alloc, release (dealloc), and DMA copy.

Each handler takes ``(gen, op)`` where ``gen`` is the ``_ModuleEmitter`` and
``op`` is the upstream-MLIR op, emits zero or more lines via ``gen.em``, and
returns ``True`` if it produced an executable statement.

Handlers are wired into the walker's dispatch table by :func:`register`.
"""

from __future__ import annotations

from . import irutils
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
    memspace = irutils.memref_memspace(ty)
    dtype = gen.api.dtype(irutils.memref_elem_type(ty))
    space = gen.api.memory_space(memspace)

    name = gen.em.fresh_name(gen.tile_hint(op, memspace))
    gen.names[result] = name
    gen.em.line(f"{name} = {gen.api.alloc(shape, dtype, space)}")
    return True


def _emit_dealloc(gen, op) -> bool:
    """``memref.dealloc`` -> ``nb.compiler.release(tile)``."""
    target = op.operation.operands[0]
    gen.em.line(gen.api.release(memref_expr(gen, target)))
    return True


def _emit_copy(gen, op) -> bool:
    """``memref.copy src, dst`` -> ``nisa.dma_copy(dst, src)``.

    MLIR's ``memref.copy`` is ``(source, target)``; kb's ``dma_copy`` is
    ``(dst, src)``, so the operands are swapped.
    """
    src = op.operation.operands[0]
    dst = op.operation.operands[1]
    src_expr = memref_expr(gen, src)
    dst_expr = memref_expr(gen, dst)
    gen.em.line(gen.api.dma_copy(dst_expr, src_expr))
    return True


def register(dispatch: dict) -> None:
    dispatch["memref.alloc"] = _emit_alloc
    dispatch["memref.dealloc"] = _emit_dealloc
    dispatch["memref.copy"] = _emit_copy
