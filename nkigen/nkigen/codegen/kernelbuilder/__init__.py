"""
KernelBuilder Python codegen backend.

Walks the post-Phase-4 IR (memref + scf + linalg + arith, with nkipy
annotations) and emits equivalent ``kernel_builder`` Python source code as
text. See ``linalg_to_kernelbuilder``.
"""

__all__ = []
