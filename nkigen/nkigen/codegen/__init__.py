"""
Codegen backends: post-Phase-4 IR -> target.

Two backends consume the same post-Phase-4 IR (after pass 23, before the
``py:linalg-to-nisa`` pass):

- ``nisa``: lowers to NISA MLIR (consumed by ``ncc_driver`` -> NEFF).
- ``kernelbuilder``: emits readable ``kernel_builder`` Python source.

Submodules are imported lazily by their callers; importing this package does
not pull in the NKI wheel.
"""

__all__ = []
