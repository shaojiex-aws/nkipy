"""
Execution: runtime paths for compiled kernels.

LLVM JIT CPU execution (``llvm``), NEFF compilation (``compile``), and the
NumPy verification dispatcher (``verify_against_numpy``).
"""

from .execution import verify_against_numpy

__all__ = ["verify_against_numpy"]
