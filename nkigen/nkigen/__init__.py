"""
NKIPyKernelGen - Lowering from NumPy to NKI compiler

This package provides tools to trace Python functions with NumPy operations
and convert them to MLIR for compilation with neuronxcc.
"""

from .frontend.trace import trace
from .frontend.traced_array import TracedArray
from .frontend.knob import knob
from .frontend.control_flow import fori_loop
from .execution.execution import verify_against_numpy
from .driver.pass_manager import apply_passes

__version__ = "0.1.0"
__author__ = "Your Name"

# NOTE: user-defined kernels are spliced in via `knob(inputs..., outputs...).use(kernel)`.
# The internal `CustomOp` bridge (frontend/custom_op.py) is what `.use()` uses to
# compile a kernel_builder function to NISA; it is intentionally not a public export.

__all__ = [
    "trace",
    "knob",
    "fori_loop",
    "TracedArray",
    "verify_against_numpy",
    "apply_passes",
]
