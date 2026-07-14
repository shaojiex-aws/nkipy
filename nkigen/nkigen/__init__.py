"""
NKIPyKernelGen - Lowering from NumPy to NKI compiler

This package provides tools to trace Python functions with NumPy operations
and convert them to MLIR for compilation with neuronxcc.
"""

from .frontend.trace import trace
from .frontend.traced_array import TracedArray
from .frontend.knob import knob
from .frontend.control_flow import fori_loop
from .frontend.agent import KernelAgent, EchoAgent
from .execution.execution import verify_against_numpy
from .driver.pass_manager import apply_passes

__version__ = "0.1.0"
__author__ = "Your Name"

# NOTE: user-defined kernels are spliced in via `knob(inputs..., outputs...).use(impl)`,
# where `impl` is a kernel_builder function (spliced directly) or a KernelAgent that
# transforms the region's kernel_builder source (str -> str). The internal `CustomOp`
# bridge (frontend/custom_op.py) compiles the resulting kernel to NISA; it is
# intentionally not a public export.

__all__ = [
    "trace",
    "knob",
    "fori_loop",
    "TracedArray",
    "KernelAgent",
    "EchoAgent",
    "verify_against_numpy",
    "apply_passes",
]
