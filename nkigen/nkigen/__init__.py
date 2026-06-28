"""
NKIPyKernelGen - Lowering from NumPy to NKI compiler

This package provides tools to trace Python functions with NumPy operations
and convert them to MLIR for compilation with neuronxcc.
"""

from .frontend.trace import trace
from .frontend.traced_array import TracedArray
from .frontend.custom_op import CustomOp
from .frontend.knob import knob
from .frontend.control_flow import fori_loop
from .execution.execution import verify_against_numpy
from .driver.pass_manager import apply_passes

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    "trace",
    "knob",
    "fori_loop",
    "TracedArray",
    "CustomOp",
    "verify_against_numpy",
    "apply_passes",
]
