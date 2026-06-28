"""
NKIPyKernelGen - Lowering from NumPy to NKI compiler

This package provides tools to trace Python functions with NumPy operations
and convert them to MLIR for compilation with neuronxcc.
"""

from .frontend.trace import trace
from .frontend.traced_array import TracedArray
from .frontend.custom_op import CustomOp
from .execution.execution import verify_against_numpy
from .driver.pass_manager import apply_passes
from . import apis

# Re-export the ``knob`` submodule so callers can do
# ``from nkigen import knob`` then ``knob.knob(x).tile_op(...)``
# or ``knob.knob(a, b).fuse()``.
from .frontend import knob

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    "trace",
    "TracedArray",
    "CustomOp",
    "verify_against_numpy",
    "apply_passes",
    "apis",
    "knob",
]
