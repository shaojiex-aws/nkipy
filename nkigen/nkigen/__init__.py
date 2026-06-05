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

# Re-export the ``knob`` submodule at the top level so existing callers can do
# ``from nkigen import knob`` and then ``knob.knob(...)`` / ``knob.fuse(...)``,
# matching the pre-restructure layout where ``nkigen.knob`` was the module and
# ``nkigen.apis.knob`` the function. ``frontend`` deliberately does not import
# the ``knob`` function into its namespace, so this resolves to the module.
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
