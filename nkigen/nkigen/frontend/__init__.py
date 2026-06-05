"""
Frontend: NumPy -> linalg MLIR tracing.

Traces ``@trace``-decorated NumPy functions into MLIR (linalg/memref/scf/arith)
via TracedArray, the op vtable, and the IR builder.

The in-trace API *functions* (``knob``, ``fori_loop``) are intentionally NOT
re-exported here: binding the ``knob`` function in this namespace would shadow
the ``knob`` submodule, breaking ``from nkigen import knob`` (module) usage.
Import those functions from :mod:`nkigen.apis` instead.
"""

from .trace import trace
from .traced_array import TracedArray
from .custom_op import CustomOp

__all__ = [
    "trace",
    "TracedArray",
    "CustomOp",
]
