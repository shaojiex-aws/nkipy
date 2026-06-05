"""
Driver: compilation pipeline orchestration.

Wraps the ``nkipy-opt`` binary and coordinates the multi-phase pass pipeline
(``apply_complete_knob_pipeline``), and exposes the legacy in-process
``apply_passes`` pass manager.
"""

__all__ = []

# Pass manager (in-process upstream MLIR PassManager wrapper).
from .pass_manager import apply_passes

__all__.append("apply_passes")

# nkipy-opt wrapper for passes that can't run from Python. Imported guardedly
# so environments without the binary can still import lighter submodules.
try:
    from .pipeline import (
        get_nkipy_opt_path,
        run_nkipy_opt_passes,
        apply_complete_knob_pipeline,
    )

    __all__.extend([
        "get_nkipy_opt_path",
        "run_nkipy_opt_passes",
        "apply_complete_knob_pipeline",
    ])
except ImportError as e:
    import warnings
    warnings.warn(
        f"nkipy-opt wrapper not available: {e}",
        ImportWarning,
    )
