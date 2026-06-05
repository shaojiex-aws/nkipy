"""Single source of truth for the NKI wheel's private MLIR bindings.

The NISA codegen reaches into ``nki.compiler._internal`` for the IR bindings
and the ``nisa`` dialect builders. Those import paths are private to the wheel
and have moved before; centralizing them here means a future path change is a
one-line edit instead of a 16-file sweep. Modules do::

    from ._vendor import nk_ir, nisa

``up_ir`` (the *upstream* MLIR bindings, used only by ``context`` to bridge
between the two contexts) is intentionally NOT re-exported here — it is a
different package (``mlir``), not part of the NKI wheel.
"""

from nki.compiler._internal import ir as nk_ir  # type: ignore[import-not-found]
from nki.compiler._internal.dialects import nisa  # type: ignore[import-not-found]

__all__ = ["nk_ir", "nisa"]
