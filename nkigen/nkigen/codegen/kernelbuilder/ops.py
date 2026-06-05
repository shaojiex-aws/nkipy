"""Single source of truth for op classification.

Every fact about how a linalg op maps to a NISA operation — its arith_op /
activation_function member *and* the role-name stem used for the tile it writes
— lives here, so adding an op means touching one table, not four.

Two tables:

- ``LINALG_OPS``: named linalg ops (linalg.add, linalg.exp, linalg.matmul_…)
  -> :class:`OpInfo`.
- ``ARITH_BODY_OPS``: the arith op inside a single-op ``linalg.generic`` body
  (arith.addf, arith.mulf, …) -> :class:`OpInfo`, so a generic is classified
  directly without a fake "linalg.add" intermediary.

``OpInfo.member`` is the bare enum member name (``"Add"``, ``"exp"``); the
``api`` layer prefixes it with ``nisa.arith_op.`` / ``nisa.activation_function.``
when rendering. The emit layer never spells out a ``nisa.*`` string.
"""

from __future__ import annotations

from dataclasses import dataclass


# Op categories — which NISA builder a linalg op lowers to.
ARITH = "arith"          # nisa.tensor_tensor_arith / tensor_scalar_arith / reduce
ACTIVATION = "activation"  # nisa.activation
MATMUL = "matmul"        # nisa.matmul
TRANSPOSE = "transpose"  # nisa.dma_transpose
FILL = "fill"            # nisa.memset


@dataclass(frozen=True)
class OpInfo:
    kind: str           # one of the category constants above
    member: str = ""    # enum member name (e.g. "Add" for arith_op, "exp" for activation)
    role: str = ""      # tile-name stem for the value this op writes (e.g. "add", "exp")


LINALG_OPS: dict[str, OpInfo] = {
    # Binary arithmetic -> nisa.arith_op.<member>
    "linalg.add": OpInfo(ARITH, "Add", "add"),
    "linalg.sub": OpInfo(ARITH, "Subtract", "sub"),
    "linalg.mul": OpInfo(ARITH, "Multiply", "mul"),
    "linalg.max": OpInfo(ARITH, "Max", "max"),
    "linalg.min": OpInfo(ARITH, "Min", "min"),
    "linalg.div": OpInfo(ARITH, "Divide", "div"),
    # Unary activations -> nisa.activation_function.<member>
    "linalg.exp": OpInfo(ACTIVATION, "exp", "exp"),
    "linalg.tanh": OpInfo(ACTIVATION, "tanh", "tanh"),
    "linalg.log": OpInfo(ACTIVATION, "log", "log"),
    "linalg.sqrt": OpInfo(ACTIVATION, "sqrt", "sqrt"),
    "linalg.abs": OpInfo(ACTIVATION, "abs", "abs"),
    "linalg.square": OpInfo(ACTIVATION, "square", "square"),
    "linalg.reciprocal": OpInfo(ACTIVATION, "reciprocal", "recip"),
    "linalg.rsqrt": OpInfo(ACTIVATION, "rsqrt", "rsqrt"),
    "linalg.sigmoid": OpInfo(ACTIVATION, "sigmoid", "sigmoid"),
    # Structural compute ops
    "linalg.matmul_transpose_a": OpInfo(MATMUL, role="matmul"),
    "linalg.transpose": OpInfo(TRANSPOSE, role="transpose"),
    "linalg.fill": OpInfo(FILL, role="fill"),
}


# arith op inside a single-op linalg.generic body -> the arith_op it realizes.
# Used for both elementwise (tensor_tensor / tensor_scalar) and reduction
# generics; both render the member via nisa.arith_op.<member>.
ARITH_BODY_OPS: dict[str, OpInfo] = {
    "arith.addf": OpInfo(ARITH, "Add", "add"),
    "arith.addi": OpInfo(ARITH, "Add", "add"),
    "arith.subf": OpInfo(ARITH, "Subtract", "sub"),
    "arith.subi": OpInfo(ARITH, "Subtract", "sub"),
    "arith.mulf": OpInfo(ARITH, "Multiply", "mul"),
    "arith.muli": OpInfo(ARITH, "Multiply", "mul"),
    "arith.divf": OpInfo(ARITH, "Divide", "div"),
    "arith.maximumf": OpInfo(ARITH, "Max", "max"),
    "arith.minimumf": OpInfo(ARITH, "Min", "min"),
}
