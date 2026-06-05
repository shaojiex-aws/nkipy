"""Emit control flow: scf.for (loops).

Phase 2 provides a provisional rendering as a plain Python ``for`` loop so the
walker descends into loop bodies and memory/compute ops inside loops get
emitted. Phase 4 replaces this with ``nb.fori_loop`` and proper loop-carried
value handling.
"""

from __future__ import annotations

from . import irutils
from .emit_indexing import index_expr


def _emit_for(gen, op) -> bool:
    """``scf.for %iv = %lb to %ub step %s { body }`` -> a Python ``for`` loop.

    Provisional (Phase 2): emits ``for iv in range(lb, ub, step):`` and recurses
    into the body. Loop-carried iter_args are not yet handled.
    """
    lb, ub, step = op.operation.operands[0], op.operation.operands[1], op.operation.operands[2]
    body = op.regions[0].blocks[0]
    iv = body.arguments[0]

    iv_name = gen.em.fresh_name("i")
    gen.names[iv] = iv_name

    lb_e = index_expr(gen, lb)
    ub_e = index_expr(gen, ub)
    step_e = index_expr(gen, step)
    # Drop a redundant step of 1 for readability.
    if step_e == "1":
        range_args = f"{lb_e}, {ub_e}" if lb_e != "0" else f"{ub_e}"
    else:
        range_args = f"{lb_e}, {ub_e}, {step_e}"

    gen.em.line(f"for {iv_name} in range({range_args}):")
    with gen.em.indent():
        gen.emit_block(body)
    return True


def register(dispatch: dict) -> None:
    dispatch["scf.for"] = _emit_for
