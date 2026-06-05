"""Emit control flow: scf.for -> nb.fori_loop.

The pipeline's canonicalize-loop-step pass normalizes every loop to
``0..N step 1``, which maps directly onto ``nb.fori_loop(N, body_fn)``. We emit
the decorator form::

    @nb.fori_loop(N)
    def _(i):
        <body>

Inside a fori_loop the induction variable is a runtime ``Reg``, so plain Python
slices (``i*128 : i*128+128``) are rejected by kb — the indexing layer renders
IV-dependent offsets as ``nb.ds(offset, size)`` instead (see emit_indexing).
"""

from __future__ import annotations

from . import irutils


def _emit_for(gen, op) -> bool:
    """``scf.for %iv = 0 to %n step 1 { body }`` -> ``@nb.fori_loop(n)`` + body.

    Loops are normalized to lb=0, step=1 by canonicalize-loop-step, so only the
    upper bound ``n`` is emitted. (A non-normalized loop would need explicit
    start/step handling, which the current pipeline never produces.)
    """
    ub = op.operation.operands[1]
    body = op.regions[0].blocks[0]
    iv = body.arguments[0]

    n = irutils.const_int(ub)
    n_expr = str(n) if n is not None else "  # TODO: dynamic loop bound"

    iv_name = gen.em.fresh_name("i")
    gen.names[iv] = iv_name
    # Mark the IV as a fori_loop Reg so the indexing layer emits nb.ds(...) for
    # slices whose offset depends on it.
    gen.loop_regs.add(iv)

    gen.em.line(f"@nb.fori_loop({n_expr})")
    gen.em.line(f"def _{iv_name}({iv_name}):")
    with gen.em.indent():
        gen.emit_block(body)
    return True


def register(dispatch: dict) -> None:
    dispatch["scf.for"] = _emit_for
