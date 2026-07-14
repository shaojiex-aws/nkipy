"""
Unit tests for the user-defined-kernel path: the ``knob(...).use(kernel)`` verb
and the internal ``CustomOp`` bridge it builds on.

``CustomOp`` is no longer a public API (users write ``knob(...).use(kernel)``),
so these tests exercise:
  * the internal bridge ``CustomOp.from_kernel_builder`` + ``emit_custom_op_declaration``
  * ``.use()`` boundary classification, extraction, and its error paths

Run with: python -m pytest tests/unit/test_custom_op.py -v
"""

import pytest
import numpy as np

from nkigen import trace, knob
from nkigen.frontend.custom_op import CustomOp, emit_custom_op_declaration

import nki.compiler.kernel_builder as nb


# ============================================================================
# Internal bridge: from_kernel_builder
# ============================================================================


def _relu_kernel(x_hbm, out_hbm):
    x_sbuf = nb.ndarray((128, 128), x_hbm.dtype, nb.sbuf)
    nb.isa.dma_copy(dst=x_sbuf, src=x_hbm[0:128, 0:128])
    out_sbuf = nb.ndarray((128, 128), x_hbm.dtype, nb.sbuf)
    bias = nb.ndarray((128, 1), x_hbm.dtype, nb.sbuf)
    nb.isa.memset(dst=bias, value=0.0)
    scale = nb.ndarray((128, 1), x_hbm.dtype, nb.sbuf)
    nb.isa.memset(dst=scale, value=1.0)
    nb.isa.activation(dst=out_sbuf, src=x_sbuf, bias=bias, scale=scale,
                      op=nb.isa.activation_function.relu)
    nb.isa.dma_copy(dst=out_hbm[0:128, 0:128], src=out_sbuf)


def test_from_kernel_builder_produces_nisa():
    """The bridge compiles a kernel_builder fn to NISA MLIR + records shapes."""
    op = CustomOp.from_kernel_builder(
        kernel_func=_relu_kernel,
        input_specs={"x_hbm": nb.Tensor((128, 128), nb.float32, nb.shared_hbm)},
        output_specs={"out_hbm": nb.Tensor((128, 128), nb.float32, nb.shared_hbm)},
    )
    assert op.func_name.startswith("__custom_op___relu_kernel_")
    assert op.input_shapes == [(128, 128)]
    assert op.output_shapes == [(128, 128)]
    for op_name in ["nisa.dma_copy", "nisa.activation", "nisa.memset"]:
        assert op_name in op.nisa_mlir or f'"{op_name}"' in op.nisa_mlir


def test_emit_custom_op_declaration():
    """Declaration is a body-less private func with the nkipy.custom_op marker."""
    from mlir import ir

    op = CustomOp(
        nisa_mlir="module {}",
        func_name="act_128x128_128x128",
        input_names=["x"],
        output_names=["y"],
        input_shapes=[(128, 128)],
        output_shapes=[(128, 128)],
        input_dtypes=["f32"],
        output_dtypes=["f32"],
    )
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            emit_custom_op_declaration(op)
        mlir_str = str(module)
        assert "private" in mlir_str
        assert "@__custom_op__act_128x128_128x128" in mlir_str
        assert "nkipy.custom_op" in mlir_str
        assert "128x128xf32" in mlir_str


# ============================================================================
# .use(): tracing emits a resolved func.call
# ============================================================================


def test_use_emits_call_and_stashes_body():
    """knob(x, y).use(kernel) extracts the region → func.call + stashed body."""

    @trace(input_specs=[((128, 128), "f32")])
    def kernel(x):
        y = np.maximum(x, 0.0)
        knob(x, y).use(_relu_kernel)
        return y

    mlir_str = str(kernel.to_mlir())
    assert "call @__custom_op___relu_kernel" in mlir_str
    assert "nkipy.custom_op" in mlir_str
    assert "nkipy.custom_op_bodies" in mlir_str
    # The region ops are gone (replaced by the call).
    assert "linalg.max" not in mlir_str and "linalg.generic" not in mlir_str


def test_use_dedup_same_kernel_shapes():
    """Two .use() calls with the same kernel+shapes → one decl, two call sites."""

    @trace(input_specs=[((128, 128), "f32")])
    def kernel(x):
        a = np.maximum(x, 0.0)
        knob(x, a).use(_relu_kernel)
        b = np.maximum(a, 0.0)
        knob(a, b).use(_relu_kernel)
        return b

    mlir_str = str(kernel.to_mlir())
    assert mlir_str.count("call @__custom_op___relu_kernel") == 2
    assert mlir_str.count("func.func private @__custom_op___relu_kernel") == 1


# ============================================================================
# .use(): error paths
# ============================================================================


def test_use_no_output_raises():
    """All boundaries are inputs (no produced tensor) → error."""

    @trace(input_specs=[((128, 128), "f32"), ((128, 128), "f32")])
    def kernel(x, w):
        y = x + w
        knob(x, w).use(_relu_kernel)  # both inputs, no output between them
        return y

    with pytest.raises(ValueError, match="no outputs found"):
        kernel.to_mlir()


def test_use_arity_mismatch_raises():
    """Kernel param count must equal #inputs + #outputs."""

    @trace(input_specs=[((128, 128), "f32"), ((128, 128), "f32")])
    def kernel(x, w):
        mm = np.matmul(x, w)
        knob(x, w, mm).use(_relu_kernel)  # 3 boundaries, kernel takes 2
        return mm

    with pytest.raises(ValueError, match="parameters but the region"):
        kernel.to_mlir()


def test_use_undeclared_input_raises():
    """A value that feeds the region but isn't a declared input → error."""

    @trace(input_specs=[((128, 128), "f32"), ((128, 128), "f32")])
    def kernel(x, b):
        mm = np.matmul(x, x)
        y = mm + b  # y depends on b, but knob only lists x
        knob(x, y).use(_relu_kernel)
        return y

    with pytest.raises(ValueError, match="feeds the region but"):
        kernel.to_mlir()


def test_use_verify_true_not_implemented():
    """verify=True is reserved for numeric verification (not yet implemented)."""

    @trace(input_specs=[((128, 128), "f32")])
    def kernel(x):
        y = np.maximum(x, 0.0)
        knob(x, y).use(_relu_kernel, verify=True)
        return y

    with pytest.raises(NotImplementedError, match="verify"):
        kernel.to_mlir()


def test_use_subview_output_raises():
    """An output that is a slice of a larger, still-used buffer is rejected
    (replacing only the view's uses would leave a dangling base)."""

    @trace(input_specs=[((128, 128), "f32")])
    def kernel(x):
        big = np.maximum(x, 0.0)
        y = big[0:64, :]            # output is a subview of `big`
        knob(x, y).use(_relu_kernel)
        return big                  # `big` still used → shared base

    with pytest.raises(ValueError, match="view of a buffer"):
        kernel.to_mlir()


def test_use_interior_layout_hint_erased():
    """A knob().layout() hint on an interior temp inside a .use() region is
    erased with the region (regression: it used to leave a dangling op → crash)."""

    @trace(input_specs=[((128, 128), "f32")])
    def kernel(x):
        t = np.maximum(x, 0.0)
        knob(t).layout(mem_space="Sbuf")   # hint on interior temp
        y = np.maximum(t, 0.0)
        knob(x, y).use(_relu_kernel)
        return y

    mlir_str = str(kernel.to_mlir())
    assert "call @__custom_op___relu_kernel" in mlir_str
    assert "nkipy.layout" not in mlir_str  # interior hint erased with region


def test_use_eager_is_noop():
    """Called outside tracing (real arrays), .use() is a no-op."""
    x = np.ones((128, 128), dtype=np.float32)
    y = np.maximum(x, 0.0)
    # knob() with plain ndarrays returns a no-op builder; .use() must not raise.
    knob(x, y).use(_relu_kernel)


# ============================================================================
# Agent path: .use(agent) and .tune()
# ============================================================================


def test_use_echo_agent_traces():
    """.use(EchoAgent()) routes the region through emit → agent → re-materialize
    → splice, producing the same func.call + stashed body as a direct kernel."""
    from nkigen import EchoAgent

    @trace(input_specs=[((128, 128), "f32")])
    def model(x):
        y = np.maximum(x, 0.0)
        knob(x, y).use(EchoAgent())
        return y

    mlir_str = str(model.to_mlir())
    assert "call @__custom_op__region" in mlir_str
    assert "nkipy.custom_op_bodies" in mlir_str
    assert "linalg." not in mlir_str  # region replaced by the call


def test_use_plain_str_agent():
    """A plain ``str -> str`` callable works as an agent (lightweight form)."""

    seen = {}

    def agent(source: str) -> str:
        seen["src"] = source
        return source  # identity

    @trace(input_specs=[((128, 128), "f32")])
    def model(x):
        y = np.maximum(x, 0.0)
        knob(x, y).use(agent)
        return y

    mlir_str = str(model.to_mlir())
    assert "call @__custom_op__region" in mlir_str
    assert "nb.compiler.alloc" in seen["src"]  # agent saw kernel_builder source


def test_tune_creates_agent_workspace(tmp_path):
    """prog.tune(db=...) gives each agent site a persistent <db>/<key>/ folder
    with the emitted region source, and lets the agent write into it."""
    from nkigen.frontend.agent import AgentContext

    class NotingAgent:
        def transform(self, ctx: AgentContext) -> str:
            (ctx.workspace / "memory.txt").write_text(f"key={ctx.key}")
            return ctx.source

    @trace(input_specs=[((128, 128), "f32")])
    def model(x):
        y = x * (1.0 / (1.0 + np.exp(-x)))
        knob(x, y).use(NotingAgent(), key="silu_region")
        return y

    db = tmp_path / "tune_db"
    model.tune(db=str(db), target="trn2")

    site = db / "silu_region"
    assert (site / "region.py").exists()          # nkigen's emitted source
    assert (site / "kernel.py").exists()           # agent's returned source
    assert (site / "memory.txt").read_text() == "key=silu_region"  # agent scratch


# ============================================================================
# CustomOp public API is retired
# ============================================================================


def test_customop_not_public():
    """CustomOp is an internal bridge, not a public export."""
    import nkigen
    assert not hasattr(nkigen, "CustomOp")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
