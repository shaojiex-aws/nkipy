"""
End-to-end tests for user-defined kernels via ``knob(...).use(kernel)``.

A ``knob(inputs..., outputs...).use(kernel)`` call names the boundary tensors of
a traced subgraph; the region between them is extracted and replaced by a call to
a plain kernel_builder ``kernel`` (params = inputs then outputs). The kernel's
NISA body is inlined by the resolve-custom-ops step inside ``linalg-to-nisa``.

Run with: pytest tests/e2e/test_custom_op.py -v
"""

import pytest
import numpy as np

from nkigen import trace, knob
from harness import run_kernel_test, Mode

import nki.compiler.kernel_builder as nb


# ============================================================================
# kernel_builder kernels used as .use() targets
# ============================================================================


def _silu_kernel_256(in_hbm, out_hbm):
    """SiLU over a 256x256 buffer, tiled 128x128 internally."""
    import nki.language as nl

    M = N = 256
    tile = 128
    for r in nl.affine_range(M // tile):
        for t in nl.affine_range(N // tile):
            x_sb = nb.ndarray((tile, tile), in_hbm.dtype, nb.sbuf)
            nb.isa.dma_copy(dst=x_sb,
                            src=in_hbm[r * tile:(r + 1) * tile, t * tile:(t + 1) * tile])
            o_sb = nb.ndarray((tile, tile), in_hbm.dtype, nb.sbuf)
            bias = nb.ndarray((tile, 1), in_hbm.dtype, nb.sbuf)
            nb.isa.memset(dst=bias, value=0.0)
            scale = nb.ndarray((tile, 1), in_hbm.dtype, nb.sbuf)
            nb.isa.memset(dst=scale, value=1.0)
            nb.isa.activation(dst=o_sb, src=x_sb, bias=bias, scale=scale,
                              op=nb.isa.activation_function.silu)
            nb.isa.dma_copy(
                dst=out_hbm[r * tile:(r + 1) * tile, t * tile:(t + 1) * tile],
                src=o_sb)


def _relu_kernel_128(x_hbm, out_hbm):
    """Single-tile ReLU: load, activate, store."""
    x_sb = nb.ndarray((128, 128), x_hbm.dtype, nb.sbuf)
    nb.isa.dma_copy(dst=x_sb, src=x_hbm[0:128, 0:128])
    o_sb = nb.ndarray((128, 128), x_hbm.dtype, nb.sbuf)
    bias = nb.ndarray((128, 1), x_hbm.dtype, nb.sbuf)
    nb.isa.memset(dst=bias, value=0.0)
    scale = nb.ndarray((128, 1), x_hbm.dtype, nb.sbuf)
    nb.isa.memset(dst=scale, value=1.0)
    nb.isa.activation(dst=o_sb, src=x_sb, bias=bias, scale=scale,
                      op=nb.isa.activation_function.relu)
    nb.isa.dma_copy(dst=out_hbm[0:128, 0:128], src=o_sb)


# ============================================================================
# Test: .use() replaces an elementwise region (SiLU) after a matmul
# ============================================================================


def test_use_replaces_silu_region():
    """A matmul stays in NumPy-traced land; the SiLU that follows is replaced by
    a kernel_builder SiLU kernel via ``knob(mm, y).use(...)``.

    ``mm`` is produced by the (external) matmul so it is the region *input*;
    ``y`` is produced by the SiLU ops so it is the *output*. Numerically the
    replaced region computes exactly SiLU(mm), matching the traced ops.
    """

    @trace(input_specs=[((256, 256), "f32"), ((256, 256), "f32")])
    def matmul_silu_kernel(x, weight):
        mm = np.matmul(x, weight)
        knob(mm).tile_op(tile_size=[128, 128, 128]).layout(mem_space="SharedHbm")
        y = mm * (1.0 / (1.0 + np.exp(-mm)))  # SiLU
        knob(mm, y).use(_silu_kernel_256)
        return y

    run_kernel_test(
        matmul_silu_kernel,
        check_ir_contains=[
            "nisa.matmul",       # main-kernel matmul survives
            "nisa.activation",   # inlined SiLU
            "nisa.dma_copy",
            "nisa.memset",
            "nisa.target",
        ],
        check_ir_not_contains=[
            "nkipy.custom_op_bodies",
            "nkipy.custom_op",
            "call @__custom_op",
        ],
        rtol=1e-3,
        atol=1e-3,
        modes=Mode.HW | Mode.STRING_CHECK | Mode.CODEGEN,
    )


# ============================================================================
# Test: whole single-input region replaced by one kernel (block-arg input)
# ============================================================================


def test_use_replaces_elementwise_cone():
    """``knob(x, y).use(k)`` replaces an entire elementwise region: x is the
    block-arg input, y is the output. The kernel computes SiLU(x), matching the
    traced region exactly."""

    @trace(input_specs=[((256, 256), "f32")])
    def model(x):
        y = x * (1.0 / (1.0 + np.exp(-x)))  # SiLU(x)
        knob(x, y).use(_silu_kernel_256)
        return y

    run_kernel_test(
        model,
        check_ir_contains=["nisa.activation", "nisa.dma_copy", "nisa.memset", "nisa.target"],
        check_ir_not_contains=[
            "nkipy.custom_op_bodies", "nkipy.custom_op", "call @__custom_op",
        ],
        rtol=1e-3,
        atol=1e-3,
        modes=Mode.HW | Mode.STRING_CHECK | Mode.CODEGEN,
    )


# ============================================================================
# Test: chained .use() (result of one region feeds the next)
# ============================================================================


def test_use_chained():
    """Two .use() regions where the first output feeds the second. Exercises
    call-result mem_space consistency and dedup (same kernel+shapes → one decl,
    two call sites)."""

    @trace(input_specs=[((128, 128), "f32")])
    def model(x):
        a = np.maximum(x, 0.0)          # relu-ish region 1
        knob(x, a).use(_relu_kernel_128)
        b = np.maximum(a, 0.0)          # relu-ish region 2
        knob(a, b).use(_relu_kernel_128)
        return b

    run_kernel_test(
        model,
        check_ir_contains=["nisa.activation", "nisa.dma_copy", "nisa.target"],
        check_ir_not_contains=[
            "nkipy.custom_op_bodies", "nkipy.custom_op", "call @__custom_op",
        ],
        rtol=1e-3,
        atol=1e-3,
        modes=Mode.HW | Mode.STRING_CHECK | Mode.CODEGEN,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
