"""
End-to-end tests for .cache() primitive.

Tests that explicit .cache() annotations correctly control SBUF promotion
for elementwise, reduction, and matmul operations.

Run with: pytest tests/e2e/test_cache.py -v
"""

import pytest
import numpy as np

from nkigen import trace, knob
from harness import run_kernel_test, Mode


# ============================================================================
# Elementwise with .cache()
# ============================================================================


def test_elementwise_cache_all_inputs():
    """Cache all inputs of a binary elementwise op (HW requirement)."""

    @trace(input_specs=[((256, 256), "f32"), ((256, 256), "f32")])
    def kernel(a, b):
        c = a + b
        knob(c).tile_op(tile_size=[128, 128]).layout(
            mem_space="SharedHbm"
        ).cache(a, axis=[-1]).cache(b, axis=[-1])
        return c

    run_kernel_test(
        kernel,
        check_ir_contains=["nisa.tensor_tensor_arith"],
        modes=Mode.HW | Mode.STRING_CHECK,
    )


def test_elementwise_cache_both_inputs():
    """Cache both inputs of a binary elementwise op."""

    @trace(input_specs=[((256, 256), "f32"), ((256, 256), "f32")])
    def kernel(a, b):
        c = a + b
        knob(c).tile_op(tile_size=[128, 128]).layout(
            mem_space="SharedHbm"
        ).cache(a, axis=[-1]).cache(b, axis=[-1])
        return c

    run_kernel_test(
        kernel,
        check_ir_contains=["nisa.tensor_tensor_arith"],
        modes=Mode.HW | Mode.STRING_CHECK,
    )


def test_elementwise_cache_broadcast():
    """Cache a broadcast input (bias) at outermost loop level for reuse.

    Uses (128, 1) to broadcast along free dim only (partition dim must match).
    See docs/2026-06-24-broadcast-rank-mismatch-bug.md for rank-mismatch issue.
    """

    @trace(input_specs=[((256, 256), "f32"), ((256, 1), "f32")])
    def kernel(x, bias):
        y = x + bias
        knob(y).tile_op(tile_size=[128, 128]).layout(
            mem_space="SharedHbm"
        ).cache(x, axis=[-1]).cache(bias, axis=[-1])
        return y

    run_kernel_test(
        kernel,
        check_ir_contains=["nisa.tensor_scalar_arith"],
        modes=Mode.HW | Mode.STRING_CHECK,
    )


# ============================================================================
# Reduction with .cache()
# ============================================================================


def test_reduction_cache_input():
    """Cache the input of a reduction at innermost level."""

    @trace(input_specs=[((256, 256), "f32")])
    def kernel(x):
        sq = np.square(x)
        knob(sq).tile_op(tile_size=[128, 128]).layout(mem_space="Sbuf")

        result = np.sum(sq, axis=-1, keepdims=True)
        knob(result).tile_op(tile_size=[128, 128]).layout(
            mem_space="SharedHbm"
        ).cache(sq, axis=[-1])
        return result

    run_kernel_test(
        kernel,
        check_ir_contains=["nisa.tensor_reduce_arith"],
        modes=Mode.HW | Mode.STRING_CHECK,
        rtol=1e-3,
        atol=1e-3,
    )


# ============================================================================
# Matmul with .cache()
# ============================================================================


def test_matmul_cache_both():
    """Matmul with explicit .cache() on both LHS and RHS."""

    @trace(input_specs=[((256, 128), "f32"), ((128, 256), "f32")])
    def kernel(a, b):
        c = a @ b
        knob(c).tile_op(tile_size=[128, 128, 128]).layout(
            mem_space="SharedHbm"
        ).cache(a, axis=[0]).cache(b, axis=[2])
        return c

    run_kernel_test(
        kernel,
        check_ir_contains=["nisa.matmul"],
        modes=Mode.HW | Mode.STRING_CHECK,
    )


def test_matmul_large_cache_both():
    """Larger matmul with .cache() to exercise blocking (blocksM=2, blocksN=2)."""

    @trace(input_specs=[((512, 256), "f32"), ((256, 512), "f32")])
    def kernel(a, b):
        c = a @ b
        knob(c).tile_op(tile_size=[128, 128, 128]).layout(
            mem_space="SharedHbm"
        ).cache(a, axis=[0]).cache(b, axis=[2])
        return c

    run_kernel_test(
        kernel,
        check_ir_contains=["nisa.matmul"],
        modes=Mode.HW | Mode.STRING_CHECK,
    )


# ============================================================================
# Validation tests (should raise at trace time)
# ============================================================================


def test_cache_without_tile_op_raises():
    """Cache without a preceding tile_op should raise ValueError."""

    @trace(input_specs=[((128, 128), "f32"), ((128, 128), "f32")])
    def kernel(a, b):
        c = a + b
        knob(c).cache(a, axis=[-1])
        return c

    with pytest.raises(ValueError, match="requires a preceding .tile_op"):
        kernel.to_mlir()


def test_cache_axis_out_of_bounds_raises():
    """Cache with axis beyond the post-tiling loop levels should raise."""

    @trace(input_specs=[((128, 128), "f32"), ((128, 128), "f32")])
    def kernel(a, b):
        c = a + b
        knob(c).tile_op(tile_size=[128, 128]).cache(a, axis=[5])
        return c

    with pytest.raises(ValueError, match="out of bounds"):
        kernel.to_mlir()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
