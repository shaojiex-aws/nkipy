"""
Tests for knob-driven-tiling pass with elementwise operations.

These tests verify simple N-dimensional tiling for elementwise ops like add, sub, etc.
Run with: python -m pytest tests/passes/knob_driven_tiling/test_elementwise.py -v
Or directly: python tests/passes/knob_driven_tiling/test_elementwise.py
"""

import pytest
import numpy as np

from nkigen import trace, knob
from harness import run_kernel_test, Mode

# ============================================================================
# Test Configurations
# ============================================================================

# (shape, tile_size, test_id)
ADD_TEST_CONFIGS = [
    pytest.param((256, 256), [128, 128], id="add_256x256_tile128"),
    pytest.param((512, 512), [128, 128], id="add_512x512_tile128"),
    pytest.param((256, 256), [64, 64], id="add_256x256_tile64"),
    pytest.param((128, 256, 64), [64, 128, 32], id="add_3d_128x256x64"),
]


# ============================================================================
# Test Functions
# ============================================================================

@pytest.mark.parametrize("shape,tile_size", ADD_TEST_CONFIGS)
def test_add_tiling(shape, tile_size):
    """
    Test knob-driven-tiling on linalg.add.

    Pattern: result = A + B

    Elementwise ops use simple single-level tiling (no blocking).
    For 2D: for i in [0, dim0, tile0): for j in [0, dim1, tile1): add_tile
    """
    @trace(input_specs=[(shape, "f32"), (shape, "f32")])
    def add_kernel(a, b):
        result = a + b
        knob(result).tile_op(tile_size=tile_size)
        return result

    # Build FileCheck patterns based on dimensionality
    # Memref backend produces:
    # 1. scf.for loops for tiling
    # 2. memref.subview for each input/output
    # 3. memref.alloc in SBUF + memref.copy for promotion
    # 4. linalg.add on SBUF memrefs
    # 5. memref.copy to write back
    check_patterns = "CHECK: func.func\n"
    for i, (dim, tile) in enumerate(zip(shape, tile_size)):
        check_patterns += f"    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{dim}{{{{.*}}}} step %c{tile}\n"

    tile_shape_x = "x".join(str(t) for t in tile_size)

    # With SBUF promotion: subview + alloc + copy for each operand
    check_patterns += f"    CHECK: memref.subview\n"
    check_patterns += f"    CHECK: memref.alloc() : memref<{tile_shape_x}xf32, #nkipy.mem<Sbuf>>\n"
    check_patterns += f"    CHECK: memref.subview\n"
    check_patterns += f"    CHECK: memref.alloc() : memref<{tile_shape_x}xf32, #nkipy.mem<Sbuf>>\n"
    check_patterns += f"    CHECK: memref.subview\n"
    check_patterns += f"    CHECK: memref.alloc() : memref<{tile_shape_x}xf32, #nkipy.mem<Sbuf>>\n"
    check_patterns += f"    CHECK: linalg.add {{{{.*}}}} memref<{tile_shape_x}xf32, #nkipy.mem<Sbuf>>\n"
    check_patterns += f"    CHECK: linalg.copy\n"

    run_kernel_test(
        add_kernel,
        stop_after='knob-driven-tiling',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


def test_sub_2d():
    """
    Test knob-driven-tiling on linalg.sub.
    """
    shape = (256, 256)
    tile_size = [128, 128]

    @trace(input_specs=[(shape, "f32"), (shape, "f32")])
    def sub_kernel(a, b):
        result = a - b
        knob(result).tile_op(tile_size=tile_size)
        return result

    run_kernel_test(
        sub_kernel,
        stop_after='knob-driven-tiling',
        modes=Mode.LLVM,
    )


def test_mul_2d():
    """
    Test knob-driven-tiling on linalg.mul (element-wise multiplication).
    """
    shape = (256, 256)
    tile_size = [128, 128]

    @trace(input_specs=[(shape, "f32"), (shape, "f32")])
    def mul_kernel(a, b):
        result = a * b
        knob(result).tile_op(tile_size=tile_size)
        return result

    run_kernel_test(
        mul_kernel,
        stop_after='knob-driven-tiling',
        modes=Mode.LLVM,
    )


def test_add_simple():
    """
    Simple test for 256x256 add to verify basic elementwise functionality.

    For 256x256 with tile_size=[128, 128]:
      for i in [0, 256, 128): for j in [0, 256, 128): add([128, 128])
    """
    shape = (256, 256)
    tile_size = [128, 128]
    dim0, dim1 = shape
    tile0, tile1 = tile_size

    @trace(input_specs=[(shape, "f32"), (shape, "f32")])
    def add_kernel(a, b):
        result = a + b
        knob(result).tile_op(tile_size=tile_size)
        return result

    check_patterns = f"""
    CHECK: func.func
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{dim0}{{{{.*}}}} step %c{tile0}
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{dim1}{{{{.*}}}} step %c{tile1}
    CHECK: memref.subview
    CHECK: memref.alloc() : memref<{tile0}x{tile1}xf32, #nkipy.mem<Sbuf>>
    CHECK: memref.subview
    CHECK: memref.alloc() : memref<{tile0}x{tile1}xf32, #nkipy.mem<Sbuf>>
    CHECK: memref.subview
    CHECK: memref.alloc() : memref<{tile0}x{tile1}xf32, #nkipy.mem<Sbuf>>
    CHECK: linalg.add {{{{.*}}}} memref<{tile0}x{tile1}xf32, #nkipy.mem<Sbuf>>
    CHECK: linalg.copy
    """
    run_kernel_test(
        add_kernel,
        stop_after='knob-driven-tiling',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


# ============================================================================
# Scalar Constant Tests
# ============================================================================
# When one operand is a scalar constant, the tracer generates a linalg.generic
# with the arith.constant embedded in the region body. This has 1 DPS input
# (the memref) and 1 DPS init (the output), so tiling promotes 1 input + 1
# output to SBUF (2 memref.alloc, not 3 like binary tensor-tensor ops).


def test_tensor_add_scalar():
    """
    Test tiling of tensor + scalar constant.

    Pattern: result = x + 2.0
    Generated IR: linalg.generic with arith.addf and embedded arith.constant.
    """
    shape = (256, 256)
    tile_size = [128, 128]
    dim0, dim1 = shape
    tile0, tile1 = tile_size

    @trace(input_specs=[(shape, "f32")])
    def kernel(x):
        result = x + 2.0
        knob(result).tile_op(tile_size=tile_size)
        return result

    check_patterns = f"""
    CHECK: func.func
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{dim0}{{{{.*}}}} step %c{tile0}
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{dim1}{{{{.*}}}} step %c{tile1}
    CHECK: memref.subview
    CHECK: memref.alloc() : memref<{tile0}x{tile1}xf32, #nkipy.mem<Sbuf>>
    CHECK: memref.subview
    CHECK: memref.alloc() : memref<{tile0}x{tile1}xf32, #nkipy.mem<Sbuf>>
    CHECK: linalg.generic
    CHECK: linalg.copy
    """
    run_kernel_test(
        kernel,
        stop_after='knob-driven-tiling',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


def test_scalar_minus_tensor():
    """
    Test tiling of scalar - tensor (non-commutative, scalar on LHS).

    Pattern: result = 5.0 - x
    Generated IR: linalg.generic with arith.subf, scalar_is_lhs=True.
    """
    shape = (256, 256)
    tile_size = [128, 128]
    dim0, dim1 = shape
    tile0, tile1 = tile_size

    @trace(input_specs=[(shape, "f32")])
    def kernel(x):
        result = 5.0 - x
        knob(result).tile_op(tile_size=tile_size)
        return result

    check_patterns = f"""
    CHECK: func.func
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{dim0}{{{{.*}}}} step %c{tile0}
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{dim1}{{{{.*}}}} step %c{tile1}
    CHECK: memref.subview
    CHECK: memref.alloc() : memref<{tile0}x{tile1}xf32, #nkipy.mem<Sbuf>>
    CHECK: memref.subview
    CHECK: memref.alloc() : memref<{tile0}x{tile1}xf32, #nkipy.mem<Sbuf>>
    CHECK: linalg.generic
    CHECK: linalg.copy
    """
    run_kernel_test(
        kernel,
        stop_after='knob-driven-tiling',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


def test_tensor_mul_scalar():
    """
    Test tiling of tensor * scalar constant.

    Pattern: result = x * 3.0
    """
    shape = (256, 256)
    tile_size = [128, 128]
    dim0, dim1 = shape
    tile0, tile1 = tile_size

    @trace(input_specs=[(shape, "f32")])
    def kernel(x):
        result = x * 3.0
        knob(result).tile_op(tile_size=tile_size)
        return result

    check_patterns = f"""
    CHECK: func.func
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{dim0}{{{{.*}}}} step %c{tile0}
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{dim1}{{{{.*}}}} step %c{tile1}
    CHECK: memref.subview
    CHECK: memref.alloc() : memref<{tile0}x{tile1}xf32, #nkipy.mem<Sbuf>>
    CHECK: memref.subview
    CHECK: memref.alloc() : memref<{tile0}x{tile1}xf32, #nkipy.mem<Sbuf>>
    CHECK: linalg.generic
    CHECK: linalg.copy
    """
    run_kernel_test(
        kernel,
        stop_after='knob-driven-tiling',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


def test_tensor_div_scalar():
    """
    Test tiling of tensor / scalar constant.

    Pattern: result = x / 2.0
    """
    shape = (256, 256)
    tile_size = [128, 128]
    dim0, dim1 = shape
    tile0, tile1 = tile_size

    @trace(input_specs=[(shape, "f32")])
    def kernel(x):
        result = x / 2.0
        knob(result).tile_op(tile_size=tile_size)
        return result

    check_patterns = f"""
    CHECK: func.func
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{dim0}{{{{.*}}}} step %c{tile0}
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{dim1}{{{{.*}}}} step %c{tile1}
    CHECK: memref.subview
    CHECK: memref.alloc() : memref<{tile0}x{tile1}xf32, #nkipy.mem<Sbuf>>
    CHECK: memref.subview
    CHECK: memref.alloc() : memref<{tile0}x{tile1}xf32, #nkipy.mem<Sbuf>>
    CHECK: linalg.generic
    CHECK: linalg.copy
    """
    run_kernel_test(
        kernel,
        stop_after='knob-driven-tiling',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


def test_scalar_div_tensor():
    """
    Test tiling of scalar / tensor (non-commutative, scalar on LHS).

    Pattern: result = 1.0 / x  (reciprocal-like pattern used in sigmoid)
    prepare-arithmetic converts this to linalg.reciprocal.
    """
    shape = (256, 256)
    tile_size = [128, 128]
    dim0, dim1 = shape
    tile0, tile1 = tile_size

    @trace(input_specs=[(shape, "f32")])
    def kernel(x):
        result = 1.0 / x
        knob(result).tile_op(tile_size=tile_size)
        return result

    check_patterns = f"""
    CHECK: func.func
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{dim0}{{{{.*}}}} step %c{tile0}
    CHECK: scf.for %{{{{.*}}}} = %c0{{{{.*}}}} to %c{dim1}{{{{.*}}}} step %c{tile1}
    CHECK: memref.subview
    CHECK: memref.alloc() : memref<{tile0}x{tile1}xf32, #nkipy.mem<Sbuf>>
    CHECK: memref.subview
    CHECK: memref.alloc() : memref<{tile0}x{tile1}xf32, #nkipy.mem<Sbuf>>
    CHECK: linalg.reciprocal
    CHECK: linalg.copy
    """
    run_kernel_test(
        kernel,
        stop_after='knob-driven-tiling',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
