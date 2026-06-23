"""
End-to-end tests for >2D SBUF tensor handling.

Tests that the NISA emitter correctly maps >2D SBUF allocs to physical 2D
[partition, free]. Covers:
- Temp SBUF with leading unit dims (from loop tiling over batch dims)
- User-specified SBUF with partition_dim as intermediate

Run with: pytest tests/e2e/test_sbuf_multidim.py -v
"""

import pytest

from nkigen import trace, knob
from harness import run_kernel_test, Mode


def test_3d_temp_sbuf_single_leading_one():
    """
    tile_size=[1, 128, 64] on shape (4, 128, 64) produces memref<1x128x64, sbuf>.
    The leading 1 is the batch loop variable — emitter strips it → 128x64.
    """
    BH = 4
    M, N = 128, 64

    @trace(input_specs=[((BH, M, N), "f32"), ((BH, M, N), "f32")])
    def kernel(a, b):
        result = a + b
        knob.knob(result).tile_op(tile_size=[1, 128, 64]).layout(mem_space="SharedHbm")
        return result

    run_kernel_test(
        kernel,
        check_ir_contains=["nisa.alloc", "nisa.tensor_tensor_arith"],
        check_ir_not_contains=["transform.named_sequence"],
        modes=Mode.HW | Mode.STRING_CHECK,
    )


def test_4d_temp_sbuf_multiple_leading_ones():
    """
    tile_size=[1, 1, 128, 64] on shape (2, 2, 128, 64) produces
    memref<1x1x128x64, sbuf>. Both leading 1s stripped → 128x64.
    """
    B1, B2 = 2, 2
    M, N = 128, 64

    @trace(input_specs=[((B1, B2, M, N), "f32"), ((B1, B2, M, N), "f32")])
    def kernel(a, b):
        result = a + b
        knob.knob(result).tile_op(tile_size=[1, 1, 128, 64]).layout(mem_space="SharedHbm")
        return result

    run_kernel_test(
        kernel,
        check_ir_contains=["nisa.alloc", "nisa.tensor_tensor_arith"],
        check_ir_not_contains=["transform.named_sequence"],
        modes=Mode.HW | Mode.STRING_CHECK,
    )


@pytest.mark.xfail(reason="user SBUF with partition_dim needs upstream pass fixes (4b/4c)")
def test_3d_user_sbuf_partition_dim0():
    """
    User pins intermediate to SBUF with partition_dim=0.
    Shape (128, 4, 64), tile [128, 1, 64]. Dim 0 is partition.
    Intermediate lives in SBUF, final result goes to HBM.
    """
    M, BH, N = 128, 4, 64

    @trace(input_specs=[
        ((M, BH, N), "f32"), ((M, BH, N), "f32"), ((M, BH, N), "f32")
    ])
    def kernel(a, b, c):
        intermediate = a + b
        knob.knob(intermediate).tile_op(tile_size=[128, 1, 64]).layout(
            mem_space="Sbuf", partition_dim=0
        )

        result = intermediate + c
        knob.knob(result).tile_op(tile_size=[128, 1, 64]).layout(mem_space="SharedHbm")
        return result

    run_kernel_test(
        kernel,
        check_ir_contains=["nisa.alloc", "nisa.tensor_tensor_arith"],
        check_ir_not_contains=["transform.named_sequence"],
        modes=Mode.HW | Mode.STRING_CHECK,
    )


@pytest.mark.xfail(reason="user SBUF with partition_dim needs upstream pass fixes (4b/4c)")
def test_3d_user_sbuf_partition_dim1():
    """
    User pins intermediate to SBUF with partition_dim=1.
    Shape (4, 128, 64), tile [1, 128, 64]. canonicalize-partition-dim
    permutes so dim 0 = partition (128). Intermediate in SBUF, result in HBM.
    """
    BH, M, N = 4, 128, 64

    @trace(input_specs=[
        ((BH, M, N), "f32"), ((BH, M, N), "f32"), ((BH, M, N), "f32")
    ])
    def kernel(a, b, c):
        intermediate = a + b
        knob.knob(intermediate).tile_op(tile_size=[1, 128, 64]).layout(
            mem_space="Sbuf", partition_dim=1
        )

        result = intermediate + c
        knob.knob(result).tile_op(tile_size=[1, 128, 64]).layout(mem_space="SharedHbm")
        return result

    run_kernel_test(
        kernel,
        check_ir_contains=["nisa.alloc", "nisa.tensor_tensor_arith"],
        check_ir_not_contains=["transform.named_sequence"],
        modes=Mode.HW | Mode.STRING_CHECK,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
