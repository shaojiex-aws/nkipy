import numpy as np

from nkigen import trace, knob
from harness import run_kernel_test, Mode


def test_high_rank_transpose_lowers_via_2d_dma_transpose():
    batch = 2
    heads = 2
    seq = 128
    head_dim = 128

    @trace(input_specs=[((batch * heads, seq, head_dim), "f32")])
    def kernel(x):
        x = np.reshape(x, (batch, heads, seq, head_dim))
        y = np.transpose(x, (0, 2, 1, 3))
        knob(y).layout(mem_space="SharedHbm")
        return y

    run_kernel_test(
        kernel,
        stop_after="linalg-to-nisa",
        check_ir_contains=[
            "nisa.dma_copy",
            "nisa.dma_transpose",
            "permutation=[1, 0]",
            "dst<2| 1>",
            "src<2| 1>",
            "dst<128| 1>",
            "src<128| 1>",
            "dst<128| 2>",
            "src<2| 128>",
        ],
        check_ir_not_contains=[
            "permutation=[0, 2, 1, 3]",
        ],
        modes=Mode.STRING_CHECK,
    )


def test_high_rank_transpose_with_identity_reduced_tile_lowers_to_copy():
    @trace(input_specs=[((2, 128, 128), "f32")])
    def kernel(x):
        y = np.transpose(x, (1, 2, 0))
        knob(y).layout(mem_space="SharedHbm")
        return y

    run_kernel_test(
        kernel,
        stop_after="linalg-to-nisa",
        check_ir_contains=[
            "nisa.dma_copy",
            "nisa.tensor_copy",
        ],
        check_ir_not_contains=[
            "nisa.dma_transpose",
            "permutation=[1, 2, 0]",
        ],
        modes=Mode.STRING_CHECK,
    )


def test_rank3_cycle_transpose_lowers_via_reduced_2d_dma_transpose():
    @trace(input_specs=[((2, 4, 128), "f32")])
    def kernel(x):
        y = np.transpose(x, (2, 0, 1))
        knob(y).layout(mem_space="SharedHbm")
        return y

    run_kernel_test(
        kernel,
        stop_after="linalg-to-nisa",
        check_ir_contains=[
            "nisa.dma_copy",
            "nisa.dma_transpose",
            "permutation=[1, 0]",
        ],
        check_ir_not_contains=[
            "permutation=[2, 0, 1]",
        ],
        modes=Mode.STRING_CHECK,
    )
