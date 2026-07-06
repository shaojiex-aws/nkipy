import numpy as np

from nkigen import trace, knob
from harness import run_kernel_test, Mode


def test_high_rank_transpose_defaults_to_effective_2d_tile():
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
        stop_after="infer-layout",
        check_ir_contains=[
            "loop_tile_size = array<i64: 1, 128, 2, 1>",
        ],
        modes=Mode.STRING_CHECK,
    )
