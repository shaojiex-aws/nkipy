"""
Tests for the memref-native backend (WI-1 + WI-2).

Verifies @trace(backend="memref") produces correct linalg-on-memref IR
and that tiling + promotion works on memref-typed linalg ops.
"""

import pytest
import numpy as np

from nkigen import trace, knob
from harness import run_kernel_test, Mode


def _make_kernel(specs, fn):
    @trace(backend="memref", input_specs=specs)
    def kernel(*args):
        a = args[0]
        b = args[1] if len(args) > 1 else None
        return fn(a, b)
    return kernel


# ============================================================================
# Frontend IR + LLVM execution
# ============================================================================

FRONTEND_CASES = [
    pytest.param(
        [((128, 64), "f32"), ((128, 64), "f32")],
        lambda a, b: a + b,
        ["memref<128x64xf32>", "linalg.add"],
        id="add",
    ),
    pytest.param(
        [((128, 64), "f32"), ((64, 128), "f32")],
        lambda a, b: a @ b,
        ["memref<128x128xf32>", "linalg.matmul", "linalg.fill"],
        id="matmul",
    ),
    pytest.param(
        [((128, 256), "f32")],
        lambda x, _: np.sum(x, axis=-1),
        ["memref<128xf32>", "linalg.generic", "arith.addf"],
        id="reduction",
    ),
    pytest.param(
        [((128, 64), "f32")],
        lambda x, _: np.transpose(x),
        ["memref<64x128xf32>", "linalg.transpose"],
        id="transpose",
    ),
    pytest.param(
        [((128, 64), "f32")],
        lambda x, _: x.reshape(1, 128, 64),
        ["memref.reinterpret_cast", "memref<1x128x64xf32>"],
        id="reshape",
    ),
    pytest.param(
        [((128, 64), "f32")],
        lambda x, _: x.astype(np.float16),
        ["memref<128x64xf16>", "arith.truncf"],
        id="astype",
    ),
    pytest.param(
        [((64, 32), "f32"), ((64, 32), "f32")],
        lambda a, b: a * b,
        ["memref<64x32xf32>", "linalg.mul"],
        id="mul",
    ),
]


@pytest.mark.parametrize("specs,fn,expected_ir", FRONTEND_CASES)
def test_memref_frontend(specs, fn, expected_ir):
    kernel = _make_kernel(specs, fn)
    run_kernel_test(
        kernel, stop_after="trace",
        check_ir_contains=expected_ir,
        check_ir_not_contains=["tensor"],
        modes=Mode.STRING_CHECK | Mode.LLVM,
    )


# ============================================================================
# Tiling: verify tile_using_for + promotion on memref
# ============================================================================

TILING_CASES = [
    pytest.param(
        [((256, 256), "f32"), ((256, 256), "f32")],
        lambda a, b: a + b,
        [128, 128],
        ["scf.for", "memref.subview", "memref<128x128xf32, 3 : i32>", "linalg.add"],
        id="elementwise_add",
    ),
    pytest.param(
        [((256, 128), "f32"), ((128, 256), "f32")],
        lambda a, b: a @ b,
        [128, 128, 64],
        ["scf.for", "memref.subview", "3 : i32", "linalg.matmul"],
        id="matmul",
    ),
    pytest.param(
        [((256, 512), "f32")],
        lambda x, _: np.sum(x, axis=-1),
        [128, 256],
        ["scf.for", "memref.subview", "3 : i32", "linalg.generic"],
        id="reduction",
    ),
]


@pytest.mark.parametrize("specs,fn,tile_size,expected_ir", TILING_CASES)
def test_memref_tiling(specs, fn, tile_size, expected_ir):
    @trace(backend="memref", input_specs=specs)
    def kernel(*args):
        a = args[0]
        b = args[1] if len(args) > 1 else None
        result = fn(a, b)
        knob.knob(result).tile_op(tile_size=tile_size)
        return result

    run_kernel_test(
        kernel, stop_after="apply-and-strip-transforms",
        check_ir_contains=expected_ir,
        check_ir_not_contains=["tensor"],
        modes=Mode.STRING_CHECK | Mode.LLVM,
    )
