"""
Tests for the memref-native backend (WI-1 through WI-6).

Verifies @trace produces correct linalg-on-memref IR,
tiling + promotion + fusion work, and full codegen pipeline succeeds.
"""

import pytest
import numpy as np

from nkigen import trace, knob
from harness import run_kernel_test, Mode


def _make_kernel(specs, fn):
    @trace(input_specs=specs)
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
        ["scf.for", "memref.subview", "#nkipy.mem<Sbuf>", "linalg.add"],
        id="elementwise_add",
    ),
    pytest.param(
        [((256, 128), "f32"), ((128, 256), "f32")],
        lambda a, b: a @ b,
        [128, 128, 64],
        ["scf.for", "memref.subview", "#nkipy.mem<Sbuf>", "linalg.matmul"],
        id="matmul",
    ),
    pytest.param(
        [((256, 512), "f32")],
        lambda x, _: np.sum(x, axis=-1),
        [128, 256],
        ["scf.for", "memref.subview", "#nkipy.mem<Sbuf>", "linalg.generic"],
        id="reduction",
    ),
]


@pytest.mark.parametrize("specs,fn,tile_size,expected_ir", TILING_CASES)
def test_memref_tiling(specs, fn, tile_size, expected_ir):
    @trace(input_specs=specs)
    def kernel(*args):
        a = args[0]
        b = args[1] if len(args) > 1 else None
        result = fn(a, b)
        knob.knob(result).tile_op(tile_size=tile_size)
        return result

    run_kernel_test(
        kernel, stop_after="knob-driven-tiling",
        check_ir_contains=expected_ir,
        check_ir_not_contains=["tensor"],
        modes=Mode.STRING_CHECK | Mode.LLVM,
    )


# ============================================================================
# Fusion: verify sibling loop fusion on memref
# ============================================================================


def test_memref_fusion():
    @trace(input_specs=[
        ((256, 256), "f32"), ((256, 256), "f32"),
        ((256, 256), "f32"), ((256, 256), "f32"),
    ])
    def kernel(a, b, c, d):
        x = a + b
        knob.knob(x).tile_op(tile_size=[128, 128])
        y = c + d
        knob.knob(y).tile_op(tile_size=[128, 128])
        knob.fuse(x, y)
        return x, y

    run_kernel_test(
        kernel, stop_after="knob-driven-fusion",
        check_ir_contains=["scf.for", "linalg.add", "memref.subview"],
        check_ir_not_contains=["tensor"],
        modes=Mode.STRING_CHECK | Mode.LLVM,
    )


# ============================================================================
# Full pipeline e2e: HW execution + KernelBuilder codegen on memref IR
# ============================================================================

E2E_CASES = [
    pytest.param(
        [((256, 256), "f32"), ((256, 256), "f32")],
        lambda a, b: a + b,
        [128, 128],
        id="add",
    ),
    pytest.param(
        [((256, 128), "f32"), ((128, 256), "f32")],
        lambda a, b: a @ b,
        [128, 128, 64],
        id="matmul",
    ),
]


@pytest.mark.parametrize("specs,fn,tile_size", E2E_CASES)
def test_memref_e2e(specs, fn, tile_size):
    @trace(input_specs=specs)
    def kernel(*args):
        a = args[0]
        b = args[1] if len(args) > 1 else None
        result = fn(a, b)
        knob.knob(result).tile_op(tile_size=tile_size)
        return result

    run_kernel_test(
        kernel,
        modes=Mode.HW | Mode.CODEGEN,
    )
