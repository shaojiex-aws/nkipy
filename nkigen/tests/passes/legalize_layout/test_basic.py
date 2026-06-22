"""
Tests for legalize-layout pass.

The legalize-layout pass:
1. Attaches #nkipy.sbuf_map<tile: [...], blocks: [...]> to SBUF memref allocs
   (keeps the logical shape unchanged)
2. Generates block-loop nests for HBM↔SBUF transfers (tiled copy/transpose)
3. Decomposes HBM fill ops into tiled copies

Run with: python -m pytest tests/passes/legalize_layout/test_basic.py -v
"""

import pytest
import numpy as np

from nkigen import trace, knob
from harness import run_kernel_test, Mode

# ============================================================================
# Test Cases
# ============================================================================

def test_matmul_sbuf_add_hbm():
    """
    Test matmul-add chain with:
    - matmul output -> SBUF (intermediate)
    - add output -> SharedHbm (returned result)

    This tests that the pass correctly:
    1. Attaches #nkipy.sbuf_map to SBUF allocs (logical shape preserved)
    2. Generates block-loop nests for HBM→SBUF transfers
    3. Keeps downstream ops (matmul, add) using the logical-shape memrefs
    """
    M, N, K = 256, 256, 256
    matmul_tile = [128, 128]       # TILE_M, TILE_N
    matmul_reduction_tile = [128]  # TILE_K
    add_tile = [128, 128]          # TILE_M, TILE_N

    @trace(input_specs=[((M, K), "f32"), ((K, N), "f32"), ((M, N), "f32")])
    def matmul_add_kernel(a, b, bias):
        c = np.matmul(a, b)
        knob.knob(c).tile_op(tile_size=matmul_tile + matmul_reduction_tile).layout(mem_space="Sbuf")

        result = c + bias
        knob.knob(result).tile_op(tile_size=add_tile).layout(mem_space="SharedHbm")

        return result

    # After legalize-layout, SBUF allocs get #nkipy.sbuf_map with logical shape
    # preserved, and block loops are generated for HBM↔SBUF transfers.
    check_patterns = '''
CHECK: func.func @matmul_add_kernel
CHECK-SAME: 4 : i32
CHECK: memref.alloc(){{.*}}: memref<256x256xf32, #nkipy.sbuf_map<tile: [128, 128], blocks: [2, 2]>, 3 : i32>
CHECK: scf.for
CHECK: scf.for
CHECK: memref.alloc(){{.*}}: memref<128x128xf32, 2 : i32>
CHECK: linalg.matmul
CHECK: memref.alloc(){{.*}}: memref<256x256xf32, 4 : i32>
CHECK: scf.for
CHECK: scf.for
CHECK: linalg.add
CHECK: return{{.*}}memref<256x256xf32, 4 : i32>
'''
    run_kernel_test(
        matmul_add_kernel,
        stop_after='legalize-layout',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


def test_3d_add_chain_sbuf():
    """
    Test 3D add chain with SBUF intermediate:
    - Shape: (256, 2, 256) — 3D tensor where dim0 > 128 triggers legalization
    - tile_size: [128, 1, 128] — middle dim must have tile=1 (design constraint)
    - intermediate (a+b) goes to SBUF, result (intermediate+c) goes to SharedHbm

    This tests that the pass correctly handles rank-3 tensors:
    1. Attaches #nkipy.sbuf_map<tile: [128, 1, 128], blocks: [2, 2, 2]> to SBUF alloc
    2. Generates 3-level nested block loops for HBM↔SBUF transfers
    """
    B, M, N = 256, 2, 256
    tile_size = [128, 1, 128]

    @trace(input_specs=[((B, M, N), "f32"), ((B, M, N), "f32"), ((B, M, N), "f32")])
    def add_chain_3d(a, b, c):
        intermediate = a + b
        knob.knob(intermediate).tile_op(tile_size=tile_size).layout(mem_space="Sbuf")

        result = intermediate + c
        knob.knob(result).tile_op(tile_size=tile_size).layout(mem_space="SharedHbm")

        return result

    # 3D SBUF alloc gets sbuf_map with tile/blocks for each dim.
    # Block loops are generated for the HBM↔SBUF transfers.
    check_patterns = '''
CHECK: func.func @add_chain_3d
CHECK: memref.alloc(){{.*}}: memref<256x2x256xf32, #nkipy.sbuf_map<tile: [128, 1, 128], blocks: [2, 2, 2]>, 3 : i32>
CHECK: scf.for
CHECK: scf.for
CHECK: scf.for
CHECK: memref.alloc(){{.*}}: memref<256x2x256xf32, 4 : i32>
CHECK: scf.for
CHECK: scf.for
CHECK: scf.for
CHECK: linalg.add
CHECK: return{{.*}}memref<256x2x256xf32, 4 : i32>
'''
    run_kernel_test(
        add_chain_3d,
        stop_after='legalize-layout',
        check_patterns=check_patterns,
        modes=Mode.FILECHECK,
    )


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
