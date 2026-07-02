"""
Tests for legalize-layout handling of Phase 0 (foldReshapeIntoAlloc) patterns.

When an SBUF alloc is followed by a copy then a reshape, Phase 0 folds the
reshape into the alloc, creating a collapse_shape for the copy.  Phases 1-3
must follow through the collapse_shape to:
  - discover tile sizes (Phase 1 -- traceToLinalgOperands)
  - resolve the legalized SBUF alloc (Phase 3 -- tileMemrefCopy)
  - handle the HBM/SBUF rank mismatch when the reshape inserted dims (Phase 3)

These tests use crafted MLIR input (the output of annotate-memory-space) and
run only the legalize-layout pass to verify the transformation in isolation.

Run with: python -m pytest tests/passes/legalize_layout/test_fold_reshape_copy.py -v
"""

import pytest

from nkigen.driver.pipeline import run_nkipy_opt_passes
from passes.pass_utils import run_filecheck


# ============================================================================
# Helpers
# ============================================================================

def run_legalize_layout(mlir_input: str) -> str:
    """Run only the legalize-layout pass on the given MLIR."""
    return run_nkipy_opt_passes(mlir_input, ['legalize-layout'])


# ============================================================================
# Test: 2D SBUF alloc + copy + reshape to 3D
# ============================================================================

# This is the pattern produced by the upstream pipeline for:
#   cos_sbuf = alloc(256x64, sbuf); copy(hbm -> cos_sbuf); reshape -> 256x1x64
# The reshape is then subviewed [128,1,64] inside a tiled loop.

# Note: an earlier test_2d_copy_reshape_3d_legalized exercised the
# foldReshapeIntoAlloc pre-pass, which handled `memref.reshape` directly on
# SBUF allocs.  That pattern is not produced by the current pipeline
# (canonicalize-reshape rewrites all SBUF reshapes upstream), so the
# pre-pass and its companion test were removed.

# ============================================================================
# Test: 2D SBUF alloc (no reshape) — baseline regression
# ============================================================================

MLIR_2D_SBUF_BASELINE = '''
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @test_2d_baseline(
      %arg0: memref<256x128xf32, strided<[?, ?], offset: ?>, #nkipy.mem<SharedHbm>>,
      %arg1: memref<256x128xf32, strided<[?, ?], offset: ?>, #nkipy.mem<SharedHbm>>
  ) -> memref<256x128xf32, #nkipy.mem<SharedHbm>> {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index

    %alloc_out = memref.alloc() {alignment = 64 : i64} : memref<256x128xf32, #nkipy.mem<Sbuf>>
    nkipy.layout(%alloc_out : memref<256x128xf32, #nkipy.mem<Sbuf>>) {mem_space = #nkipy.mem<Sbuf>, partition_dim = 0 : ui32, tile_size = array<i64: 128, 128>}

    scf.for %iv = %c0 to %c2 step %c1 {
      %off = arith.muli %iv, %c128 : index

      %sv_a = memref.subview %arg0[%off, 0] [128, 128] [1, 1]
        : memref<256x128xf32, strided<[?, ?], offset: ?>, #nkipy.mem<SharedHbm>>
        to memref<128x128xf32, strided<[?, ?], offset: ?>, #nkipy.mem<SharedHbm>>
      %tile_a = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32, #nkipy.mem<Sbuf>>
      nkipy.layout(%tile_a : memref<128x128xf32, #nkipy.mem<Sbuf>>) {mem_space = #nkipy.mem<Sbuf>, partition_dim = 0 : ui32, tile_size = array<i64: 128, 128>}
      memref.copy %sv_a, %tile_a
        : memref<128x128xf32, strided<[?, ?], offset: ?>, #nkipy.mem<SharedHbm>>
        to memref<128x128xf32, #nkipy.mem<Sbuf>>

      %sv_b = memref.subview %arg1[%off, 0] [128, 128] [1, 1]
        : memref<256x128xf32, strided<[?, ?], offset: ?>, #nkipy.mem<SharedHbm>>
        to memref<128x128xf32, strided<[?, ?], offset: ?>, #nkipy.mem<SharedHbm>>
      %tile_b = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32, #nkipy.mem<Sbuf>>
      nkipy.layout(%tile_b : memref<128x128xf32, #nkipy.mem<Sbuf>>) {mem_space = #nkipy.mem<Sbuf>, partition_dim = 0 : ui32, tile_size = array<i64: 128, 128>}
      memref.copy %sv_b, %tile_b
        : memref<128x128xf32, strided<[?, ?], offset: ?>, #nkipy.mem<SharedHbm>>
        to memref<128x128xf32, #nkipy.mem<Sbuf>>

      %tile_out = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32, #nkipy.mem<Sbuf>>
      nkipy.layout(%tile_out : memref<128x128xf32, #nkipy.mem<Sbuf>>) {mem_space = #nkipy.mem<Sbuf>, partition_dim = 0 : ui32, tile_size = array<i64: 128, 128>}
      linalg.add ins(%tile_a, %tile_b
        : memref<128x128xf32, #nkipy.mem<Sbuf>>,
          memref<128x128xf32, #nkipy.mem<Sbuf>>)
        outs(%tile_out : memref<128x128xf32, #nkipy.mem<Sbuf>>)

      %sv_out = memref.subview %alloc_out[%off, 0] [128, 128] [1, 1]
        : memref<256x128xf32, #nkipy.mem<Sbuf>>
        to memref<128x128xf32, strided<[128, 1], offset: ?>, #nkipy.mem<Sbuf>>
      memref.copy %tile_out, %sv_out
        : memref<128x128xf32, #nkipy.mem<Sbuf>>
        to memref<128x128xf32, strided<[128, 1], offset: ?>, #nkipy.mem<Sbuf>>
    }

    %alloc_hbm = memref.alloc() {alignment = 64 : i64} : memref<256x128xf32, #nkipy.mem<SharedHbm>>
    memref.copy %alloc_out, %alloc_hbm
      : memref<256x128xf32, #nkipy.mem<Sbuf>>
      to memref<256x128xf32, #nkipy.mem<SharedHbm>>
    return %alloc_hbm : memref<256x128xf32, #nkipy.mem<SharedHbm>>
  }
}
'''


def test_2d_sbuf_baseline():
    """
    Baseline: 2D SBUF alloc (256x128) without reshape legalizes normally.

    The 256x128 alloc gets #nkipy.sbuf_map<tile: [128, 128], blocks: [2, 1]>.
    Logical shape is preserved.
    """
    result = run_legalize_layout(MLIR_2D_SBUF_BASELINE)

    check_patterns = '''
CHECK: func.func @test_2d_baseline
CHECK: memref.alloc(){{.*}}: memref<256x128xf32, #nkipy.sbuf_map<tile: [128, 128], blocks: [2, 1]>, #nkipy.mem<Sbuf>>
CHECK: scf.for
CHECK: linalg.add
CHECK: return{{.*}}#nkipy.mem<SharedHbm>
'''
    run_filecheck(result, check_patterns)


# ============================================================================
# Test: 3D SBUF alloc with full-buffer copy from 3D HBM (no reshape, no rank mismatch)
# ============================================================================

MLIR_3D_SBUF_COPY = '''
module {
  func.func @test_3d_copy(
      %arg0: memref<256x2x64xf32, strided<[?, ?, ?], offset: ?>, #nkipy.mem<SharedHbm>>,
      %arg1: memref<256x2x64xf32, strided<[?, ?, ?], offset: ?>, #nkipy.mem<SharedHbm>>
  ) -> memref<256x2x64xf32, #nkipy.mem<SharedHbm>> {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index

    // Full SBUF alloc loaded from HBM (no reshape -- ranks match)
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x2x64xf32, #nkipy.mem<Sbuf>>
    nkipy.layout(%alloc : memref<256x2x64xf32, #nkipy.mem<Sbuf>>) {mem_space = #nkipy.mem<Sbuf>, partition_dim = 0 : ui32, tile_size = array<i64: 128, 1, 64>}
    memref.copy %arg1, %alloc
      : memref<256x2x64xf32, strided<[?, ?, ?], offset: ?>, #nkipy.mem<SharedHbm>>
      to memref<256x2x64xf32, #nkipy.mem<Sbuf>>

    %alloc_out = memref.alloc() {alignment = 64 : i64} : memref<256x2x64xf32, #nkipy.mem<Sbuf>>
    nkipy.layout(%alloc_out : memref<256x2x64xf32, #nkipy.mem<Sbuf>>) {mem_space = #nkipy.mem<Sbuf>, partition_dim = 0 : ui32, tile_size = array<i64: 128, 1, 64>}

    // Linalg ops use 3D operands; legalize-layout flattens tile allocs to 2D
    scf.for %i = %c0 to %c2 step %c1 {
      %off = arith.muli %i, %c128 : index
      scf.for %j = %c0 to %c2 step %c1 {
        %sv_a = memref.subview %arg0[%off, %j, 0] [128, 1, 64] [1, 1, 1]
          : memref<256x2x64xf32, strided<[?, ?, ?], offset: ?>, #nkipy.mem<SharedHbm>>
          to memref<128x1x64xf32, strided<[?, ?, ?], offset: ?>, #nkipy.mem<SharedHbm>>
        %tile_a = memref.alloc() {alignment = 64 : i64} : memref<128x1x64xf32, #nkipy.mem<Sbuf>>
        nkipy.layout(%tile_a : memref<128x1x64xf32, #nkipy.mem<Sbuf>>) {mem_space = #nkipy.mem<Sbuf>, partition_dim = 0 : ui32, tile_size = array<i64: 128, 1, 64>}
        memref.copy %sv_a, %tile_a
          : memref<128x1x64xf32, strided<[?, ?, ?], offset: ?>, #nkipy.mem<SharedHbm>>
          to memref<128x1x64xf32, #nkipy.mem<Sbuf>>

        %sv_b = memref.subview %alloc[%off, %j, 0] [128, 1, 64] [1, 1, 1]
          : memref<256x2x64xf32, #nkipy.mem<Sbuf>>
          to memref<128x1x64xf32, strided<[128, 64, 1], offset: ?>, #nkipy.mem<Sbuf>>

        %tile_out = memref.alloc() {alignment = 64 : i64} : memref<128x1x64xf32, #nkipy.mem<Sbuf>>
        nkipy.layout(%tile_out : memref<128x1x64xf32, #nkipy.mem<Sbuf>>) {mem_space = #nkipy.mem<Sbuf>, partition_dim = 0 : ui32, tile_size = array<i64: 128, 1, 64>}
        linalg.add ins(%tile_a, %sv_b
            : memref<128x1x64xf32, #nkipy.mem<Sbuf>>,
              memref<128x1x64xf32, strided<[128, 64, 1], offset: ?>, #nkipy.mem<Sbuf>>)
          outs(%tile_out : memref<128x1x64xf32, #nkipy.mem<Sbuf>>)

        %sv_out = memref.subview %alloc_out[%off, %j, 0] [128, 1, 64] [1, 1, 1]
          : memref<256x2x64xf32, #nkipy.mem<Sbuf>>
          to memref<128x1x64xf32, strided<[128, 64, 1], offset: ?>, #nkipy.mem<Sbuf>>
        memref.copy %tile_out, %sv_out
          : memref<128x1x64xf32, #nkipy.mem<Sbuf>>
          to memref<128x1x64xf32, strided<[128, 64, 1], offset: ?>, #nkipy.mem<Sbuf>>
      }
    }

    %hbm = memref.alloc() {alignment = 64 : i64} : memref<256x2x64xf32, #nkipy.mem<SharedHbm>>
    memref.copy %alloc_out, %hbm
      : memref<256x2x64xf32, #nkipy.mem<Sbuf>>
      to memref<256x2x64xf32, #nkipy.mem<SharedHbm>>
    return %hbm : memref<256x2x64xf32, #nkipy.mem<SharedHbm>>
  }
}
'''


def test_3d_sbuf_full_copy():
    """
    3D SBUF alloc (256x2x64) with full-buffer copy from 3D HBM.

    No reshape involved — HBM and SBUF have the same rank (3). legalize-layout
    attaches #nkipy.sbuf_map to the alloc (logical shape preserved). It no
    longer tiles the copies itself — as of the 3a/3b redesign all copies reach
    legalize already tiled (knob-driven-tiling), so the full-size copy here
    passes through unchanged; only the sbuf_map is added.
    """
    result = run_legalize_layout(MLIR_3D_SBUF_COPY)

    check_patterns = '''
CHECK: func.func @test_3d_copy
CHECK: memref.alloc(){{.*}}: memref<256x2x64xf32, #nkipy.sbuf_map<tile: [128, 1, 64], blocks: [2, 2, 1]>, #nkipy.mem<Sbuf>>
CHECK: memref.copy{{.*}}#nkipy.mem<SharedHbm>>{{.*}}to{{.*}}#nkipy.sbuf_map<tile: [128, 1, 64], blocks: [2, 2, 1]>, #nkipy.mem<Sbuf>>
CHECK: linalg.add
CHECK: return{{.*}}#nkipy.mem<SharedHbm>
'''
    run_filecheck(result, check_patterns)


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
