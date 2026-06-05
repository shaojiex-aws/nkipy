"""Unit tests for kernelbuilder memory + indexing emission (Phase 2).

FileCheck-style assertions on the generated Python text for small, hand-written
post-Phase-4 IR snippets. These exercise alloc/release/dma_copy, subview ->
slice rendering, and index arithmetic.
"""

import pytest

from nkigen.codegen.kernelbuilder import linalg_to_kernelbuilder


def gen(mlir: str, name: str = "k") -> str:
    return linalg_to_kernelbuilder(mlir, kernel_name=name)


def test_alloc_and_release_sbuf():
    mlir = """
    module {
      func.func @k(%arg0: memref<128x256xf32, 4 : i32>) {
        %a = memref.alloc() {alignment = 64 : i64} : memref<128x256xf32, 3 : i32>
        memref.dealloc %a : memref<128x256xf32, 3 : i32>
        return
      }
    }
    """
    code = gen(mlir)
    assert "nb.compiler.alloc((128, 256), dtype=nb.float32, space=nb.sbuf)" in code
    assert "nb.compiler.release(" in code
    compile(code, "<t>", "exec")


def test_alloc_psum_space():
    mlir = """
    module {
      func.func @k() {
        %a = memref.alloc() : memref<128x512xf32, 2 : i32>
        memref.dealloc %a : memref<128x512xf32, 2 : i32>
        return
      }
    }
    """
    code = gen(mlir)
    assert "space=nb.psum" in code
    assert "(128, 512)" in code


def test_dma_copy_operand_order_swapped():
    # memref.copy(source, target) -> nisa.dma_copy(dst, src)
    mlir = """
    module {
      func.func @k(%arg0: memref<128x256xf32, 4 : i32>) {
        %a = memref.alloc() : memref<128x256xf32, 3 : i32>
        memref.copy %arg0, %a : memref<128x256xf32, 4 : i32> to memref<128x256xf32, 3 : i32>
        memref.dealloc %a : memref<128x256xf32, 3 : i32>
        return
      }
    }
    """
    code = gen(mlir)
    # dst (the alloc) comes first, src (the arg) second.
    assert "nisa.dma_copy(sbuf, input_0)" in code


def test_output_alloc_becomes_param():
    # An alloc that is returned should become an output_ parameter, not an
    # alloc statement.
    mlir = """
    module {
      func.func @k(%arg0: memref<128x256xf32, 4 : i32>) -> memref<128x256xf32, 4 : i32> {
        %a = memref.alloc() : memref<128x256xf32, 4 : i32>
        memref.copy %arg0, %a : memref<128x256xf32, 4 : i32> to memref<128x256xf32, 4 : i32>
        return %a : memref<128x256xf32, 4 : i32>
      }
    }
    """
    code = gen(mlir)
    assert "def k(input_0, output_0):" in code
    # The returned alloc is the output param; no alloc statement for it.
    assert "output_0 = nb.compiler.alloc" not in code
    assert "nisa.dma_copy(output_0, input_0)" in code


def test_subview_full_dim_is_colon():
    mlir = """
    module {
      func.func @k(%arg0: memref<256x256xf32, 4 : i32>) {
        %c0 = arith.constant 0 : index
        %c128 = arith.constant 128 : index
        %off = arith.muli %c0, %c128 : index
        %sv = memref.subview %arg0[%off, 0] [128, 256] [1, 1] : memref<256x256xf32, 4 : i32> to memref<128x256xf32, strided<[256, 1], offset: ?>, 4 : i32>
        %a = memref.alloc() : memref<128x256xf32, 3 : i32>
        memref.copy %sv, %a : memref<128x256xf32, strided<[256, 1], offset: ?>, 4 : i32> to memref<128x256xf32, 3 : i32>
        memref.dealloc %a : memref<128x256xf32, 3 : i32>
        return
      }
    }
    """
    code = gen(mlir)
    # Second dim is full extent (256 of 256) -> ":"; first dim is sliced.
    assert ":128" in code or "128" in code
    assert ", :]" in code  # the full free dim renders as ':'


def test_subview_dynamic_offset_from_loop():
    mlir = """
    module {
      func.func @k(%arg0: memref<256x256xf32, 4 : i32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c128 = arith.constant 128 : index
        scf.for %i = %c0 to %c2 step %c1 {
          %off = arith.muli %i, %c128 : index
          %sv = memref.subview %arg0[%off, 0] [128, 256] [1, 1] : memref<256x256xf32, 4 : i32> to memref<128x256xf32, strided<[256, 1], offset: ?>, 4 : i32>
          %a = memref.alloc() : memref<128x256xf32, 3 : i32>
          memref.copy %sv, %a : memref<128x256xf32, strided<[256, 1], offset: ?>, 4 : i32> to memref<128x256xf32, 3 : i32>
          memref.dealloc %a : memref<128x256xf32, 3 : i32>
        }
        return
      }
    }
    """
    code = gen(mlir)
    assert "for i in range(2):" in code
    # Dynamic offset i*128 rendered as a slice.
    assert "i * 128:i * 128 + 128" in code


def test_no_todo_for_subview_standalone():
    # subviews are consumed lazily; a standalone subview must not leave a TODO.
    mlir = """
    module {
      func.func @k(%arg0: memref<256x256xf32, 4 : i32>) {
        %sv = memref.subview %arg0[0, 0] [128, 256] [1, 1] : memref<256x256xf32, 4 : i32> to memref<128x256xf32, strided<[256, 1]>, 4 : i32>
        %a = memref.alloc() : memref<128x256xf32, 3 : i32>
        memref.copy %sv, %a : memref<128x256xf32, strided<[256, 1]>, 4 : i32> to memref<128x256xf32, 3 : i32>
        memref.dealloc %a : memref<128x256xf32, 3 : i32>
        return
      }
    }
    """
    code = gen(mlir)
    assert "TODO unhandled op: memref.subview" not in code
