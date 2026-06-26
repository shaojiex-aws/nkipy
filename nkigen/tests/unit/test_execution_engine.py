"""
Tests for LLVM execution engine.

These tests verify that MLIR modules can be executed directly using the LLVM
execution engine, testing low-level MLIR operations.
"""

import pytest

import numpy as np
from mlir.ir import Context, Module
from nkigen.execution.llvm import LLVMModule


class TestExecutionEngine:
    """Test LLVM execution engine with MLIR modules."""
    
    def test_memref_subtract(self):
        """Test element-wise subtraction using linalg.elementwise on memrefs."""
        with Context():
            module = Module.parse(
                """
        module {
          func.func @main(%arg0: memref<8x32xf32>, %arg1: memref<8x32xf32>) -> memref<8x32xf32> attributes { llvm.emit_c_interface } {
            %0 = memref.alloc() : memref<8x32xf32>
            linalg.elementwise kind=#linalg.elementwise_kind<sub> ins(%arg0, %arg1 : memref<8x32xf32>, memref<8x32xf32>) outs(%0 : memref<8x32xf32>)
            return %0 : memref<8x32xf32>
          }
        } """
            )
        
        runner = LLVMModule(module, "main")
        
        # Inputs and NumPy reference
        A = np.random.rand(8, 32).astype(np.float32)
        B = np.random.rand(8, 32).astype(np.float32)
        ref = A - B
        
        # Run and compare
        out = runner(A.copy(), B.copy())
        
        assert np.allclose(out, ref, rtol=1e-5, atol=1e-6), \
            f"MLIR result does not match NumPy result for memref subtraction"
    
    def test_memref_reduce_all(self):
        """Test reduction over all dimensions on memrefs."""
        with Context():
            module = Module.parse(
                r"""
module {
  func.func @main(%arg0: memref<3x4xf32>) -> f32 {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.alloc() : memref<f32>
    linalg.fill ins(%cst : f32) outs(%0 : memref<f32>)
    linalg.reduce { arith.addf } ins(%arg0 : memref<3x4xf32>) outs(%0 : memref<f32>) dimensions = [0, 1]
    %s = memref.load %0[] : memref<f32>
    return %s : f32
  }
}
"""
            )
        
        runner = LLVMModule(module, "main")
        
        # Input and NumPy reference
        A = np.random.rand(3, 4).astype(np.float32)
        ref = np.array(A.sum().astype(np.float32))
        
        # Run and compare
        out = runner(A.copy())
        
        assert np.allclose(out, ref, rtol=1e-5, atol=1e-6), \
            f"MLIR result does not match NumPy result for memref reduction"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
