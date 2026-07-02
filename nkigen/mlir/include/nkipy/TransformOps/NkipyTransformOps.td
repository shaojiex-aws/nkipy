#ifndef NKIPY_TRANSFORM_OPS
#define NKIPY_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// PromoteTensorOp
//===----------------------------------------------------------------------===//

def Nkipy_PromoteTensorOp : Op<Transform_Dialect, "nkipy.promote_tensor",
                         [DeclareOpInterfaceMethods<TransformOpInterface>,
                          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Request a tensor value to live in a specific memory space "
                "after bufferization";
  let description = [{
    Requests that a tensor value lives in a specific memory space for its
    lifetime. This is achieved by allocating a new tensor in the desired
    memory space with `bufferization.alloc_tensor` and optionally materializing
    the source value into that allocation with
    `bufferization.materialize_in_destination`. All uses of the original value
    are then redirected to the promoted value.

    The generated code for promoting tensor value %0 resembles the following:

      %1 = bufferization.alloc_tensor(<dynamic dims of %0>)
           { memory_space = memory_space }
      // Note: the materialization is omitted if %0 is never read and is only
      // written into (i.e., it behaves as a result tensor).
      %2 = bufferization.materialize_in_destination %0 in %1
      // ...
      <all users of %0 now use %2 instead>

    Deallocation is not handled by this transform.

    Return modes:
    - Produces a silenceable failure if the given handle does not point to
      tensor-typed values.
    - Succeeds otherwise and returns a handle to the promoted value(s), i.e.,
      the result of materialization if present and the allocation otherwise.

    The second result, `copy_in`, is an op handle to the stage-in copies
    created while promoting (one per promoted value that is read). It is empty
    when nothing needs a copy-in (e.g. the value already lives in the target
    memory space). Callers tile it as a first-class step; copy-back stores are
    not included.
  }];

  let arguments = (ins TransformValueHandleTypeInterface:$tensor,
      OptionalAttr<AnyAttr>:$memory_space,
      OptionalAttr<DenseI64ArrayAttr>:$tile_size);
  let results = (outs TransformValueHandleTypeInterface:$promoted,
      TransformHandleTypeInterface:$copy_in);

  let assemblyFormat =
      "(`to` $memory_space^)? $tensor attr-dict `:` functional-type($tensor, results)";
}

//===----------------------------------------------------------------------===//
// NkipyTransposeMatmulOp
//===----------------------------------------------------------------------===//

def NkipyTransposeMatmulOp : Op<Transform_Dialect, "nkipy.transpose_matmul",
                         [DeclareOpInterfaceMethods<TransformOpInterface>,
                          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Rewrite linalg.matmul to linalg.matmul_transpose_a with "
                "explicit LHS transpose (supports both tensor and memref)";
  let description = [{
    Rewrites `linalg.matmul(A, B, C)` into
    `linalg.transpose(A, A_T)` + `linalg.matmul_transpose_a(A_T, B, C)`.

    For tensor operands: creates `tensor.empty` for the transpose output.
    For memref operands: creates `memref.alloc` for the transpose output.

    Returns two handles: `transformed` points to the new
    `linalg.matmul_transpose_a` op, and `transpose` points to the inserted
    LHS `linalg.transpose` op (so the caller can tile it as a first-class
    step).
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
      OptionalAttr<DenseI64ArrayAttr>:$tile_size);
  let results = (outs TransformHandleTypeInterface:$transformed,
      TransformHandleTypeInterface:$transpose);

  let assemblyFormat =
      "$target attr-dict `:` functional-type($target, results)";
}

#endif // NKIPY_TRANSFORM_OPS
