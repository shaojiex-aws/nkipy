//===- CanonicalizeReshape.cpp - Apply mem_space and materialize outputs --===//
//
// This pass does two things:
//
// 1. Apply nkipy.layout mem_space annotations to memref types, stamp
//    SharedHbm on func args/returns, and propagate through view ops.
//
// 2. Ensure function outputs are separate allocations (NISA requires it).
//    When a return value is a view of a func arg (via reinterpret_cast,
//    subview, etc.), insert alloc+copy to materialize a new buffer.
//
//===----------------------------------------------------------------------===//

#include "nkipy/Transforms/Passes.h"
#include "nkipy/Transforms/IRHelpers.h"
#include "nkipy/Dialect/NkipyAttrs.h"
#include "nkipy/Dialect/NkipyOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

/// Trace through memref view ops to find the base memref.
static Value traceToBase(Value v) {
  while (auto defOp = v.getDefiningOp()) {
    if (auto op = dyn_cast<memref::CollapseShapeOp>(defOp))
      v = op.getSrc();
    else if (auto op = dyn_cast<memref::ExpandShapeOp>(defOp))
      v = op.getSrc();
    else if (auto op = dyn_cast<memref::CastOp>(defOp))
      v = op.getSource();
    else if (auto op = dyn_cast<memref::SubViewOp>(defOp))
      v = op.getSource();
    else if (auto op = dyn_cast<memref::ReinterpretCastOp>(defOp))
      v = op.getSource();
    else
      break;
  }
  return v;
}

struct CanonicalizeReshapePass
    : public PassWrapper<CanonicalizeReshapePass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizeReshapePass)

  StringRef getArgument() const final {
    return "canonicalize-reshape";
  }

  StringRef getDescription() const final {
    return "Apply mem_space annotations and materialize output allocations";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }

  /// Apply mem_space from nkipy.layout ops to memref types and stamp
  /// SharedHbm on func args/returns that lack a mem_space.
  void applyMemSpaceAnnotations(func::FuncOp func) {
    MLIRContext *ctx = func.getContext();
    auto sharedHbm =
        nkipy::MemSpaceAttr::get(ctx, nkipy::MemSpaceEnum::SharedHbm);

    func.walk([&](nkipy::LayoutOp layoutOp) {
      Value target = layoutOp.getTarget();
      auto memSpace = layoutOp.getMemSpace();
      if (!memSpace) return;
      auto memrefType = dyn_cast<MemRefType>(target.getType());
      if (!memrefType) return;
      target.setType(MemRefType::get(memrefType.getShape(),
                                     memrefType.getElementType(),
                                     memrefType.getLayout(),
                                     Attribute(*memSpace)));
    });

    for (auto arg : func.getArguments()) {
      auto mt = dyn_cast<MemRefType>(arg.getType());
      if (!mt || mt.getMemorySpace()) continue;
      arg.setType(MemRefType::get(mt.getShape(), mt.getElementType(),
                                  mt.getLayout(), sharedHbm));
    }

    auto returnOp =
        cast<func::ReturnOp>(func.getBody().front().getTerminator());
    for (auto operand : returnOp.getOperands()) {
      auto mt = dyn_cast<MemRefType>(operand.getType());
      if (!mt || mt.getMemorySpace()) continue;
      operand.setType(MemRefType::get(mt.getShape(), mt.getElementType(),
                                      mt.getLayout(), sharedHbm));
    }

    // Propagate mem_space through view ops until convergence.
    bool changed = true;
    while (changed) {
      changed = false;
      func.walk([&](Operation *op) {
        if (!isa<memref::SubViewOp, memref::CastOp, memref::CollapseShapeOp,
                 memref::ExpandShapeOp, memref::ReinterpretCastOp>(op))
          return;
        auto srcType = dyn_cast<MemRefType>(op->getOperand(0).getType());
        auto resType = dyn_cast<MemRefType>(op->getResult(0).getType());
        if (!srcType || !resType) return;
        if (resType.getMemorySpace() || !srcType.getMemorySpace()) return;
        op->getResult(0).setType(MemRefType::get(
            resType.getShape(), resType.getElementType(),
            resType.getLayout(), srcType.getMemorySpace()));
        changed = true;
      });
    }

    SmallVector<Type> argTypes, resTypes;
    for (auto arg : func.getArguments()) argTypes.push_back(arg.getType());
    for (auto operand : returnOp.getOperands())
      resTypes.push_back(operand.getType());
    func.setType(FunctionType::get(ctx, argTypes, resTypes));
  }

  /// Ensure each return value has its own allocation.
  /// NISA requires function outputs to be separate HBM buffers, not
  /// views of inputs. If a return value traces back to a func arg,
  /// insert alloc+copy.
  void materializeOutputAllocations(func::FuncOp func) {
    auto returnOp =
        cast<func::ReturnOp>(func.getBody().front().getTerminator());

    for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
      Value retVal = returnOp.getOperand(i);
      if (!isa<MemRefType>(retVal.getType()))
        continue;

      Value base = traceToBase(retVal);
      if (!isa<BlockArgument>(base))
        continue;

      auto retType = cast<MemRefType>(retVal.getType());
      auto allocType = MemRefType::get(
          retType.getShape(), retType.getElementType(),
          MemRefLayoutAttrInterface{}, retType.getMemorySpace());

      OpBuilder builder(returnOp);
      Location loc = returnOp.getLoc();

      auto allocOp = builder.create<memref::AllocOp>(loc, allocType);
      builder.create<memref::CopyOp>(loc, retVal, allocOp.getResult());
      returnOp.setOperand(i, allocOp.getResult());

      auto funcType = func.getFunctionType();
      SmallVector<Type> newResultTypes(funcType.getResults());
      newResultTypes[i] = allocType;
      func.setFunctionType(FunctionType::get(
          func.getContext(), funcType.getInputs(), newResultTypes));
    }
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    applyMemSpaceAnnotations(func);
    materializeOutputAllocations(func);

    // Epilogue: canonicalize to clean up dead allocs and subviews.
    RewritePatternSet patterns(&getContext());
    for (auto *dialect : getContext().getLoadedDialects())
      dialect->getCanonicalizationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};

} // namespace

namespace mlir {
namespace nkipy {

std::unique_ptr<OperationPass<func::FuncOp>>
createCanonicalizeReshapePass() {
  return std::make_unique<CanonicalizeReshapePass>();
}

} // namespace nkipy
} // namespace mlir
