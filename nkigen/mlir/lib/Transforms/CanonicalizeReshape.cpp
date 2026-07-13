//===- CanonicalizeReshape.cpp - Materialize copies and apply mem_space ----===//
//
// This pass:
// 1. Materializes copies where a view (reinterpret_cast, subview) cannot
//    remain zero-cost: mem_space conflict, or return value aliasing input.
// 2. Applies nkipy.layout mem_space to memref types, stamps SharedHbm on
//    func args/returns, and propagates through remaining views.
//
//===----------------------------------------------------------------------===//

#include "nkipy/Transforms/Passes.h"
#include "nkipy/Transforms/IRHelpers.h"
#include "nkipy/Dialect/NkipyAttrs.h"
#include "nkipy/Dialect/NkipyOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

/// Trace through view ops to find the base memref.
static Value traceToBase(Value v) {
  while (auto defOp = v.getDefiningOp()) {
    if (auto op = dyn_cast<memref::SubViewOp>(defOp))
      v = op.getSource();
    else if (auto op = dyn_cast<memref::ReinterpretCastOp>(defOp))
      v = op.getSource();
    else if (auto op = dyn_cast<memref::CastOp>(defOp))
      v = op.getSource();
    else
      break;
  }
  return v;
}

/// Find the nkipy.layout mem_space annotation on a value, if any.
static std::optional<nkipy::MemSpaceEnum> findLayoutMemSpace(Value v) {
  for (Operation *user : v.getUsers()) {
    auto layout = dyn_cast<nkipy::LayoutOp>(user);
    if (!layout || layout.getTarget() != v) continue;
    if (auto ms = layout.getMemSpace())
      return ms->getValue();
  }
  return std::nullopt;
}

/// Check if a view op has a mem_space conflict: source and result
/// have different mem_space annotations.
static bool hasMemSpaceConflict(Operation *viewOp) {
  Value source = viewOp->getOperand(0);
  Value result = viewOp->getResult(0);

  auto resultMs = findLayoutMemSpace(result);
  if (!resultMs) return false;

  Value base = traceToBase(source);
  auto baseMs = findLayoutMemSpace(base);
  if (!baseMs) baseMs = findLayoutMemSpace(source);
  if (!baseMs) return false;

  return *baseMs != *resultMs;
}

struct CanonicalizeReshapePass
    : public PassWrapper<CanonicalizeReshapePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizeReshapePass)

  StringRef getArgument() const final { return "canonicalize-reshape"; }

  StringRef getDescription() const final {
    return "Materialize copies for views that cross mem_space boundaries "
           "or alias func args at return, then apply mem_space annotations";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
    registry.insert<linalg::LinalgDialect>();
  }

  /// Walk views and insert alloc+copy where a zero-cost view is not possible:
  /// - View crosses a mem_space boundary (e.g., SBUF source → HBM result)
  /// - Return value is a view of a func arg (NISA needs separate output allocs)
  void materializeCopies(func::FuncOp func) {
    // Collect return operands for the "returned view of arg" check.
    auto returnOp =
        cast<func::ReturnOp>(func.getBody().front().getTerminator());
    llvm::SmallPtrSet<Value, 4> returnedValues;
    for (auto operand : returnOp.getOperands())
      returnedValues.insert(operand);

    // Walk all view ops and decide if a copy is needed.
    SmallVector<Operation *> viewsNeedingCopy;
    func.walk([&](Operation *op) {
      if (!isa<memref::ReinterpretCastOp, memref::SubViewOp,
               memref::CastOp>(op))
        return;
      Value result = op->getResult(0);

      // Case 1: mem_space conflict between source and result annotations.
      if (hasMemSpaceConflict(op)) {
        viewsNeedingCopy.push_back(op);
        return;
      }

      // Case 2: returned view of a func arg.
      if (returnedValues.contains(result) &&
          isa<BlockArgument>(traceToBase(result))) {
        viewsNeedingCopy.push_back(op);
        return;
      }
    });

    // Insert alloc+copy for each conflicting view.
    for (auto *op : viewsNeedingCopy) {
      Value result = op->getResult(0);
      auto resType = cast<MemRefType>(result.getType());

      // For mem_space conflict: use the result's annotated mem_space.
      // For returned-view-of-arg: no mem_space yet (will be stamped later).
      Attribute memSpace;
      if (auto ms = findLayoutMemSpace(result))
        memSpace = nkipy::MemSpaceAttr::get(func.getContext(), *ms);

      auto allocType = MemRefType::get(
          resType.getShape(), resType.getElementType(),
          MemRefLayoutAttrInterface{}, memSpace);

      OpBuilder builder(op->getNextNode());
      Location loc = op->getLoc();

      auto allocOp = builder.create<memref::AllocOp>(loc, allocType);

      // Attach sbuf_tile_size for SBUF allocs.
      if (auto nkipyMs = dyn_cast_or_null<nkipy::MemSpaceAttr>(memSpace)) {
        if (nkipyMs.getValue() == nkipy::MemSpaceEnum::Sbuf &&
            allocType.getRank() >= 2) {
          auto pdim0 = builder.getIntegerAttr(
              builder.getIntegerType(32, /*isSigned=*/false), 0);
          SmallVector<int64_t> tile(allocType.getShape().begin(),
                                    allocType.getShape().end());
          tile[0] = std::min(tile[0], (int64_t)128);
          auto tileAttr = DenseI64ArrayAttr::get(func.getContext(), tile);
          builder.create<nkipy::LayoutOp>(
              loc, allocOp.getResult(), nkipyMs, pdim0, tileAttr);
        }
      }

      auto copyOp = builder.create<linalg::CopyOp>(
          loc, ValueRange{result}, ValueRange{allocOp.getResult()});
      llvm::SmallPtrSet<Operation *, 2> exceptions;
      exceptions.insert(op);
      exceptions.insert(copyOp);
      result.replaceAllUsesExcept(allocOp.getResult(), exceptions);
    }
  }

  /// Apply nkipy.layout mem_space to memref types, stamp SharedHbm on
  /// func args/returns that lack mem_space, propagate through views.
  void applyMemSpaceAnnotations(func::FuncOp func) {
    MLIRContext *ctx = func.getContext();
    auto sharedHbm =
        nkipy::MemSpaceAttr::get(ctx, nkipy::MemSpaceEnum::SharedHbm);

    // Apply nkipy.layout mem_space to targets.
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

    // Stamp SharedHbm on func args that lack mem_space.
    for (auto arg : func.getArguments()) {
      auto mt = dyn_cast<MemRefType>(arg.getType());
      if (!mt || mt.getMemorySpace()) continue;
      arg.setType(MemRefType::get(mt.getShape(), mt.getElementType(),
                                  mt.getLayout(), sharedHbm));
    }

    // Stamp SharedHbm on return operands that lack mem_space.
    auto returnOp =
        cast<func::ReturnOp>(func.getBody().front().getTerminator());
    for (auto operand : returnOp.getOperands()) {
      auto mt = dyn_cast<MemRefType>(operand.getType());
      if (!mt || mt.getMemorySpace()) continue;
      operand.setType(MemRefType::get(mt.getShape(), mt.getElementType(),
                                      mt.getLayout(), sharedHbm));
    }

    // Stamp SharedHbm on custom-op call results that lack mem_space. A custom
    // op is an HBM↔HBM boundary (its bridged body is compiled over shared_hbm
    // memrefs), so its results live in HBM.  Doing this before propagation
    // lets the mem_space flow into downstream consumers — including a chained
    // call `op(op(x))`, where the inner result feeds the outer call operand
    // and would otherwise stay unstamped and mismatch the reconciled decl.
    ModuleOp module = func->getParentOfType<ModuleOp>();
    func.walk([&](func::CallOp call) {
      auto callee = module.lookupSymbol<func::FuncOp>(call.getCalleeAttr());
      if (!callee || !callee->hasAttr("nkipy.custom_op")) return;
      for (auto result : call.getResults()) {
        auto mt = dyn_cast<MemRefType>(result.getType());
        if (!mt || mt.getMemorySpace()) continue;
        result.setType(MemRefType::get(mt.getShape(), mt.getElementType(),
                                       mt.getLayout(), sharedHbm));
      }
    });

    // Propagate mem_space through view ops until convergence.
    bool changed = true;
    while (changed) {
      changed = false;
      func.walk([&](Operation *op) {
        if (!isa<memref::SubViewOp, memref::CastOp,
                 memref::ReinterpretCastOp>(op))
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

    // Update function type.
    SmallVector<Type> argTypes, resTypes;
    for (auto arg : func.getArguments()) argTypes.push_back(arg.getType());
    for (auto operand : returnOp.getOperands())
      resTypes.push_back(operand.getType());
    func.setType(FunctionType::get(ctx, argTypes, resTypes));
  }

  /// After mem_space annotations mutate the types flowing into func.call
  /// operands (e.g. a matmul output stamped SharedHbm by knob().layout()),
  /// the callee's body-less declaration still carries its trace-time
  /// signature.  Re-sync each custom-op declaration's FunctionType to its
  /// call sites so the module verifies.  The stashed NISA body (parsed later
  /// by resolve-custom-ops) already uses shared_hbm memrefs, so the inlined
  /// body matches these reconciled boundary types.
  ///
  /// This is a module-level fixup: a func::FuncOp pass may not legally mutate
  /// a sibling declaration, which is why this pass operates on the ModuleOp.
  void reconcileCustomOpDecls(ModuleOp module) {
    for (auto decl : module.getOps<func::FuncOp>()) {
      if (!decl.isDeclaration() || !decl->hasAttr("nkipy.custom_op"))
        continue;

      // Find a call site and adopt its operand/result types.
      std::optional<SymbolTable::UseRange> uses =
          SymbolTable::getSymbolUses(decl, module);
      if (!uses)
        continue;
      for (SymbolTable::SymbolUse use : *uses) {
        auto call = dyn_cast<func::CallOp>(use.getUser());
        if (!call)
          continue;
        SmallVector<Type> argTypes(call.getOperandTypes());
        SmallVector<Type> resTypes(call.getResultTypes());
        decl.setType(
            FunctionType::get(module.getContext(), argTypes, resTypes));
        break; // All call sites share one traced boundary; first suffices.
      }
    }
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    for (auto func : module.getOps<func::FuncOp>()) {
      // Skip body-less custom-op declarations; their NISA bodies are inlined
      // later by the Python resolve-custom-ops step. The per-function logic
      // below walks the body and would dereference a nonexistent block
      // terminator (e.g. cast<func::ReturnOp> on an empty region → SIGSEGV).
      if (func.isDeclaration())
        continue;

      materializeCopies(func);
      applyMemSpaceAnnotations(func);

      // Epilogue: canonicalize to clean up dead views.
      RewritePatternSet patterns(&getContext());
      for (auto *dialect : getContext().getLoadedDialects())
        dialect->getCanonicalizationPatterns(patterns);
      (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
    }

    // Re-sync custom-op declarations after mem_space annotations changed the
    // boundary types their call sites now carry.
    reconcileCustomOpDecls(module);
  }
};

} // namespace

namespace mlir {
namespace nkipy {

std::unique_ptr<OperationPass<ModuleOp>>
createCanonicalizeReshapePass() {
  return std::make_unique<CanonicalizeReshapePass>();
}

} // namespace nkipy
} // namespace mlir
