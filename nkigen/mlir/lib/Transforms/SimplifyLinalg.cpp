//===- SimplifyLinalg.cpp - Simplify linalg ops for NISA lowering ---------===//
//
// Pre-processing pass that simplifies linalg operations before linalg-to-nisa.
//
// 1. Converts trivial-broadcast linalg.generic ops to named linalg ops.
// 2. Replaces SBUF gather operands with HBM originals for dma_copy_indirect.
//
//===----------------------------------------------------------------------===//

#include "nkipy/Transforms/Passes.h"
#include "nkipy/Transforms/IRHelpers.h"
#include "nkipy/Dialect/NkipyOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

using nkipy::isHbm;
using nkipy::isSbuf;

/// Elementwise arith kinds recognized in linalg.generic bodies.  Kept local
/// to this file so we do not depend on nki::nisa::ArithOpKind.
enum class LocalArithKind {
  ADD, SUBTRACT, MULTIPLY, DIVIDE, MOD, MODINT,
  ISEQ, ISGT, ISGE, ISLE, ISLT, ISNE,
};

/// Map arith dialect binary op to a local arith kind used to classify the
/// body of linalg.generic ops.
static std::optional<LocalArithKind>
getArithOpKindFromBodyOp(Operation *op) {
  auto kind = llvm::TypeSwitch<Operation *, std::optional<LocalArithKind>>(op)
      .Case<arith::AddFOp, arith::AddIOp>(
          [](auto) { return LocalArithKind::ADD; })
      .Case<arith::SubFOp, arith::SubIOp>(
          [](auto) { return LocalArithKind::SUBTRACT; })
      .Case<arith::MulFOp, arith::MulIOp>(
          [](auto) { return LocalArithKind::MULTIPLY; })
      .Case<arith::DivFOp, arith::DivSIOp, arith::DivUIOp>(
          [](auto) { return LocalArithKind::DIVIDE; })
      .Case<arith::RemFOp>([](auto) { return LocalArithKind::MOD; })
      .Case<arith::RemSIOp>([](auto) { return LocalArithKind::MODINT; })
      .Default([](Operation *) { return std::nullopt; });
  if (kind)
    return kind;

  // Comparison pattern: arith.uitofp(arith.cmpf(...))
  if (auto uitofp = dyn_cast<arith::UIToFPOp>(op)) {
    if (auto cmpf = uitofp.getIn().getDefiningOp<arith::CmpFOp>()) {
      switch (cmpf.getPredicate()) {
      case arith::CmpFPredicate::OEQ: return LocalArithKind::ISEQ;
      case arith::CmpFPredicate::OGT: return LocalArithKind::ISGT;
      case arith::CmpFPredicate::OGE: return LocalArithKind::ISGE;
      case arith::CmpFPredicate::OLE: return LocalArithKind::ISLE;
      case arith::CmpFPredicate::OLT: return LocalArithKind::ISLT;
      case arith::CmpFPredicate::ONE: return LocalArithKind::ISNE;
      default: return std::nullopt;
      }
    }
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Preprocessing: Canonicalize trivial-broadcast generics to named ops
//===----------------------------------------------------------------------===//

/// Convert linalg.generic with trivial broadcast to named ops.
/// After tiling, a broadcast like (128x4x64) * (128x1x64) becomes
/// (128x1x64) * (128x1x64) -- the broadcast dim is now size 1 on both sides.
/// The generic still carries broadcast indexing maps but is effectively
/// a same-shape elementwise op. Convert it to a named op (linalg.mul, etc.)
/// so the existing LinalgElementwiseToNisaPattern can handle it.
static void canonicalizeTrivialBroadcastGenerics(func::FuncOp func) {
  SmallVector<linalg::GenericOp> toConvert;
  func.walk([&](linalg::GenericOp op) {
    // 2 inputs, 1 output, all parallel
    if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
      return;
    if (!llvm::all_of(op.getIteratorTypesArray(),
            [](utils::IteratorType t) {
              return t == utils::IteratorType::parallel;
            }))
      return;

    // Single binary arith op in body
    Operation *binaryOp = nullptr;
    for (Operation &bodyOp : op.getRegion().front().without_terminator()) {
      if (getArithOpKindFromBodyOp(&bodyOp)) {
        if (binaryOp) return; // multiple ops
        binaryOp = &bodyOp;
      }
    }
    if (!binaryOp) return;

    // Must be a direct binary op (not a wrapped pattern like uitofp(andi(...)))
    if (binaryOp->getNumOperands() != 2) return;

    // Both operands must be block args (not constants)
    if (binaryOp->getOperand(0).getDefiningOp<arith::ConstantOp>() ||
        binaryOp->getOperand(1).getDefiningOp<arith::ConstantOp>())
      return;

    // Check all indexing maps are identity or trivial broadcast
    auto maps = op.getIndexingMapsArray();
    auto outType = dyn_cast<ShapedType>(op.getDpsInits()[0].getType());
    if (!outType) return;

    for (auto &map : maps) {
      if (map.isIdentity()) continue;
      for (unsigned i = 0; i < map.getNumResults(); ++i) {
        auto expr = map.getResult(i);
        if (auto constExpr = dyn_cast<AffineConstantExpr>(expr)) {
          if (constExpr.getValue() == 0 && outType.getDimSize(i) == 1)
            continue; // trivial broadcast
          return; // non-trivial
        }
        if (!isa<AffineDimExpr>(expr)) return;
      }
    }

    toConvert.push_back(op);
  });

  for (auto op : toConvert) {
    Operation *binaryOp = nullptr;
    for (Operation &bodyOp : op.getRegion().front().without_terminator()) {
      if (getArithOpKindFromBodyOp(&bodyOp)) {
        binaryOp = &bodyOp;
        break;
      }
    }

    // Figure out operand order: body may swap block args
    Block &body = op.getRegion().front();
    Value lhs = op.getDpsInputs()[0];
    Value rhs = op.getDpsInputs()[1];
    if (binaryOp->getOperand(0) == body.getArgument(1) &&
        binaryOp->getOperand(1) == body.getArgument(0))
      std::swap(lhs, rhs);

    OpBuilder builder(op);
    Value output = op.getDpsInits()[0];

    Operation *namedOp = nullptr;
    auto kind = *getArithOpKindFromBodyOp(binaryOp);
    switch (kind) {
    case LocalArithKind::ADD:
      namedOp = builder.create<linalg::AddOp>(
          op.getLoc(), ValueRange{lhs, rhs}, ValueRange{output});
      break;
    case LocalArithKind::SUBTRACT:
      namedOp = builder.create<linalg::SubOp>(
          op.getLoc(), ValueRange{lhs, rhs}, ValueRange{output});
      break;
    case LocalArithKind::MULTIPLY:
      namedOp = builder.create<linalg::MulOp>(
          op.getLoc(), ValueRange{lhs, rhs}, ValueRange{output});
      break;
    case LocalArithKind::DIVIDE:
      namedOp = builder.create<linalg::DivOp>(
          op.getLoc(), ValueRange{lhs, rhs}, ValueRange{output});
      break;
    default:
      continue; // skip unsupported ops
    }

    // Copy over any relevant attrs (like nkipy.op_id)
    if (auto opId = op->getAttr("nkipy.op_id"))
      namedOp->setAttr("nkipy.op_id", opId);

    op.replaceAllUsesWith(namedOp->getResults());
    op.erase();
  }
}

//===----------------------------------------------------------------------===//
// Preprocessing: Replace SBUF gather operands with HBM originals
//===----------------------------------------------------------------------===//

/// For nkipy.gather ops, replace SBUF source/indices with their HBM origins.
/// nisa.dma_copy_indirect requires the source table in HBM. The annotation
/// pass may have copied source/indices to SBUF — undo that so linalg-to-nisa
/// sees HBM operands and can emit dma_copy_indirect directly.
static void prepareGatherForNisaLowering(func::FuncOp func) {
  SmallVector<nkipy::GatherOp> gatherOps;
  func.walk([&](nkipy::GatherOp op) { gatherOps.push_back(op); });

  for (auto gatherOp : gatherOps) {
    // Check source (operand 0) and indices (operand 1).
    for (unsigned idx : {0u, 1u}) {
      Value operand = gatherOp->getOperand(idx);
      auto memrefType = dyn_cast<MemRefType>(operand.getType());
      if (!memrefType || !isSbuf(memrefType))
        continue;

      // Find the linalg.copy that writes HBM data into this SBUF alloc.
      Value hbmSource = nullptr;
      Operation *deadCopy = nullptr;
      for (auto *user : operand.getUsers()) {
        auto copyOp = dyn_cast<linalg::CopyOp>(user);
        if (!copyOp)
          continue;
        Value copySrc = copyOp.getInputs()[0];
        Value copyDst = copyOp.getOutputs()[0];
        if (copyDst != operand)
          continue;
        Value base = nkipy::getBaseMemRef(copySrc);
        if (isHbm(cast<MemRefType>(base.getType()))) {
          hbmSource = copySrc;
          deadCopy = user;
          break;
        }
      }
      if (!hbmSource)
        continue;

      // Replace the gather operand with the HBM source.
      gatherOp->setOperand(idx, hbmSource);

      // Erase the dead copy.
      deadCopy->erase();

      // If the SBUF alloc has no remaining readers, erase it + dealloc.
      if (auto allocOp = operand.getDefiningOp<memref::AllocOp>()) {
        SmallVector<Operation *> toErase;
        bool canErase = true;
        for (auto *user : allocOp->getResult(0).getUsers()) {
          if (isa<memref::DeallocOp>(user))
            toErase.push_back(user);
          else {
            canErase = false;
            break;
          }
        }
        if (canErase) {
          for (auto *op : toErase)
            op->erase();
          allocOp->erase();
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct SimplifyLinalgPass
    : public PassWrapper<SimplifyLinalgPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimplifyLinalgPass)

  StringRef getArgument() const final { return "simplify-linalg"; }

  StringRef getDescription() const final {
    return "Prepare linalg operations for NISA lowering";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    canonicalizeTrivialBroadcastGenerics(func);

    // Replace SBUF gather operands with HBM originals for dma_copy_indirect
    prepareGatherForNisaLowering(func);
  }
};

} // namespace

namespace mlir {
namespace nkipy {

std::unique_ptr<OperationPass<func::FuncOp>> createSimplifyLinalgPass() {
  return std::make_unique<SimplifyLinalgPass>();
}

} // namespace nkipy
} // namespace mlir
