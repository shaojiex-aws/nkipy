//===- KnobDrivenFusion.cpp - Fuse sibling scf.for loops via fuse_op -------===//
//
// Walks each `nkipy.fuse_op(%a, %b, ...)` in the function and fuses the
// `scf.for` loops associated with the listed values into a single loop using
// upstream MLIR's fuseIndependentSiblingForLoops helper.
//
// Runs after apply-and-strip-transforms (so the per-op scf.for loops exist)
// and before canonicalize-loop-step (so loop bounds are still in their
// original step=tile form).  The helper itself does no legality check —
// we verify matching lower/upper/step bounds before calling it.
//
//===----------------------------------------------------------------------===//

#include "PassGen.h"
#include "nkipy/Transforms/Passes.h"
#include "nkipy/Transforms/IRHelpers.h"
#include "nkipy/Dialect/NkipyDialect.h"
#include "nkipy/Dialect/NkipyOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace nkipy;

namespace mlir {
namespace nkipy {
namespace {

/// Find the scf.for associated with `val`.
/// Tensor mode: the value is produced by scf.for (loop-carried result).
/// Memref mode: the value is used inside scf.for (via subview in loop body);
/// walk up to the outermost scf.for nested inside the same func.
static scf::ForOp findProducingForLoop(Value val) {
  if (Operation *def = val.getDefiningOp())
    if (auto forOp = dyn_cast<scf::ForOp>(def))
      return forOp;

  // Memref path: find the outermost scf.for that contains a user.
  for (OpOperand &use : val.getUses()) {
    Operation *user = use.getOwner();
    scf::ForOp result;
    for (auto parentFor = user->getParentOfType<scf::ForOp>(); parentFor;
         parentFor = parentFor->getParentOfType<scf::ForOp>())
      result = parentFor;
    if (result)
      return result;
  }
  return nullptr;
}

/// Returns true if two `scf.for` loops have matching lower bound, upper
/// bound, and step (the precondition for fuseIndependentSiblingForLoops).
/// Constants emitted by independent tilings are distinct SSA values even
/// when their numeric value is identical, so prefer constant-value equality.
static bool sameBound(Value a, Value b) {
  if (a == b)
    return true;
  std::optional<int64_t> ca = getConstantIntValue(a);
  std::optional<int64_t> cb = getConstantIntValue(b);
  return ca.has_value() && cb.has_value() && *ca == *cb;
}

static bool sameBounds(scf::ForOp a, scf::ForOp b) {
  return sameBound(a.getLowerBound(), b.getLowerBound()) &&
         sameBound(a.getUpperBound(), b.getUpperBound()) &&
         sameBound(a.getStep(), b.getStep());
}

/// Hoist ops textually between `first` and `second` above `first`, so
/// their defs dominate the fused loop's position after sibling fusion
/// merges `second` into `first`.  Only hoists ops whose operands all
/// dominate `first` (i.e., don't depend on `first`'s results).
static void hoistSetupOpsBetween(Operation *first, Operation *second) {
  SmallVector<Operation *> toHoist;
  for (Operation *op = first->getNextNode(); op && op != second;
       op = op->getNextNode()) {
    if (isa<scf::ForOp>(op))
      continue;
    bool canHoist = true;
    for (Value operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (defOp && !defOp->isBeforeInBlock(first)) {
        canHoist = false;
        break;
      }
    }
    if (canHoist)
      toHoist.push_back(op);
  }
  for (Operation *op : toHoist)
    op->moveBefore(first);
}

/// After fusing zero-result scf.for loops, duplicate scf.yield ops may
/// appear in the merged body. Remove all but the final (terminator) yield.
static void eraseExtraYields(scf::ForOp loop) {
  Block *body = loop.getBody();
  Operation *terminator = body->getTerminator();
  SmallVector<Operation *> toErase;
  for (Operation &op : *body) {
    if (isa<scf::YieldOp>(&op) && &op != terminator)
      toErase.push_back(&op);
  }
  for (Operation *op : toErase)
    op->erase();
}

/// Recursively fuse pairs of consecutive same-bounds sibling scf.for loops
/// inside `parent`'s body.  Returns the (possibly rewritten) outer loop.
/// Used after outer-level fusion to merge the inner loops that came along
/// for the ride.
static void fuseInnerSiblings(scf::ForOp parent, IRRewriter &rewriter) {
  bool changed = true;
  while (changed) {
    changed = false;
    Block *body = parent.getBody();
    scf::ForOp prev;
    for (Operation &op : *body) {
      auto curr = dyn_cast<scf::ForOp>(&op);
      if (!curr) {
        prev = nullptr;
        continue;
      }
      if (prev && sameBounds(prev, curr)) {
        hoistSetupOpsBetween(prev, curr);
        scf::ForOp fused =
            ::mlir::fuseIndependentSiblingForLoops(curr, prev, rewriter);
        eraseExtraYields(fused);
        fuseInnerSiblings(fused, rewriter);
        changed = true;
        break;
      }
      prev = curr;
    }
  }
}

struct NkipyKnobDrivenFusionPass
    : public KnobDrivenFusionBase<NkipyKnobDrivenFusionPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    SmallVector<nkipy::FuseOp> fuseOps;
    func.walk([&](nkipy::FuseOp op) { fuseOps.push_back(op); });

    if (!fuseOps.empty()) {
    IRRewriter rewriter(&getContext());

    for (nkipy::FuseOp fuseOp : fuseOps) {
      // Resolve each target to its producing scf.for.
      SmallVector<scf::ForOp> loops;
      for (Value target : fuseOp.getTargets()) {
        scf::ForOp loop = findProducingForLoop(target);
        if (!loop) {
          fuseOp.emitError() << "fuse_op target is not produced by an "
                                "scf.for (was it tiled?)";
          return signalPassFailure();
        }
        loops.push_back(loop);
      }

      // All loops must share lower/upper/step.  Check against loops[0].
      for (size_t i = 1; i < loops.size(); i++) {
        if (!sameBounds(loops[0], loops[i])) {
          fuseOp.emitError()
              << "fuse_op targets have mismatched loop bounds (target #0 "
              << "vs #" << i << "); ensure their tile_op tile_size matches";
          return signalPassFailure();
        }
      }

      // Fuse outer loops into loops[0] left-to-right.  Each independently
      // tiled loop and its setup ops sit textually after loops[0]; hoist
      // them before fusing so their defs dominate the fused loop's position.
      // fuseIndependentSiblingForLoops takes (target, source): target is
      // merged INTO source, which becomes the surviving fused loop.
      scf::ForOp fused = loops[0];
      for (size_t i = 1; i < loops.size(); i++) {
        hoistSetupOpsBetween(fused, loops[i]);
        fused = ::mlir::fuseIndependentSiblingForLoops(loops[i], fused,
                                                       rewriter);
        eraseExtraYields(fused);
      }

      // Outer-level fusion brought along each loop's inner nest as a
      // sibling inside the merged outer loop.  Recursively fuse any
      // consecutive same-bounds inner siblings so the merged body has a
      // single nest at every level (not a loop-of-loops at level 1
      // followed by another loop-of-loops at level 2).
      fuseInnerSiblings(fused, rewriter);

      llvm::errs() << "[KnobDrivenFusion] Fused " << loops.size()
                   << " scf.for nest(s)\n";

      rewriter.eraseOp(fuseOp);
    }
    } // end if (!fuseOps.empty())

    // Epilogue: canonicalize loop steps to 1 and run greedy canonicalization.
    canonicalizeLoopSteps(func);
    RewritePatternSet patterns(&getContext());
    for (auto *dialect : getContext().getLoadedDialects())
      dialect->getCanonicalizationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }

  void canonicalizeLoopSteps(func::FuncOp func) {
    func.walk<WalkOrder::PostOrder>([&](scf::ForOp forOp) {
      auto stepConst = getConstantInt(forOp.getStep());
      if (!stepConst || *stepConst == 1)
        return;
      auto lbConst = getConstantInt(forOp.getLowerBound());
      auto ubConst = getConstantInt(forOp.getUpperBound());
      if (lbConst && ubConst && (*ubConst - *lbConst) % *stepConst != 0)
        return;

      OpBuilder builder(forOp);
      Location loc = forOp.getLoc();
      Value lb = forOp.getLowerBound();
      Value ub = forOp.getUpperBound();
      Value step = forOp.getStep();

      Value range = builder.create<arith::SubIOp>(loc, ub, lb);
      Value tripCount = builder.create<arith::DivUIOp>(loc, range, step);
      Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
      Value one = builder.create<arith::ConstantIndexOp>(loc, 1);

      builder.setInsertionPointToStart(forOp.getBody());
      Value iv = forOp.getInductionVar();
      Value scaled = builder.create<arith::MulIOp>(loc, iv, step);
      Value originalIV = builder.create<arith::AddIOp>(loc, lb, scaled);
      SmallPtrSet<Operation *, 2> exceptions;
      exceptions.insert(scaled.getDefiningOp());
      exceptions.insert(originalIV.getDefiningOp());
      iv.replaceAllUsesExcept(originalIV, exceptions);

      forOp.setLowerBound(zero);
      forOp.setUpperBound(tripCount);
      forOp.setStep(one);
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createKnobDrivenFusionPass() {
  return std::make_unique<NkipyKnobDrivenFusionPass>();
}

} // namespace nkipy
} // namespace mlir
