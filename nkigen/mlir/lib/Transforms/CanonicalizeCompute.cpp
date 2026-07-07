//===- CanonicalizeCompute.cpp - Rewrite linalg ops for NISA hardware -----===//
//
// Combined pass `--canonicalize-compute` that prepares all linalg compute ops
// for NISA hardware.  Runs before infer-layout / tiling.
//
// Transformations:
//   1. Convert division to reciprocal+multiply (NISA has no divide).
//   2. Decompose batch_matmul → scf.for + matmul (NISA only has 2D matmul).
//   3. Remove fill(0) before matmul-like ops (NISA matmul auto-zeros PSUM).
//
// Replaces the former `prepare-arithmetic` + `prepare-matmul` pass pair.
//
//===----------------------------------------------------------------------===//

#include "nkipy/Transforms/Passes.h"
#include "nkipy/Dialect/NkipyOps.h"
#include "nkipy/Dialect/NkipyDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Shared helpers
//===----------------------------------------------------------------------===//

static void cloneAnnotations(Value oldValue, Value newValue,
                             PatternRewriter &rewriter) {
  for (Operation *user : oldValue.getUsers()) {
    if (auto layoutOp = dyn_cast<nkipy::LayoutOp>(user)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfterValue(newValue);
      rewriter.create<nkipy::LayoutOp>(
          layoutOp.getLoc(), newValue,
          layoutOp.getMemSpaceAttr(), layoutOp.getPartitionDimAttr(),
          layoutOp.getTileSizeAttr());
    } else if (auto tileOp = dyn_cast<nkipy::TileOp>(user)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfterValue(newValue);
      rewriter.create<nkipy::TileOp>(
          tileOp.getLoc(), newValue, tileOp.getLoopTileSizeAttr());
    }
  }
}

static bool isMatmulLikeOp(Operation *op) {
  return isa<linalg::MatmulOp, linalg::MatmulTransposeAOp,
             linalg::MatmulTransposeBOp, linalg::BatchMatmulOp,
             linalg::BatchMatmulTransposeAOp,
             linalg::BatchMatmulTransposeBOp>(op);
}

static bool isZeroConstant(Value value) {
  auto constOp = value.getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return false;
  if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
    return intAttr.getValue().isZero();
  if (auto fpAttr = dyn_cast<FloatAttr>(constOp.getValue()))
    return fpAttr.getValue().isZero();
  return false;
}

//===----------------------------------------------------------------------===//
// Pattern: Division -> reciprocal + multiply
//===----------------------------------------------------------------------===//

struct ConvertDivToReciprocal
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (auto divOp = dyn_cast<linalg::DivOp>(linalgOp.getOperation()))
      return handleNamedDiv(divOp, rewriter);

    auto genericOp = dyn_cast<linalg::GenericOp>(linalgOp.getOperation());
    if (!genericOp || genericOp.getNumDpsInits() != 1 ||
        !isAllParallel(genericOp))
      return failure();

    arith::DivFOp divFOp = findUniqueDivF(genericOp);
    if (!divFOp)
      return failure();

    unsigned numInputs = genericOp.getNumDpsInputs();
    if (numInputs == 1)
      return handleScalarDiv(genericOp, divFOp, rewriter);
    if (numInputs == 2)
      return handleBroadcastDiv(genericOp, divFOp, rewriter);
    return failure();
  }

private:
  static arith::DivFOp findUniqueDivF(linalg::GenericOp op) {
    arith::DivFOp found = nullptr;
    for (Operation &bodyOp : op.getRegion().front().without_terminator()) {
      if (auto d = dyn_cast<arith::DivFOp>(&bodyOp)) {
        if (found) return nullptr;
        found = d;
      }
    }
    return found;
  }

  static Value createReciprocal(PatternRewriter &rewriter, Location loc,
                                Value input, Value outputBuf = nullptr) {
    auto memrefType = cast<MemRefType>(input.getType());
    Value out = outputBuf ? outputBuf
                          : rewriter.create<memref::AllocOp>(loc, memrefType).getResult();
    rewriter.create<linalg::ReciprocalOp>(
        loc, TypeRange{}, ValueRange{input}, ValueRange{out});
    return out;
  }

  static bool isAllParallel(linalg::GenericOp op) {
    return llvm::all_of(op.getIteratorTypesArray(),
        [](utils::IteratorType t) { return t == utils::IteratorType::parallel; });
  }

  void replaceWithReciprocalMul(Operation *origOp, Value numerator,
                                Value divisor, Value output,
                                PatternRewriter &rewriter) const {
    Location loc = origOp->getLoc();
    Value recipResult = createReciprocal(rewriter, loc, divisor);
    cloneAnnotations(output, recipResult, rewriter);
    rewriter.create<linalg::MulOp>(loc, TypeRange{},
        ValueRange{numerator, recipResult}, ValueRange{output});
    rewriter.eraseOp(origOp);
  }

  LogicalResult handleNamedDiv(linalg::DivOp op, PatternRewriter &rewriter) const {
    replaceWithReciprocalMul(op, op.getInputs()[0], op.getInputs()[1],
                             op.getOutputs()[0], rewriter);
    return success();
  }

  LogicalResult handleScalarDiv(linalg::GenericOp op, arith::DivFOp divOp,
                                PatternRewriter &rewriter) const {
    Value divLhs = divOp.getLhs(), divRhs = divOp.getRhs();
    auto lhsConst = divLhs.getDefiningOp<arith::ConstantOp>();
    auto rhsConst = divRhs.getDefiningOp<arith::ConstantOp>();
    if ((!lhsConst && !rhsConst) || (lhsConst && rhsConst))
      return failure();

    if (rhsConst) {
      auto floatAttr = dyn_cast<FloatAttr>(rhsConst.getValue());
      if (!floatAttr || floatAttr.getValueAsDouble() == 0.0) return failure();
      double recipVal = 1.0 / floatAttr.getValueAsDouble();
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(divOp);
      auto recipConst = rewriter.create<arith::ConstantOp>(
          divOp.getLoc(), rewriter.getFloatAttr(floatAttr.getType(), recipVal));
      rewriter.replaceOpWithNewOp<arith::MulFOp>(divOp, divLhs, recipConst.getResult());
      return success();
    }

    auto floatAttr = dyn_cast<FloatAttr>(lhsConst.getValue());
    if (!floatAttr) return failure();
    double scalarVal = floatAttr.getValueAsDouble();
    Location loc = op.getLoc();
    Value input = op.getDpsInputs()[0];
    Value output = op.getDpsInits()[0];
    auto outputType = cast<ShapedType>(output.getType());

    if (scalarVal == 1.0) {
      createReciprocal(rewriter, loc, input, output);
      rewriter.eraseOp(op);
    } else {
      auto memrefType = cast<MemRefType>(output.getType());
      Value fillOut = rewriter.create<memref::AllocOp>(loc, memrefType);
      auto scalarCst = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getFloatAttr(
              cast<FloatType>(outputType.getElementType()), scalarVal));
      rewriter.create<linalg::FillOp>(
          loc, TypeRange{}, ValueRange{scalarCst.getResult()}, ValueRange{fillOut});
      replaceWithReciprocalMul(op, fillOut, input, output, rewriter);
    }
    return success();
  }

  LogicalResult handleBroadcastDiv(linalg::GenericOp op, arith::DivFOp divOp,
                                   PatternRewriter &rewriter) const {
    auto rhsArg = dyn_cast<BlockArgument>(divOp.getRhs());
    if (!rhsArg || !isa<BlockArgument>(divOp.getLhs())) return failure();
    unsigned rhsArgNum = rhsArg.getArgNumber();
    if (rhsArgNum >= op.getNumDpsInputs()) return failure();

    Location loc = op.getLoc();
    Value rhsInput = op.getDpsInputs()[rhsArgNum];
    Value recipResult = createReciprocal(rewriter, loc, rhsInput);

    SmallVector<Value> newInputs(op.getDpsInputs());
    newInputs[rhsArgNum] = recipResult;
    auto newGeneric = rewriter.create<linalg::GenericOp>(
        loc, op.getResultTypes(), newInputs, op.getDpsInits(),
        op.getIndexingMapsArray(), op.getIteratorTypesArray());
    rewriter.cloneRegionBefore(op.getRegion(), newGeneric.getRegion(),
                               newGeneric.getRegion().end());
    for (Operation &bodyOp : newGeneric.getRegion().front().without_terminator()) {
      if (auto d = dyn_cast<arith::DivFOp>(&bodyOp)) {
        rewriter.setInsertionPoint(d);
        rewriter.replaceOpWithNewOp<arith::MulFOp>(d, d.getLhs(), d.getRhs());
        break;
      }
    }
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Batch matmul decomposition
//===----------------------------------------------------------------------===//

static LogicalResult decomposeOneBatchMatmul(linalg::BatchMatmulOp bmmOp) {
  Value lhs = bmmOp.getInputs()[0];
  Value rhs = bmmOp.getInputs()[1];
  Value init = bmmOp.getOutputs()[0];

  auto initType = cast<ShapedType>(init.getType());
  if (initType.getRank() != 3 || !initType.hasStaticShape())
    return bmmOp.emitError("unsupported batch_matmul shape");

  // Collect annotations on the output buffer.
  SmallVector<nkipy::LayoutOp> layoutOps;
  SmallVector<nkipy::TileOp> tileOps;
  for (Operation *user : init.getUsers()) {
    if (user == bmmOp) continue;
    if (auto lay = dyn_cast<nkipy::LayoutOp>(user)) layoutOps.push_back(lay);
    else if (auto t = dyn_cast<nkipy::TileOp>(user)) tileOps.push_back(t);
  }

  for (auto lay : layoutOps)
    if (auto ts = lay.getTileSizeAttr())
      if (!ts.asArrayRef().empty() && ts.asArrayRef()[0] != 1)
        return lay.emitError("layout tile_size[0] on batch_matmul must be 1");
  for (auto t : tileOps)
    if (auto ts = t.getLoopTileSizeAttr())
      if (!ts.asArrayRef().empty() && ts.asArrayRef()[0] != 1)
        return t.emitError("tile_op tile_size[0] on batch_matmul must be 1");

  Location loc = bmmOp.getLoc();
  auto shape = initType.getShape();
  int64_t B = shape[0], M = shape[1], N = shape[2];
  int64_t K = cast<ShapedType>(lhs.getType()).getShape()[2];
  Type elemTy = initType.getElementType();
  OpBuilder builder(bmmOp);

  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value cB = builder.create<arith::ConstantIndexOp>(loc, B);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);

  // scf.for over batch dim, writing matmul results into init[b,:,:] in-place.
  // ForOp with no iter_args auto-creates a yield terminator.
  auto forOp = builder.create<scf::ForOp>(loc, c0, cB, c1);
  builder.setInsertionPoint(forOp.getBody()->getTerminator());
  Value iv = forOp.getInductionVar();

  // Rank-reducing subview: [B,d1,d2][iv,0,0][1,d1,d2] → memref<d1xd2>
  auto subview2D = [&](Value src, int64_t d1, int64_t d2) -> Value {
    auto srcType = cast<MemRefType>(src.getType());
    auto resultType = MemRefType::get(
        {d1, d2}, srcType.getElementType(),
        StridedLayoutAttr::get(builder.getContext(),
            ShapedType::kDynamic, {srcType.getShape()[2], 1}),
        srcType.getMemorySpace());
    SmallVector<OpFoldResult> offsets = {iv, builder.getIndexAttr(0),
                                         builder.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes = {builder.getIndexAttr(1),
                                       builder.getIndexAttr(d1),
                                       builder.getIndexAttr(d2)};
    SmallVector<OpFoldResult> strides(3, builder.getIndexAttr(1));
    return builder.create<memref::SubViewOp>(
        loc, resultType, src, offsets, sizes, strides);
  };

  Value lhsSlice = subview2D(lhs, M, K);
  Value rhsSlice = subview2D(rhs, K, N);
  Value initSlice = subview2D(init, M, N);

  auto matmulOp = builder.create<linalg::MatmulOp>(
      loc, TypeRange{}, ValueRange{lhsSlice, rhsSlice}, ValueRange{initSlice});

  if (auto opIdAttr = bmmOp->getAttrOfType<IntegerAttr>("nkipy.op_id"))
    matmulOp->setAttr("nkipy.op_id", opIdAttr);

  // Transfer annotations to initSlice, dropping the batch dim (first entry).
  for (auto t : tileOps)
    if (auto ts = t.getLoopTileSizeAttr()) {
      auto arr = ts.asArrayRef();
      if (arr.size() >= 2) {
        SmallVector<int64_t> inner(arr.begin() + 1, arr.end());
        auto innerTileSize = DenseI64ArrayAttr::get(bmmOp.getContext(), inner);
        builder.create<nkipy::TileOp>(t.getLoc(), initSlice, innerTileSize);
      }
    }
  for (auto lay : layoutOps) {
    DenseI64ArrayAttr innerTs;
    if (auto ts = lay.getTileSizeAttr()) {
      auto arr = ts.asArrayRef();
      if (arr.size() >= 2) {
        SmallVector<int64_t> inner(arr.begin() + 1, arr.end());
        innerTs = DenseI64ArrayAttr::get(bmmOp.getContext(), inner);
      }
    }
    builder.create<nkipy::LayoutOp>(lay.getLoc(), initSlice,
                                    lay.getMemSpaceAttr(),
                                    lay.getPartitionDimAttr(), innerTs);
  }

  // Re-annotate the base output alloc so downstream passes (infer-layout,
  // legalize-layout) see it with the correct mem_space and tile_size.
  // Derive tile from layout's tile_size or from tile_op's loop_tile_size.
  DenseI64ArrayAttr baseTileSize;
  for (auto lay : layoutOps) {
    if (auto ts = lay.getTileSizeAttr()) {
      auto arr = ts.asArrayRef();
      int64_t take = std::min((int64_t)arr.size(), (int64_t)initType.getRank());
      baseTileSize = DenseI64ArrayAttr::get(bmmOp.getContext(),
          SmallVector<int64_t>(arr.begin(), arr.begin() + take));
      break;
    }
  }
  if (!baseTileSize) {
    for (auto t : tileOps) {
      if (auto ts = t.getLoopTileSizeAttr()) {
        auto arr = ts.asArrayRef();
        int64_t take = std::min((int64_t)arr.size(), (int64_t)initType.getRank());
        baseTileSize = DenseI64ArrayAttr::get(bmmOp.getContext(),
            SmallVector<int64_t>(arr.begin(), arr.begin() + take));
        break;
      }
    }
  }

  builder.setInsertionPointAfter(forOp);
  for (auto lay : layoutOps) {
    builder.create<nkipy::LayoutOp>(lay.getLoc(), init,
                                    lay.getMemSpaceAttr(),
                                    lay.getPartitionDimAttr(), baseTileSize);
  }

  // Erase old annotations and the batch_matmul op.
  for (auto lay : layoutOps) lay.erase();
  for (auto t : tileOps) t.erase();
  bmmOp.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Pattern: Remove fill(0) before matmul
//===----------------------------------------------------------------------===//

struct RemoveZeroFillBeforeMatmul : public OpRewritePattern<linalg::FillOp> {
  using OpRewritePattern<linalg::FillOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::FillOp fillOp,
                                PatternRewriter &rewriter) const override {
    if (!isZeroConstant(fillOp.getInputs()[0]))
      return failure();
    if (fillOp.getNumResults() == 1) {
      Value fillResult = fillOp.getResult(0);
      for (Operation *user : fillResult.getUsers())
        if (!isMatmulLikeOp(user)) return failure();
      rewriter.replaceOp(fillOp, fillOp.getOutputs()[0]);
      return success();
    }
    Value outMemref = fillOp.getOutputs()[0];
    bool hasMatmulConsumer = false;
    for (OpOperand &use : outMemref.getUses()) {
      Operation *user = use.getOwner();
      if (user == fillOp) continue;
      if (!isMatmulLikeOp(user)) continue;
      auto dpsUser = dyn_cast<DestinationStyleOpInterface>(user);
      if (dpsUser && dpsUser.isDpsInit(&use)) { hasMatmulConsumer = true; break; }
    }
    if (!hasMatmulConsumer) return failure();
    rewriter.eraseOp(fillOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Combined pass
//===----------------------------------------------------------------------===//

struct CanonicalizeComputePass
    : public PassWrapper<CanonicalizeComputePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizeComputePass)

  StringRef getArgument() const final { return "canonicalize-compute"; }
  StringRef getDescription() const final {
    return "Canonicalize linalg compute ops for NISA: div->recip*mul, "
           "decompose batch_matmul, remove fill(0) before matmul";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<nkipy::NkipyDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    // Step 1: Convert division to reciprocal+multiply.
    RewritePatternSet arithPatterns(ctx);
    arithPatterns.add<ConvertDivToReciprocal>(ctx);
    if (failed(applyPatternsGreedily(module, std::move(arithPatterns)))) {
      signalPassFailure();
      return;
    }

    // Step 2: Remove fill(0) before matmul-like ops (before decomposition,
    // so batch_matmul is still a direct user of the fill output).
    RewritePatternSet matmulPatterns(ctx);
    matmulPatterns.add<RemoveZeroFillBeforeMatmul>(ctx);
    if (failed(applyPatternsGreedily(module, std::move(matmulPatterns)))) {
      signalPassFailure();
      return;
    }

    // Step 3: Decompose batch_matmul → scf.for + matmul.
    module.walk([&](func::FuncOp func) {
      SmallVector<linalg::BatchMatmulOp> bmms;
      func.walk([&](linalg::BatchMatmulOp op) { bmms.push_back(op); });
      for (auto bmm : bmms)
        if (failed(decomposeOneBatchMatmul(bmm))) {
          signalPassFailure();
          return;
        }
    });
  }
};

} // namespace

namespace mlir {
namespace nkipy {

std::unique_ptr<OperationPass<ModuleOp>> createCanonicalizeComputePass() {
  return std::make_unique<CanonicalizeComputePass>();
}

} // namespace nkipy
} // namespace mlir
