//===- MatmulPrep.cpp - Prepare matmul-shaped ops for NISA ---------------===//
//
// Single combined pass `--prepare-matmul` that prepares all matmul-like ops
// for NISA hardware.  Steps (in order):
//
//   1. Decompose `linalg.batch_matmul [B,M,N]` → `scf.for` + `linalg.matmul`
//      (NISA only has 2D matmul).
//   2. Remove `linalg.fill(0)` when all users are matmul-like
//      (NISA matmul auto-zeros PSUM).
//
// Works on both tensor and memref IR.  Must run before tiling.
//
//===----------------------------------------------------------------------===//

#include "nkipy/Transforms/Passes.h"
#include "nkipy/Dialect/NkipyOps.h"
#include "nkipy/Dialect/NkipyDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

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
// Step 1: Decompose batch_matmul → scf.for + matmul
//===----------------------------------------------------------------------===//

static LogicalResult decomposeOneBatchMatmul(linalg::BatchMatmulOp bmmOp) {
  Value lhs = bmmOp.getInputs()[0];
  Value rhs = bmmOp.getInputs()[1];
  Value init = bmmOp.getOutputs()[0];

  auto initType = cast<RankedTensorType>(init.getType());
  if (initType.getRank() != 3 || !initType.hasStaticShape()) {
    return bmmOp.emitError(
        "prepare-matmul: unsupported batch_matmul shape "
        "(only static rank-3 [B,M,N] is supported)");
  }

  SmallVector<nkipy::LayoutOp> layoutOps;
  SmallVector<nkipy::TileOp> tileOps;
  for (Operation *user : bmmOp.getResult(0).getUsers()) {
    if (auto lay = dyn_cast<nkipy::LayoutOp>(user))
      layoutOps.push_back(lay);
    else if (auto t = dyn_cast<nkipy::TileOp>(user))
      tileOps.push_back(t);
  }

  for (auto lay : layoutOps) {
    if (auto ts = lay.getTileSizeAttr()) {
      auto arr = ts.asArrayRef();
      if (!arr.empty() && arr[0] != 1)
        return lay.emitError(
            "prepare-matmul: layout tile_size[0] on a "
            "batch_matmul must be 1 (batch dim cannot be tiled)");
    }
  }
  for (auto t : tileOps) {
    if (auto ts = t.getLoopTileSizeAttr()) {
      auto arr = ts.asArrayRef();
      if (!arr.empty() && arr[0] != 1)
        return t.emitError(
            "prepare-matmul: tile_op tile_size[0] on a "
            "batch_matmul must be 1 (batch dim cannot be tiled)");
    }
  }

  Location loc = bmmOp.getLoc();
  int64_t B = initType.getShape()[0];
  int64_t M = initType.getShape()[1];
  int64_t N = initType.getShape()[2];
  Type elemTy = initType.getElementType();

  OpBuilder builder(bmmOp);

  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value cB = builder.create<arith::ConstantIndexOp>(loc, B);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);

  auto forOp = builder.create<scf::ForOp>(loc, c0, cB, c1, ValueRange{init});

  builder.setInsertionPointToStart(forOp.getBody());
  Value iv = forOp.getInductionVar();
  Value acc = forOp.getRegionIterArg(0);

  auto extract2D = [&](Value src) -> Value {
    auto srcType = cast<RankedTensorType>(src.getType());
    auto shape = srcType.getShape();
    auto sliceType = RankedTensorType::get({shape[1], shape[2]},
                                            srcType.getElementType());
    SmallVector<OpFoldResult> offsets = {iv, builder.getIndexAttr(0),
                                          builder.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes = {builder.getIndexAttr(1),
                                        builder.getIndexAttr(shape[1]),
                                        builder.getIndexAttr(shape[2])};
    SmallVector<OpFoldResult> strides(3, builder.getIndexAttr(1));
    return builder.create<tensor::ExtractSliceOp>(loc, sliceType, src,
                                                   offsets, sizes, strides);
  };

  Value lhsSlice = extract2D(lhs);
  Value rhsSlice = extract2D(rhs);

  auto mmType = RankedTensorType::get({M, N}, elemTy);
  SmallVector<OpFoldResult> sliceOffsets = {iv, builder.getIndexAttr(0),
                                             builder.getIndexAttr(0)};
  SmallVector<OpFoldResult> sliceSizes = {builder.getIndexAttr(1),
                                           builder.getIndexAttr(M),
                                           builder.getIndexAttr(N)};
  SmallVector<OpFoldResult> sliceStrides(3, builder.getIndexAttr(1));

  Value initSlice = builder.create<tensor::ExtractSliceOp>(
      loc, mmType, acc, sliceOffsets, sliceSizes, sliceStrides);
  auto matmulOp = builder.create<linalg::MatmulOp>(
      loc, TypeRange{mmType}, ValueRange{lhsSlice, rhsSlice},
      ValueRange{initSlice});
  Value inserted = builder.create<tensor::InsertSliceOp>(
      loc, matmulOp.getResult(0), acc, sliceOffsets, sliceSizes,
      sliceStrides);

  if (auto opIdAttr = bmmOp->getAttrOfType<IntegerAttr>("nkipy.op_id"))
    matmulOp->setAttr("nkipy.op_id", opIdAttr);

  builder.create<scf::YieldOp>(loc, ValueRange{inserted});

  Value forResult = forOp.getResult(0);

  DenseI64ArrayAttr derivedLayoutTile;
  if (!tileOps.empty()) {
    if (auto ts = tileOps.front().getLoopTileSizeAttr()) {
      auto arr = ts.asArrayRef();
      if (arr.size() >= 2) {
        SmallVector<int64_t> dropK(arr.begin(), arr.end() - 1);
        derivedLayoutTile = DenseI64ArrayAttr::get(bmmOp.getContext(), dropK);
      }
    }
  }
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(forOp);
  for (auto lay : layoutOps) {
    DenseI64ArrayAttr layoutTile = lay.getTileSizeAttr();
    if (!layoutTile)
      layoutTile = derivedLayoutTile;
    builder.create<nkipy::LayoutOp>(
        lay.getLoc(), forResult, lay.getMemSpaceAttr(),
        lay.getPartitionDimAttr(), layoutTile);
  }

  for (auto t : tileOps) {
    DenseI64ArrayAttr innerTileSize;
    if (auto ts = t.getLoopTileSizeAttr()) {
      auto arr = ts.asArrayRef();
      if (arr.size() >= 2) {
        SmallVector<int64_t> inner(arr.begin() + 1, arr.end());
        innerTileSize = DenseI64ArrayAttr::get(bmmOp.getContext(), inner);
      }
    }
    if (innerTileSize) {
      OpBuilder innerBuilder(matmulOp);
      innerBuilder.setInsertionPointAfter(matmulOp);
      innerBuilder.create<nkipy::TileOp>(
          t.getLoc(), matmulOp.getResult(0), innerTileSize);
    }
  }

  SmallVector<OpOperand *> usesToReplace;
  for (OpOperand &use : bmmOp.getResult(0).getUses()) {
    Operation *owner = use.getOwner();
    if (!isa<nkipy::LayoutOp>(owner) && !isa<nkipy::TileOp>(owner))
      usesToReplace.push_back(&use);
  }
  for (OpOperand *use : usesToReplace)
    use->set(forResult);

  for (Operation *user : llvm::make_early_inc_range(
           bmmOp.getResult(0).getUsers())) {
    if (isa<nkipy::LayoutOp>(user) || isa<nkipy::TileOp>(user))
      user->erase();
  }
  bmmOp.erase();

  llvm::errs() << "[PrepareMatmul] Decomposed batch_matmul B=" << B
               << " M=" << M << " N=" << N << "\n";
  return success();
}

//===----------------------------------------------------------------------===//
// Step 2: Remove fill(0) before matmul-like ops
//===----------------------------------------------------------------------===//

struct RemoveZeroFillBeforeMatmul : public OpRewritePattern<linalg::FillOp> {
  using OpRewritePattern<linalg::FillOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::FillOp fillOp,
                                PatternRewriter &rewriter) const override {
    if (!isZeroConstant(fillOp.getInputs()[0]))
      return failure();

    // Tensor path: fill produces a result that feeds into matmul.
    if (fillOp.getNumResults() == 1) {
      Value fillResult = fillOp.getResult(0);
      for (Operation *user : fillResult.getUsers()) {
        if (!isMatmulLikeOp(user))
          return failure();
      }
      Value outputTensor = fillOp.getOutputs()[0];
      llvm::errs() << "[PrepareMatmul] Removing zero fill before matmul\n";
      rewriter.replaceOp(fillOp, outputTensor);
      return success();
    }

    // Memref path: fill writes in-place, no results. Verify that at least
    // one matmul-like op uses the same output memref as its DPS init
    // (NISA matmul auto-zeros PSUM, so the fill is redundant).
    Value outMemref = fillOp.getOutputs()[0];
    bool hasMatmulConsumer = false;
    for (OpOperand &use : outMemref.getUses()) {
      Operation *user = use.getOwner();
      if (user == fillOp)
        continue;
      if (!isMatmulLikeOp(user))
        continue;
      auto dpsUser = dyn_cast<DestinationStyleOpInterface>(user);
      if (dpsUser && dpsUser.isDpsInit(&use)) {
        hasMatmulConsumer = true;
        break;
      }
    }
    if (!hasMatmulConsumer)
      return failure();
    llvm::errs() << "[PrepareMatmul] Removing zero fill before matmul (memref)\n";
    rewriter.eraseOp(fillOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Combined pass
//===----------------------------------------------------------------------===//

struct PrepareMatmulPass
    : public PassWrapper<PrepareMatmulPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareMatmulPass)

  StringRef getArgument() const final { return "prepare-matmul"; }

  StringRef getDescription() const final {
    return "Prepare matmul ops for NISA: decompose batch_matmul, "
           "remove redundant zero fills";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<nkipy::NkipyDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();

    // Step 1: Decompose batch_matmul → scf.for + matmul.
    SmallVector<linalg::BatchMatmulOp> bmms;
    func.walk([&](linalg::BatchMatmulOp op) { bmms.push_back(op); });
    for (auto bmm : bmms) {
      if (failed(decomposeOneBatchMatmul(bmm))) {
        signalPassFailure();
        return;
      }
    }

    // Step 2: Remove fill(0) before matmul-like ops.
    RewritePatternSet patterns(ctx);
    patterns.add<RemoveZeroFillBeforeMatmul>(ctx);
    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    llvm::errs() << "[PrepareMatmul] Pass completed\n";
  }
};

} // namespace

namespace mlir {
namespace nkipy {

std::unique_ptr<OperationPass<func::FuncOp>> createPrepareMatmulPass() {
  return std::make_unique<PrepareMatmulPass>();
}

} // namespace nkipy
} // namespace mlir
