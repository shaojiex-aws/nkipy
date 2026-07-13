//===- LegalizeLayout.cpp - Legalize SBUF tensor layouts ------------------===//
//
// This pass attaches #nkipy.sbuf_map to SBUF memref types, encoding
// the physical factorization (tile/blocks) without reshaping the memref.
// The logical shape is preserved — downstream passes read the attr to
// generate tiled DMA and tile-access ops.
//
// The pass also tiles HBM↔SBUF copies/transposes into block loops and
// decomposes linalg.fill on HBM into SBUF fill + tiled copy.
//
// Prerequisites:
// - Runs after knob-driven-tiling + transform-interpreter
// - Runs after canonicalize-loop-step (loops have step=1)
//
//===----------------------------------------------------------------------===//

#include "PassGen.h"
#include "nkipy/Transforms/Passes.h"
#include "nkipy/Transforms/HardwareConstants.h"
#include "nkipy/Transforms/IRHelpers.h"
#include "nkipy/Dialect/NkipyAttrs.h"
#include "nkipy/Dialect/NkipyDialect.h"
#include "nkipy/Dialect/NkipyOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"


#define DEBUG_TYPE "legalize-layout"

using namespace mlir;
using namespace nkipy;

namespace mlir {
namespace nkipy {

namespace {

/// Structure to hold layout transformation info for an SBUF tensor.
struct LayoutInfo {
  Value originalValue;
  SmallVector<int64_t> origShape;
  SmallVector<int64_t> tileSize;
  SmallVector<int64_t> numBlocks;

  int64_t rank() const { return origShape.size(); }
};

static bool isSbuf(Attribute memSpaceAttr) {
  if (auto a = dyn_cast_or_null<nkipy::MemSpaceAttr>(memSpaceAttr))
    return a.getValue() == nkipy::MemSpaceEnum::Sbuf;
  return false;
}

static bool isHbm(Attribute memSpaceAttr) {
  if (auto a = dyn_cast_or_null<nkipy::MemSpaceAttr>(memSpaceAttr))
    return a.getValue() == nkipy::MemSpaceEnum::Hbm ||
           a.getValue() == nkipy::MemSpaceEnum::SharedHbm;
  return false;
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct NkipyLegalizeLayoutPass
    : public LegalizeLayoutBase<NkipyLegalizeLayoutPass> {

  bool hasError = false;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<NkipyDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<bufferization::BufferizationDialect>();
  }

  int64_t maxPartitionDim() const {
    StringRef t = target.empty() ? StringRef("trn2") : StringRef(target);
    if (auto n = nkipy::getSbufNumPartitions(t))
      return *n;
    if (auto n = nkipy::getSbufNumPartitions("trn2"))
      return *n;
    return 128;
  }

  static void eraseAllLayoutOps(func::FuncOp func) {
    SmallVector<nkipy::LayoutOp> layoutOps;
    func.walk([&](nkipy::LayoutOp op) { layoutOps.push_back(op); });
    for (auto op : layoutOps)
      op.erase();
  }


  void runOnOperation() override {
    func::FuncOp func = getOperation();
    hasError = false;

    // Body-less custom-op declarations have no SBUF tensors to legalize.
    if (func.isDeclaration())
      return;

    LLVM_DEBUG(llvm::dbgs() << "[LegalizeLayout] Processing function: "
                 << func.getName() << "\n");

    // Phase 1: Find SBUF allocs and compute factorization
    SmallVector<LayoutInfo> layoutInfos = findSbufTensorsToLegalize(func);

    if (hasError) {
      signalPassFailure();
      return;
    }

    if (layoutInfos.empty()) {
      decomposeHbmFills(func);
      eraseAllLayoutOps(func);
      return;
    }

    llvm::errs() << "[LegalizeLayout] Found " << layoutInfos.size()
                 << " SBUF tensor(s) to legalize:\n";
    for (auto &info : layoutInfos) {
      llvm::errs() << "  tensor<";
      llvm::interleave(info.origShape, llvm::errs(), "x");
      llvm::errs() << "> tile=[";
      llvm::interleave(info.tileSize, llvm::errs(), ",");
      llvm::errs() << "] numBlocks=[";
      llvm::interleave(info.numBlocks, llvm::errs(), ",");
      llvm::errs() << "]\n";
    }

    // Phase 2: Attach #nkipy.sbuf_map to memref types (logical shape preserved)
    attachSbufMapAttrs(func, layoutInfos);

    if (hasError) {
      signalPassFailure();
      return;
    }

    // Phase 4: Decompose HBM fills into SBUF fill + tiled DMA.
    // HBM has no direct fill engine — we stage through an SBUF buffer:
    //   linalg.fill(scalar, hbm_buf)
    //     → alloc sbuf_tile
    //     → linalg.fill(scalar, sbuf_tile)
    //     → loop { linalg.copy sbuf_tile[block] → hbm_buf[block] }
    decomposeHbmFills(func);

    if (hasError) {
      signalPassFailure();
      return;
    }

    // Note: >2D tile-sized SBUF allocs are NOT flattened in the IR.
    // The NISA emitter handles 2D projection at emission time (dim0=partition,
    // product(dim1..R-1)=free). This keeps the IR clean and avoids
    // collapse_shape/expand_shape view chains.

    eraseAllLayoutOps(func);

    // Epilogue: canonicalize to clean up dead subviews and simplify.
    RewritePatternSet patterns(&getContext());
    for (auto *dialect : getContext().getLoadedDialects())
      dialect->getCanonicalizationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

    llvm::errs() << "[LegalizeLayout] Pass completed successfully\n";
  }

  //===--------------------------------------------------------------------===//
  // Phase 1: Find SBUF tensors
  //===--------------------------------------------------------------------===//

  static nkipy::LayoutOp findLayoutOpFor(Value val) {
    for (Operation *user : val.getUsers())
      if (auto lay = dyn_cast<nkipy::LayoutOp>(user))
        return lay;
    return nullptr;
  }

  SmallVector<LayoutInfo> findSbufTensorsToLegalize(func::FuncOp func) {
    SmallVector<LayoutInfo> results;

    SmallVector<memref::AllocOp> sbufAllocs;
    func.walk([&](memref::AllocOp allocOp) {
      auto memrefType = allocOp.getType();
      if (!isSbuf(memrefType.getMemorySpace()))
        return;
      if (memrefType.getRank() < 2)
        return;
      sbufAllocs.push_back(allocOp);
    });

    for (auto allocOp : sbufAllocs) {
      auto memrefType = allocOp.getType();
      auto origShape = memrefType.getShape();
      int64_t R = memrefType.getRank();

      llvm::errs() << "[LegalizeLayout] Processing SBUF alloc: " << memrefType
                   << " at " << allocOp.getLoc() << "\n";

      SmallVector<int64_t> refTile;

      if (auto layoutOp = findLayoutOpFor(allocOp.getResult())) {
        if (auto ts = layoutOp.getTileSizeAttr()) {
          auto arr = ts.asArrayRef();
          refTile.assign(arr.begin(), arr.end());

          if ((int64_t)refTile.size() < R) {
            SmallVector<int64_t> lifted;
            size_t src = 0;
            for (int64_t i = 0; i < R; i++) {
              if (origShape[i] == 1)
                lifted.push_back(1);
              else if (src < refTile.size())
                lifted.push_back(refTile[src++]);
              else
                lifted.push_back(origShape[i]);
            }
            refTile = lifted;
          }

          llvm::errs() << "  Using tile from nkipy.layout: [";
          llvm::interleave(refTile, llvm::errs(), ",");
          llvm::errs() << "]\n";
        }
      }

      if (refTile.empty()) {
        llvm::errs() << "[LegalizeLayout] Error: SBUF alloc at "
                     << allocOp.getLoc()
                     << " missing tile_size on nkipy.layout\n";
        hasError = true;
        return results;
      }

      if ((int64_t)refTile.size() != R) {
        llvm::errs() << "[LegalizeLayout] Error: tile rank " << refTile.size()
                     << " != alloc rank " << R << "\n";
        hasError = true;
        return results;
      }
      for (int64_t i = 0; i < R; i++) {
        if (origShape[i] % refTile[i] != 0) {
          llvm::errs() << "[LegalizeLayout] Error: dim " << i << " size "
                       << origShape[i] << " not divisible by tile " << refTile[i] << "\n";
          hasError = true;
          return results;
        }
      }

      SmallVector<int64_t> numBlocks;
      for (int64_t i = 0; i < R; i++)
        numBlocks.push_back(origShape[i] / refTile[i]);

      if (llvm::all_of(numBlocks, [](int64_t n) { return n == 1; })) {
        llvm::errs() << "  -> Skipping (numBlocks all 1, already tile-sized)\n";
        continue;
      }


      LayoutInfo info;
      info.originalValue = allocOp.getResult();
      info.origShape = SmallVector<int64_t>(origShape.begin(), origShape.end());
      info.tileSize = refTile;
      info.numBlocks = numBlocks;
      results.push_back(info);
    }

    return results;
  }

  //===--------------------------------------------------------------------===//
  // Phase 2: Attach #nkipy.sbuf_map to memref types
  //===--------------------------------------------------------------------===//

  void attachSbufMapAttrs(func::FuncOp func, SmallVector<LayoutInfo> &layoutInfos) {
    for (auto &info : layoutInfos) {
      auto allocOp = info.originalValue.getDefiningOp<memref::AllocOp>();
      if (!allocOp) {
        hasError = true;
        return;
      }

      auto origType = allocOp.getType();
      auto sbufMap = SbufMapAttr::get(
          allocOp.getContext(), info.tileSize, info.numBlocks);

      auto newType = MemRefType::get(
          origType.getShape(),
          origType.getElementType(),
          sbufMap,
          origType.getMemorySpace());

      OpBuilder builder(allocOp);
      builder.setInsertionPoint(allocOp);
      auto newAlloc = builder.create<memref::AllocOp>(
          allocOp.getLoc(), newType, allocOp.getAlignmentAttr());

      allocOp.replaceAllUsesWith(newAlloc.getResult());
      allocOp.erase();

      info.originalValue = newAlloc.getResult();

      LLVM_DEBUG(llvm::dbgs() << " Attached sbuf_map: " << newType << "\n");
    }
  }


  //===--------------------------------------------------------------------===//
  // Phase 4: Decompose HBM fills
  // HBM has no direct fill engine. This decomposes linalg.fill on HBM into:
  //   1. alloc an SBUF tile
  //   2. linalg.fill the SBUF tile
  //   3. loop { DMA copy SBUF tile → HBM block }
  //===--------------------------------------------------------------------===//

  void decomposeHbmFills(func::FuncOp func) {
    SmallVector<linalg::FillOp> fillsToDecompose;

    func.walk([&](linalg::FillOp fillOp) {
      Value output = fillOp.getOutputs()[0];
      auto outputType = dyn_cast<MemRefType>(output.getType());
      if (!outputType)
        return;
      if (!isHbm(outputType.getMemorySpace()))
        return;
      if (!outputType.hasStaticShape())
        return;
      if (outputType.getRank() < 2)
        return;
      fillsToDecompose.push_back(fillOp);
    });

    if (fillsToDecompose.empty())
      return;

    OpBuilder builder(func.getContext());

    for (auto fillOp : fillsToDecompose) {
      builder.setInsertionPoint(fillOp);
      Location loc = fillOp.getLoc();

      Value scalarValue = fillOp.getInputs()[0];
      Value hbmBuf = fillOp.getOutputs()[0];
      auto hbmType = cast<MemRefType>(hbmBuf.getType());
      auto hbmShape = hbmType.getShape();
      int64_t rank = hbmType.getRank();

      int64_t partDim = hbmShape[0];
      int64_t partTile = std::min(partDim, maxPartitionDim());
      int64_t numBlocks = (partDim + partTile - 1) / partTile;

      SmallVector<int64_t> sbufShape;
      sbufShape.push_back(partTile);
      sbufShape.push_back(numBlocks * hbmShape[1]);
      for (int64_t i = 2; i < rank; ++i)
        sbufShape.push_back(hbmShape[i]);

      auto sbufMemSpace = nkipy::MemSpaceAttr::get(
          builder.getContext(), nkipy::MemSpaceEnum::Sbuf);
      auto sbufType = MemRefType::get(
          sbufShape, hbmType.getElementType(), nullptr, sbufMemSpace);
      auto sbufAlloc = builder.create<memref::AllocOp>(loc, sbufType);

      builder.create<linalg::FillOp>(loc, scalarValue, sbufAlloc.getResult());

      if (numBlocks == 1) {
        builder.create<linalg::CopyOp>(loc, ValueRange{sbufAlloc.getResult()},
                                        ValueRange{hbmBuf});
      } else {
        int64_t freeDim = hbmShape[1];

        Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
        Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
        Value numBlocksVal = builder.create<arith::ConstantIndexOp>(loc, numBlocks);
        Value partTileVal = builder.create<arith::ConstantIndexOp>(loc, partTile);
        Value freeDimVal = builder.create<arith::ConstantIndexOp>(loc, freeDim);

        auto loop = builder.create<scf::ForOp>(loc, c0, numBlocksVal, c1);
        builder.setInsertionPointToStart(loop.getBody());
        Value iv = loop.getInductionVar();

        Value sbufDim1Offset = builder.create<arith::MulIOp>(loc, iv, freeDimVal);
        SmallVector<OpFoldResult> sbufOffsets, sbufSizes, sbufStrides;
        sbufOffsets.push_back(builder.getIndexAttr(0));
        sbufOffsets.push_back(OpFoldResult(sbufDim1Offset));
        sbufSizes.push_back(builder.getIndexAttr(partTile));
        sbufSizes.push_back(builder.getIndexAttr(freeDim));
        sbufStrides.push_back(builder.getIndexAttr(1));
        sbufStrides.push_back(builder.getIndexAttr(1));
        for (int64_t i = 2; i < rank; ++i) {
          sbufOffsets.push_back(builder.getIndexAttr(0));
          sbufSizes.push_back(builder.getIndexAttr(hbmShape[i]));
          sbufStrides.push_back(builder.getIndexAttr(1));
        }
        auto sbufTile = builder.create<memref::SubViewOp>(
            loc, sbufAlloc.getResult(), sbufOffsets, sbufSizes, sbufStrides);

        Value hbmPartOffset = builder.create<arith::MulIOp>(loc, iv, partTileVal);
        SmallVector<OpFoldResult> hbmOffsets, hbmSizes, hbmStrides;
        hbmOffsets.push_back(OpFoldResult(hbmPartOffset));
        hbmSizes.push_back(builder.getIndexAttr(partTile));
        hbmStrides.push_back(builder.getIndexAttr(1));
        for (int64_t i = 1; i < rank; ++i) {
          hbmOffsets.push_back(builder.getIndexAttr(0));
          hbmSizes.push_back(builder.getIndexAttr(hbmShape[i]));
          hbmStrides.push_back(builder.getIndexAttr(1));
        }
        auto hbmTile = builder.create<memref::SubViewOp>(
            loc, hbmBuf, hbmOffsets, hbmSizes, hbmStrides);

        builder.create<linalg::CopyOp>(loc, ValueRange{sbufTile},
                                        ValueRange{hbmTile});
        builder.setInsertionPointAfter(loop);
      }

      fillOp.erase();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeLayoutPass() {
  return std::make_unique<NkipyLegalizeLayoutPass>();
}

} // namespace nkipy
} // namespace mlir
