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

static bool needsTiledTransfer(MemRefType srcType, MemRefType dstType) {
  bool srcH = isHbm(srcType.getMemorySpace());
  bool srcS = isSbuf(srcType.getMemorySpace());
  bool dstH = isHbm(dstType.getMemorySpace());
  bool dstS = isSbuf(dstType.getMemorySpace());
  return (srcH && dstS) || (srcS && dstH);
}

/// Look through memref.cast ops to find the base value.
static Value lookThroughCast(Value v) {
  while (auto castOp = v.getDefiningOp<memref::CastOp>())
    v = castOp.getSource();
  return v;
}

/// Result of createBlockLoopNest.
struct BlockLoopNest {
  SmallVector<Value> ivs;
};

/// Create an R-level scf.for loop nest iterating over block indices.
static BlockLoopNest createBlockLoopNest(
    OpBuilder &builder, Location loc, ArrayRef<int64_t> numBlocks) {
  BlockLoopNest result;
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  for (int64_t nb : numBlocks) {
    Value ub = builder.create<arith::ConstantIndexOp>(loc, nb);
    auto loop = builder.create<scf::ForOp>(loc, c0, ub, c1);
    builder.setInsertionPointToStart(loop.getBody());
    result.ivs.push_back(loop.getInductionVar());
  }
  return result;
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

    // Phase 3: Tile HBM↔SBUF copies and transposes
    tileCopyAndTranspose(func, layoutInfos);

    if (hasError) {
      signalPassFailure();
      return;
    }

    // Phase 4: Decompose HBM fills into SBUF fill + tiled DMA.
    // HBM has no direct fill engine — we stage through an SBUF buffer:
    //   linalg.fill(scalar, hbm_buf)
    //     → alloc sbuf_tile
    //     → linalg.fill(scalar, sbuf_tile)
    //     → loop { memref.copy sbuf_tile[block] → hbm_buf[block] }
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
  // Phase 3: Tile HBM↔SBUF copies and transposes
  //===--------------------------------------------------------------------===//

  /// Find the LayoutInfo for a given SBUF value (looks through subviews/casts).
  LayoutInfo *findLayoutForValue(Value val, SmallVector<LayoutInfo> &layoutInfos) {
    Value base = val;
    while (true) {
      if (auto sv = base.getDefiningOp<memref::SubViewOp>()) {
        base = sv.getSource();
        continue;
      }
      if (auto cast = base.getDefiningOp<memref::CastOp>()) {
        base = cast.getSource();
        continue;
      }
      break;
    }
    for (auto &info : layoutInfos) {
      if (info.originalValue == base)
        return &info;
    }
    return nullptr;
  }

  /// Get SbufMapAttr from a memref value (walks through subview/cast).
  SbufMapAttr getSbufMapFor(Value val) {
    Value base = lookThroughCast(val);
    while (auto sv = base.getDefiningOp<memref::SubViewOp>())
      base = lookThroughCast(sv.getSource());
    auto memrefType = dyn_cast<MemRefType>(base.getType());
    if (!memrefType)
      return nullptr;
    return dyn_cast_or_null<SbufMapAttr>(memrefType.getLayout());
  }

  /// Extract (source, target) from a memref.copy or linalg.copy. Returns
  /// false if `op` is neither. (Interim: both copy kinds coexist until the
  /// staging copies are tiled at emit time; this function is removed then.)
  static bool getCopyOperands(Operation *op, Value &src, Value &dst) {
    if (auto c = dyn_cast<memref::CopyOp>(op)) {
      src = c.getSource();
      dst = c.getTarget();
      return true;
    }
    if (auto c = dyn_cast<linalg::CopyOp>(op)) {
      src = c.getInputs()[0];
      dst = c.getOutputs()[0];
      return true;
    }
    return false;
  }

  void tileCopyAndTranspose(func::FuncOp func, SmallVector<LayoutInfo> &layoutInfos) {
    OpBuilder builder(func.getContext());

    SmallVector<Operation *> copiesToTile;
    SmallVector<linalg::TransposeOp> transposesToTile;

    func.walk([&](Operation *op) {
      Value copySrc, copyDst;
      if (getCopyOperands(op, copySrc, copyDst)) {
        auto srcType = cast<MemRefType>(copySrc.getType());
        auto dstType = cast<MemRefType>(copyDst.getType());
        if (needsTiledTransfer(srcType, dstType)) {
          // Only tile if the SBUF side has sbuf_map
          Value sbufSide = isSbuf(srcType.getMemorySpace())
              ? copySrc : copyDst;
          if (getSbufMapFor(sbufSide))
            copiesToTile.push_back(op);
        }
      } else if (auto transposeOp = dyn_cast<linalg::TransposeOp>(op)) {
        Value input = transposeOp.getDpsInputs()[0];
        Value output = transposeOp.getDpsInits()[0];
        Value inputBase = lookThroughCast(input);
        auto inputBaseType = cast<MemRefType>(inputBase.getType());
        auto outputType = cast<MemRefType>(output.getType());

        if (needsTiledTransfer(inputBaseType, outputType) ||
            (isSbuf(inputBaseType.getMemorySpace()) &&
             isSbuf(outputType.getMemorySpace()))) {
          if (getSbufMapFor(input) || getSbufMapFor(output))
            transposesToTile.push_back(transposeOp);
        }
      }
    });

    for (auto copyOp : copiesToTile) {
      tileMemrefCopy(builder, copyOp, layoutInfos);
      if (hasError) return;
    }

    for (auto transposeOp : transposesToTile) {
      tileTranspose(builder, transposeOp, layoutInfos);
      if (hasError) return;
    }
  }

  void tileMemrefCopy(OpBuilder &builder, Operation *op,
                      SmallVector<LayoutInfo> &layoutInfos) {
    Value src, dst;
    getCopyOperands(op, src, dst);
    auto srcType = cast<MemRefType>(src.getType());
    auto dstType = cast<MemRefType>(dst.getType());

    bool srcIsSbuf = isSbuf(srcType.getMemorySpace());
    Value bufSBUF = srcIsSbuf ? src : dst;
    Value bufHBM = srcIsSbuf ? dst : src;

    LayoutInfo *info = findLayoutForValue(bufSBUF, layoutInfos);
    if (!info) {
      LLVM_DEBUG(llvm::dbgs() << " Skipping copy (no layout info)\n");
      return;
    }

    int64_t R = info->rank();
    builder.setInsertionPoint(op);
    Location loc = op->getLoc();

    auto nest = createBlockLoopNest(builder, loc, info->numBlocks);

    // HBM subview: [iv0*t0, iv1*t1, ...][t0, t1, ...]
    SmallVector<OpFoldResult> offsetsHBM, sizesHBM, stridesHBM;
    for (int64_t i = 0; i < R; i++) {
      if (info->tileSize[i] == 1) {
        offsetsHBM.push_back(OpFoldResult(nest.ivs[i]));
      } else {
        Value ts = builder.create<arith::ConstantIndexOp>(loc, info->tileSize[i]);
        Value offset = builder.create<arith::MulIOp>(loc, nest.ivs[i], ts);
        offsetsHBM.push_back(OpFoldResult(offset));
      }
      sizesHBM.push_back(builder.getIndexAttr(info->tileSize[i]));
      stridesHBM.push_back(builder.getIndexAttr(1));
    }
    auto hbmTile = builder.create<memref::SubViewOp>(
        loc, bufHBM, offsetsHBM, sizesHBM, stridesHBM);

    // SBUF subview: same offsets/sizes (logical shape preserved)
    SmallVector<OpFoldResult> offsetsSBUF, sizesSBUF, stridesSBUF;
    for (int64_t i = 0; i < R; i++) {
      offsetsSBUF.push_back(offsetsHBM[i]);
      sizesSBUF.push_back(builder.getIndexAttr(info->tileSize[i]));
      stridesSBUF.push_back(builder.getIndexAttr(1));
    }
    auto sbufTile = builder.create<memref::SubViewOp>(
        loc, bufSBUF, offsetsSBUF, sizesSBUF, stridesSBUF);

    if (srcIsSbuf)
      builder.create<linalg::CopyOp>(loc, ValueRange{sbufTile},
                                      ValueRange{hbmTile});
    else
      builder.create<linalg::CopyOp>(loc, ValueRange{hbmTile},
                                      ValueRange{sbufTile});

    LLVM_DEBUG(llvm::dbgs() << " Tiled copy: " << srcType << " -> " << dstType << "\n");
    op->erase();
  }

  void tileTranspose(OpBuilder &builder, linalg::TransposeOp op,
                     SmallVector<LayoutInfo> &layoutInfos) {
    Value input = op.getDpsInputs()[0];
    Value output = op.getDpsInits()[0];
    Value inputBase = lookThroughCast(input);
    auto inputBaseType = cast<MemRefType>(inputBase.getType());
    auto outputType = cast<MemRefType>(output.getType());

    bool inputIsSbuf = isSbuf(inputBaseType.getMemorySpace());
    bool outputIsSbuf = isSbuf(outputType.getMemorySpace());

    auto permutation = op.getPermutation();

    // Find layout info from whichever side is SBUF
    LayoutInfo *info = nullptr;
    if (outputIsSbuf)
      info = findLayoutForValue(output, layoutInfos);
    if (!info && inputIsSbuf)
      info = findLayoutForValue(inputBase, layoutInfos);
    if (!info) {
      LLVM_DEBUG(llvm::dbgs() << " Skipping transpose (no layout info)\n");
      return;
    }

    int64_t R = info->rank();
    builder.setInsertionPoint(op);
    Location loc = op.getLoc();

    // Compute effective numBlocks from the SBUF operand's actual shape
    // (not the parent alloc). The operand may be a subview that's already
    // tile-sized, in which case no block loop is needed.
    Value sbufOperand = outputIsSbuf ? output : inputBase;
    auto sbufOperandType = cast<MemRefType>(sbufOperand.getType());
    SmallVector<int64_t> effectiveNumBlocks;
    for (int64_t i = 0; i < R; i++)
      effectiveNumBlocks.push_back(sbufOperandType.getShape()[i] / info->tileSize[i]);

    // If all numBlocks are 1, the operand is already tile-sized — no loop needed.
    bool allOne = llvm::all_of(effectiveNumBlocks, [](int64_t n) { return n == 1; });

    if (allOne) {
      // Just collapse to 2D and do the copy/transpose directly.
      SmallVector<int64_t> invPerm(R);
      for (int64_t i = 0; i < R; i++)
        invPerm[permutation[i]] = i;
      SmallVector<int64_t> perm2D = (invPerm[0] < invPerm[R - 1])
          ? SmallVector<int64_t>{0, 1}
          : SmallVector<int64_t>{1, 0};

      if (perm2D[0] == 0 && perm2D[1] == 1)
        builder.create<linalg::CopyOp>(loc, ValueRange{inputBase},
                                        ValueRange{output});
      else
        builder.create<linalg::TransposeOp>(loc, inputBase, output, perm2D);

      op.erase();
      return;
    }

    auto nest = createBlockLoopNest(builder, loc, effectiveNumBlocks);

    // Inverse permutation: invPerm[out_dim] = in_dim
    SmallVector<int64_t> invPerm(R);
    for (int64_t i = 0; i < R; i++)
      invPerm[permutation[i]] = i;

    // Output (SBUF) subview: iterate over SBUF blocks using the SBUF tile size
    // directly. The SBUF is the reference buffer whose layout we're tiling.
    SmallVector<OpFoldResult> offsetsOut, sizesOut, stridesOut;
    for (int64_t i = 0; i < R; i++) {
      if (info->tileSize[i] == 1) {
        offsetsOut.push_back(OpFoldResult(nest.ivs[i]));
      } else {
        Value ts = builder.create<arith::ConstantIndexOp>(loc, info->tileSize[i]);
        Value offset = builder.create<arith::MulIOp>(loc, nest.ivs[i], ts);
        offsetsOut.push_back(OpFoldResult(offset));
      }
      sizesOut.push_back(builder.getIndexAttr(info->tileSize[i]));
      stridesOut.push_back(builder.getIndexAttr(1));
    }

    // Input (HBM) subview: apply inverse permutation to map output block
    // indices back to input dimensions. Input tile size at dim i is the
    // output tile size at the corresponding output dim (permutation[i]).
    SmallVector<OpFoldResult> offsetsIn, sizesIn, stridesIn;
    for (int64_t i = 0; i < R; i++) {
      // Input dim i corresponds to output dim permutation[i].
      // The block index for that output dim is nest.ivs[permutation[i]].
      int64_t outDim = permutation[i];
      int64_t inTile = info->tileSize[outDim];
      if (inTile == 1) {
        offsetsIn.push_back(OpFoldResult(nest.ivs[outDim]));
      } else {
        Value ts = builder.create<arith::ConstantIndexOp>(loc, inTile);
        Value offset = builder.create<arith::MulIOp>(loc, nest.ivs[outDim], ts);
        offsetsIn.push_back(OpFoldResult(offset));
      }
      sizesIn.push_back(builder.getIndexAttr(inTile));
      stridesIn.push_back(builder.getIndexAttr(1));
    }

    Value dstBuf = outputIsSbuf ? output : input;
    Value srcBuf = outputIsSbuf ? inputBase : output;

    auto outTile = builder.create<memref::SubViewOp>(
        loc, dstBuf, offsetsOut, sizesOut, stridesOut);
    auto inTile = builder.create<memref::SubViewOp>(
        loc, srcBuf, offsetsIn, sizesIn, stridesIn);

    // Determine 2D permutation (partition vs free swap)
    SmallVector<int64_t> perm2D = (invPerm[0] < invPerm[R - 1])
        ? SmallVector<int64_t>{0, 1}
        : SmallVector<int64_t>{1, 0};

    if (perm2D[0] == 0 && perm2D[1] == 1)
      builder.create<linalg::CopyOp>(loc, ValueRange{inTile},
                                      ValueRange{outTile});
    else
      builder.create<linalg::TransposeOp>(loc, inTile, outTile, perm2D);

    LLVM_DEBUG(llvm::dbgs() << " Tiled transpose: " << inputBaseType
                 << " -> " << outputType << "\n");
    op.erase();
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
