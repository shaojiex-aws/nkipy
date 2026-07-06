//===- InferLayout.cpp - Default tile_op and layout for unannotated ops ---===//
//
// Simple pass that fills in missing annotations:
// 1. defaultTileOps: emit tile_op with hardware defaults for unannotated ops
// 2. propagateTileOps: copy tile_op backward through elementwise chains
// 3. defaultLayouts: emit nkipy.layout(mem_space) on unannotated allocs
//
//===----------------------------------------------------------------------===//

#include "PassGen.h"
#include "nkipy/Transforms/Passes.h"
#include "nkipy/Transforms/HardwareConstants.h"
#include "nkipy/Transforms/IRHelpers.h"
#include "nkipy/Transforms/OpClassification.h"
#include "nkipy/Dialect/NkipyAttrs.h"
#include "nkipy/Dialect/NkipyDialect.h"
#include "nkipy/Dialect/NkipyOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace nkipy;

namespace mlir {
namespace nkipy {

namespace {

/// Get the output value for a linalg op (DPS init in memref mode).
static Value getOutputValue(linalg::LinalgOp op) {
  if (op->getNumResults() > 0)
    return op->getResult(0);
  SmallVector<Value> inits(op.getDpsInits());
  return inits.empty() ? Value() : inits[0];
}

/// Check if a value already has a nkipy.tile_op attached.
static bool hasTileOp(Value val) {
  for (Operation *user : val.getUsers())
    if (isa<nkipy::TileOp>(user))
      return true;
  return false;
}

/// Check if a value already has a nkipy.layout with mem_space attached.
static bool hasLayoutMemSpace(Value val) {
  for (Operation *user : val.getUsers()) {
    auto layout = dyn_cast<nkipy::LayoutOp>(user);
    if (layout && layout.getTarget() == val && layout.getMemSpace())
      return true;
  }
  return false;
}

/// Check if a value has a nkipy.layout with partition_dim set.
static bool hasPartitionDim(Value val) {
  for (Operation *user : val.getUsers()) {
    auto layout = dyn_cast<nkipy::LayoutOp>(user);
    if (layout && layout.getTarget() == val && layout.getPartitionDimAttr())
      return true;
  }
  return false;
}

/// Get the partition_dim attribute from a value's layout, or nullptr.
static IntegerAttr getPartitionDimAttr(Value val) {
  for (Operation *user : val.getUsers()) {
    auto layout = dyn_cast<nkipy::LayoutOp>(user);
    if (layout && layout.getTarget() == val && layout.getPartitionDimAttr())
      return layout.getPartitionDimAttr();
  }
  return nullptr;
}

/// Check if a value is used in a func.return, tracing through
/// reinterpret_cast views (a reshape of a return value is still returned).
static bool isReturnValue(Value val) {
  llvm::SmallPtrSet<Value, 8> visited;
  SmallVector<Value> worklist = {val};
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!visited.insert(v).second) continue;
    for (Operation *user : v.getUsers()) {
      if (isa<func::ReturnOp>(user))
        return true;
      if (auto cast = dyn_cast<memref::ReinterpretCastOp>(user))
        worklist.push_back(cast.getResult());
    }
  }
  return false;
}

/// Find an explicit mem_space annotation on `val` or on any reinterpret_cast
/// view of it. A reinterpret_cast is a pure reinterpretation of the same
/// storage — it cannot move data between memory spaces — so a mem_space the
/// user attached to a *view* of an alloc is really a constraint on the alloc.
/// Returns the annotated space, or nullptr if no view carries one.
static MemSpaceAttr getViewMemSpace(Value val) {
  llvm::SmallPtrSet<Value, 8> visited;
  SmallVector<Value> worklist = {val};
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!visited.insert(v).second) continue;
    for (Operation *user : v.getUsers()) {
      if (auto layout = dyn_cast<nkipy::LayoutOp>(user))
        if (layout.getTarget() == v && layout.getMemSpace())
          return *layout.getMemSpace();
      if (auto cast = dyn_cast<memref::ReinterpretCastOp>(user))
        worklist.push_back(cast.getResult());
    }
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct NkipyInferLayoutPass : public InferLayoutBase<NkipyInferLayoutPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<NkipyDialect>();
    registry.insert<linalg::LinalgDialect>();
  }

  int64_t maxPartition() const {
    StringRef t = target.empty() ? StringRef("trn2") : StringRef(target);
    if (auto n = getSbufNumPartitions(t)) return *n;
    return 128;
  }

  int64_t matmulFreeCap() const {
    StringRef t = target.empty() ? StringRef("trn2") : StringRef(target);
    if (auto n = getMatmulFreeDimTileCap(t)) return *n;
    return 512;
  }

  /// Contiguity-safe loop tile for a copy. Starts from the shape default
  /// ([min(dim0,128), dim1, ..., dimN]) and sets any dim that is not
  /// contiguous with its successor — in EITHER operand — to 1, so the free
  /// fold never strides across a gap. Returns null (fall through to the
  /// generic path) if the copy is fully contiguous on both sides, so
  /// contiguous copies keep the full tile exactly like elementwise ops.
  DenseI64ArrayAttr contiguitySafeCopyTile(linalg::LinalgOp copyOp,
                                           Value outVal, int64_t partCap) {
    auto outType = dyn_cast<MemRefType>(outVal.getType());
    if (!outType || !outType.hasStaticShape() || outType.getRank() < 2)
      return nullptr;  // rank <2: no non-partition dims to fold
    int64_t R = outType.getRank();
    ArrayRef<int64_t> shape = outType.getShape();

    // contig[i] = dim i is contiguous with dim i+1 (stride[i] ==
    // size[i+1]*stride[i+1]) in every operand with a decodable strided layout.
    // A gap in any operand makes the dim non-contiguous.
    SmallVector<bool> contig(R, true);
    for (Value operand : copyOp->getOperands()) {
      auto mt = dyn_cast<MemRefType>(operand.getType());
      if (!mt || mt.getRank() != R)
        continue;
      SmallVector<int64_t> strides;
      int64_t offset;
      if (failed(mt.getStridesAndOffset(strides, offset)))
        continue;
      ArrayRef<int64_t> sh = mt.getShape();
      for (int64_t i = 0; i + 1 < R; i++) {
        if (ShapedType::isDynamic(strides[i]) ||
            ShapedType::isDynamic(strides[i + 1]) ||
            ShapedType::isDynamic(sh[i + 1]))
          continue;  // can't prove a gap → assume contiguous
        if (strides[i] != sh[i + 1] * strides[i + 1])
          contig[i] = false;
      }
    }

    // The free span folds a contiguous suffix: dim R-1 (free base) is always
    // full, and dim i folds only if it is contiguous with i+1 AND i+1 folds.
    // A dim that can't fold is looped (tile 1); dim 0 is the partition dim.
    // Build the tile in one backward pass, tracking whether the suffix folds.
    SmallVector<int64_t> tile(R, 1);
    tile[0] = std::min(shape[0], partCap);
    bool foldsSuffix = true, anyGap = false;
    for (int64_t i = R - 1; i >= 1; i--) {
      if (foldsSuffix)
        tile[i] = shape[i];
      else
        anyGap = true;
      foldsSuffix = foldsSuffix && contig[i - 1];
    }
    if (!anyGap)
      return nullptr;  // fully contiguous → use the generic default
    return DenseI64ArrayAttr::get(copyOp.getContext(), tile);
  }

  /// Step 1: emit default tile_op for ops that lack one.
  void defaultTileOps(func::FuncOp func) {
    int64_t partCap = maxPartition();
    int64_t freeCap = matmulFreeCap();

    func.walk([&](linalg::LinalgOp linalgOp) {
      if (!isAnnotatableOp(linalgOp)) return;

      Value outVal = getOutputValue(linalgOp);
      if (!outVal || hasTileOp(outVal)) return;

      SmallVector<int64_t> tile;

      // A copy lowers to a 2D partition x free DMA: the emitter folds the
      // non-partition dims into one free span. That fold is only valid when
      // those dims are contiguous in the copied buffer. A boundary copy (e.g.
      // last-axis concat) writes a strided slice where a dim has a stride gap;
      // folding it would stride across the gap into neighbouring data. So a
      // copy starts from the shape default, then sets any dim that is NOT
      // contiguous with its successor to 1 — tiling exactly the dims the fold
      // can't cross, leaving the contiguous tail folded. A fully contiguous
      // copy keeps the full tile (same as elementwise); only strided slices
      // are affected.
      if (isa<linalg::CopyOp>(linalgOp.getOperation())) {
        if (auto t = contiguitySafeCopyTile(linalgOp, outVal, partCap)) {
          OpBuilder builder(linalgOp);
          builder.setInsertionPointAfter(linalgOp);
          builder.create<nkipy::TileOp>(outVal.getLoc(), outVal, t);
          return;
        }
      }

      if (isMatmulOp(linalgOp)) {
        // Matmul C[M,N] = A[M,K] * B[K,N]: tile = [M_t, N_t, K_t]
        auto outType = cast<MemRefType>(outVal.getType());
        auto outShape = outType.getShape();
        int64_t R = outType.getRank();
        int64_t M = outShape[R - 2];
        int64_t N = outShape[R - 1];
        // K from first input's last dim
        auto inType = cast<MemRefType>(linalgOp.getDpsInputs()[0].getType());
        int64_t K = inType.getShape()[inType.getRank() - 1];
        tile = {std::min(M, partCap), std::min(N, freeCap), std::min(K, partCap)};
      } else if (isElementwiseOp(linalgOp)) {
        auto outType = cast<MemRefType>(outVal.getType());
        auto shape = outType.getShape();
        tile.push_back(std::min(shape[0], partCap));
        for (int64_t i = 1; i < outType.getRank(); i++)
          tile.push_back(shape[i]);
      } else if (isReductionGeneric(linalgOp)) {
        // One entry per iterator (parallel + reduction)
        auto inType = cast<MemRefType>(linalgOp.getDpsInputs()[0].getType());
        auto shape = inType.getShape();
        tile.push_back(std::min(shape[0], partCap));
        for (int64_t i = 1; i < inType.getRank(); i++)
          tile.push_back(shape[i]);
      } else if (isa<linalg::TransposeOp>(linalgOp.getOperation())) {
        // Transpose: 2D transposes stay 2D. Higher-rank transposes are
        // lowered as loops around an effective 2D inner op, so keep two
        // moved output dims live and tile the rest to 1.
        auto outType = cast<MemRefType>(outVal.getType());
        auto shape = outType.getShape();
        auto transposeOp = cast<linalg::TransposeOp>(linalgOp.getOperation());
        auto perm = transposeOp.getPermutation();

        if (outType.getRank() <= 2) {
          for (int64_t i = 0; i < outType.getRank(); i++)
            tile.push_back(std::min(shape[i], partCap));
        } else {
          SmallVector<unsigned> movedDims;
          for (unsigned i = 0; i < (unsigned)outType.getRank(); i++)
            if (perm[i] != i)
              movedDims.push_back(i);

          // Keep the two largest moved output dims. For head_deconcat
          // perm=[0,2,1,3], this keeps seq and head: [1,128,2,1].
          SmallVector<bool> keep(outType.getRank(), false);
          for (unsigned selected = 0; selected < 2 && !movedDims.empty();
               selected++) {
            unsigned bestPos = 0;
            for (unsigned pos = 1; pos < movedDims.size(); pos++)
              if (shape[movedDims[pos]] > shape[movedDims[bestPos]])
                bestPos = pos;
            keep[movedDims[bestPos]] = true;
            movedDims.erase(movedDims.begin() + bestPos);
          }

          for (int64_t i = 0; i < outType.getRank(); i++)
            tile.push_back(keep[i] ? std::min(shape[i], partCap) : 1);
        }
      }

      if (tile.empty()) return;

      OpBuilder builder(linalgOp);
      builder.setInsertionPointAfter(linalgOp);
      auto tileAttr = DenseI64ArrayAttr::get(func.getContext(), tile);
      builder.create<nkipy::TileOp>(outVal.getLoc(), outVal, tileAttr);
    });
  }

  /// Step 2: propagate tile_op and partition_dim backward through
  /// elementwise chains. Stops at ops that already have a tile_op or
  /// a layout with partition_dim.
  void propagateTileAndLayout(func::FuncOp func) {
    MLIRContext *ctx = func.getContext();
    auto sbuf = MemSpaceAttr::get(ctx, MemSpaceEnum::Sbuf);

    SmallVector<linalg::LinalgOp> ops;
    func.walk([&](linalg::LinalgOp op) { ops.push_back(op); });

    // Reverse: visit consumers before producers so annotations propagate
    // through the entire chain in one pass.
    for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
      linalg::LinalgOp linalgOp = *it;
      if (!isElementwiseOp(linalgOp)) continue;
      Value outVal = getOutputValue(linalgOp);
      if (!outVal || !hasTileOp(outVal)) continue;

      nkipy::TileOp existingTile;
      for (Operation *user : outVal.getUsers())
        if (auto t = dyn_cast<nkipy::TileOp>(user)) {
          existingTile = t;
          break;
        }
      if (!existingTile) continue;

      IntegerAttr pdimAttr = getPartitionDimAttr(outVal);

      for (Value input : linalgOp.getDpsInputs()) {
        // In memref DPS mode, the input is an alloc. Find the linalg op
        // that writes to it (uses it as a DPS init).
        linalg::LinalgOp producerOp;
        for (OpOperand &use : input.getUses()) {
          auto candidate = dyn_cast<linalg::LinalgOp>(use.getOwner());
          if (candidate && candidate.isDpsInit(&use)) {
            producerOp = candidate;
            break;
          }
        }
        if (!producerOp || !isElementwiseOp(producerOp)) continue;

        Value producerOut = getOutputValue(producerOp);
        if (!producerOut) continue;
        if (hasTileOp(producerOut) || hasPartitionDim(producerOut)) continue;

        auto prodType = dyn_cast<MemRefType>(producerOut.getType());
        auto consType = dyn_cast<MemRefType>(outVal.getType());
        if (!prodType || !consType) continue;
        if (prodType.getShape() != consType.getShape()) continue;

        OpBuilder builder(producerOp);
        builder.setInsertionPointAfter(producerOp);
        builder.create<nkipy::TileOp>(producerOut.getLoc(), producerOut,
                                      existingTile.getLoopTileSizeAttr());

        if (pdimAttr && !hasLayoutMemSpace(producerOut)) {
          builder.create<nkipy::LayoutOp>(producerOut.getLoc(), producerOut,
              sbuf, pdimAttr, /*tile_size=*/nullptr);
        }
      }
    }
  }

  /// Step 3: emit nkipy.layout(mem_space) on unannotated values.
  void defaultLayouts(func::FuncOp func) {
    MLIRContext *ctx = func.getContext();
    auto sharedHbm = MemSpaceAttr::get(ctx, MemSpaceEnum::SharedHbm);
    auto sbuf = MemSpaceAttr::get(ctx, MemSpaceEnum::Sbuf);
    auto pdim0 = IntegerAttr::get(IntegerType::get(ctx, 32, IntegerType::Unsigned), 0);

    // Func args are always SharedHbm (hardware constraint).
    OpBuilder argBuilder(func);
    argBuilder.setInsertionPointToStart(&func.getBody().front());
    for (Value arg : func.getArguments()) {
      if (!hasLayoutMemSpace(arg)) {
        argBuilder.create<nkipy::LayoutOp>(arg.getLoc(), arg,
            sharedHbm, /*partition_dim=*/nullptr, /*tile_size=*/nullptr);
      }
    }

    func.walk([&](memref::AllocOp allocOp) {
      Value alloc = allocOp.getResult();
      if (allocOp.getType().getMemorySpace()) return;
      if (hasLayoutMemSpace(alloc)) return;

      OpBuilder builder(allocOp);
      builder.setInsertionPointAfter(allocOp);

      // A reinterpret_cast view of this alloc may carry an explicit
      // mem_space (e.g. a user knob on a reshaped boundary value). Since a
      // cast can't change memory space, that constraint belongs to the
      // alloc — honor it so the alloc and its views agree. partition_dim is
      // left unset: the space may be off-chip, and only SBUF needs it.
      if (auto viewSpace = getViewMemSpace(alloc)) {
        builder.create<nkipy::LayoutOp>(alloc.getLoc(), alloc,
            viewSpace, /*partition_dim=*/nullptr, /*tile_size=*/nullptr);
      } else if (isReturnValue(alloc)) {
        // Return values → SharedHbm; everything else → Sbuf.
        builder.create<nkipy::LayoutOp>(alloc.getLoc(), alloc,
            sharedHbm, /*partition_dim=*/nullptr, /*tile_size=*/nullptr);
      } else {
        builder.create<nkipy::LayoutOp>(alloc.getLoc(), alloc,
            sbuf, pdim0, /*tile_size=*/nullptr);
      }
    });
  }

  /// Step 4: if a return value is SBUF, insert SBUF→HBM copy.
  void materializeReturnCopies(func::FuncOp func) {
    MLIRContext *ctx = func.getContext();
    auto sharedHbm = MemSpaceAttr::get(ctx, MemSpaceEnum::SharedHbm);

    func.walk([&](func::ReturnOp returnOp) {
      OpBuilder builder(returnOp);
      for (unsigned i = 0; i < returnOp.getNumOperands(); i++) {
        Value val = returnOp.getOperand(i);
        auto memrefType = dyn_cast<MemRefType>(val.getType());
        if (!memrefType)
          continue;

        // Check if this return value is SBUF.
        bool isSbuf = false;
        for (Operation *user : val.getUsers()) {
          auto layout = dyn_cast<nkipy::LayoutOp>(user);
          if (layout && layout.getTarget() == val && layout.getMemSpace()) {
            if (layout.getMemSpace()->getValue() == MemSpaceEnum::Sbuf)
              isSbuf = true;
            break;
          }
        }
        if (!isSbuf)
          continue;

        // Insert HBM alloc + copy.
        auto hbmType = MemRefType::get(
            memrefType.getShape(), memrefType.getElementType());
        Value hbmAlloc = builder.create<memref::AllocOp>(
            val.getLoc(), hbmType);
        builder.create<nkipy::LayoutOp>(val.getLoc(), hbmAlloc,
            sharedHbm, /*partition_dim=*/nullptr, /*tile_size=*/nullptr);
        builder.create<linalg::CopyOp>(val.getLoc(), ValueRange{val},
                                        ValueRange{hbmAlloc});

        // Give the copy a tile derived from its own (elementwise) output
        // shape, not the producer's tile_op. The producer may be a reduction
        // whose tile_op is iterator-space (rank = input rank, includes the
        // contracted dim) and does not describe this copy's output shape.
        // The copy's shape always matches the returned value, so the
        // elementwise rule [min(shape[0],128), shape[1], ...] is always valid.
        SmallVector<int64_t> tile;
        tile.push_back(std::min(memrefType.getShape()[0], maxPartition()));
        for (int64_t i = 1; i < memrefType.getRank(); i++)
          tile.push_back(memrefType.getShape()[i]);
        builder.create<nkipy::TileOp>(val.getLoc(), hbmAlloc,
            DenseI64ArrayAttr::get(ctx, tile));

        returnOp.setOperand(i, hbmAlloc);
      }
    });
  }

  /// Step 5: compute sbuf_tile_size for SBUF allocs from tile_op + indexing maps.
  void computeSbufTileSizes(func::FuncOp func) {
    func.walk([&](memref::AllocOp allocOp) {
      Value alloc = allocOp.getResult();

      // Only SBUF allocs need sbuf_tile_size.
      nkipy::LayoutOp layoutOp;
      for (Operation *user : alloc.getUsers()) {
        auto l = dyn_cast<nkipy::LayoutOp>(user);
        if (l && l.getTarget() == alloc && l.getMemSpace()) {
          if (l.getMemSpace()->getValue() == MemSpaceEnum::Sbuf)
            layoutOp = l;
          break;
        }
      }
      if (!layoutOp || layoutOp.getTileSizeAttr())
        return;

      // Find tile_op on this alloc.
      nkipy::TileOp tileOp;
      for (Operation *user : alloc.getUsers())
        if (auto t = dyn_cast<nkipy::TileOp>(user)) {
          tileOp = t;
          break;
        }
      if (!tileOp)
        return;

      auto loopTile = tileOp.getLoopTileSizeAttr().asArrayRef();

      // Find the linalg op that writes to this alloc (DPS init).
      linalg::LinalgOp producer;
      for (OpOperand &use : alloc.getUses()) {
        auto candidate = dyn_cast<linalg::LinalgOp>(use.getOwner());
        if (candidate && candidate.isDpsInit(&use)) {
          producer = candidate;
          break;
        }
      }
      if (!producer)
        return;

      // Get output indexing map and derive sbuf_tile_size.
      OpOperand *initOperand = nullptr;
      for (OpOperand &operand : producer.getDpsInitsMutable()) {
        if (operand.get() == alloc) {
          initOperand = &operand;
          break;
        }
      }
      if (!initOperand)
        return;

      AffineMap outMap = producer.getMatchingIndexingMap(initOperand);
      SmallVector<int64_t> sbufTile;
      for (unsigned i = 0; i < outMap.getNumResults(); i++) {
        auto expr = outMap.getResult(i);
        if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
          unsigned pos = dimExpr.getPosition();
          if (pos < loopTile.size())
            sbufTile.push_back(loopTile[pos]);
          else
            sbufTile.push_back(1);
        } else {
          sbufTile.push_back(1);
        }
      }

      layoutOp.setTileSizeAttr(
          DenseI64ArrayAttr::get(func.getContext(), sbufTile));
    });
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    propagateTileAndLayout(func);
    defaultTileOps(func);
    defaultLayouts(func);
    materializeReturnCopies(func);
    computeSbufTileSizes(func);
    llvm::errs() << "[InferLayout] Done\n";
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createInferLayoutPass() {
  return std::make_unique<NkipyInferLayoutPass>();
}

} // namespace nkipy
} // namespace mlir
