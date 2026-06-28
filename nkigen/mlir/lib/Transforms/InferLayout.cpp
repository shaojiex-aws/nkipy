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

/// Check if a value is the output of a matmul-like op.
static bool isMatmulOutput(Value val) {
  for (OpOperand &use : val.getUses()) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(use.getOwner());
    if (!linalgOp) continue;
    if (isMatmulOp(linalgOp) && linalgOp.isDpsInit(&use))
      return true;
  }
  return false;
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

  /// Step 1: emit default tile_op for ops that lack one.
  void defaultTileOps(func::FuncOp func) {
    int64_t partCap = maxPartition();
    int64_t freeCap = matmulFreeCap();

    func.walk([&](linalg::LinalgOp linalgOp) {
      if (!isAnnotatableOp(linalgOp)) return;

      Value outVal = getOutputValue(linalgOp);
      if (!outVal || hasTileOp(outVal)) return;

      SmallVector<int64_t> tile;

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
        // Transpose: tile identity dims to 1, keep swapped dims full.
        auto outType = cast<MemRefType>(outVal.getType());
        auto shape = outType.getShape();
        auto transposeOp = cast<linalg::TransposeOp>(linalgOp.getOperation());
        auto perm = transposeOp.getPermutation();
        for (int64_t i = 0; i < outType.getRank(); i++) {
          if (perm[i] == i)
            tile.push_back(1);  // identity dim → tile to 1
          else
            tile.push_back(shape[i]);  // swapped dim → full
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

      // Return values and matmul outputs → SharedHbm
      if (isReturnValue(alloc) || isMatmulOutput(alloc)) {
        builder.create<nkipy::LayoutOp>(alloc.getLoc(), alloc,
            sharedHbm, /*partition_dim=*/nullptr, /*tile_size=*/nullptr);
      } else {
        builder.create<nkipy::LayoutOp>(alloc.getLoc(), alloc,
            sbuf, pdim0, /*tile_size=*/nullptr);
      }
    });
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    propagateTileAndLayout(func);
    defaultTileOps(func);
    defaultLayouts(func);
    llvm::errs() << "[InferLayout] Done\n";
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createInferLayoutPass() {
  return std::make_unique<NkipyInferLayoutPass>();
}

} // namespace nkipy
} // namespace mlir
