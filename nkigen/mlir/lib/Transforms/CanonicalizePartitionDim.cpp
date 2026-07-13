//===- CanonicalizePartitionDim.cpp - Ensure partition_dim=0 everywhere ----===//
//
// This pass inserts transposes so that partition_dim=0 holds for all annotated
// tensors.  NISA hardware assumes dimension 0 is always the partition
// dimension, and every downstream pass relies on this.
//
// Algorithm:
//   1. Collect all nkipy.annotate ops with partition_dim != 0.
//   2. For each such annotation, BFS through the connected elementwise
//      component to find all values that share the same non-zero partition_dim.
//   3. At component boundaries (inputs from non-elementwise producers, outputs
//      to non-elementwise consumers), insert linalg.transpose to move
//      partition_dim to position 0.
//   4. Rewrite all elementwise ops inside the component with permuted shapes.
//   5. Update all nkipy.annotate ops: partition_dim -> 0, permute tile_size.
//
// The pass runs BEFORE assign-linalg-op-ids so that new transpose ops get IDs
// (needed for knob-driven tiling). It runs AFTER infer-layout so that all
// tensors in the chain already have partition_dim annotations.
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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace nkipy;

namespace mlir {
namespace nkipy {

namespace {

//===----------------------------------------------------------------------===//
// Memref helpers
//===----------------------------------------------------------------------===//

/// Linalg ops on memref write in place (zero results). The value that carries
/// the op's "output" is its DPS init operand (the buffer it writes).
static Value getLinalgOutputValue(linalg::LinalgOp op) {
  SmallVector<Value> inits(op.getDpsInits());
  return inits.empty() ? Value() : inits[0];
}

/// Find the linalg op that writes `val`: the user that has `val` as a DPS init
/// operand. (For memref IR the defining op of `val` is a memref.alloc, not the
/// linalg op that produces the data.)
static linalg::LinalgOp findProducerLinalgOp(Value val) {
  for (Operation *user : val.getUsers()) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(user);
    if (!linalgOp)
      continue;
    for (Value init : linalgOp.getDpsInits())
      if (init == val)
        return linalgOp;
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Build the permutation that moves dimension `partDim` to position 0
/// and shifts the others right.  E.g. for rank=4, partDim=2:
///   [2, 0, 1, 3]
static SmallVector<int64_t> buildPermutation(int64_t rank, int64_t partDim) {
  SmallVector<int64_t> perm;
  perm.push_back(partDim);
  for (int64_t i = 0; i < rank; ++i) {
    if (i != partDim)
      perm.push_back(i);
  }
  return perm;
}

/// Build the inverse permutation.  E.g. for perm=[2,0,1,3]:
///   inv=[1,2,0,3]
static SmallVector<int64_t> invertPermutation(ArrayRef<int64_t> perm) {
  SmallVector<int64_t> inv(perm.size());
  for (size_t i = 0; i < perm.size(); ++i)
    inv[perm[i]] = i;
  return inv;
}

/// Apply a permutation to a vector.
template <typename T>
static SmallVector<T> permuteVector(ArrayRef<T> vec, ArrayRef<int64_t> perm) {
  SmallVector<T> result;
  for (int64_t p : perm)
    result.push_back(vec[p]);
  return result;
}

/// For >2D transposes: equalize tile sizes of swapped dim pairs so that
/// after tiling, the tiled transpose has ≤2 non-unit dims (required by
/// linalg-to-nisa, since NISA ops are 2D).
/// E.g. perm=[1,0,2], tile=[4,128,128] → tile=[4,4,128]
///
/// Exception: when one of the swapped dims already has tile=1, the
/// transpose along that pair is trivial (a reshape), so equalization
/// would be harmful — it would shrink the non-unit dim to 1 and create
/// tile size mismatches with downstream consumers.
static void equalizeSwappedTileDims(SmallVector<int64_t> &tile,
                                    ArrayRef<int64_t> perm) {
  for (int64_t i = 0; i < static_cast<int64_t>(tile.size()); ++i) {
    int64_t j = perm[i];
    if (j != i && j < static_cast<int64_t>(tile.size())) {
      if (tile[i] == 1 || tile[j] == 1)
        continue;
      int64_t minTile = std::min(tile[i], tile[j]);
      tile[i] = minTile;
      tile[j] = minTile;
    }
  }
}

/// Permute a reduced-rank tile_size (from a reduction op) through a
/// full-rank permutation. Expands to full rank using 1s for size-1 dims,
/// permutes, then strips back to parallel dims only.
static DenseI64ArrayAttr permuteReducedTileSize(
    ArrayRef<int64_t> oldTileSize, ArrayRef<int64_t> perm,
    ArrayRef<int64_t> invPerm, ArrayRef<int64_t> permutedShape,
    int64_t rank, MLIRContext *ctx) {
  // Recover original shape to find parallel dims.
  SmallVector<int64_t> origShape = permuteVector<int64_t>(permutedShape, invPerm);
  SmallVector<int64_t> origParDims;
  for (int64_t i = 0; i < rank; ++i) {
    if (origShape[i] > 1)
      origParDims.push_back(i);
  }

  if (static_cast<int64_t>(oldTileSize.size()) !=
      static_cast<int64_t>(origParDims.size()))
    return {};

  // Build full-rank tile with 1s for size-1 dims, then permute.
  SmallVector<int64_t> fullTile(rank, 1);
  for (size_t i = 0; i < origParDims.size(); ++i)
    fullTile[origParDims[i]] = oldTileSize[i];
  SmallVector<int64_t> permFull = permuteVector<int64_t>(fullTile, perm);

  // Strip back to only parallel dims in the permuted shape.
  SmallVector<int64_t> newTileSize;
  for (int64_t i = 0; i < rank; ++i) {
    if (permutedShape[i] > 1)
      newTileSize.push_back(permFull[i]);
  }
  return DenseI64ArrayAttr::get(ctx, newTileSize);
}

/// Emit nkipy.layout + nkipy.tile_op for a boundary transpose result.
/// Applies >2D tile equalization, then creates the annotations with
/// partition_dim=0 and the given mem_space/tile_size.
static void annotateBoundaryTranspose(OpBuilder &builder, Location loc,
                                      Value transposed,
                                      MemSpaceAttr memSpace,
                                      DenseI64ArrayAttr tileSize,
                                      ArrayRef<int64_t> perm, int64_t rank) {
  DenseI64ArrayAttr finalTileSize;
  if (tileSize) {
    SmallVector<int64_t> tileSizeVec(tileSize.asArrayRef());
    if (rank > 2)
      equalizeSwappedTileDims(tileSizeVec, perm);
    finalTileSize = DenseI64ArrayAttr::get(builder.getContext(), tileSizeVec);
  }
  auto zeroPdAttr = builder.getIntegerAttr(
      builder.getIntegerType(32, /*isSigned=*/false), 0);
  builder.create<nkipy::LayoutOp>(
      loc, transposed, memSpace, zeroPdAttr, finalTileSize);
  if (finalTileSize)
    builder.create<nkipy::TileOp>(loc, transposed, finalTileSize);
}

/// Wrappers that accept Operation* for use in BFS traversal where we
/// iterate over generic Operations rather than typed LinalgOps.
static bool isElementwiseOp(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  return linalgOp && ::mlir::nkipy::isElementwiseOp(linalgOp);
}

static bool isReductionGeneric(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  return linalgOp && ::mlir::nkipy::isReductionGeneric(linalgOp);
}

static bool isMatmulOp(Operation *op) {
  return ::mlir::nkipy::isMatmulOp(op->getName().getStringRef());
}

/// Get the partition_dim for a value from its nkipy.layout op, if any.
/// Returns -1 if no annotation or no partition_dim.
static int64_t getPartitionDim(Value val,
                               DenseMap<Value, int64_t> &partDimMap) {
  auto it = partDimMap.find(val);
  if (it != partDimMap.end())
    return it->second;
  return -1;
}

/// Find the nkipy.tile_op (if any) attached to `target`.
static nkipy::TileOp findTileOp(Value target) {
  for (Operation *user : target.getUsers())
    if (auto t = dyn_cast<nkipy::TileOp>(user))
      return t;
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct NkipyCanonicalizePartitionDimPass
    : public CanonicalizePartitionDimBase<NkipyCanonicalizePartitionDimPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<nkipy::NkipyDialect>();
  }

  /// SBUF partition limit for the configured target.  Falls back to trn2 if
  /// the option is unset or the target is unknown.
  int64_t maxPartitionDim() const {
    StringRef t = target.empty() ? StringRef("trn2") : StringRef(target);
    if (auto n = nkipy::getSbufNumPartitions(t))
      return *n;
    if (auto n = nkipy::getSbufNumPartitions("trn2"))
      return *n;
    return 128;
  }

  /// Collect annotations with partition_dim info from the function.
  void collectAnnotations(
      func::FuncOp func,
      DenseMap<Value, int64_t> &partDimMap,
      DenseMap<Value, nkipy::LayoutOp> &valueAnnotateMap,
      SmallVector<nkipy::LayoutOp> &nonZeroAnnotations) {
    func.walk([&](nkipy::LayoutOp layoutOp) {
      valueAnnotateMap[layoutOp.getTarget()] = layoutOp;
      auto partDimAttr = layoutOp.getPartitionDimAttr();
      if (!partDimAttr)
        return;
      uint32_t partDim = partDimAttr.getUInt();
      partDimMap[layoutOp.getTarget()] = partDim;
      if (partDim != 0)
        nonZeroAnnotations.push_back(layoutOp);
    });
  }

  /// BFS from seedOp to find connected elementwise/reduction component.
  llvm::SetVector<Operation *> findComponent(Operation *seedOp) {
    llvm::SetVector<Operation *> componentOps;

    auto canInclude = [](Operation *op) {
      return isElementwiseOp(op) || isReductionGeneric(op);
    };

    if (!seedOp || !canInclude(seedOp))
      return componentOps;

    SmallVector<Operation *> bfsQueue;
    bfsQueue.push_back(seedOp);
    componentOps.insert(seedOp);

    while (!bfsQueue.empty()) {
      Operation *op = bfsQueue.pop_back_val();

      // Backward through DPS inputs: the producer of a memref input is the
      // linalg op that writes it (its DPS init), not its defining op.
      if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
        for (Value input : linalgOp.getDpsInputs()) {
          linalg::LinalgOp producer = findProducerLinalgOp(input);
          if (producer && canInclude(producer) &&
              !componentOps.count(producer)) {
            componentOps.insert(producer);
            bfsQueue.push_back(producer);
          }
        }

        // Forward through users of this op's output buffer.
        Value outVal = getLinalgOutputValue(linalgOp);
        if (outVal) {
          for (Operation *user : outVal.getUsers()) {
            if (isa<nkipy::LayoutOp>(user) || isa<nkipy::TileOp>(user))
              continue;
            // A user that reads outVal as an input is a forward consumer.
            if (canInclude(user) && !componentOps.count(user)) {
              componentOps.insert(user);
              bfsQueue.push_back(user);
            }
          }
        }
      }
    }

    // Pull in any linalg.fill that initializes a component op's accumulator
    // buffer (e.g. a reduction's zero-fill). On memref the fill writes the same
    // buffer the reduction reads as its DPS init, so it must be permuted and
    // kept inside the component rather than treated as an external consumer.
    SmallVector<Operation *> fills;
    for (Operation *op : componentOps) {
      auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
      if (!linalgOp)
        continue;
      for (Value init : linalgOp.getDpsInits()) {
        for (Operation *user : init.getUsers()) {
          auto fillOp = dyn_cast<linalg::FillOp>(user);
          if (fillOp && !componentOps.count(user))
            fills.push_back(user);
        }
      }
    }
    for (Operation *fill : fills)
      componentOps.insert(fill);

    return componentOps;
  }

  /// Find boundary inputs: DPS input buffers read by component ops but written
  /// outside the component (e.g. func args, or non-component producers). The
  /// op's own DPS init (its output alloc) is never a boundary input.
  llvm::SetVector<Value> findBoundaryInputs(
      const llvm::SetVector<Operation *> &componentOps) {
    llvm::SetVector<Value> boundaryInputs;
    for (Operation *op : componentOps) {
      auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
      if (!linalgOp)
        continue;
      for (Value input : linalgOp.getDpsInputs()) {
        linalg::LinalgOp producer = findProducerLinalgOp(input);
        if (!producer || !componentOps.count(producer.getOperation()))
          boundaryInputs.insert(input);
      }
    }
    return boundaryInputs;
  }

  /// Find boundary outputs: output buffers (DPS inits) of component ops that are
  /// read outside the component (returned, copied, or read by a non-component
  /// op).
  llvm::SetVector<Value> findBoundaryOutputs(
      const llvm::SetVector<Operation *> &componentOps) {
    llvm::SetVector<Value> boundaryOutputs;
    for (Operation *op : componentOps) {
      auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
      if (!linalgOp)
        continue;
      Value outVal = getLinalgOutputValue(linalgOp);
      if (!outVal)
        continue;
      for (Operation *user : outVal.getUsers()) {
        if (isa<nkipy::LayoutOp>(user) || isa<nkipy::TileOp>(user))
          continue;
        if (!componentOps.count(user)) {
          boundaryOutputs.insert(outVal);
          break;
        }
      }
    }
    return boundaryOutputs;
  }

  /// Insert input boundary transposes (original -> permuted).
  void insertInputTransposes(
      OpBuilder &builder, const llvm::SetVector<Value> &boundaryInputs,
      ArrayRef<int64_t> perm, int64_t rank,
      DenseI64ArrayAttr seedTileSizeAttr,
      IRMapping &valueMapping) {
    for (Value input : boundaryInputs) {
      auto inputType = dyn_cast<MemRefType>(input.getType());
      if (!inputType || inputType.getRank() != rank)
        continue;

      SmallVector<int64_t> newShape =
          permuteVector<int64_t>(inputType.getShape(), perm);

      // Insert the transpose after the source buffer is fully populated.
      // A boundary input may be written through subviews inside loop nests
      // (e.g. batch_matmul decomposition), so the completion point can be an
      // enclosing scf.for rather than a direct linalg producer.
      if (Operation *writer = findWriteCompletionOp(input))
        builder.setInsertionPointAfter(writer);
      else if (input.getDefiningOp())
        builder.setInsertionPointAfter(input.getDefiningOp());
      else
        builder.setInsertionPointToStart(input.getParentBlock());

      Location loc = input.getLoc();
      Value init = builder.create<memref::AllocOp>(
          loc, MemRefType::get(newShape, inputType.getElementType()));
      builder.create<linalg::TransposeOp>(loc, input, init, perm);
      Value transposed = init;
      valueMapping.map(input, transposed);

      DenseI64ArrayAttr transposeTileSize;
      if (seedTileSizeAttr) {
        SmallVector<int64_t> permutedTile =
            permuteVector<int64_t>(seedTileSizeAttr.asArrayRef(), perm);
        transposeTileSize =
            DenseI64ArrayAttr::get(builder.getContext(), permutedTile);
      }
      auto sbufAttr = nkipy::MemSpaceAttr::get(
          builder.getContext(), nkipy::MemSpaceEnum::Sbuf);
      annotateBoundaryTranspose(builder, loc, transposed,
                                sbufAttr, transposeTileSize, perm, rank);
    }
  }

  /// Rewrite component ops with permuted shapes.
  void rewriteComponentOps(
      OpBuilder &builder, func::FuncOp func,
      const llvm::SetVector<Operation *> &componentOps,
      ArrayRef<int64_t> perm, int64_t rank,
      IRMapping &valueMapping) {
    // Process in topological order.
    SmallVector<Operation *> topoOrder;
    func.walk([&](Operation *op) {
      if (componentOps.count(op))
        topoOrder.push_back(op);
    });

    // Retype each distinct DPS init buffer in place to the permuted shape,
    // exactly once. On memref the init is the op's output value, so retyping
    // the alloc carries the permuted shape to every user (annotations,
    // consumers). A buffer shared as init by two component ops (e.g. a
    // reduction accumulator and its zero-fill) must only be permuted once.
    llvm::SetVector<Value> initBuffers;
    for (Operation *op : topoOrder) {
      if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op))
        for (Value init : linalgOp.getDpsInits())
          initBuffers.insert(init);
    }
    for (Value initOperand : initBuffers) {
      auto memrefType = dyn_cast<MemRefType>(initOperand.getType());
      if (!memrefType || memrefType.getRank() != rank)
        continue;
      SmallVector<int64_t> newShape =
          permuteVector<int64_t>(memrefType.getShape(), perm);
      auto newType = MemRefType::get(
          newShape, memrefType.getElementType(),
          memrefType.getLayout(), memrefType.getMemorySpace());
      initOperand.setType(newType);
    }

    for (Operation *op : topoOrder) {
      auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
      if (!linalgOp)
        continue;

      // Replace operands with mapped (transposed) values.
      for (unsigned i = 0; i < op->getNumOperands(); ++i) {
        if (Value mapped = valueMapping.lookupOrNull(op->getOperand(i)))
          op->setOperand(i, mapped);
      }

      // Permute indexing maps of linalg.generic ops.
      // Named ops (add, mul, exp, etc.) have implicit identity maps that
      // remain valid after consistent shape permutation.  But generic ops
      // may have non-identity maps (e.g. broadcast: (d0,d1,d2)->(0,d1,d2))
      // that must be permuted to match the new shape layout.
      if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
        MLIRContext *ctx = genericOp.getContext();
        SmallVector<int64_t> invPerm = invertPermutation(perm);

        // Build dimension remapping: d_j -> d_{invPerm[j]}.
        SmallVector<AffineExpr> dimReplacements;
        for (int64_t j = 0; j < rank; ++j)
          dimReplacements.push_back(getAffineDimExpr(invPerm[j], ctx));

        SmallVector<AffineMap> newMaps;
        for (AffineMap map : genericOp.getIndexingMapsArray()) {
          unsigned numResults = map.getNumResults();

          SmallVector<AffineExpr> exprs;
          if (static_cast<int64_t>(numResults) == rank) {
            // Full-rank map: reorder result positions by perm, then
            // remap dimension references.
            for (int64_t i = 0; i < rank; ++i)
              exprs.push_back(map.getResult(perm[i]));
          } else {
            // Reduced-rank map (e.g. reduction output): keep result
            // order, only remap dimension references.
            for (unsigned i = 0; i < numResults; ++i)
              exprs.push_back(map.getResult(i));
          }

          SmallVector<AffineExpr> finalExprs;
          for (AffineExpr expr : exprs)
            finalExprs.push_back(
                expr.replaceDimsAndSymbols(dimReplacements, {}));

          newMaps.push_back(
              AffineMap::get(map.getNumDims(), 0, finalExprs, ctx));
        }
        genericOp.setIndexingMapsAttr(
            builder.getAffineMapArrayAttr(newMaps));
      }
    }
  }

  /// Insert output boundary transposes (permuted -> original) and rewire uses.
  void insertOutputTransposes(
      OpBuilder &builder,
      const llvm::SetVector<Value> &boundaryOutputs,
      const llvm::SetVector<Operation *> &componentOps,
      ArrayRef<int64_t> invPerm, int64_t rank,
      DenseI64ArrayAttr seedTileSizeAttr,
      DenseMap<Value, nkipy::LayoutOp> &valueAnnotateMap,
      nkipy::LayoutOp annotateOp, int64_t partDim) {
    for (Value output : boundaryOutputs) {
      auto outputType = dyn_cast<MemRefType>(output.getType());
      if (!outputType)
        continue;

      SmallVector<int64_t> origShape =
          permuteVector<int64_t>(outputType.getShape(), invPerm);

      // Insert the inverse transpose right after the component op that writes
      // `output`, so the permuted buffer is populated before we read it.
      if (linalg::LinalgOp producer = findProducerLinalgOp(output))
        builder.setInsertionPointAfter(producer.getOperation());
      else
        builder.setInsertionPointAfterValue(output);
      Location loc = output.getLoc();
      Value init = builder.create<memref::AllocOp>(
          loc, MemRefType::get(origShape, outputType.getElementType()));
      auto transposeOp =
          builder.create<linalg::TransposeOp>(loc, output, init, invPerm);
      Value transposedBack = init;

      // Derive tile_size and mem_space for the output annotation.
      DenseI64ArrayAttr outputTileSize;
      MemSpaceAttr outputMemSpace;
      auto annIt = valueAnnotateMap.find(output);
      if (annIt != valueAnnotateMap.end()) {
        outputMemSpace = annIt->second.getMemSpaceAttr();
        outputTileSize = annIt->second.getTileSizeAttr();
      }
      if (!outputTileSize && seedTileSizeAttr)
        outputTileSize = seedTileSizeAttr;

      // Expand reduced-rank tile_size to full rank.
      if (outputTileSize &&
          static_cast<int64_t>(outputTileSize.size()) < rank) {
        SmallVector<int64_t> expanded;
        ArrayRef<int64_t> reduced = outputTileSize.asArrayRef();
        size_t ri = 0;
        for (int64_t i = 0; i < rank; ++i) {
          if (origShape[i] > 1 && ri < reduced.size())
            expanded.push_back(reduced[ri++]);
          else
            expanded.push_back(1);
        }
        outputTileSize =
            DenseI64ArrayAttr::get(builder.getContext(), expanded);
      }

      // Clamp the tile to this transpose's own output shape. The seed tile
      // may come from a producer with a different iteration space (e.g. a
      // reduction's [.., K] where the reduced dim is 1 here), so a per-dim
      // clamp keeps the tile valid for the transpose's parallel output.
      // Dim 0 is additionally capped at maxPartitionDim().
      if (outputTileSize) {
        SmallVector<int64_t> tileSizeVec(outputTileSize.asArrayRef());
        for (int64_t i = 0; i < rank; ++i)
          tileSizeVec[i] = std::min(tileSizeVec[i], origShape[i]);
        if (tileSizeVec[0] > maxPartitionDim()) {
          annotateOp.emitWarning("partition_dim=")
              << partDim << ": clamping boundary transpose tile_size[0] "
              << "from " << tileSizeVec[0] << " to " << maxPartitionDim();
          tileSizeVec[0] = maxPartitionDim();
        }
        outputTileSize =
            DenseI64ArrayAttr::get(builder.getContext(), tileSizeVec);
      }

      annotateBoundaryTranspose(builder, loc, transposedBack,
                                outputMemSpace, outputTileSize,
                                invPerm, rank);

      // Rewire non-component uses to the inverse-transposed value.
      SmallVector<OpOperand *> usesToReplace;
      for (OpOperand &use : output.getUses()) {
        Operation *user = use.getOwner();
        if (user == transposeOp ||
            isa<nkipy::LayoutOp>(user) || isa<nkipy::TileOp>(user))
          continue;
        if (!componentOps.count(user))
          usesToReplace.push_back(&use);
      }
      for (OpOperand *use : usesToReplace)
        use->set(transposedBack);
    }
  }

  /// Update nkipy.layout and nkipy.tile_op ops for values in the component:
  /// set partition_dim=0 and permute tile sizes.
  void updateComponentAnnotations(
      OpBuilder &builder, func::FuncOp func,
      const llvm::SetVector<Operation *> &componentOps,
      ArrayRef<int64_t> perm, ArrayRef<int64_t> invPerm, int64_t rank) {
    auto permuteTileAttr = [&](DenseI64ArrayAttr tileSizeAttr,
                               Value target) -> DenseI64ArrayAttr {
      ArrayRef<int64_t> oldTileSize = tileSizeAttr.asArrayRef();
      if (static_cast<int64_t>(oldTileSize.size()) == rank) {
        SmallVector<int64_t> newTileSize =
            permuteVector<int64_t>(oldTileSize, perm);
        return DenseI64ArrayAttr::get(builder.getContext(), newTileSize);
      }
      auto annTargetType = dyn_cast<ShapedType>(target.getType());
      if (!annTargetType)
        return {};
      return permuteReducedTileSize(oldTileSize, perm, invPerm,
                                    annTargetType.getShape(), rank,
                                    builder.getContext());
    };

    // An annotation belongs to the component if the linalg op that writes its
    // target buffer is in the component.
    auto inComponent = [&](Value target) {
      linalg::LinalgOp producer = findProducerLinalgOp(target);
      return producer && componentOps.count(producer.getOperation());
    };

    func.walk([&](nkipy::LayoutOp annOp) {
      if (!inComponent(annOp.getTarget()))
        return;

      annOp.setPartitionDimAttr(builder.getIntegerAttr(
          builder.getIntegerType(32, /*isSigned=*/false), 0));

      if (auto tileSizeAttr = annOp.getTileSizeAttr()) {
        if (auto newTs = permuteTileAttr(tileSizeAttr, annOp.getTarget()))
          annOp.setTileSizeAttr(newTs);
      }
    });

    func.walk([&](nkipy::TileOp tileOp) {
      Value tgt = tileOp.getTarget();
      if (!inComponent(tgt))
        return;
      if (auto tileSizeAttr = tileOp.getLoopTileSizeAttr()) {
        if (auto newTs = permuteTileAttr(tileSizeAttr, tgt))
          tileOp.setLoopTileSizeAttr(newTs);
      }
    });
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Body-less custom-op declarations have no body to canonicalize; their
    // NISA bodies are inlined later by resolve-custom-ops. Bail before any
    // walk that would deref a nonexistent block terminator.
    if (func.isDeclaration())
      return;

    llvm::errs() << "[CanonicalizePartitionDim] Processing function: "
                 << func.getName() << "\n";

    // Phase 1: Collect partition_dim annotations.
    DenseMap<Value, int64_t> partDimMap;
    DenseMap<Value, nkipy::LayoutOp> valueAnnotateMap;
    SmallVector<nkipy::LayoutOp> nonZeroAnnotations;
    collectAnnotations(func, partDimMap, valueAnnotateMap, nonZeroAnnotations);

    if (nonZeroAnnotations.empty()) {
      llvm::errs() << "[CanonicalizePartitionDim] No non-zero partition_dim "
                      "annotations found\n";
      return;
    }

    // Phase 2: Process each non-zero partition_dim component.
    OpBuilder builder(func.getContext());
    int numTransposed = 0;
    DenseSet<Operation *> processedOps;

    for (nkipy::LayoutOp annotateOp : nonZeroAnnotations) {
      Value target = annotateOp.getTarget();
      // On memref IR the linalg op that writes the target buffer is the seed,
      // not the buffer's defining op (a memref.alloc).
      linalg::LinalgOp producer = findProducerLinalgOp(target);
      Operation *seedOp = producer ? producer.getOperation() : nullptr;

      if (seedOp && processedOps.count(seedOp))
        continue;

      auto shapedType = dyn_cast<ShapedType>(target.getType());
      if (!shapedType) {
        annotateOp.emitError("partition_dim != 0 on non-shaped type");
        return signalPassFailure();
      }

      int64_t partDim = partDimMap[target];
      int64_t rank = shapedType.getRank();
      if (partDim >= rank) {
        annotateOp.emitError("partition_dim ")
            << partDim << " >= rank " << rank;
        return signalPassFailure();
      }

      SmallVector<int64_t> perm = buildPermutation(rank, partDim);
      SmallVector<int64_t> invPerm = invertPermutation(perm);
      auto seedTileSizeAttr = annotateOp.getTileSizeAttr();
      if (!seedTileSizeAttr) {
        if (auto tileOp = findTileOp(target))
          seedTileSizeAttr = tileOp.getLoopTileSizeAttr();
      }

      // Validate partition tile size fits hardware.
      if (seedTileSizeAttr) {
        ArrayRef<int64_t> tileVals = seedTileSizeAttr.asArrayRef();
        if (static_cast<int64_t>(tileVals.size()) > partDim &&
            tileVals[partDim] > maxPartitionDim()) {
          annotateOp.emitError("tile_size[")
              << partDim << "] = " << tileVals[partDim]
              << " exceeds hardware partition limit " << maxPartitionDim();
          return signalPassFailure();
        }
      }

      // No linalg op writes this buffer (e.g. a directly-annotated func arg):
      // partition_dim is informational only, nothing to transpose.
      if (!seedOp)
        continue;

      // Error on matmul with partition_dim != 0.
      if (seedOp && isMatmulOp(seedOp)) {
        annotateOp.emitError(
            "partition_dim != 0 on matmul/bmm is not supported. "
            "Please annotate downstream elementwise ops instead.");
        return signalPassFailure();
      }

      // Step 1: Find connected component.
      auto componentOps = findComponent(seedOp);
      for (Operation *op : componentOps)
        processedOps.insert(op);

      if (componentOps.empty()) {
        annotateOp.emitError(
            "partition_dim != 0 on an unsupported op. "
            "Supported: elementwise and reduction ops.");
        return signalPassFailure();
      }

      // Step 2: Find boundaries.
      auto boundaryInputs = findBoundaryInputs(componentOps);
      auto boundaryOutputs = findBoundaryOutputs(componentOps);

      // Step 3: Insert input boundary transposes.
      IRMapping valueMapping;
      insertInputTransposes(builder, boundaryInputs, perm, rank,
                            seedTileSizeAttr, valueMapping);

      // Step 4: Rewrite component ops with permuted shapes.
      rewriteComponentOps(builder, func, componentOps, perm, rank,
                          valueMapping);

      // Step 5: Insert output boundary transposes.
      insertOutputTransposes(builder, boundaryOutputs, componentOps,
                             invPerm, rank, seedTileSizeAttr,
                             valueAnnotateMap, annotateOp, partDim);

      // Step 6: Update annotations.
      updateComponentAnnotations(builder, func, componentOps,
                                 perm, invPerm, rank);

      numTransposed += componentOps.size();
      llvm::errs() << "[CanonicalizePartitionDim] Processed component of "
                   << componentOps.size() << " ops with partition_dim="
                   << partDim << "\n";
    }

    llvm::errs() << "[CanonicalizePartitionDim] Rewritten " << numTransposed
                 << " op(s) total\n";
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createCanonicalizePartitionDimPass() {
  return std::make_unique<NkipyCanonicalizePartitionDimPass>();
}

} // namespace nkipy
} // namespace mlir
