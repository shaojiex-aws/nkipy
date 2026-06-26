#include "nkipy/Dialect/NkipyOps.h"
#include "nkipy/Dialect/NkipyDialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/TilingInterface.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// FuseOp
//===----------------------------------------------------------------------===//

LogicalResult nkipy::FuseOp::verify() {
  if (getTargets().size() < 2)
    return emitOpError("requires at least 2 target tensors to fuse");
  return success();
}

//===----------------------------------------------------------------------===//
// GatherOp — DestinationStyleOpInterface
//===----------------------------------------------------------------------===//

MutableOperandRange nkipy::GatherOp::getDpsInitsMutable() {
  return getOutputMutable();
}

//===----------------------------------------------------------------------===//
// GatherOp — TilingInterface
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> nkipy::GatherOp::getLoopIteratorTypes() {
  auto resultType = cast<ShapedType>(getResult().getType());
  return SmallVector<utils::IteratorType>(
      resultType.getRank(), utils::IteratorType::parallel);
}

SmallVector<Range> nkipy::GatherOp::getIterationDomain(OpBuilder &b) {
  auto resultType = cast<ShapedType>(getResult().getType());
  SmallVector<Range> domain;
  for (int64_t i = 0; i < resultType.getRank(); ++i) {
    domain.push_back(Range{b.getIndexAttr(0),
                           b.getIndexAttr(resultType.getDimSize(i)),
                           b.getIndexAttr(1)});
  }
  return domain;
}

FailureOr<TilingResult>
nkipy::GatherOp::getTiledImplementation(
    OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  Location loc = getLoc();

  auto sourceType = cast<ShapedType>(getSource().getType());
  auto resultType = cast<ShapedType>(getResult().getType());
  int64_t rank = resultType.getRank();

  // --- Slice indices: indices[i_off : i_off + tN] ---
  SmallVector<OpFoldResult> idxOffsets = {offsets[0]};
  SmallVector<OpFoldResult> idxSizes = {sizes[0]};
  SmallVector<OpFoldResult> idxStrides = {b.getIndexAttr(1)};
  Value indicesTile = b.create<tensor::ExtractSliceOp>(
      loc, getIndices(), idxOffsets, idxSizes, idxStrides);

  // --- Slice source: source[0:V, j_off : j_off+tH] ---
  SmallVector<OpFoldResult> srcOffsets(rank, b.getIndexAttr(0));
  SmallVector<OpFoldResult> srcSizes;
  SmallVector<OpFoldResult> srcStrides(rank, b.getIndexAttr(1));
  srcSizes.push_back(b.getIndexAttr(sourceType.getDimSize(0)));
  for (int64_t d = 1; d < rank; ++d) {
    srcOffsets[d] = offsets[d];
    srcSizes.push_back(sizes[d]);
  }
  Value sourceTile = b.create<tensor::ExtractSliceOp>(
      loc, getSource(), srcOffsets, srcSizes, srcStrides);

  // --- Slice output (DPS init): output[i_off:, j_off:] ---
  SmallVector<OpFoldResult> outOffsets(offsets.begin(), offsets.end());
  SmallVector<OpFoldResult> outSizes(sizes.begin(), sizes.end());
  SmallVector<OpFoldResult> outStrides(rank, b.getIndexAttr(1));
  Value outputTile = b.create<tensor::ExtractSliceOp>(
      loc, getOutput(), outOffsets, outSizes, outStrides);

  // --- Build tiled result type ---
  SmallVector<int64_t> tiledShape;
  for (auto s : sizes) {
    if (auto attr = getConstantIntValue(s))
      tiledShape.push_back(*attr);
    else
      tiledShape.push_back(ShapedType::kDynamic);
  }
  auto tiledResultType = RankedTensorType::get(
      tiledShape, resultType.getElementType());

  // --- Create tiled gather ---
  auto tiledGather = b.create<nkipy::GatherOp>(
      loc, tiledResultType, sourceTile, indicesTile, outputTile);

  // --- Clone reference_impl into the tiled gather ---
  Region &origRegion = getReferenceImpl();
  if (!origRegion.empty()) {
    Region &newRegion = tiledGather.getReferenceImpl();
    Block &origBlock = origRegion.front();

    Block *newBlock = new Block();
    newRegion.push_back(newBlock);
    newBlock->addArgument(sourceTile.getType(), loc);
    newBlock->addArgument(indicesTile.getType(), loc);

    IRMapping mapping;
    mapping.map(origBlock.getArgument(0), newBlock->getArgument(0));
    mapping.map(origBlock.getArgument(1), newBlock->getArgument(1));

    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(newBlock);

    for (Operation &op : origBlock) {
      if (isa<tensor::EmptyOp>(&op)) {
        auto newEmpty = b.create<tensor::EmptyOp>(
            loc, tiledResultType.getShape(),
            tiledResultType.getElementType());
        mapping.map(op.getResult(0), newEmpty.getResult());
      } else {
        Operation *cloned = b.clone(op, mapping);
        if (auto linalgOp = dyn_cast<linalg::LinalgOp>(cloned)) {
          auto inits = linalgOp.getDpsInits();
          for (unsigned i = 0; i < cloned->getNumResults(); ++i) {
            if (i < inits.size())
              cloned->getResult(i).setType(inits[i].getType());
          }
        }
      }
    }
  }

  return TilingResult{{tiledGather.getOperation()},
                      {tiledGather.getResult()},
                      {}};
}

LogicalResult nkipy::GatherOp::getResultTilePosition(
    OpBuilder &b, unsigned resultNumber,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  if (resultNumber != 0)
    return failure();
  resultOffsets.assign(offsets.begin(), offsets.end());
  resultSizes.assign(sizes.begin(), sizes.end());
  return success();
}

#define GET_OP_CLASSES
#include "nkipy/Dialect/NkipyOps.cpp.inc"
