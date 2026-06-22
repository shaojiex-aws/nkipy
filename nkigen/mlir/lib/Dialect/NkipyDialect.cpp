#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/StringExtras.h"

#include "llvm/ADT/TypeSwitch.h"

#include "nkipy/Dialect/NkipyDialect.h"
#include "nkipy/Dialect/NkipyAttrs.h"
#include "nkipy/Dialect/NkipyOps.h"

using namespace mlir;
using namespace mlir::nkipy;

#include "nkipy/Dialect/NkipyDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "nkipy/Dialect/NkipyAttrs.cpp.inc"

#include "nkipy/Dialect/NkipyEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// SbufMapAttr implementation
//===----------------------------------------------------------------------===//

SmallVector<int64_t> SbufMapAttr::getPhysicalShape() const {
  auto tile = getTileSize();
  auto blocks = getNumBlocks();
  SmallVector<int64_t> shape;
  shape.push_back(tile[0]);
  for (int64_t b : blocks)
    shape.push_back(b);
  shape.push_back(tile[tile.size() - 1]);
  return shape;
}

AffineMap SbufMapAttr::getAffineMap() const {
  // Return an identity map over the logical rank. The memref keeps its
  // logical shape; sbuf_map is metadata about physical factorization.
  return AffineMap::getMultiDimIdentityMap(getLogicalRank(),
                                           getContext());
}

bool SbufMapAttr::isIdentity() const { return false; }

LogicalResult SbufMapAttr::verifyLayout(
    ArrayRef<int64_t> shape,
    function_ref<InFlightDiagnostic()> emitError) const {
  // The shape on the memref is the LOGICAL shape. Verify that
  // tileSize and numBlocks are consistent with it.
  if (static_cast<int64_t>(shape.size()) != getLogicalRank()) {
    return emitError() << "sbuf_map has " << getLogicalRank()
                       << " dims but memref has rank " << shape.size();
  }
  auto tile = getTileSize();
  auto blocks = getNumBlocks();
  for (int64_t i = 0, e = getLogicalRank(); i < e; ++i) {
    if (shape[i] == ShapedType::kDynamic)
      continue;
    if (tile[i] * blocks[i] != shape[i]) {
      return emitError() << "sbuf_map dim " << i << ": tile(" << tile[i]
                         << ") * blocks(" << blocks[i] << ") != shape("
                         << shape[i] << ")";
    }
  }
  return success();
}

LogicalResult SbufMapAttr::getStridesAndOffset(
    ArrayRef<int64_t> shape, SmallVectorImpl<int64_t> &strides,
    int64_t &offset) const {
  // Return standard row-major strides for the logical shape.
  // The sbuf_map encodes physical factorization but the memref's
  // logical layout is still contiguous row-major.
  offset = 0;
  int64_t rank = shape.size();
  strides.resize(rank);
  int64_t stride = 1;
  for (int64_t i = rank - 1; i >= 0; --i) {
    strides[i] = stride;
    if (shape[i] == ShapedType::kDynamic)
      stride = ShapedType::kDynamic;
    else
      stride *= shape[i];
  }
  return success();
}

LogicalResult SbufMapAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    ArrayRef<int64_t> tileSize, ArrayRef<int64_t> numBlocks) {
  if (tileSize.size() != numBlocks.size()) {
    return emitError() << "sbuf_map: tileSize and numBlocks must have "
                          "the same number of elements";
  }
  if (tileSize.empty()) {
    return emitError() << "sbuf_map: must have at least one dimension";
  }
  for (int64_t t : tileSize) {
    if (t <= 0)
      return emitError() << "sbuf_map: tile sizes must be positive";
  }
  for (int64_t b : numBlocks) {
    if (b <= 0)
      return emitError() << "sbuf_map: block counts must be positive";
  }
  return success();
}

Attribute SbufMapAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess())
    return {};

  // Parse: tile: [t0, ..., tR], blocks: [b0, ..., bR]
  SmallVector<int64_t> tileSize, numBlocks;

  if (parser.parseKeyword("tile") || parser.parseColon() ||
      parser.parseLSquare())
    return {};
  if (parser.parseCommaSeparatedList([&]() {
        int64_t val;
        if (parser.parseInteger(val))
          return failure();
        tileSize.push_back(val);
        return success();
      }))
    return {};
  if (parser.parseRSquare() || parser.parseComma())
    return {};

  if (parser.parseKeyword("blocks") || parser.parseColon() ||
      parser.parseLSquare())
    return {};
  if (parser.parseCommaSeparatedList([&]() {
        int64_t val;
        if (parser.parseInteger(val))
          return failure();
        numBlocks.push_back(val);
        return success();
      }))
    return {};
  if (parser.parseRSquare() || parser.parseGreater())
    return {};

  return SbufMapAttr::getChecked(
      parser.getEncodedSourceLoc(parser.getCurrentLocation()),
      parser.getContext(), tileSize, numBlocks);
}

void SbufMapAttr::print(AsmPrinter &printer) const {
  printer << "<tile: [";
  llvm::interleaveComma(getTileSize(), printer);
  printer << "], blocks: [";
  llvm::interleaveComma(getNumBlocks(), printer);
  printer << "]>";
}

//===----------------------------------------------------------------------===//
// Dialect initialization
//===----------------------------------------------------------------------===//

void NkipyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "nkipy/Dialect/NkipyOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "nkipy/Dialect/NkipyAttrs.cpp.inc"
      >();
}

mlir::Type NkipyDialect::parseType(DialectAsmParser &parser) const {
  parser.emitError(parser.getCurrentLocation(),
                   "nkipy dialect has no custom types");
  return mlir::Type();
}

void NkipyDialect::printType(Type type, DialectAsmPrinter &printer) const {
  llvm_unreachable("nkipy dialect has no custom types");
}