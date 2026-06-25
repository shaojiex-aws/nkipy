#include "nkipy-c/Dialect/NkipyAttributes.h"
#include "nkipy/Dialect/NkipyAttrs.h"
#include "nkipy/Dialect/NkipyDialect.h"

#include "mlir/CAPI/Registration.h"
#include "mlir/IR/Attributes.h"

using namespace mlir;
using namespace nkipy;

bool mlirAttributeIsAMemSpace(MlirAttribute attr) {
  return mlir::isa<MemSpaceAttr>(unwrap(attr));
}

MlirAttribute mlirMemSpaceGet(MlirContext ctx, MlirAttribute space) {
  auto attr = llvm::cast<mlir::IntegerAttr>(unwrap(space));
  MemSpaceEnum spaceEnum =
      static_cast<MemSpaceEnum>(attr.getInt());
  return wrap(MemSpaceAttr::get(unwrap(ctx), spaceEnum));
}

MlirStringRef mlirMemSpaceGetValue(MlirAttribute attr) {
  auto msAttr = mlir::cast<MemSpaceAttr>(unwrap(attr));
  llvm::StringRef str = ConvertToMemSpaceString(msAttr.getValue());
  return wrap(str);
}

bool mlirAttributeIsASbufMap(MlirAttribute attr) {
  return mlir::isa<SbufMapAttr>(unwrap(attr));
}

intptr_t mlirSbufMapAttrGetRank(MlirAttribute attr) {
  return mlir::cast<SbufMapAttr>(unwrap(attr)).getLogicalRank();
}

int64_t mlirSbufMapAttrGetTileSize(MlirAttribute attr, intptr_t idx) {
  return mlir::cast<SbufMapAttr>(unwrap(attr)).getTileSize()[idx];
}

int64_t mlirSbufMapAttrGetNumBlocks(MlirAttribute attr, intptr_t idx) {
  return mlir::cast<SbufMapAttr>(unwrap(attr)).getNumBlocks()[idx];
}
