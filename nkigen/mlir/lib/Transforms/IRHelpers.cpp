//===- IRHelpers.cpp - Shared IR utility functions -------------------------===//

#include "nkipy/Transforms/IRHelpers.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir {
namespace nkipy {

std::optional<int64_t> getConstantInt(Value v) {
  if (auto constOp = v.getDefiningOp<arith::ConstantOp>())
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
      return intAttr.getInt();
  if (auto constOp = v.getDefiningOp<arith::ConstantIndexOp>())
    return constOp.value();
  return std::nullopt;
}

Value getBaseMemRef(Value v) {
  while (auto *def = v.getDefiningOp()) {
    if (auto view = dyn_cast<ViewLikeOpInterface>(def)) {
      v = view.getViewSource();
      continue;
    }
    break;
  }
  return v;
}

void collectMemRefAliases(Value base, llvm::SetVector<Value> &aliases) {
  if (!aliases.insert(base))
    return;
  for (Operation *user : base.getUsers()) {
    auto view = dyn_cast<ViewLikeOpInterface>(user);
    if (!view || user->getNumResults() == 0 || view.getViewSource() != base)
      continue;
    collectMemRefAliases(user->getResult(0), aliases);
  }
}

Operation *findWriteCompletionOp(Value buffer) {
  Block *defBlock = nullptr;
  if (Operation *defOp = buffer.getDefiningOp())
    defBlock = defOp->getBlock();
  else
    defBlock = buffer.getParentBlock();
  if (!defBlock)
    return nullptr;

  llvm::SetVector<Value> aliases;
  collectMemRefAliases(buffer, aliases);

  Operation *last = nullptr;
  for (Value alias : aliases) {
    for (Operation *user : alias.getUsers()) {
      auto linalgOp = dyn_cast<linalg::LinalgOp>(user);
      if (!linalgOp)
        continue;
      bool writesAlias = llvm::any_of(linalgOp.getDpsInits(),
                                      [&](Value init) { return init == alias; });
      if (!writesAlias)
        continue;
      Operation *completion = getAncestorInBlock(user, defBlock);
      if (!completion)
        continue;
      if (!last || last->isBeforeInBlock(completion))
        last = completion;
    }
  }
  return last;
}

std::optional<nkipy::MemSpaceEnum> getNkipyMemSpace(Type type) {
  auto memrefType = dyn_cast<MemRefType>(type);
  if (!memrefType)
    return std::nullopt;
  auto memSpaceAttr = memrefType.getMemorySpace();
  if (!memSpaceAttr)
    return std::nullopt;
  if (auto ms = dyn_cast<nkipy::MemSpaceAttr>(memSpaceAttr))
    return ms.getValue();
  return std::nullopt;
}

Operation *getAncestorInBlock(Operation *op, Block *block) {
  while (op && op->getBlock() != block)
    op = op->getParentOp();
  return op;
}

} // namespace nkipy
} // namespace mlir
