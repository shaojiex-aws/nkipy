
#ifndef NKIPY_TRANSFORMS_PASSES_H
#define NKIPY_TRANSFORMS_PASSES_H

#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace nkipy {

std::unique_ptr<OperationPass<ModuleOp>> createCanonicalizeReshapePass();
std::unique_ptr<OperationPass<func::FuncOp>> createCanonicalizePartitionDimPass();
std::unique_ptr<OperationPass<func::FuncOp>> createAssignLinalgOpIdsPass();
std::unique_ptr<OperationPass<func::FuncOp>> createInferLayoutPass();
std::unique_ptr<OperationPass<ModuleOp>> createKnobDrivenTilingPass();
std::unique_ptr<OperationPass<func::FuncOp>> createKnobDrivenFusionPass();
std::unique_ptr<OperationPass<func::FuncOp>> createInsertSpillReloadPass();
std::unique_ptr<OperationPass<func::FuncOp>> createInsertMemRefDeallocPass();
std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeLayoutPass();
std::unique_ptr<OperationPass<func::FuncOp>> createSimplifyLinalgPass();
std::unique_ptr<OperationPass<ModuleOp>> createCanonicalizeComputePass();
std::unique_ptr<OperationPass<func::FuncOp>> createInlineNkipyReferencePass();


/// Registers all transformation passes
void registerNkipyPasses();

} // namespace nkipy
} // namespace mlir

#endif // NKIPY_TRANSFORMS_PASSES_H
