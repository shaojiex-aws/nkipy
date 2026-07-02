//===- NkipyTransformOps.cpp - Nkipy Transform Operations -----------------===//
//
// Implementation of custom transform dialect operations for NKIPyKernelGen.
//
//===----------------------------------------------------------------------===//

#include "nkipy/TransformOps/NkipyTransformOps.h"

#include "nkipy/Dialect/NkipyAttrs.h"
#include "nkipy/Dialect/NkipyOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;

#define GET_OP_CLASSES
#include "nkipy/TransformOps/NkipyTransformOps.cpp.inc"

namespace {

//===----------------------------------------------------------------------===//
// PromoteTensorOp helpers
//===----------------------------------------------------------------------===//

/// Return true if the operand may be read from by its owner.
static bool mayBeRead(OpOperand &operand) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(operand.getOwner());
  if (!linalgOp)
    return true;
  Value blockArgument = linalgOp.getMatchingBlockArgument(&operand);
  return !blockArgument.use_empty();
}

/// Return true if the value may be read through any of its uses.
static bool mayBeRead(Value value) {
  if (!isa<MemRefType, FloatType, IntegerType>(value.getType()))
    return true;
  return llvm::any_of(value.getUses(),
                      static_cast<bool (&)(OpOperand &)>(mayBeRead));
}

/// Return the mem_space that `value` already lives in, if known.
/// Walks back through no-alloc aliasing ops so promote_tensor can skip
/// alloc+copy when the value already lives in the target memory space.
static std::optional<nkipy::MemSpaceEnum>
findExistingMemSpace(Value value) {
  Value v = value;
  while (v) {
    for (Operation *user : v.getUsers()) {
      auto layout = dyn_cast<nkipy::LayoutOp>(user);
      if (!layout || layout.getTarget() != v)
        continue;
      if (auto ms = layout.getMemSpace())
        return ms->getValue();
    }
    if (auto allocOp = v.getDefiningOp<memref::AllocOp>()) {
      auto memrefType = cast<MemRefType>(allocOp.getType());
      if (auto msAttr = memrefType.getMemorySpace()) {
        if (auto nkipyMs = dyn_cast<nkipy::MemSpaceAttr>(msAttr))
          return nkipyMs.getValue();
      }
      return std::nullopt;
    }
    Operation *defOp = v.getDefiningOp();
    if (!defOp)
      return std::nullopt;
    if (auto subview = dyn_cast<memref::SubViewOp>(defOp))
      v = subview.getSource();
    else if (auto transposeOp = dyn_cast<linalg::TransposeOp>(defOp))
      v = transposeOp.getInit();
    else
      return std::nullopt;
  }
  return std::nullopt;
}

} // namespace

//===----------------------------------------------------------------------===//
// PromoteTensorOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::PromoteTensorOp::apply(transform::TransformRewriter &rewriter,
                                  transform::TransformResults &results,
                                  transform::TransformState &state) {
  SmallVector<Value> promoted;
  SmallVector<Operation *> copyIns;
  std::optional<nkipy::MemSpaceEnum> targetMs;
  if (auto msAttr = getMemorySpaceAttr())
    if (auto nkipyMs = dyn_cast<nkipy::MemSpaceAttr>(msAttr))
      targetMs = nkipyMs.getValue();

  for (Value value : state.getPayloadValues(getTensor())) {
    if (targetMs) {
      auto sourceMs = findExistingMemSpace(value);
      if (sourceMs && *sourceMs == *targetMs) {
        promoted.push_back(value);
        continue;
      }
    }

    auto memrefType = dyn_cast<MemRefType>(value.getType());
    if (!memrefType) {
      return emitSilenceableError() << "unsupported type: " << value;
    }

    bool needsCopyIn = mayBeRead(value);
    Operation *dpsConsumer = nullptr;
    for (OpOperand &use : value.getUses()) {
      auto dstOp = dyn_cast<DestinationStyleOpInterface>(use.getOwner());
      if (dstOp && dstOp.isDpsInit(&use)) {
        dpsConsumer = dstOp;
        break;
      }
    }

    Operation *definingOp = value.getDefiningOp();
    if (definingOp)
      rewriter.setInsertionPointAfter(definingOp);
    else
      rewriter.setInsertionPointToStart(
          cast<BlockArgument>(value).getOwner());

    auto newMemrefType = MemRefType::get(
        memrefType.getShape(), memrefType.getElementType(),
        MemRefLayoutAttrInterface{}, getMemorySpaceAttr());
    SmallVector<Value> dynamicDims;
    for (auto [pos, dim] : llvm::enumerate(memrefType.getShape())) {
      if (!ShapedType::isDynamic(dim))
        continue;
      Value idx = rewriter.create<arith::ConstantIndexOp>(
          value.getLoc(), static_cast<int64_t>(pos));
      dynamicDims.push_back(
          rewriter.create<memref::DimOp>(value.getLoc(), value, idx));
    }
    auto alloc = rewriter.create<memref::AllocOp>(
        value.getLoc(), newMemrefType, dynamicDims);

    llvm::SmallPtrSet<Operation *, 4> preservedOps;
    preservedOps.insert(alloc);

    if (auto nkipyMs = dyn_cast_or_null<nkipy::MemSpaceAttr>(
            newMemrefType.getMemorySpace())) {
      if (nkipyMs.getValue() == nkipy::MemSpaceEnum::Sbuf &&
          newMemrefType.getRank() >= 2) {
        auto pdim0 = rewriter.getIntegerAttr(
            rewriter.getIntegerType(32, /*isSigned=*/false), 0);
        // Caller-supplied tile (matmul operands, read at tile granularity)
        // takes precedence. Otherwise the operand is already leaf-sized
        // (elementwise promotion runs after tiling), so cap dim 0 at 128.
        DenseI64ArrayAttr tileAttr = getTileSizeAttr();
        if (!tileAttr) {
          SmallVector<int64_t> tile(newMemrefType.getShape().begin(),
                                    newMemrefType.getShape().end());
          tile[0] = std::min(tile[0], (int64_t)128);
          tileAttr = DenseI64ArrayAttr::get(rewriter.getContext(), tile);
        }
        rewriter.create<nkipy::LayoutOp>(
            value.getLoc(), alloc.getResult(), nkipyMs, pdim0, tileAttr);
      }
    }

    if (needsCopyIn) {
      auto copyOp = rewriter.create<linalg::CopyOp>(
          value.getLoc(), ValueRange{value}, ValueRange{alloc.getResult()});
      preservedOps.insert(copyOp);
      copyIns.push_back(copyOp);
    }

    if (dpsConsumer) {
      rewriter.setInsertionPointAfter(dpsConsumer);
      auto copyBack = rewriter.create<linalg::CopyOp>(
          value.getLoc(), ValueRange{alloc.getResult()}, ValueRange{value});
      preservedOps.insert(copyBack);
    } else if (!needsCopyIn) {
      rewriter.setInsertionPoint(value.getParentBlock()->getTerminator());
      auto copyBack = rewriter.create<linalg::CopyOp>(
          value.getLoc(), ValueRange{alloc.getResult()}, ValueRange{value});
      preservedOps.insert(copyBack);
    }

    promoted.push_back(alloc.getResult());
    rewriter.replaceAllUsesExcept(value, promoted.back(), preservedOps);
  }
  results.setValues(cast<OpResult>(getPromoted()), promoted);
  results.set(cast<OpResult>(getCopyIn()), copyIns);
  return DiagnosedSilenceableFailure::success();
}

void transform::PromoteTensorOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTensorMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// NkipyTransposeMatmulOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::NkipyTransposeMatmulOp::apply(transform::TransformRewriter &rewriter,
                                          transform::TransformResults &results,
                                          transform::TransformState &state) {
  SmallVector<Operation *> transformed;
  SmallVector<Operation *> transposes;

  for (Operation *op : state.getPayloadOps(getTarget())) {
    auto matmulOp = dyn_cast<linalg::MatmulOp>(op);
    if (!matmulOp) {
      return emitSilenceableError()
             << "expected linalg.matmul, got " << op->getName();
    }

    rewriter.setInsertionPoint(matmulOp);
    Location loc = matmulOp.getLoc();

    Value lhs = matmulOp.getInputs()[0];
    Value rhs = matmulOp.getInputs()[1];
    Value out = matmulOp.getOutputs()[0];

    auto lhsType = cast<ShapedType>(lhs.getType());
    int64_t M = lhsType.getDimSize(0);
    int64_t K = lhsType.getDimSize(1);
    Type elemTy = lhsType.getElementType();

    auto sbufAttr = nkipy::MemSpaceAttr::get(
        rewriter.getContext(), nkipy::MemSpaceEnum::Sbuf);
    auto transposedType = MemRefType::get(
        {K, M}, elemTy, MemRefLayoutAttrInterface{}, sbufAttr);
    Value transposedInit = rewriter.create<memref::AllocOp>(loc, transposedType);

    // Attach sbuf_tile_size. The transpose output (A^T) is consumed by
    // matmul_transpose_a at tile granularity [tileK, tileM], so the caller
    // passes that tile. Fall back to [min(K,128), M] when unset.
    {
      auto pdim0 = rewriter.getIntegerAttr(
          rewriter.getIntegerType(32, /*isSigned=*/false), 0);
      DenseI64ArrayAttr tileAttr = getTileSizeAttr();
      if (!tileAttr) {
        SmallVector<int64_t> tile = {std::min(K, (int64_t)128), M};
        tileAttr = DenseI64ArrayAttr::get(rewriter.getContext(), tile);
      }
      rewriter.create<nkipy::LayoutOp>(
          loc, transposedInit, sbufAttr, pdim0, tileAttr);
    }

    auto transposeOp = rewriter.create<linalg::TransposeOp>(
        loc, lhs, transposedInit, ArrayRef<int64_t>{1, 0});

    auto newMatmul = rewriter.create<linalg::MatmulTransposeAOp>(
        loc, matmulOp.getResultTypes(), ValueRange{transposedInit, rhs},
        ValueRange{out});

    rewriter.replaceOp(matmulOp, newMatmul.getResults());
    transformed.push_back(newMatmul);
    transposes.push_back(transposeOp);
  }

  results.set(cast<OpResult>(getTransformed()), transformed);
  results.set(cast<OpResult>(getTranspose()), transposes);
  return DiagnosedSilenceableFailure::success();
}

void transform::NkipyTransposeMatmulOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTargetMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// Transform dialect extension registration
//===----------------------------------------------------------------------===//

namespace {

class NkipyTransformDialectExtension
    : public transform::TransformDialectExtension<
          NkipyTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NkipyTransformDialectExtension)

  NkipyTransformDialectExtension() {
    registerTransformOps<
#define GET_OP_LIST
#include "nkipy/TransformOps/NkipyTransformOps.cpp.inc"
        >();
  }
};

} // namespace

void mlir::nkipy::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<NkipyTransformDialectExtension>();
}
