#ifndef NKIPY_ATTRS
#define NKIPY_ATTRS

include "nkipy/Dialect/NkipyDialect.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

// https://mlir.llvm.org/docs/OpDefinitions/#enum-attributes
// Memory space enumeration for annotate operation.
//
// IMPORTANT: All values must be NON-ZERO. MemSpaceEnumAttr is built on
// IntegerAttr, and mlir::MemRefType::get() treats a memorySpace attribute
// whose integer value is 0 as "no memory space" — it is silently dropped
// during uniquing. That would make `root.setType(memref<..., Hbm>)` a no-op
// for HBM=0, which in turn breaks the fixpoint loop in
// AnnotateMemorySpace::inferHbmForCopySources (it keeps re-applying HBM
// every iteration and never converges).
// Values 1..5 keep all memspaces representable on memref types.
def Hbm: I32EnumAttrCase<"Hbm", 1>;
def Psum: I32EnumAttrCase<"Psum", 2>;
def Sbuf: I32EnumAttrCase<"Sbuf", 3>;
def SharedHbm: I32EnumAttrCase<"SharedHbm", 4>;
def Constant: I32EnumAttrCase<"Constant", 5>;  // Scalar broadcast constants (marker, not real memory)

def MemSpaceEnum: I32EnumAttr<"MemSpaceEnum",
        "Memory space enumeration",
        [Hbm, Psum, Sbuf, SharedHbm, Constant]> {
    let cppNamespace = "mlir::nkipy";
    let stringToSymbolFnName = "ConvertToMemSpaceEnum";
    let symbolToStringFnName = "ConvertToMemSpaceString";
}

//===----------------------------------------------------------------------===//
// NkipyMemSpaceAttr
//===----------------------------------------------------------------------===//

def NkipyMemSpaceAttr : Nkipy_Attr<"MemSpace"> {
  let mnemonic = "mem";
  let summary = "NKIPy memory space attribute for memref types";
  let description = [{
    A typed memory space attribute that prints as `#nkipy.mem<sbuf>` instead
    of a raw integer. Used as the memory_space on memref types throughout
    the pipeline.

    Syntax: `#nkipy.mem<sbuf>` | `#nkipy.mem<psum>` | `#nkipy.mem<hbm>` |
            `#nkipy.mem<shared_hbm>` | `#nkipy.mem<constant>`
  }];

  let parameters = (ins
    EnumParameter<MemSpaceEnum>:$value
  );

  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// NkipySbufMapAttr
//===----------------------------------------------------------------------===//

def NkipySbufMapAttr : Nkipy_Attr<"SbufMap",
    [DeclareAttrInterfaceMethods<MemRefLayoutAttrInterface,
                                 ["verifyLayout", "getStridesAndOffset",
                                  "isIdentity"]>]> {
  let mnemonic = "sbuf_map";
  let summary = "SBUF physical layout factorization for a memref";
  let description = [{
    Encodes the physical layout of an SBUF-resident memref as a factorization
    of each logical dimension into (tileSize, numBlocks) pairs.

    Physical shape: [tileSize[0], numBlocks[0], ..., numBlocks[R-1], tileSize[R-1]]

    Roles are positional (canonicalize-partition-dim guarantees dim 0 = partition):
      - dim 0's tileSize[0] = partition lanes (≤128)
      - dim R-1's tileSize[R-1] = free elements
      - numBlocks[i] for all i = block dimensions

    Syntax: `#nkipy.sbuf_map<tile: [t0, ..., tR], blocks: [b0, ..., bR]>`

    Example: a memref<512x512xf32> with tile [128, 128]:
      #nkipy.sbuf_map<tile: [128, 128], blocks: [4, 4]>
      Physical shape: [128, 4, 4, 128]
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t", "tile sizes per logical dim">:$tileSize,
    ArrayRefParameter<"int64_t", "num blocks per logical dim">:$numBlocks
  );

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Physical shape: [tileSize[0], numBlocks[0], ..., numBlocks[R-1], tileSize[R-1]]
    SmallVector<int64_t> getPhysicalShape() const;

    /// Logical rank (number of original dimensions).
    int64_t getLogicalRank() const { return getTileSize().size(); }

    /// Physical rank = logical rank + 2.
    int64_t getPhysicalRank() const { return getLogicalRank() + 2; }
  }];
}

#endif // NKIPY_ATTRS
