
#ifndef NKIPY_MLIR_PASSES
#define NKIPY_MLIR_PASSES

include "mlir/Pass/PassBase.td"

def AnnotateMemorySpace : Pass<"annotate-memory-space", "func::FuncOp"> {
  let summary = "Annotate memrefs with memory space attributes from knobs";
  let description = [{
    This pass annotates memref types with memory space attributes based on
    nkipy.annotate operations. It performs two main tasks:
    
    1. Annotates function inputs/outputs with SharedHbm memory space
    2. Processes nkipy.annotate ops to apply memory spaces (Sbuf, Psum, etc.)
       to internal memrefs
    3. Propagates memory space attributes through memref.subview operations
    4. Removes nkipy.annotate ops after processing
    
    This prepares memrefs for subsequent lowering to NISA dialect.
  }];
  let constructor = "mlir::nkipy::createAnnotateMemorySpacePass()";
}


def CanonicalizeReshape : Pass<"canonicalize-reshape", "func::FuncOp"> {
  let summary = "Classify and canonicalize memref reshape ops by mem_space and partition_dim";
  let description = [{
    Post-bufferization pass that classifies expand_shape / collapse_shape ops
    based on memory space and partition dim involvement:

    - HBM reshapes: always views (no partition concept)
    - SBUF collapse (merge): always views (contiguous in memory)
    - SBUF expand (split) of non-partition dims: views
    - SBUF expand (split) of partition dim (dim 0): needs alloc+copy
      (NISA has no modulo op), except trivial splits N->(N,1)/(1,N)
    - Returned expand_shape views of func args: alloc+copy (NISA requires
      separate output allocations)

    Runs after annotate-memory-space so all memrefs have explicit memory
    spaces and partition_dim is guaranteed to be 0.
  }];
  let constructor = "mlir::nkipy::createCanonicalizeReshapePass()";
}

def CanonicalizePartitionDim : Pass<"canonicalize-partition-dim", "func::FuncOp"> {
  let summary = "Insert transposes to ensure partition_dim=0 everywhere";
  let description = [{
    This pass inserts linalg.transpose operations so that partition_dim=0
    holds for all annotated tensors. NISA hardware assumes dimension 0 is
    always the partition dimension, and every downstream pass relies on this.

    For each connected component of elementwise ops sharing a non-zero
    partition_dim, the pass:
    1. Inserts transposes on inputs entering the component
    2. Rewrites elementwise ops with permuted shapes
    3. Inserts inverse transposes on outputs leaving the component
    4. Updates nkipy.annotate ops: partition_dim -> 0, permutes tile_size

    Non-elementwise ops (matmul, reduction) with partition_dim != 0 are
    rejected with an error — the user must fix their annotation.

    This pass runs after infer-layout (so partition_dim is propagated to all
    ops in the chain) and before assign-linalg-op-ids (so new transposes
    get op IDs for tiling).
  }];
  let constructor = "mlir::nkipy::createCanonicalizePartitionDimPass()";
  let options = [
    Option<"target", "target", "std::string", "\"trn2\"",
           "Target hardware (trn1, trn2, trn3). Determines the partition "
           "dim limit applied when clamping boundary tile_size.">,
  ];
}

def AssignLinalgOpIds : Pass<"assign-linalg-op-ids", "func::FuncOp"> {
  let summary = "Assign unique nkipy.op_id attributes to all linalg operations";
  let description = [{
    This pass assigns unique nkipy.op_id attributes to ALL linalg operations.
    
    The op_id enables per-instance matching during transform dialect application,
    allowing different tile sizes to be applied to different instances of the
    same linalg operation type.
    
    Example transformation:
      Before:
        %0 = linalg.matmul ins(%A, %B) outs(%C) -> tensor<256x256xf32>
        %1 = linalg.add ins(%0, %D) outs(%E) -> tensor<256x256xf32>
      
      After:
        %0 = linalg.matmul {nkipy.op_id = 0 : i64} ins(%A, %B) outs(%C) -> tensor<256x256xf32>
        %1 = linalg.add {nkipy.op_id = 1 : i64} ins(%0, %D) outs(%E) -> tensor<256x256xf32>
  }];
  let constructor = "mlir::nkipy::createAssignLinalgOpIdsPass()";
}

def InferLayout : Pass<"infer-layout", "func::FuncOp"> {
  let summary = "Infer layout annotations (tiling and placement) for unannotated elementwise ops";
  let description = [{
    This pass infers layout information (tile_size and mem_space placement) for
    elementwise operations that lack explicit annotations. It propagates these
    layout attributes from annotated elementwise ops to adjacent unannotated
    ones along the SSA use-def chain.

    Tiling and placement (e.g., SBUF vs PSUM) are the minimum layout info
    required to make code runnable on hardware. Without them, ops remain
    untiled and unplaced after bufferization, causing failures in linalg-to-nisa.

    This enables users to write compound expressions like:
      gated = gate / (1.0 + exp(-gate)) * up
      knob(gated, mem_space="Sbuf", tile_size=[128, 128])
    and have the intermediate ops (negate, exp, add, reciprocal, mul)
    automatically inherit the tiling and placement from the annotated output.

    The pass walks backwards from each annotated elementwise op through its
    input operands. For each predecessor that is also elementwise and has the
    same output shape, it creates a new nkipy.annotate op with the same
    tile_size and mem_space.

    Propagation rules:
    - Only propagates to elementwise ops (named or generic with all-parallel)
    - Only propagates when producer output shape matches consumer output shape
    - Does not override existing annotations
    - Only infers tile_size and mem_space (not other advanced knobs)

    This pass runs after prepare-arithmetic and assign-linalg-op-ids, but
    before knob-driven-tiling.
  }];
  let constructor = "mlir::nkipy::createInferLayoutPass()";
  let options = [
    Option<"target", "target", "std::string", "\"trn2\"",
           "Target hardware (trn1, trn2, trn3). Determines the SBUF "
           "partition limit and PSUM bank size used when inferring "
           "tile_size on auto-annotated ops.">,
  ];
}

def KnobDrivenTiling : Pass<"knob-driven-tiling", "ModuleOp"> {
  let summary = "Generate Transform dialect sequence for tiling linalg operations with blocking";
  let description = [{
    This pass generates Transform dialect operations to tile linalg operations using
    tile sizes specified in nkipy.annotate knobs. It uses a two-level blocking strategy
    for improved data reuse.
    
    The pass adds a `transform.named_sequence @__transform_main` to the module that
    contains the tiling schedule. The transforms are NOT applied by this pass - use
    the `apply-and-strip-transforms` pass afterwards to apply them and erase the
    transform module.
    
    Blocking strategy (TILES_IN_BLOCK = 2 for both M and N):
    - BLOCK_M = TILE_M × 2
    - BLOCK_N = TILE_N × 2
    
    Generated loop structure for matmul:
      for block_m in [0, M, BLOCK_M):       // Level 1: Block-M loop
        LOAD LHS to SBUF                    // Reused across all N-blocks
        for block_n in [0, N, BLOCK_N):     // Level 1: Block-N loop
          LOAD RHS to SBUF                  // Reused within block
          for tile_m in [0, BLOCK_M, TILE_M):   // Level 2: Tile-M loop
            for tile_n in [0, BLOCK_N, TILE_N): // Level 2: Tile-N loop
              ALLOC psum buffer
              for k in [0, K, TILE_K):          // Level 2: K loop
                psum += matmul(lhs_tile, rhs_tile)
              STORE result tile
    
    Usage:
      1. Run `--knob-driven-tiling` to generate transform sequence
      2. Run `--apply-and-strip-transforms` to apply and strip the transforms

    Example:
      nkipy-opt input.mlir --knob-driven-tiling --apply-and-strip-transforms
  }];
  let constructor = "mlir::nkipy::createKnobDrivenTilingPass()";
}

def ApplyAndStripTransforms : Pass<"apply-and-strip-transforms", "ModuleOp"> {
  let summary = "Run @__transform_main, then erase the transform module";
  let description = [{
    Fused replacement for `--transform-interpreter` followed by manual
    stripping of the transform module. Applies the `@__transform_main`
    named sequence to the enclosing ModuleOp (same semantics as upstream
    `--transform-interpreter`), then erases every top-level op in the
    transform dialect and clears the `transform.with_named_sequence`
    module attribute.

    This lives in our pipeline in place of `--transform-interpreter`.
    After this pass runs, the IR contains no transform-dialect ops, so
    the downstream Python linalg→NISA phase can parse the module through
    upstream MLIR bindings (which don't know about our custom transform
    op `transform.nkipy.promote_tensor`).

    No-op if no `@__transform_main` entry point is present.
  }];
  let constructor = "mlir::nkipy::createApplyAndStripTransformsPass()";
}



def InsertSpillReload : Pass<"insert-spill-reload", "func::FuncOp"> {
  let summary = "Insert spill/reload operations for SBUF memory pressure";
  let description = [{
    This pass analyzes per-partition SBUF memory pressure and automatically
    inserts spill (SBUF→HBM) and reload (HBM→SBUF) operations when capacity
    is exceeded.

    SBUF has 128 partitions; usable size per partition is ~176 KB on TRN1,
    ~208 KB on TRN2, and ~240 KB on TRN3.  This pass runs after
    legalize-layout, so SBUF memrefs are in physical layout
    [partTile, numBlocks..., freeTile].  Per-partition size is computed as
    total_size / shape[0] (partTile).

    When a kernel's live SBUF usage exceeds the per-partition capacity:
    1. Spill: Copy less-frequently-used data from SBUF → HBM
    2. Reload: Copy it back when needed HBM → SBUF

    Algorithm:
    1. Collect all SBUF allocations and compute their sizes
    2. Perform liveness analysis to find peak memory pressure points
    3. At high-pressure points, select victims to spill using a heuristic
       (size-based, LRU, or Belady's MIN)
    4. Insert memref.copy operations for spill/reload

    These copies are later lowered to nisa.dma_copy in the linalg-to-nisa pass.

    Example transformation:
      Before (3 × 1 MB SBUF allocations, exceed per-partition capacity):
        %a = memref.alloc() : memref<512x512xf32, #sbuf>  // 1MB
        %b = memref.alloc() : memref<512x512xf32, #sbuf>  // 1MB
        %c = memref.alloc() : memref<512x512xf32, #sbuf>  // 1MB
        linalg.copy ins(%arg0) outs(%a)
        linalg.exp ins(%a) outs(%b)
        linalg.mul ins(%b, %b) outs(%c)

      After (with %a spilled after use):
        %a = memref.alloc() : memref<512x512xf32, #sbuf>
        %a_spill = memref.alloc() : memref<512x512xf32, #hbm>  // Spill slot
        %b = memref.alloc() : memref<512x512xf32, #sbuf>
        %c = memref.alloc() : memref<512x512xf32, #sbuf>
        linalg.copy ins(%arg0) outs(%a)
        memref.copy %a, %a_spill  // SPILL: %a to HBM
        linalg.exp ins(%a) outs(%b)
        linalg.mul ins(%b, %b) outs(%c)  // Peak now 2MB (only %b, %c)

    Pipeline Position:
    - Insert after legalize-layout + canonicalize (pass #17)
    - Before insert-memref-dealloc (pass #19)
    - SBUF allocs are in physical per-partition layout; sizes are per-partition
  }];
  let constructor = "mlir::nkipy::createInsertSpillReloadPass()";
  let options = [
    Option<"target", "target", "std::string", "\"trn2\"",
           "Target hardware (trn1, trn2, trn3). Used to query SBUF partition "
           "usable size from NisaTargetInfo.">,
    Option<"sbufCapacityOverride", "sbuf-capacity", "int64_t", "-1",
           "Override SBUF capacity in bytes (-1 = query from target). "
           "Useful for testing.">
  ];
}

def InsertMemRefDealloc : Pass<"insert-memref-dealloc", "func::FuncOp"> {
  let summary = "Insert memref.dealloc operations to mark allocation lifetime ends";
  let description = [{
    Analyzes lifetime of memref.alloc operations with NISA memory space and
    inserts memref.dealloc at appropriate points. These are later lowered to
    nisa.release by the linalg-to-nisa pass.

    - Only deallocates memrefs with NISA memory space (SBUF, PSUM, HBM)
    - Skips SHAREDHBM (externally managed)
    - Skips allocations that escape (returned from function)
    - Uses scope-based release: dealloc at end of allocation's scope
  }];
  let constructor = "mlir::nkipy::createInsertMemRefDeallocPass()";
}


def LegalizeLayout : Pass<"legalize-layout", "func::FuncOp"> {
  let summary = "Legalize SBUF tensor layouts to satisfy NKI hardware constraints";
  let description = [{
    This pass transforms SBUF tensor layouts to satisfy NKI hardware constraints
    where the first dimension (partition dimension) must be ≤128.
    
    The pass identifies SBUF tensors via two mechanisms:
    1. bufferization.alloc_tensor with memory_space = #nisa.mem<sbuf>
    2. nkipy.annotate ops with mem_space = Sbuf (traced back to tensor.empty)
    
    For each SBUF tensor needing legalization, the pass:
    1. Computes the target 4D shape based on tile_size annotation
    2. Propagates the shape change through the entire use-def chain via BFS
    3. Updates scf.for init_args, block args, and results
    4. Transforms extract_slice to 4D indexing + collapse_shape
    5. Transforms insert_slice with expand_shape + 4D indexing
    
    Example transformation for tensor<512x512xf32> with tile_size=[128,128,128]:
      Before:
        %0 = tensor.empty() : tensor<512x512xf32>
        %1 = scf.for ... iter_args(%arg = %0) {
          %e = tensor.extract_slice %arg[%i*128, %j*128] [128, 128]
          ...
          %ins = tensor.insert_slice %tile into %arg[%i*128, %j*128]
          scf.yield %ins
        }
      
      After:
        %0 = tensor.empty() : tensor<128x4x4x128xf32>
        %1 = scf.for ... iter_args(%arg = %0) : tensor<128x4x4x128xf32> {
          %e_4d = tensor.extract_slice %arg[0, %i, %j, 0] [128, 1, 1, 128]
          %e = tensor.collapse_shape %e_4d [[0,1],[2,3]]
          ...
          %tile_4d = tensor.expand_shape %tile [[0,1],[2,3]]
          %ins = tensor.insert_slice %tile_4d into %arg[0, %i, %j, 0]
          scf.yield %ins
        }
    
    Prerequisites:
    - Runs after knob-driven-tiling + transform-interpreter
    - Runs after canonicalize-loop-step (loops have step=1)
    
    This pass replaces the separate legalize-sbuf-outputs and legalize-sbuf-inputs
    passes by performing both transformations atomically.
  }];
  let constructor = "mlir::nkipy::createLegalizeLayoutPass()";
  let options = [
    Option<"target", "target", "std::string", "\"trn2\"",
           "Target hardware (trn1, trn2, trn3). Determines the SBUF "
           "partition tile cap when picking partTile for HBM allocs.">,
  ];
}

def SimplifyLinalg : Pass<"simplify-linalg", "func::FuncOp"> {
  let summary = "Simplify linalg operations for NISA lowering";
  let description = [{
    Simplification pass that rewrites linalg operations before linalg-to-nisa.

    1. Rewrites >2D SBUF linalg.transpose with unit dims to 2D.
       NISA dma_transpose only supports [1,0] (2D) or [2,1,0] (3D full reverse).
       Collapses SBUF allocs to 2D + expand_shape views, rewrites transpose or
       emits copy (when non-unit dims keep order, i.e. just a reshape).

    2. Converts trivial-broadcast linalg.generic ops to named linalg ops.
       After tiling, broadcasts become same-shape operations. This converts
       them to named ops (linalg.mul, etc.) so LinalgToNisa patterns can match.
  }];
  let constructor = "mlir::nkipy::createSimplifyLinalgPass()";
}

def CanonicalizeCompute : Pass<"canonicalize-compute", "ModuleOp"> {
  let summary = "Canonicalize linalg compute ops for NISA hardware";
  let description = [{
    Combined pass that rewrites linalg ops for NISA hardware constraints:

    1. Convert division to reciprocal+multiply (NISA has no divide).
    2. Decompose batch_matmul → scf.for + matmul (NISA only has 2D matmul).
    3. Remove fill(0) before matmul-like ops (NISA matmul auto-zeros PSUM).

    Runs before infer-layout / tiling.
  }];
  let constructor = "mlir::nkipy::createCanonicalizeComputePass()";
}

def InlineNkipyReference : Pass<"inline-nkipy-reference", "func::FuncOp"> {
  let summary = "Inline reference_impl regions from nkipy ops";
  let description = [{
    nkipy ops like nkipy.gather carry an optional reference_impl region
    containing standard linalg/tensor ops that express the same computation.

    This pass replaces each such op with the inlined contents of its region,
    enabling LLVM CPU simulation on IR that would otherwise contain custom
    dialect ops the LLVM JIT cannot lower.

    The pass is NOT part of the NISA compilation pipeline.  It is used only
    for the LLVM JIT verification path.
  }];
  let constructor = "mlir::nkipy::createInlineNkipyReferencePass()";
}

def KnobDrivenFusion : Pass<"knob-driven-fusion", "func::FuncOp"> {
  let summary = "Fuse sibling scf.for loops based on nkipy.fuse_op annotations";
  let description = [{
    Walks each `nkipy.fuse_op(%a, %b, ...)` annotation in the function and
    fuses the `scf.for` loops producing the listed tensors into a single
    loop using `fuseIndependentSiblingForLoops`.

    Runs after `apply-and-strip-transforms` (so the per-op `scf.for` loops
    exist) and before `canonicalize-loop-step` (so loop bounds are still
    in their original `step=tile` form).  All listed loops must have
    matching lower/upper/step bounds; otherwise the pass emits an error.

    The annotated values are expected to alias to `scf.for` results after
    tiling (each value's defining op is the tiled `scf.for`).  The pass
    erases each `nkipy.fuse_op` after applying the fusion.
  }];
  let constructor = "mlir::nkipy::createKnobDrivenFusionPass()";
  let dependentDialects = ["scf::SCFDialect"];
}

#endif // NKIPY_MLIR_PASSES
