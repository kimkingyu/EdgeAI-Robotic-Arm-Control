// M4 schedule experiments: bounded tiling and an explicit vectorization attempt.
// Both are opt-in and must leave unmatched IR untouched. A successful vector
// rewrite is not a speed claim: generated ARM code still has to be inspected.
#include "edgeai/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/SmallVector.h"

namespace edgeai {
namespace {
using namespace mlir;

// Only the checked Q11 preprocessing generic is eligible; anything else keeps
// its current schedule rather than being reshaped by an unproven heuristic.
bool isPreprocessKernel(linalg::GenericOp op) {
  if (!op->hasAttr("edge.resize") || op.getNumLoops() != 3 ||
      op->getNumResults() != 1)
    return false;
  auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  return type && type.hasStaticShape() && type.getRank() == 3 &&
         type.getDimSize(2) == 3 && type.getElementType().isSignlessInteger(8);
}

struct TileOptions {
  int64_t rows = 0, columns = 0;
};

struct TilePass : PassWrapper<TilePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TilePass)
  TilePass() = default;
  TilePass(const TilePass &other) : PassWrapper(other) {}

  Option<int64_t> rows{*this, "rows", llvm::cl::desc("Row tile size; 0 disables"),
                       llvm::cl::init(0)};
  Option<int64_t> columns{*this, "columns",
                          llvm::cl::desc("Column tile size; 0 disables"),
                          llvm::cl::init(0)};

  StringRef getArgument() const final { return "edge-tile-preprocess"; }
  StringRef getDescription() const final {
    return "Tile the checked preprocessing kernel by fixed row/column sizes";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    linalg::LinalgDialect, scf::SCFDialect, tensor::TensorDialect>();
  }
  void runOnOperation() override {
    if (rows < 0 || columns < 0) {
      getOperation().emitError("tile sizes must be nonnegative");
      return signalPassFailure();
    }
    SmallVector<linalg::GenericOp> candidates;
    getOperation().walk([&](linalg::GenericOp op) {
      if (isPreprocessKernel(op))
        candidates.push_back(op);
    });
    IRRewriter rewriter(&getContext());
    for (linalg::GenericOp op : candidates) {
      scf::SCFTilingOptions options;
      // A zero tile size means "do not tile this loop"; the channel loop is
      // never tiled, so three-channel pixels stay together.
      options.setTileSizes({rewriter.getIndexAttr(rows), rewriter.getIndexAttr(columns),
                            rewriter.getIndexAttr(0)});
      rewriter.setInsertionPoint(op);
      auto tiled = scf::tileUsingSCF(rewriter, cast<TilingInterface>(op.getOperation()), options);
      if (failed(tiled)) {
        op.emitError("requested tiling failed for this kernel");
        return signalPassFailure();
      }
      rewriter.replaceOp(op, tiled->mergeResult.replacements);
    }
  }
};

struct VectorizePass : PassWrapper<VectorizePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorizePass)
  VectorizePass() = default;
  VectorizePass(const VectorizePass &other) : PassWrapper(other) {}

  // 20.1.8 requires input vector sizes to cover the whole static iteration
  // space, so vector width comes from tiling, not from an independent size
  // here. Passing no sizes lets the vectorizer use those static ranges.
  Option<int64_t> maxElements{*this, "max-elements",
                              llvm::cl::desc("Refuse kernels whose static iteration space exceeds this"),
                              llvm::cl::init(1024)};
  Option<bool> ndExtract{*this, "nd-extract",
                         llvm::cl::desc("Allow vectorizing non-contiguous tensor.extract"),
                         llvm::cl::init(true)};
  // Measured on real kernels: with nd-extract the vectorizer emits a
  // vector.gather whose offsets assume a densely packed source. That is wrong
  // for the row-padded ROIs this ABI accepts, so it must be opted into.
  Option<bool> allowPackedSourceGather{
      *this, "allow-packed-source-gather",
      llvm::cl::desc("Permit gathers that require a source without row padding"),
      llvm::cl::init(false)};

  StringRef getArgument() const final { return "edge-vectorize-preprocess"; }
  StringRef getDescription() const final {
    return "Attempt explicit vectorization of the checked preprocessing kernel";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect, linalg::LinalgDialect,
                    scf::SCFDialect, tensor::TensorDialect, vector::VectorDialect>();
  }
  void runOnOperation() override {
    if (maxElements <= 0) {
      getOperation().emitError("max-elements must be positive");
      return signalPassFailure();
    }
    SmallVector<linalg::GenericOp> candidates;
    getOperation().walk([&](linalg::GenericOp op) {
      if (isPreprocessKernel(op))
        candidates.push_back(op);
    });
    IRRewriter rewriter(&getContext());
    for (linalg::GenericOp op : candidates) {
      auto type = cast<RankedTensorType>(op->getResult(0).getType());
      const int64_t elements =
          type.getDimSize(0) * type.getDimSize(1) * type.getDimSize(2);
      if (elements > maxElements) {
        op.emitError("refusing to vectorize a ")
            << elements << "-element iteration space; tile it first";
        return signalPassFailure();
      }
      Location loc = op.getLoc();
      Operation *parent = op->getParentOp();
      if (failed(linalg::vectorize(rewriter, op.getOperation(), /*inputVectorSizes=*/{},
                                   /*inputScalableVecDims=*/{}, ndExtract))) {
        op.emitError("vectorization of this kernel failed for shape ")
            << type.getDimSize(0) << "x" << type.getDimSize(1) << "x"
            << type.getDimSize(2);
        return signalPassFailure();
      }
      if (!allowPackedSourceGather) {
        WalkResult found = parent->walk([](vector::GatherOp gather) {
          auto source = dyn_cast<ShapedType>(gather.getBase().getType());
          // A dynamic source shape here means flattened offsets were derived
          // from dims alone, which silently ignores a padded row stride.
          return source && !source.hasStaticShape() ? WalkResult::interrupt()
                                                    : WalkResult::advance();
        });
        if (found.wasInterrupted()) {
          // The original op was already replaced, so report against its location.
          emitError(loc) << "vectorization produced a gather that assumes a densely "
                            "packed source; row-padded inputs would read wrong pixels";
          return signalPassFailure();
        }
      }
    }
  }
};
} // namespace

void registerScheduleExperimentPasses() {
  PassRegistration<TilePass>();
  PassRegistration<VectorizePass>();
}

} // namespace edgeai
