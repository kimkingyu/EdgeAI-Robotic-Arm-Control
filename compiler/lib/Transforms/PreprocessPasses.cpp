#include "edgeai/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"

#include <cstdint>
#include <limits>

namespace edgeai {
namespace {
using namespace mlir;

constexpr llvm::StringLiteral kResize = "linear_half_pixel_q11_v1";

bool isHwcI8(RankedTensorType type, bool requireStatic) {
  return type && type.getRank() == 3 && type.getDimSize(2) == 3 &&
         type.getElementType().isSignlessInteger(8) && !type.getEncoding() &&
         (!requireStatic ||
          (type.hasStaticShape() && type.getDimSize(0) > 0 &&
           type.getDimSize(1) > 0 &&
           type.getDimSize(0) <= std::numeric_limits<int64_t>::max() / 2 &&
           type.getDimSize(1) <= std::numeric_limits<int64_t>::max() / 2));
}

bool isConstant(Value value, int64_t expected) {
  llvm::APInt integer;
  return value && matchPattern(value, m_ConstantInt(&integer)) &&
         integer.getSExtValue() == expected;
}

// Return the nonconstant operand of a binary op. Only explicitly commutative
// operations call this with commute=true; sub/div/shift keep their operand order.
template <typename Op>
Value otherThanConstant(Value value, int64_t constant, bool commute = false) {
  auto op = value ? value.getDefiningOp<Op>() : Op();
  if (!op)
    return {};
  if (isConstant(op->getOperand(1), constant))
    return op->getOperand(0);
  if (commute && isConstant(op->getOperand(0), constant))
    return op->getOperand(1);
  return {};
}

bool hasResizeAttributes(tensor::GenerateOp op, bool fused) {
  if (op->getAttrOfType<StringAttr>("edge.resize") !=
      StringAttr::get(op.getContext(), kResize))
    return false;
  if (fused != static_cast<bool>(op->getAttrOfType<UnitAttr>("edge.fused")))
    return false;
  for (NamedAttribute attr : op->getAttrs()) {
    StringRef name = attr.getName().getValue();
    if (name != "edge.resize" && !(fused && name == "edge.fused"))
      return false;
  }
  return true;
}

bool hasColorAttributes(linalg::GenericOp op) {
  // A library_call may have semantics beyond this region. Unknown semantic
  // annotations are not taken as permission to change a preprocessing method.
  for (NamedAttribute attr : op->getAttrs()) {
    StringRef name = attr.getName().getValue();
    if (name == "edge.color") {
      if (dyn_cast<StringAttr>(attr.getValue()) !=
          StringAttr::get(op.getContext(), "bgr_to_rgb"))
        return false;
    } else if (name != "indexing_maps" && name != "iterator_types" &&
               name != "operandSegmentSizes" && name != "doc") {
      return false;
    }
  }
  return true;
}

bool isAllowedScalar(Operation &op) {
  if (op.getNumRegions() || !isMemoryEffectFree(&op))
    return false;
  if (!llvm::StringSwitch<bool>(op.getName().getStringRef())
           .Cases("arith.constant", "arith.index_cast", "arith.addi", true)
           .Cases("arith.subi", "arith.muli", "arith.floordivsi", true)
           .Cases("arith.minsi", "arith.maxsi", "arith.extui", true)
           .Cases("arith.trunci", "arith.shrui", "arith.minui", true)
           .Case("tensor.extract", true)
           .Default(false))
    return false;
  for (NamedAttribute attr : op.getAttrs())
    if (attr.getName().getValue().starts_with("edge."))
      return false;
  auto scalarType = [](Type type) {
    return type.isIndex() || type.isSignlessInteger(8) ||
           type.isSignlessInteger(32) || type.isSignlessInteger(64);
  };
  for (Type type : op.getResultTypes())
    if (!scalarType(type))
      return false;
  for (auto operand : llvm::enumerate(op.getOperands())) {
    if (isa<tensor::ExtractOp>(op) && operand.index() == 0)
      continue;
    if (!scalarType(operand.value().getType()))
      return false;
  }
  return true;
}

// Match the exact integer dataflow emitted by edge-preprocess-gen. This is
// intentionally not an algebraic normalizer: an unrecognized equivalent body is
// left alone, rather than trusting a semantic-version string or changing its
// rounding, border policy, or channel dependencies.
class Q11BodyMatcher {
public:
  explicit Q11BodyMatcher(tensor::GenerateOp op) : op(op) {}

  bool match(bool fused) {
    auto type = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!isHwcI8(type, true) || op->getNumOperands() ||
        !hasResizeAttributes(op, fused) || !llvm::hasSingleElement(op.getBody()))
      return false;
    Block &body = op.getBody().front();
    if (body.getNumArguments() != 3 ||
        !llvm::all_of(body.getArgumentTypes(),
                      [](Type type) { return type.isIndex(); }))
      return false;
    auto yield = dyn_cast<tensor::YieldOp>(body.getTerminator());
    if (!yield || yield->getNumOperands() != 1)
      return false;
    SmallVector<tensor::ExtractOp, 4> extracts;
    for (Operation &scalar : body.without_terminator()) {
      if (!isAllowedScalar(scalar))
        return false;
      if (auto extract = dyn_cast<tensor::ExtractOp>(scalar))
        extracts.push_back(extract);
    }
    if (extracts.size() != 4)
      return false;
    source = extracts.front().getTensor();
    auto sourceType = dyn_cast<RankedTensorType>(source.getType());
    auto toTensor = source.getDefiningOp<bufferization::ToTensorOp>();
    if (!isHwcI8(sourceType, false) || !toTensor ||
        !toTensor->hasAttr("restrict") || toTensor->hasAttr("writable") ||
        toTensor->getBlock() == &body)
      return false;
    for (tensor::ExtractOp extract : extracts)
      if (extract.getTensor() != source || extract.getIndices().size() != 3 ||
          !extract.getResult().getType().isSignlessInteger(8))
        return false;

    Value channel = extracts.front().getIndices()[2];
    if (fused) {
      auto reverse = channel.getDefiningOp<arith::SubIOp>();
      if (!reverse || reverse->getBlock() != &body ||
          !isConstant(reverse.getLhs(), 2) ||
          reverse.getRhs() != body.getArgument(2) ||
          !body.getArgument(2).hasOneUse())
        return false;
    } else if (channel != body.getArgument(2)) {
      return false;
    }
    for (tensor::ExtractOp extract : extracts)
      if (extract.getIndices()[2] != channel)
        return false;
    for (OpOperand &use : channel.getUses())
      if (!isa<tensor::ExtractOp>(use.getOwner()) ||
          use.getOwner()->getBlock() != &body || use.getOperandNumber() != 3)
        return false;

    // Extract order is p00, p01, p10, p11. This ordering is part of the
    // conservative matcher, not a requirement of bilinear interpolation itself.
    Value y0 = extracts[0].getIndices()[0];
    Value x0 = extracts[0].getIndices()[1];
    Value y1 = extracts[3].getIndices()[0];
    Value x1 = extracts[3].getIndices()[1];
    if (extracts[1].getIndices()[0] != y0 ||
        extracts[1].getIndices()[1] != x1 ||
        extracts[2].getIndices()[0] != y1 ||
        extracts[2].getIndices()[1] != x0)
      return false;

    auto narrow = yield->getOperand(0).getDefiningOp<arith::TruncIOp>();
    if (!narrow || !narrow.getIn().getType().isSignlessInteger(32) ||
        !narrow.getOut().getType().isSignlessInteger(8))
      return false;
    Value shifted = otherThanConstant<arith::MinUIOp>(narrow.getIn(), 255, true);
    Value rounded = otherThanConstant<arith::ShRUIOp>(shifted, 22);
    Value sum = otherThanConstant<arith::AddIOp>(rounded, 2097152, true);
    auto add = sum ? sum.getDefiningOp<arith::AddIOp>() : arith::AddIOp();
    if (!add || !sum.getType().isSignlessInteger(32))
      return false;
    Value wx0, wx1, wy0, row1Wx0, row1Wx1, wy1;
    if (!matchRow(add.getLhs(), extracts[0], extracts[1], wx0, wx1, wy0) ||
        !matchRow(add.getRhs(), extracts[2], extracts[3], row1Wx0, row1Wx1,
                  wy1) ||
        wx0 != row1Wx0 || wx1 != row1Wx1)
      return false;
    return matchAxis(y0, y1, wy0, wy1, body.getArgument(0), 0,
                     type.getDimSize(0)) &&
           matchAxis(x0, x1, wx0, wx1, body.getArgument(1), 1,
                     type.getDimSize(1));
  }

private:
  static bool matchPixelProduct(Value value, tensor::ExtractOp extract,
                                Value &weight) {
    auto mul = value.getDefiningOp<arith::MulIOp>();
    if (!mul || !value.getType().isSignlessInteger(32))
      return false;
    for (unsigned pixelOperand = 0; pixelOperand != 2; ++pixelOperand) {
      auto pixel = mul->getOperand(pixelOperand).getDefiningOp<arith::ExtUIOp>();
      if (pixel && pixel.getIn() == extract.getResult() &&
          pixel.getOut().getType().isSignlessInteger(32)) {
        weight = mul->getOperand(1 - pixelOperand);
        return true;
      }
    }
    return false;
  }

  static bool matchRow(Value value, tensor::ExtractOp p0, tensor::ExtractOp p1,
                       Value &wx0, Value &wx1, Value &wy) {
    auto mul = value.getDefiningOp<arith::MulIOp>();
    if (!mul)
      return false;
    for (unsigned rowOperand = 0; rowOperand != 2; ++rowOperand) {
      auto row = mul->getOperand(rowOperand).getDefiningOp<arith::AddIOp>();
      if (row && matchPixelProduct(row.getLhs(), p0, wx0) &&
          matchPixelProduct(row.getRhs(), p1, wx1)) {
        wy = mul->getOperand(1 - rowOperand);
        return true;
      }
    }
    return false;
  }

  static Value wideWeight(Value value) {
    auto cast = value.getDefiningOp<arith::TruncIOp>();
    return cast && cast.getIn().getType().isSignlessInteger(64) &&
                   cast.getOut().getType().isSignlessInteger(32)
               ? cast.getIn()
               : Value();
  }

  // Accept only min(max(sample, 0), sourceSize - 1), then i64 -> index.
  static Value unclamp(Value index, Value &size) {
    auto cast = index.getDefiningOp<arith::IndexCastOp>();
    if (!cast || !cast.getIn().getType().isSignlessInteger(64) ||
        !cast.getOut().getType().isIndex())
      return {};
    auto min = cast.getIn().getDefiningOp<arith::MinSIOp>();
    if (!min)
      return {};
    for (unsigned boundedOperand = 0; boundedOperand != 2; ++boundedOperand) {
      Value sample = otherThanConstant<arith::MaxSIOp>(
          min->getOperand(boundedOperand), 0, true);
      Value candidateSize = otherThanConstant<arith::SubIOp>(
          min->getOperand(1 - boundedOperand), 1);
      if (sample && candidateSize) {
        size = candidateSize;
        return sample;
      }
    }
    return {};
  }

  bool matchSourceSize(Value size, int64_t axis) const {
    auto cast = size.getDefiningOp<arith::IndexCastOp>();
    if (!cast || !cast.getIn().getType().isIndex() ||
        !cast.getOut().getType().isSignlessInteger(64))
      return false;
    auto dim = cast.getIn().getDefiningOp<tensor::DimOp>();
    return dim && dim.getSource() == source && isConstant(dim.getIndex(), axis);
  }

  bool matchAxis(Value lo, Value hi, Value w0, Value w1, Value outputIndex,
                 int64_t axis, int64_t destinationSize) const {
    Value sourceSize, upperSize;
    Value lower = unclamp(lo, sourceSize);
    Value upper = unclamp(hi, upperSize);
    if (!lower || !upper || sourceSize != upperSize ||
        !matchSourceSize(sourceSize, axis) ||
        otherThanConstant<arith::AddIOp>(upper, 1, true) != lower)
      return false;
    Value numerator =
        otherThanConstant<arith::FloorDivSIOp>(lower, 2 * destinationSize);
    auto subtract =
        numerator ? numerator.getDefiningOp<arith::SubIOp>() : arith::SubIOp();
    if (!subtract || !isConstant(subtract.getRhs(), destinationSize))
      return false;
    auto scaled = subtract.getLhs().getDefiningOp<arith::MulIOp>();
    if (!scaled)
      return false;
    Value odd;
    if (scaled.getLhs() == sourceSize)
      odd = scaled.getRhs();
    else if (scaled.getRhs() == sourceSize)
      odd = scaled.getLhs();
    Value doubled = otherThanConstant<arith::AddIOp>(odd, 1, true);
    Value coordinate = otherThanConstant<arith::MulIOp>(doubled, 2, true);
    auto coordinateCast = coordinate
                              ? coordinate.getDefiningOp<arith::IndexCastOp>()
                              : arith::IndexCastOp();
    if (!coordinateCast || coordinateCast.getIn() != outputIndex ||
        !coordinateCast.getOut().getType().isSignlessInteger(64))
      return false;

    Value wide0 = wideWeight(w0), wide1 = wideWeight(w1);
    auto complement = wide0 ? wide0.getDefiningOp<arith::SubIOp>()
                            : arith::SubIOp();
    if (!wide1 || !complement || !isConstant(complement.getLhs(), 2048) ||
        complement.getRhs() != wide1)
      return false;
    Value rounded =
        otherThanConstant<arith::FloorDivSIOp>(wide1, 2 * destinationSize);
    Value weightedRemainder =
        otherThanConstant<arith::AddIOp>(rounded, destinationSize, true);
    Value remainder =
        otherThanConstant<arith::MulIOp>(weightedRemainder, 2048, true);
    auto rem = remainder ? remainder.getDefiningOp<arith::SubIOp>()
                         : arith::SubIOp();
    return rem && rem.getLhs() == numerator &&
           otherThanConstant<arith::MulIOp>(rem.getRhs(), 2 * destinationSize,
                                            true) == lower;
  }

  tensor::GenerateOp op;
  Value source;
};

// The init must be genuinely unused. In particular, yielding the output block
// argument (or doing any arithmetic/normalization in this body) is not a color
// permutation, even when an edge.color attribute claims otherwise.
bool matchColor(linalg::GenericOp op) {
  if (op.getInputs().size() != 1 || op.getOutputs().size() != 1 ||
      op->getNumResults() != 1 || !hasColorAttributes(op) ||
      !llvm::hasSingleElement(op.getRegion()))
    return false;
  auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!isHwcI8(type, true) || op.getInputs()[0].getType() != type ||
      op.getOutputs()[0].getType() != type ||
      !op.getOutputs()[0].getDefiningOp<tensor::EmptyOp>())
    return false;
  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  MLIRContext *ctx = op.getContext();
  AffineExpr y = getAffineDimExpr(0, ctx), x = getAffineDimExpr(1, ctx),
             c = getAffineDimExpr(2, ctx);
  AffineMap reverse = AffineMap::get(3, 0, {y, x, 2 - c}, ctx);
  if (maps.size() != 2 || maps[0] != reverse ||
      maps[1] != AffineMap::getMultiDimIdentityMap(3, ctx))
    return false;
  auto iterators = op.getIteratorTypesArray();
  if (iterators.size() != 3 ||
      !llvm::all_of(iterators, [](utils::IteratorType iterator) {
        return iterator == utils::IteratorType::parallel;
      }))
    return false;
  Block &body = op.getRegion().front();
  if (body.getNumArguments() != 2 || body.getOperations().size() != 1 ||
      !body.getArgument(0).getType().isSignlessInteger(8) ||
      !body.getArgument(1).getType().isSignlessInteger(8) ||
      !body.getArgument(1).use_empty())
    return false;
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  return yield && yield->getNumOperands() == 1 &&
         yield->getOperand(0) == body.getArgument(0);
}

struct FuseResizeColorPattern : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp color,
                                PatternRewriter &rewriter) const override {
    if (!matchColor(color))
      return failure();
    auto resize = color.getInputs()[0].getDefiningOp<tensor::GenerateOp>();
    if (!resize || !resize.getResult().hasOneUse() ||
        resize->getBlock() != color->getBlock() ||
        !resize->isBeforeInBlock(color) || !Q11BodyMatcher(resize).match(false))
      return failure();

    // Keep evaluation where the original producer was. Captured dimension and
    // source values already dominate this position; no capture-defining op is
    // hoisted or cloned, and no intervening memory effect is crossed.
    rewriter.setInsertionPoint(resize);
    auto fused = rewriter.create<tensor::GenerateOp>(
        resize.getLoc(), resize.getResult().getType(), ValueRange{},
        [&](OpBuilder &builder, Location loc, ValueRange indices) {
          IRMapping mapping;
          Block &oldBody = resize.getBody().front();
          mapping.map(oldBody.getArgument(0), indices[0]);
          mapping.map(oldBody.getArgument(1), indices[1]);
          Value two = builder.create<arith::ConstantIndexOp>(loc, 2);
          Value channel = builder.create<arith::SubIOp>(loc, two, indices[2]);
          mapping.map(oldBody.getArgument(2), channel);
          for (Operation &scalar : oldBody)
            builder.clone(scalar, mapping);
        });
    fused->setAttrs(resize->getAttrs());
    fused->setAttr("edge.fused", rewriter.getUnitAttr());
    rewriter.replaceOp(color, fused.getResult());
    rewriter.eraseOp(resize);
    return success();
  }
};

bool isContiguousDestination(Value value, RankedTensorType sourceType) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type || !type.hasStaticShape() || type.getRank() != 3 ||
      type.getShape() != sourceType.getShape() ||
      type.getElementType() != sourceType.getElementType())
    return false;
  SmallVector<int64_t> strides;
  int64_t offset;
  // Nonidentity layouts are accepted only when they encode exactly the same
  // zero-offset, contiguous HWC storage; dynamic/noncontiguous layouts fail.
  return succeeded(type.getStridesAndOffset(strides, offset)) && offset == 0 &&
         strides.size() == 3 && strides[2] == 1 && strides[1] == 3 &&
         sourceType.getDimSize(1) <= std::numeric_limits<int64_t>::max() / 3 &&
         strides[0] == 3 * sourceType.getDimSize(1);
}

bool canMoveProducerTo(Operation *producer, Operation *anchor) {
  if (producer->getBlock() != anchor->getBlock() ||
      !producer->isBeforeInBlock(anchor))
    return false;
  // In particular, never delay reads of a captured to_tensor past a call/store.
  // Keeping both ops in the same block also proves that every captured SSA
  // value available at the old producer still dominates the new one.
  for (Operation *next = producer->getNextNode(); next != anchor;
       next = next->getNextNode())
    if (next->getNumRegions() || !isMemoryEffectFree(next))
      return false;
  return true;
}

struct PrepareDestinationPattern
    : OpRewritePattern<bufferization::MaterializeInDestinationOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult
  matchAndRewrite(bufferization::MaterializeInDestinationOp anchor,
                  PatternRewriter &rewriter) const override {
    Value source = anchor.getSource();
    auto type = dyn_cast<RankedTensorType>(source.getType());
    if (!anchor->getAttrOfType<UnitAttr>("restrict") ||
        !anchor->getAttrOfType<UnitAttr>("writable") ||
        anchor->getNumResults() || !source.hasOneUse() ||
        !isHwcI8(type, true) || !isContiguousDestination(anchor.getDest(), type))
      return failure();
    auto color = source.getDefiningOp<linalg::GenericOp>();
    auto fused = source.getDefiningOp<tensor::GenerateOp>();
    Operation *producer = source.getDefiningOp();
    tensor::EmptyOp oldEmpty;
    if (color) {
      if (!matchColor(color))
        return failure();
      auto resize = color.getInputs()[0].getDefiningOp<tensor::GenerateOp>();
      if (!resize || !Q11BodyMatcher(resize).match(false))
        return failure();
      oldEmpty = color.getOutputs()[0].getDefiningOp<tensor::EmptyOp>();
    } else if (!fused || !Q11BodyMatcher(fused).match(true)) {
      return failure();
    }
    if (!canMoveProducerTo(producer, anchor))
      return failure();

    // The ABI and the existing restrict assertion prove the caller destination
    // does not alias the input tensor's buffer. Transfer (do not duplicate) that
    // assertion before introducing another tensor view. Input to_tensor is
    // never made writable or otherwise changed.
    rewriter.modifyOpInPlace(anchor, [&] { anchor->removeAttr("restrict"); });
    rewriter.setInsertionPoint(anchor);
    Value destination = rewriter.create<bufferization::ToTensorOp>(
        anchor.getLoc(), anchor.getDest(), true, true);
    Value replacement;
    if (color) {
      // Clone the existing maps, input and scalar region verbatim. Only the
      // unused tensor.empty init is replaced by the caller-owned destination.
      auto direct = cast<linalg::GenericOp>(rewriter.clone(*color.getOperation()));
      direct->setOperand(1, destination);
      replacement = direct->getResult(0);
    } else {
      SmallVector<AffineMap> maps{
          AffineMap::getMultiDimIdentityMap(3, rewriter.getContext())};
      SmallVector<utils::IteratorType> iterators(3,
                                                utils::IteratorType::parallel);
      auto direct = rewriter.create<linalg::GenericOp>(
          fused.getLoc(), TypeRange{type}, ValueRange{}, ValueRange{destination},
          maps, iterators,
          [&](OpBuilder &builder, Location loc, ValueRange) {
            IRMapping mapping;
            Block &oldBody = fused.getBody().front();
            for (unsigned axis = 0; axis != 3; ++axis) {
              Value index = builder.create<linalg::IndexOp>(loc, axis);
              mapping.map(oldBody.getArgument(axis), index);
            }
            for (Operation &scalar : oldBody.without_terminator())
              builder.clone(scalar, mapping);
            Value pixel = oldBody.getTerminator()->getOperand(0);
            builder.create<linalg::YieldOp>(loc, mapping.lookupOrDefault(pixel));
          });
      direct->setAttr("edge.resize", fused->getAttr("edge.resize"));
      direct->setAttr("edge.fused", rewriter.getUnitAttr());
      replacement = direct->getResult(0);
    }
    rewriter.replaceOp(producer, replacement);
    if (oldEmpty && oldEmpty.getResult().use_empty())
      rewriter.eraseOp(oldEmpty);
    // Keep materialize_in_destination as the side-effect anchor. One-Shot
    // Bufferize + canonicalize, not fusion alone, may remove its self-copy.
    return success();
  }
};

void registerDependentDialects(DialectRegistry &registry) {
  registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                  linalg::LinalgDialect, tensor::TensorDialect>();
}

// A snapshot-based, one-shot driver keeps nonmatches byte-for-byte unchanged
// (modulo assembly printing). Greedy rewriting would also fold/CSE/DCE unrelated
// IR, while the walk driver forbids erasing these sibling producers. None of
// these patterns can erase another candidate: legal producer bodies have no
// nested regions and legal color bodies contain only a yield.
template <typename Op, typename Pattern>
void applyOnce(ModuleOp module) {
  SmallVector<Op> candidates;
  module.walk([&](Op op) { candidates.push_back(op); });
  PatternRewriter rewriter(module.getContext());
  Pattern pattern(module.getContext());
  for (Op candidate : candidates) {
    rewriter.setInsertionPoint(candidate);
    (void)pattern.matchAndRewrite(candidate, rewriter);
  }
}

struct FuseResizeColorPass
    : PassWrapper<FuseResizeColorPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseResizeColorPass)
  StringRef getArgument() const final { return "edge-fuse-resize-color"; }
  StringRef getDescription() const final {
    return "Fuse a checked Q11 resize and HWC BGR-to-RGB permutation";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registerDependentDialects(registry);
  }
  void runOnOperation() override {
    applyOnce<linalg::GenericOp, FuseResizeColorPattern>(getOperation());
  }
};

struct PreparePreprocessDestinationPass
    : PassWrapper<PreparePreprocessDestinationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PreparePreprocessDestinationPass)
  StringRef getArgument() const final {
    return "edge-prepare-preprocess-destination";
  }
  StringRef getDescription() const final {
    return "Prepare checked preprocessing for the caller-owned destination";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registerDependentDialects(registry);
  }
  void runOnOperation() override {
    applyOnce<bufferization::MaterializeInDestinationOp,
              PrepareDestinationPattern>(getOperation());
  }
};
} // namespace

void registerPreprocessPasses() {
  PassRegistration<FuseResizeColorPass>();
  PassRegistration<PreparePreprocessDestinationPass>();
}

} // namespace edgeai
