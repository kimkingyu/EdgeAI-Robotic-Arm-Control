// M2: construct the unfused, destination-style preprocessing IR with MLIR
// builders. Builder signatures follow llvmorg-20.1.8, not rolling MLIR APIs.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>

namespace {

constexpr int64_t kMaxDimension = 8192;
constexpr int64_t kCoefficientScale = 2048;
constexpr int64_t kAccumulatorBias = 2097152;
constexpr unsigned kAccumulatorShift = 22;
constexpr char kSemanticVersion[] = "linear_half_pixel_q11_v1";

struct Options {
  int64_t height = 0;
  int64_t width = 0;
  std::string functionName = "edge_resize_color_impl";
};

enum class ParseResult { Success, Help, Error };

void printUsage() {
  // Keep stdout reserved for a complete, verified MLIR module, even for help.
  llvm::errs() << "Usage: edge-preprocess-gen --height N --width N "
                  "[--function NAME]\n"
                  "  N: decimal integer in [1, 8192]\n"
                  "  NAME: [A-Za-z_][A-Za-z0-9_]* "
                  "(default: edge_resize_color_impl)\n";
}

bool isCIdentifier(llvm::StringRef name) {
  const auto isInitial = [](char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
  };
  if (name.empty() || !isInitial(name.front()))
    return false;
  for (char c : name.drop_front()) {
    if (!isInitial(c) && !(c >= '0' && c <= '9'))
      return false;
  }
  return true;
}

bool parseDimension(llvm::StringRef text, int64_t &result) {
  if (text.empty())
    return false;
  int64_t value = 0;
  for (char c : text) {
    if (c < '0' || c > '9')
      return false;
    const int64_t digit = c - '0';
    // Check before multiplying: even an arbitrarily long CLI value is safe.
    if (value > (kMaxDimension - digit) / 10)
      return false;
    value = value * 10 + digit;
  }
  if (value < 1 || value > kMaxDimension)
    return false;
  result = value;
  return true;
}

ParseResult parseOptions(int argc, char **argv, Options &options) {
  bool sawHeight = false;
  bool sawWidth = false;
  bool sawFunction = false;
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef argument(argv[i]);
    if (argument == "--help" || argument == "-h") {
      printUsage();
      return ParseResult::Help;
    }

    // Accept both --height 640 and --height=640 without interpreting IR text.
    const auto equal = argument.find('=');
    const llvm::StringRef option = argument.take_front(equal);
    bool *seen = nullptr;
    if (option == "--height")
      seen = &sawHeight;
    else if (option == "--width")
      seen = &sawWidth;
    else if (option == "--function")
      seen = &sawFunction;
    else {
      llvm::errs() << "error: unknown option '" << argument << "'\n";
      return ParseResult::Error;
    }
    if (*seen) {
      llvm::errs() << "error: duplicate option '" << option << "'\n";
      return ParseResult::Error;
    }
    *seen = true;

    llvm::StringRef value;
    if (equal != llvm::StringRef::npos) {
      value = argument.drop_front(equal + 1);
    } else {
      if (i + 1 == argc) {
        llvm::errs() << "error: missing value for '" << option << "'\n";
        return ParseResult::Error;
      }
      value = argv[++i];
    }

    if (option == "--function") {
      if (!isCIdentifier(value)) {
        llvm::errs() << "error: --function requires a C identifier "
                        "[A-Za-z_][A-Za-z0-9_]*\n";
        return ParseResult::Error;
      }
      options.functionName = value.str();
    } else {
      int64_t &dimension = option == "--height" ? options.height : options.width;
      if (!parseDimension(value, dimension)) {
        llvm::errs() << "error: " << option
                     << " requires a decimal integer in [1, 8192]\n";
        return ParseResult::Error;
      }
    }
  }
  if (!sawHeight || !sawWidth) {
    llvm::errs() << "error: both --height and --width are required\n";
    return ParseResult::Error;
  }
  return ParseResult::Success;
}

mlir::Value integerConstant(mlir::OpBuilder &builder, mlir::Location loc,
                           int64_t value, unsigned width) {
  return builder.create<mlir::arith::ConstantIntOp>(loc, value, width);
}

struct AxisInterpolation {
  mlir::Value lowerIndex;
  mlir::Value upperIndex;
  mlir::Value lowerWeight;
  mlir::Value upperWeight;
};

AxisInterpolation buildAxis(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value outputIndex, mlir::Value sourceExtent,
                            int64_t destinationExtent) {
  using namespace mlir;
  Value zero = integerConstant(builder, loc, 0, 64);
  Value one = integerConstant(builder, loc, 1, 64);
  Value two = integerConstant(builder, loc, 2, 64);
  Value scale = integerConstant(builder, loc, kCoefficientScale, 64);
  Value destination = integerConstant(builder, loc, destinationExtent, 64);
  Value denominator = integerConstant(builder, loc, 2 * destinationExtent, 64);
  Value output = builder.create<arith::IndexCastOp>(
      loc, builder.getI64Type(), outputIndex);

  // Half-pixel coordinates, including negative border coordinates. Signed
  // FLOOR division is essential: truncation toward zero changes the border.
  Value twiceOutput = builder.create<arith::MulIOp>(loc, two, output);
  Value oddOutput = builder.create<arith::AddIOp>(loc, twiceOutput, one);
  Value scaledOutput =
      builder.create<arith::MulIOp>(loc, oddOutput, sourceExtent);
  Value numerator =
      builder.create<arith::SubIOp>(loc, scaledOutput, destination);
  Value lower = builder.create<arith::FloorDivSIOp>(loc, numerator, denominator);
  Value lowerTimesDenominator =
      builder.create<arith::MulIOp>(loc, lower, denominator);
  Value remainder =
      builder.create<arith::SubIOp>(loc, numerator, lowerTimesDenominator);
  Value scaledRemainder = builder.create<arith::MulIOp>(loc, remainder, scale);
  Value roundedRemainder =
      builder.create<arith::AddIOp>(loc, scaledRemainder, destination);
  Value upperWeight64 =
      builder.create<arith::FloorDivSIOp>(loc, roundedRemainder, denominator);
  Value lowerWeight64 =
      builder.create<arith::SubIOp>(loc, scale, upperWeight64);

  // Compute weights from the UNCLAMPED coordinate. Clamp both sample indices
  // independently, so S=1 and both sides of the image replicate correctly.
  Value upper = builder.create<arith::AddIOp>(loc, lower, one);
  Value last = builder.create<arith::SubIOp>(loc, sourceExtent, one);
  const auto clampIndex = [&](Value index) -> Value {
    Value nonnegative = builder.create<arith::MaxSIOp>(loc, index, zero);
    Value clamped = builder.create<arith::MinSIOp>(loc, nonnegative, last);
    return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), clamped);
  };
  Value lowerIndex = clampIndex(lower);
  Value upperIndex = clampIndex(upper);
  Value lowerWeight =
      builder.create<arith::TruncIOp>(loc, builder.getI32Type(), lowerWeight64);
  Value upperWeight =
      builder.create<arith::TruncIOp>(loc, builder.getI32Type(), upperWeight64);
  return {lowerIndex, upperIndex, lowerWeight, upperWeight};
}

mlir::Value buildPixel(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Value source, mlir::Value sourceHeight,
                      mlir::Value sourceWidth, mlir::ValueRange indices,
                      const Options &options) {
  using namespace mlir;
  AxisInterpolation y =
      buildAxis(builder, loc, indices[0], sourceHeight, options.height);
  AxisInterpolation x =
      buildAxis(builder, loc, indices[1], sourceWidth, options.width);
  const auto sample = [&](Value row, Value column) -> Value {
    Value byte = builder.create<tensor::ExtractOp>(
        loc, builder.getI8Type(), source, ValueRange{row, column, indices[2]});
    // Signless i8 stores unsigned pixels. Do not sign-extend values >= 128.
    return builder.create<arith::ExtUIOp>(loc, builder.getI32Type(), byte);
  };
  Value p00 = sample(y.lowerIndex, x.lowerIndex);
  Value p01 = sample(y.lowerIndex, x.upperIndex);
  Value p10 = sample(y.upperIndex, x.lowerIndex);
  Value p11 = sample(y.upperIndex, x.upperIndex);
  const auto weightedRow = [&](Value left, Value right) -> Value {
    Value leftTerm = builder.create<arith::MulIOp>(loc, left, x.lowerWeight);
    Value rightTerm = builder.create<arith::MulIOp>(loc, right, x.upperWeight);
    return builder.create<arith::AddIOp>(loc, leftTerm, rightTerm);
  };
  Value row0 = weightedRow(p00, p01);
  Value row1 = weightedRow(p10, p11);
  Value upperTerm = builder.create<arith::MulIOp>(loc, row0, y.lowerWeight);
  Value lowerTerm = builder.create<arith::MulIOp>(loc, row1, y.upperWeight);
  Value sum = builder.create<arith::AddIOp>(loc, upperTerm, lowerTerm);

  // Single final rounding only. sum <= 255 * 2048^2 = 1069547520;
  // adding the Q22 bias still fits signed i32 as well as unsigned i32.
  Value bias = integerConstant(builder, loc, kAccumulatorBias, 32);
  Value shift = integerConstant(builder, loc, kAccumulatorShift, 32);
  Value maximum = integerConstant(builder, loc, 255, 32);
  Value biased = builder.create<arith::AddIOp>(loc, sum, bias);
  Value rounded = builder.create<arith::ShRUIOp>(loc, biased, shift);
  Value saturated = builder.create<arith::MinUIOp>(loc, rounded, maximum);
  return builder.create<arith::TruncIOp>(loc, builder.getI8Type(), saturated);
}

void addAuditAttributes(mlir::OpBuilder &builder, mlir::Operation *operation,
                        const Options &options) {
  // Audit metadata, not proof: downstream passes must inspect actual IR.
  operation->setAttr("edge.semantic_version",
                     builder.getStringAttr(kSemanticVersion));
  operation->setAttr("edge.height", builder.getI64IntegerAttr(options.height));
  operation->setAttr("edge.width", builder.getI64IntegerAttr(options.width));
}

mlir::OwningOpRef<mlir::ModuleOp> buildModule(mlir::MLIRContext &context,
                                             const Options &options) {
  using namespace mlir;
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  OwningOpRef<ModuleOp> module(ModuleOp::create(loc));
  addAuditAttributes(builder, module->getOperation(), options);
  builder.setInsertionPointToEnd(module->getBody());

  const llvm::SmallVector<int64_t, 3> outputShape{options.height, options.width, 3};
  Type byte = builder.getI8Type();
  auto sourceLayout =
      StridedLayoutAttr::get(&context, 0, {ShapedType::kDynamic, 3, 1});
  auto sourceType = MemRefType::get(
      {ShapedType::kDynamic, ShapedType::kDynamic, 3}, byte, sourceLayout);
  auto destinationType = MemRefType::get(outputShape, byte);
  auto outputTensorType = RankedTensorType::get(outputShape, byte);
  auto functionType = builder.getFunctionType({sourceType, destinationType}, {});
  auto function =
      builder.create<func::FuncOp>(loc, options.functionName, functionType);
  function->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  addAuditAttributes(builder, function.getOperation(), options);
  Block *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  // Source H/W are validated by the caller to be in [1, 8192]; the caller also
  // owns a disjoint, writable destination. The source remains read-only.
  Value source = builder.create<bufferization::ToTensorOp>(
      loc, entry->getArgument(0), /*restrict=*/true, /*writeable=*/false);
  Value heightIndex = builder.create<tensor::DimOp>(loc, source, int64_t{0});
  Value widthIndex = builder.create<tensor::DimOp>(loc, source, int64_t{1});
  Value sourceHeight =
      builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), heightIndex);
  Value sourceWidth =
      builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), widthIndex);

  auto resized = builder.create<tensor::GenerateOp>(
      loc, outputTensorType, ValueRange{},
      [&](OpBuilder &bodyBuilder, Location bodyLoc, ValueRange indices) {
        // Region arguments are (oy, ox, c), all index; source is captured.
        Value pixel = buildPixel(bodyBuilder, bodyLoc, source, sourceHeight,
                                 sourceWidth, indices, options);
        bodyBuilder.create<tensor::YieldOp>(bodyLoc, pixel);
      });
  resized->setAttr("edge.resize", builder.getStringAttr(kSemanticVersion));

  // BGR -> RGB is encoded in the input indexing map, not merely in a label.
  // The unused destination element argument must never be read.
  Value empty = builder.create<tensor::EmptyOp>(loc, outputShape, byte);
  AffineExpr d0 = builder.getAffineDimExpr(0);
  AffineExpr d1 = builder.getAffineDimExpr(1);
  AffineExpr d2 = builder.getAffineDimExpr(2);
  AffineMap colorMap = AffineMap::get(
      3, 0, {d0, d1, builder.getAffineConstantExpr(2) - d2}, &context);
  AffineMap identityMap = builder.getMultiDimIdentityMap(3);
  const llvm::SmallVector<AffineMap, 2> maps{colorMap, identityMap};
  const llvm::SmallVector<utils::IteratorType, 3> iterators(
      3, utils::IteratorType::parallel);
  auto rgb = builder.create<linalg::GenericOp>(
      loc, TypeRange{outputTensorType}, ValueRange{resized.getResult()},
      ValueRange{empty}, maps, iterators,
      [](OpBuilder &bodyBuilder, Location bodyLoc, ValueRange arguments) {
        bodyBuilder.create<linalg::YieldOp>(bodyLoc, arguments[0]);
      });
  rgb->setAttr("edge.color", builder.getStringAttr("bgr_to_rgb"));

  // LLVM 20.1.8's (source, dest) builder infers zero results for a memref dest.
  // Do NOT introduce a second to_tensor here: destination rewriting must be
  // able to transfer this restrict promise to its own destination to_tensor.
  auto materialized = builder.create<bufferization::MaterializeInDestinationOp>(
      loc, rgb.getResult(0), entry->getArgument(1));
  materialized->setAttr("restrict", builder.getUnitAttr());
  materialized->setAttr("writable", builder.getUnitAttr());
  builder.create<func::ReturnOp>(loc);
  return module;
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  Options options;
  const ParseResult parsed = parseOptions(argc, argv, options);
  if (parsed == ParseResult::Help)
    return 0;
  if (parsed == ParseResult::Error) {
    printUsage();
    return 2;
  }

  mlir::MLIRContext context;
  context.loadDialect<mlir::arith::ArithDialect,
                      mlir::bufferization::BufferizationDialect,
                      mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                      mlir::tensor::TensorDialect>();
  mlir::ScopedDiagnosticHandler diagnosticHandler(
      &context, [](mlir::Diagnostic &diagnostic) {
        diagnostic.print(llvm::errs());
        llvm::errs() << '\n';
        return mlir::success();
      });
  auto module = buildModule(context, options);
  if (mlir::failed(mlir::verify(module->getOperation()))) {
    llvm::errs() << "error: generated preprocessing module failed verification\n";
    return 1;
  }
  module->print(llvm::outs());
  llvm::outs() << '\n';
  llvm::outs().flush();
  if (llvm::outs().has_error()) {
    llvm::errs() << "error: failed to write MLIR to stdout: "
                 << llvm::outs().error().message() << '\n';
    llvm::outs().clear_error();
    return 1;
  }
  return 0;
}
