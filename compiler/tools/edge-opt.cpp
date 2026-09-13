// Independent optimizer; project transformations precede pinned upstream lowering.
#include "edgeai/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/SubsetOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char** argv) {
  mlir::registerAllPasses();
  edgeai::registerPreprocessPasses();
  edgeai::registerScheduleExperimentPasses();
  mlir::DialectRegistry dialects;
  dialects.insert<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                  mlir::bufferization::BufferizationDialect, mlir::cf::ControlFlowDialect,
                  mlir::func::FuncDialect, mlir::LLVM::LLVMDialect,
                  mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                  mlir::vector::VectorDialect>();
  mlir::arith::registerBufferizableOpInterfaceExternalModels(dialects);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(dialects);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(dialects);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(dialects);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(dialects);
  mlir::vector::registerBufferizableOpInterfaceExternalModels(dialects);
  // Subset (slice/transfer) models are needed once vectorized tensor IR is
  // bufferized; without them 20.1.8 aborts on a promised-but-missing interface.
  mlir::vector::registerSubsetOpInterfaceExternalModels(dialects);
  mlir::tensor::registerSubsetOpInterfaceExternalModels(dialects);
  mlir::linalg::registerSubsetOpInterfaceExternalModels(dialects);
  mlir::linalg::registerTilingInterfaceExternalModels(dialects);
  mlir::tensor::registerTilingInterfaceExternalModels(dialects);
  // Tiled scf loops need value bounds during bufferization and canonicalization.
  mlir::affine::registerValueBoundsOpInterfaceExternalModels(dialects);
  mlir::arith::registerValueBoundsOpInterfaceExternalModels(dialects);
  mlir::linalg::registerValueBoundsOpInterfaceExternalModels(dialects);
  mlir::memref::registerValueBoundsOpInterfaceExternalModels(dialects);
  mlir::scf::registerValueBoundsOpInterfaceExternalModels(dialects);
  mlir::tensor::registerValueBoundsOpInterfaceExternalModels(dialects);
  mlir::vector::registerValueBoundsOpInterfaceExternalModels(dialects);
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "EdgeAI preprocessing optimizer\n", dialects));
}
