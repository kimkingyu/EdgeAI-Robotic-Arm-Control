cmake_minimum_required(VERSION 3.20)
project(EdgeAIPreprocessCompiler LANGUAGES C CXX)

# Independent developer tool: the existing robot build does not include this.
find_package(MLIR REQUIRED CONFIG)
if(NOT LLVM_PACKAGE_VERSION VERSION_EQUAL "20.1.8")
  message(FATAL_ERROR "Use the locked LLVM/MLIR 20.1.8 toolchain, not ${LLVM_PACKAGE_VERSION}")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(LLVM_RUNTIME_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/bin")
set(LLVM_LIBRARY_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/lib")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

get_property(edge_dialects GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(edge_conversions GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
add_llvm_executable(edge-opt tools/edge-opt.cpp lib/Transforms/PreprocessPasses.cpp
  lib/Transforms/ScheduleExperiments.cpp)
target_include_directories(edge-opt PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
llvm_update_compile_flags(edge-opt)
target_link_libraries(edge-opt PRIVATE ${edge_dialects} ${edge_conversions} MLIROptLib
  MLIRArithTransforms MLIRBufferizationTransforms MLIRLinalgTransforms
  MLIRSCFTransforms MLIRTensorTransforms MLIRTransformUtils
  MLIRTilingInterface MLIRVectorDialect)
mlir_check_all_link_libraries(edge-opt)

add_llvm_executable(edge-preprocess-gen tools/edge-preprocess-gen.cpp)
llvm_update_compile_flags(edge-preprocess-gen)
target_link_libraries(edge-preprocess-gen PRIVATE MLIRArithDialect MLIRBufferizationDialect
  MLIRFuncDialect MLIRLinalgDialect MLIRMemRefDialect MLIRTensorDialect MLIRIR MLIRSupport)
mlir_check_all_link_libraries(edge-preprocess-gen)
