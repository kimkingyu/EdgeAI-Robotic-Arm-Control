#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {edge.height = 5 : i64, edge.semantic_version = "linear_half_pixel_q11_v1", edge.width = 7 : i64} {
  func.func @edge_resize_color_impl(%arg0: memref<?x?x3xi8, strided<[?, 3, 1]>>, %arg1: memref<5x7x3xi8>) attributes {edge.height = 5 : i64, edge.semantic_version = "linear_half_pixel_q11_v1", edge.width = 7 : i64, llvm.emit_c_interface} {
    %0 = bufferization.to_tensor %arg0 restrict : memref<?x?x3xi8, strided<[?, 3, 1]>> to tensor<?x?x3xi8>
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %0, %c0 : tensor<?x?x3xi8>
    %c1 = arith.constant 1 : index
    %dim_0 = tensor.dim %0, %c1 : tensor<?x?x3xi8>
    %1 = arith.index_cast %dim : index to i64
    %2 = arith.index_cast %dim_0 : index to i64
    %3 = tensor.empty() : tensor<5x7x3xi8>
    %4 = bufferization.to_tensor %arg1 restrict writable : memref<5x7x3xi8> to tensor<5x7x3xi8>
    %5 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%4 : tensor<5x7x3xi8>) attrs =  {edge.fused, edge.resize = "linear_half_pixel_q11_v1"} {
    ^bb0(%out: i8):
      %6 = linalg.index 0 : index
      %7 = linalg.index 1 : index
      %8 = linalg.index 2 : index
      %c2 = arith.constant 2 : index
      %9 = arith.subi %c2, %8 : index
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %c2048_i64 = arith.constant 2048 : i64
      %c5_i64 = arith.constant 5 : i64
      %c10_i64 = arith.constant 10 : i64
      %10 = arith.index_cast %6 : index to i64
      %11 = arith.muli %c2_i64, %10 : i64
      %12 = arith.addi %11, %c1_i64 : i64
      %13 = arith.muli %12, %1 : i64
      %14 = arith.subi %13, %c5_i64 : i64
      %15 = arith.floordivsi %14, %c10_i64 : i64
      %16 = arith.muli %15, %c10_i64 : i64
      %17 = arith.subi %14, %16 : i64
      %18 = arith.muli %17, %c2048_i64 : i64
      %19 = arith.addi %18, %c5_i64 : i64
      %20 = arith.floordivsi %19, %c10_i64 : i64
      %21 = arith.subi %c2048_i64, %20 : i64
      %22 = arith.addi %15, %c1_i64 : i64
      %23 = arith.subi %1, %c1_i64 : i64
      %24 = arith.maxsi %15, %c0_i64 : i64
      %25 = arith.minsi %24, %23 : i64
      %26 = arith.index_cast %25 : i64 to index
      %27 = arith.maxsi %22, %c0_i64 : i64
      %28 = arith.minsi %27, %23 : i64
      %29 = arith.index_cast %28 : i64 to index
      %30 = arith.trunci %21 : i64 to i32
      %31 = arith.trunci %20 : i64 to i32
      %c0_i64_1 = arith.constant 0 : i64
      %c1_i64_2 = arith.constant 1 : i64
      %c2_i64_3 = arith.constant 2 : i64
      %c2048_i64_4 = arith.constant 2048 : i64
      %c7_i64 = arith.constant 7 : i64
      %c14_i64 = arith.constant 14 : i64
      %32 = arith.index_cast %7 : index to i64
      %33 = arith.muli %c2_i64_3, %32 : i64
      %34 = arith.addi %33, %c1_i64_2 : i64
      %35 = arith.muli %34, %2 : i64
      %36 = arith.subi %35, %c7_i64 : i64
      %37 = arith.floordivsi %36, %c14_i64 : i64
      %38 = arith.muli %37, %c14_i64 : i64
      %39 = arith.subi %36, %38 : i64
      %40 = arith.muli %39, %c2048_i64_4 : i64
      %41 = arith.addi %40, %c7_i64 : i64
      %42 = arith.floordivsi %41, %c14_i64 : i64
      %43 = arith.subi %c2048_i64_4, %42 : i64
      %44 = arith.addi %37, %c1_i64_2 : i64
      %45 = arith.subi %2, %c1_i64_2 : i64
      %46 = arith.maxsi %37, %c0_i64_1 : i64
      %47 = arith.minsi %46, %45 : i64
      %48 = arith.index_cast %47 : i64 to index
      %49 = arith.maxsi %44, %c0_i64_1 : i64
      %50 = arith.minsi %49, %45 : i64
      %51 = arith.index_cast %50 : i64 to index
      %52 = arith.trunci %43 : i64 to i32
      %53 = arith.trunci %42 : i64 to i32
      %extracted = tensor.extract %0[%26, %48, %9] : tensor<?x?x3xi8>
      %54 = arith.extui %extracted : i8 to i32
      %extracted_5 = tensor.extract %0[%26, %51, %9] : tensor<?x?x3xi8>
      %55 = arith.extui %extracted_5 : i8 to i32
      %extracted_6 = tensor.extract %0[%29, %48, %9] : tensor<?x?x3xi8>
      %56 = arith.extui %extracted_6 : i8 to i32
      %extracted_7 = tensor.extract %0[%29, %51, %9] : tensor<?x?x3xi8>
      %57 = arith.extui %extracted_7 : i8 to i32
      %58 = arith.muli %54, %52 : i32
      %59 = arith.muli %55, %53 : i32
      %60 = arith.addi %58, %59 : i32
      %61 = arith.muli %56, %52 : i32
      %62 = arith.muli %57, %53 : i32
      %63 = arith.addi %61, %62 : i32
      %64 = arith.muli %60, %30 : i32
      %65 = arith.muli %63, %31 : i32
      %66 = arith.addi %64, %65 : i32
      %c2097152_i32 = arith.constant 2097152 : i32
      %c22_i32 = arith.constant 22 : i32
      %c255_i32 = arith.constant 255 : i32
      %67 = arith.addi %66, %c2097152_i32 : i32
      %68 = arith.shrui %67, %c22_i32 : i32
      %69 = arith.minui %68, %c255_i32 : i32
      %70 = arith.trunci %69 : i32 to i8
      linalg.yield %70 : i8
    } -> tensor<5x7x3xi8>
    bufferization.materialize_in_destination %5 in writable %arg1 : (tensor<5x7x3xi8>, memref<5x7x3xi8>) -> ()
    return
  }
}

