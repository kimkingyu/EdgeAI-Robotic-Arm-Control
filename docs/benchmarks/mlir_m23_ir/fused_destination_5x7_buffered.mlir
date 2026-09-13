#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {edge.height = 5 : i64, edge.semantic_version = "linear_half_pixel_q11_v1", edge.width = 7 : i64} {
  func.func @edge_resize_color_impl(%arg0: memref<?x?x3xi8, strided<[?, 3, 1]>>, %arg1: memref<5x7x3xi8>) attributes {edge.height = 5 : i64, edge.semantic_version = "linear_half_pixel_q11_v1", edge.width = 7 : i64, llvm.emit_c_interface} {
    %c255_i32 = arith.constant 255 : i32
    %c22_i32 = arith.constant 22 : i32
    %c2097152_i32 = arith.constant 2097152 : i32
    %c14_i64 = arith.constant 14 : i64
    %c7_i64 = arith.constant 7 : i64
    %c10_i64 = arith.constant 10 : i64
    %c5_i64 = arith.constant 5 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?x?x3xi8, strided<[?, 3, 1]>>
    %dim_0 = memref.dim %arg0, %c1 : memref<?x?x3xi8, strided<[?, 3, 1]>>
    %0 = arith.index_cast %dim : index to i64
    %1 = arith.index_cast %dim_0 : index to i64
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%arg1 : memref<5x7x3xi8>) attrs =  {edge.fused, edge.resize = "linear_half_pixel_q11_v1"} {
    ^bb0(%out: i8):
      %2 = linalg.index 0 : index
      %3 = linalg.index 1 : index
      %4 = linalg.index 2 : index
      %5 = arith.subi %c2, %4 : index
      %6 = arith.index_cast %2 : index to i64
      %7 = arith.muli %6, %c2_i64 : i64
      %8 = arith.addi %7, %c1_i64 : i64
      %9 = arith.muli %8, %0 : i64
      %10 = arith.subi %9, %c5_i64 : i64
      %11 = arith.floordivsi %10, %c10_i64 : i64
      %12 = arith.muli %11, %c10_i64 : i64
      %13 = arith.subi %10, %12 : i64
      %14 = arith.muli %13, %c2048_i64 : i64
      %15 = arith.addi %14, %c5_i64 : i64
      %16 = arith.floordivsi %15, %c10_i64 : i64
      %17 = arith.subi %c2048_i64, %16 : i64
      %18 = arith.addi %11, %c1_i64 : i64
      %19 = arith.subi %0, %c1_i64 : i64
      %20 = arith.maxsi %11, %c0_i64 : i64
      %21 = arith.minsi %20, %19 : i64
      %22 = arith.index_cast %21 : i64 to index
      %23 = arith.maxsi %18, %c0_i64 : i64
      %24 = arith.minsi %23, %19 : i64
      %25 = arith.index_cast %24 : i64 to index
      %26 = arith.trunci %17 : i64 to i32
      %27 = arith.trunci %16 : i64 to i32
      %28 = arith.index_cast %3 : index to i64
      %29 = arith.muli %28, %c2_i64 : i64
      %30 = arith.addi %29, %c1_i64 : i64
      %31 = arith.muli %30, %1 : i64
      %32 = arith.subi %31, %c7_i64 : i64
      %33 = arith.floordivsi %32, %c14_i64 : i64
      %34 = arith.muli %33, %c14_i64 : i64
      %35 = arith.subi %32, %34 : i64
      %36 = arith.muli %35, %c2048_i64 : i64
      %37 = arith.addi %36, %c7_i64 : i64
      %38 = arith.floordivsi %37, %c14_i64 : i64
      %39 = arith.subi %c2048_i64, %38 : i64
      %40 = arith.addi %33, %c1_i64 : i64
      %41 = arith.subi %1, %c1_i64 : i64
      %42 = arith.maxsi %33, %c0_i64 : i64
      %43 = arith.minsi %42, %41 : i64
      %44 = arith.index_cast %43 : i64 to index
      %45 = arith.maxsi %40, %c0_i64 : i64
      %46 = arith.minsi %45, %41 : i64
      %47 = arith.index_cast %46 : i64 to index
      %48 = arith.trunci %39 : i64 to i32
      %49 = arith.trunci %38 : i64 to i32
      %50 = memref.load %arg0[%22, %44, %5] : memref<?x?x3xi8, strided<[?, 3, 1]>>
      %51 = arith.extui %50 : i8 to i32
      %52 = memref.load %arg0[%22, %47, %5] : memref<?x?x3xi8, strided<[?, 3, 1]>>
      %53 = arith.extui %52 : i8 to i32
      %54 = memref.load %arg0[%25, %44, %5] : memref<?x?x3xi8, strided<[?, 3, 1]>>
      %55 = arith.extui %54 : i8 to i32
      %56 = memref.load %arg0[%25, %47, %5] : memref<?x?x3xi8, strided<[?, 3, 1]>>
      %57 = arith.extui %56 : i8 to i32
      %58 = arith.muli %51, %48 : i32
      %59 = arith.muli %53, %49 : i32
      %60 = arith.addi %58, %59 : i32
      %61 = arith.muli %55, %48 : i32
      %62 = arith.muli %57, %49 : i32
      %63 = arith.addi %61, %62 : i32
      %64 = arith.muli %60, %26 : i32
      %65 = arith.muli %63, %27 : i32
      %66 = arith.addi %64, %65 : i32
      %67 = arith.addi %66, %c2097152_i32 : i32
      %68 = arith.shrui %67, %c22_i32 : i32
      %69 = arith.minui %68, %c255_i32 : i32
      %70 = arith.trunci %69 : i32 to i8
      linalg.yield %70 : i8
    }
    return
  }
}

