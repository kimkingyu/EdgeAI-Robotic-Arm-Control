#map = affine_map<(d0, d1, d2) -> (d0, d1, -d2 + 2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
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
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?x?x3xi8, strided<[?, 3, 1]>>
    %dim_0 = memref.dim %arg0, %c1 : memref<?x?x3xi8, strided<[?, 3, 1]>>
    %0 = arith.index_cast %dim : index to i64
    %1 = arith.index_cast %dim_0 : index to i64
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<5x7x3xi8>
    linalg.map outs(%alloc : memref<5x7x3xi8>)
      () {
        %2 = linalg.index 0 : index
        %3 = linalg.index 1 : index
        %4 = linalg.index 2 : index
        %5 = arith.index_cast %2 : index to i64
        %6 = arith.muli %5, %c2_i64 : i64
        %7 = arith.addi %6, %c1_i64 : i64
        %8 = arith.muli %7, %0 : i64
        %9 = arith.subi %8, %c5_i64 : i64
        %10 = arith.floordivsi %9, %c10_i64 : i64
        %11 = arith.muli %10, %c10_i64 : i64
        %12 = arith.subi %9, %11 : i64
        %13 = arith.muli %12, %c2048_i64 : i64
        %14 = arith.addi %13, %c5_i64 : i64
        %15 = arith.floordivsi %14, %c10_i64 : i64
        %16 = arith.subi %c2048_i64, %15 : i64
        %17 = arith.addi %10, %c1_i64 : i64
        %18 = arith.subi %0, %c1_i64 : i64
        %19 = arith.maxsi %10, %c0_i64 : i64
        %20 = arith.minsi %19, %18 : i64
        %21 = arith.index_cast %20 : i64 to index
        %22 = arith.maxsi %17, %c0_i64 : i64
        %23 = arith.minsi %22, %18 : i64
        %24 = arith.index_cast %23 : i64 to index
        %25 = arith.trunci %16 : i64 to i32
        %26 = arith.trunci %15 : i64 to i32
        %27 = arith.index_cast %3 : index to i64
        %28 = arith.muli %27, %c2_i64 : i64
        %29 = arith.addi %28, %c1_i64 : i64
        %30 = arith.muli %29, %1 : i64
        %31 = arith.subi %30, %c7_i64 : i64
        %32 = arith.floordivsi %31, %c14_i64 : i64
        %33 = arith.muli %32, %c14_i64 : i64
        %34 = arith.subi %31, %33 : i64
        %35 = arith.muli %34, %c2048_i64 : i64
        %36 = arith.addi %35, %c7_i64 : i64
        %37 = arith.floordivsi %36, %c14_i64 : i64
        %38 = arith.subi %c2048_i64, %37 : i64
        %39 = arith.addi %32, %c1_i64 : i64
        %40 = arith.subi %1, %c1_i64 : i64
        %41 = arith.maxsi %32, %c0_i64 : i64
        %42 = arith.minsi %41, %40 : i64
        %43 = arith.index_cast %42 : i64 to index
        %44 = arith.maxsi %39, %c0_i64 : i64
        %45 = arith.minsi %44, %40 : i64
        %46 = arith.index_cast %45 : i64 to index
        %47 = arith.trunci %38 : i64 to i32
        %48 = arith.trunci %37 : i64 to i32
        %49 = memref.load %arg0[%21, %43, %4] : memref<?x?x3xi8, strided<[?, 3, 1]>>
        %50 = arith.extui %49 : i8 to i32
        %51 = memref.load %arg0[%21, %46, %4] : memref<?x?x3xi8, strided<[?, 3, 1]>>
        %52 = arith.extui %51 : i8 to i32
        %53 = memref.load %arg0[%24, %43, %4] : memref<?x?x3xi8, strided<[?, 3, 1]>>
        %54 = arith.extui %53 : i8 to i32
        %55 = memref.load %arg0[%24, %46, %4] : memref<?x?x3xi8, strided<[?, 3, 1]>>
        %56 = arith.extui %55 : i8 to i32
        %57 = arith.muli %50, %47 : i32
        %58 = arith.muli %52, %48 : i32
        %59 = arith.addi %57, %58 : i32
        %60 = arith.muli %54, %47 : i32
        %61 = arith.muli %56, %48 : i32
        %62 = arith.addi %60, %61 : i32
        %63 = arith.muli %59, %25 : i32
        %64 = arith.muli %62, %26 : i32
        %65 = arith.addi %63, %64 : i32
        %66 = arith.addi %65, %c2097152_i32 : i32
        %67 = arith.shrui %66, %c22_i32 : i32
        %68 = arith.minui %67, %c255_i32 : i32
        %69 = arith.trunci %68 : i32 to i8
        linalg.yield %69 : i8
      }
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<5x7x3xi8>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc : memref<5x7x3xi8>) outs(%alloc_1 : memref<5x7x3xi8>) attrs =  {edge.color = "bgr_to_rgb"} {
    ^bb0(%in: i8, %out: i8):
      linalg.yield %in : i8
    }
    memref.copy %alloc_1, %arg1 : memref<5x7x3xi8> to memref<5x7x3xi8>
    memref.dealloc %alloc : memref<5x7x3xi8>
    memref.dealloc %alloc_1 : memref<5x7x3xi8>
    return
  }
}

