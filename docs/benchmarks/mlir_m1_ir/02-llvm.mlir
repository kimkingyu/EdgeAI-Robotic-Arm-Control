module {
  llvm.func @edge_color_smoke_impl(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: !llvm.ptr, %arg10: !llvm.ptr, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %1 = llvm.insertvalue %arg9, %0[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2 = llvm.insertvalue %arg10, %1[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %3 = llvm.insertvalue %arg11, %2[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %4 = llvm.insertvalue %arg12, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %5 = llvm.insertvalue %arg15, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %6 = llvm.insertvalue %arg13, %5[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %7 = llvm.insertvalue %arg16, %6[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %8 = llvm.insertvalue %arg14, %7[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %9 = llvm.insertvalue %arg17, %8[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %10 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %11 = llvm.insertvalue %arg0, %10[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %12 = llvm.insertvalue %arg1, %11[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %13 = llvm.insertvalue %arg2, %12[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %14 = llvm.insertvalue %arg3, %13[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %15 = llvm.insertvalue %arg6, %14[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %16 = llvm.insertvalue %arg4, %15[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %17 = llvm.insertvalue %arg7, %16[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %18 = llvm.insertvalue %arg5, %17[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %19 = llvm.insertvalue %arg8, %18[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %20 = llvm.mlir.constant(0 : index) : i64
    %21 = llvm.mlir.constant(1 : index) : i64
    %22 = llvm.mlir.constant(2 : index) : i64
    %23 = llvm.extractvalue %19[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %24 = llvm.extractvalue %19[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.br ^bb1(%20 : i64)
  ^bb1(%25: i64):  // 2 preds: ^bb0, ^bb5
    %26 = llvm.icmp "slt" %25, %23 : i64
    llvm.cond_br %26, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%20 : i64)
  ^bb3(%27: i64):  // 2 preds: ^bb2, ^bb4
    %28 = llvm.icmp "slt" %27, %24 : i64
    llvm.cond_br %28, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %29 = llvm.extractvalue %19[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %30 = llvm.extractvalue %19[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %31 = llvm.mul %25, %30 : i64
    %32 = llvm.mlir.constant(3 : index) : i64
    %33 = llvm.mul %27, %32 : i64
    %34 = llvm.add %31, %33 : i64
    %35 = llvm.add %34, %20 : i64
    %36 = llvm.getelementptr %29[%35] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %37 = llvm.load %36 : !llvm.ptr -> i8
    %38 = llvm.extractvalue %19[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %39 = llvm.extractvalue %19[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %40 = llvm.mul %25, %39 : i64
    %41 = llvm.mlir.constant(3 : index) : i64
    %42 = llvm.mul %27, %41 : i64
    %43 = llvm.add %40, %42 : i64
    %44 = llvm.add %43, %21 : i64
    %45 = llvm.getelementptr %38[%44] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %46 = llvm.load %45 : !llvm.ptr -> i8
    %47 = llvm.extractvalue %19[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %48 = llvm.extractvalue %19[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %49 = llvm.mul %25, %48 : i64
    %50 = llvm.mlir.constant(3 : index) : i64
    %51 = llvm.mul %27, %50 : i64
    %52 = llvm.add %49, %51 : i64
    %53 = llvm.add %52, %22 : i64
    %54 = llvm.getelementptr %47[%53] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %55 = llvm.load %54 : !llvm.ptr -> i8
    %56 = llvm.extractvalue %9[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %57 = llvm.extractvalue %9[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %58 = llvm.mul %25, %57 : i64
    %59 = llvm.mlir.constant(3 : index) : i64
    %60 = llvm.mul %27, %59 : i64
    %61 = llvm.add %58, %60 : i64
    %62 = llvm.add %61, %20 : i64
    %63 = llvm.getelementptr %56[%62] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %55, %63 : i8, !llvm.ptr
    %64 = llvm.extractvalue %9[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %65 = llvm.extractvalue %9[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %66 = llvm.mul %25, %65 : i64
    %67 = llvm.mlir.constant(3 : index) : i64
    %68 = llvm.mul %27, %67 : i64
    %69 = llvm.add %66, %68 : i64
    %70 = llvm.add %69, %21 : i64
    %71 = llvm.getelementptr %64[%70] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %46, %71 : i8, !llvm.ptr
    %72 = llvm.extractvalue %9[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %73 = llvm.extractvalue %9[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %74 = llvm.mul %25, %73 : i64
    %75 = llvm.mlir.constant(3 : index) : i64
    %76 = llvm.mul %27, %75 : i64
    %77 = llvm.add %74, %76 : i64
    %78 = llvm.add %77, %22 : i64
    %79 = llvm.getelementptr %72[%78] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %37, %79 : i8, !llvm.ptr
    %80 = llvm.add %27, %21 : i64
    llvm.br ^bb3(%80 : i64)
  ^bb5:  // pred: ^bb3
    %81 = llvm.add %25, %21 : i64
    llvm.br ^bb1(%81 : i64)
  ^bb6:  // pred: ^bb1
    llvm.return
  }
  llvm.func @_mlir_ciface_edge_color_smoke_impl(%arg0: !llvm.ptr, %arg1: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %6 = llvm.extractvalue %0[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %7 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %8 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %9 = llvm.extractvalue %0[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %10 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %11 = llvm.extractvalue %10[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %12 = llvm.extractvalue %10[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %13 = llvm.extractvalue %10[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %14 = llvm.extractvalue %10[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %15 = llvm.extractvalue %10[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %16 = llvm.extractvalue %10[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %17 = llvm.extractvalue %10[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %18 = llvm.extractvalue %10[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %19 = llvm.extractvalue %10[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.call @edge_color_smoke_impl(%1, %2, %3, %4, %5, %6, %7, %8, %9, %11, %12, %13, %14, %15, %16, %17, %18, %19) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> ()
    llvm.return
  }
}

