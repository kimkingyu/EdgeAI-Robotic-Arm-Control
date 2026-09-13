; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define void @edge_color_smoke_impl(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, i64 %7, i64 %8, ptr %9, ptr %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15, i64 %16, i64 %17) {
  %19 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %9, 0
  %20 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %19, ptr %10, 1
  %21 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %20, i64 %11, 2
  %22 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %21, i64 %12, 3, 0
  %23 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %22, i64 %15, 4, 0
  %24 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %23, i64 %13, 3, 1
  %25 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %24, i64 %16, 4, 1
  %26 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %25, i64 %14, 3, 2
  %27 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %26, i64 %17, 4, 2
  %28 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } undef, ptr %0, 0
  %29 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %28, ptr %1, 1
  %30 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %29, i64 %2, 2
  %31 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %30, i64 %3, 3, 0
  %32 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %31, i64 %6, 4, 0
  %33 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %32, i64 %4, 3, 1
  %34 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %33, i64 %7, 4, 1
  %35 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %34, i64 %5, 3, 2
  %36 = insertvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %35, i64 %8, 4, 2
  %37 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %36, 3, 0
  %38 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %36, 3, 1
  br label %39

39:                                               ; preds = %93, %18
  %40 = phi i64 [ %94, %93 ], [ 0, %18 ]
  %41 = icmp slt i64 %40, %37
  br i1 %41, label %42, label %95

42:                                               ; preds = %39
  br label %43

43:                                               ; preds = %46, %42
  %44 = phi i64 [ %92, %46 ], [ 0, %42 ]
  %45 = icmp slt i64 %44, %38
  br i1 %45, label %46, label %93

46:                                               ; preds = %43
  %47 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %36, 1
  %48 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %36, 4, 0
  %49 = mul i64 %40, %48
  %50 = mul i64 %44, 3
  %51 = add i64 %49, %50
  %52 = add i64 %51, 0
  %53 = getelementptr i8, ptr %47, i64 %52
  %54 = load i8, ptr %53, align 1
  %55 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %36, 1
  %56 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %36, 4, 0
  %57 = mul i64 %40, %56
  %58 = mul i64 %44, 3
  %59 = add i64 %57, %58
  %60 = add i64 %59, 1
  %61 = getelementptr i8, ptr %55, i64 %60
  %62 = load i8, ptr %61, align 1
  %63 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %36, 1
  %64 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %36, 4, 0
  %65 = mul i64 %40, %64
  %66 = mul i64 %44, 3
  %67 = add i64 %65, %66
  %68 = add i64 %67, 2
  %69 = getelementptr i8, ptr %63, i64 %68
  %70 = load i8, ptr %69, align 1
  %71 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %27, 1
  %72 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %27, 4, 0
  %73 = mul i64 %40, %72
  %74 = mul i64 %44, 3
  %75 = add i64 %73, %74
  %76 = add i64 %75, 0
  %77 = getelementptr i8, ptr %71, i64 %76
  store i8 %70, ptr %77, align 1
  %78 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %27, 1
  %79 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %27, 4, 0
  %80 = mul i64 %40, %79
  %81 = mul i64 %44, 3
  %82 = add i64 %80, %81
  %83 = add i64 %82, 1
  %84 = getelementptr i8, ptr %78, i64 %83
  store i8 %62, ptr %84, align 1
  %85 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %27, 1
  %86 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %27, 4, 0
  %87 = mul i64 %40, %86
  %88 = mul i64 %44, 3
  %89 = add i64 %87, %88
  %90 = add i64 %89, 2
  %91 = getelementptr i8, ptr %85, i64 %90
  store i8 %54, ptr %91, align 1
  %92 = add i64 %44, 1
  br label %43

93:                                               ; preds = %43
  %94 = add i64 %40, 1
  br label %39

95:                                               ; preds = %39
  ret void
}

define void @_mlir_ciface_edge_color_smoke_impl(ptr %0, ptr %1) {
  %3 = load { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %0, align 8
  %4 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %3, 0
  %5 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %3, 1
  %6 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %3, 2
  %7 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %3, 3, 0
  %8 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %3, 3, 1
  %9 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %3, 3, 2
  %10 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %3, 4, 0
  %11 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %3, 4, 1
  %12 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %3, 4, 2
  %13 = load { ptr, ptr, i64, [3 x i64], [3 x i64] }, ptr %1, align 8
  %14 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %13, 0
  %15 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %13, 1
  %16 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %13, 2
  %17 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %13, 3, 0
  %18 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %13, 3, 1
  %19 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %13, 3, 2
  %20 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %13, 4, 0
  %21 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %13, 4, 1
  %22 = extractvalue { ptr, ptr, i64, [3 x i64], [3 x i64] } %13, 4, 2
  call void @edge_color_smoke_impl(ptr %4, ptr %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, ptr %14, ptr %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20, i64 %21, i64 %22)
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
