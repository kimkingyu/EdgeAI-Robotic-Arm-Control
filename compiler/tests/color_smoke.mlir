// M1 only: BGR -> RGB with caller-owned strided buffers; no resize or allocation.
module {
  func.func @edge_color_smoke_impl(
      %src: memref<?x?x3xi8, strided<[?, 3, 1], offset: 0>>,
      %dst: memref<?x?x3xi8, strided<[?, 3, 1], offset: 0>>)
      attributes {llvm.emit_c_interface} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %height = memref.dim %src, %zero : memref<?x?x3xi8, strided<[?, 3, 1], offset: 0>>
    %width = memref.dim %src, %one : memref<?x?x3xi8, strided<[?, 3, 1], offset: 0>>
    scf.for %y = %zero to %height step %one {
      scf.for %x = %zero to %width step %one {
        %b = memref.load %src[%y, %x, %zero] : memref<?x?x3xi8, strided<[?, 3, 1], offset: 0>>
        %g = memref.load %src[%y, %x, %one] : memref<?x?x3xi8, strided<[?, 3, 1], offset: 0>>
        %r = memref.load %src[%y, %x, %two] : memref<?x?x3xi8, strided<[?, 3, 1], offset: 0>>
        memref.store %r, %dst[%y, %x, %zero] : memref<?x?x3xi8, strided<[?, 3, 1], offset: 0>>
        memref.store %g, %dst[%y, %x, %one] : memref<?x?x3xi8, strided<[?, 3, 1], offset: 0>>
        memref.store %b, %dst[%y, %x, %two] : memref<?x?x3xi8, strided<[?, 3, 1], offset: 0>>
      }
    }
    return
  }
}
