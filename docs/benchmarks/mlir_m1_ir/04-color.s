	.file	"LLVMDialectModule"
	.text
	.globl	edge_color_smoke_impl           // -- Begin function edge_color_smoke_impl
	.p2align	2
	.type	edge_color_smoke_impl,@function
edge_color_smoke_impl:                  // @edge_color_smoke_impl
	.cfi_startproc
// %bb.0:
	ldr	x8, [sp, #56]
	ldr	x9, [sp, #16]
	mov	x10, xzr
	b	.LBB0_2
.LBB0_1:                                //   in Loop: Header=BB0_2 Depth=1
	add	x10, x10, #1
.LBB0_2:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_4 Depth 2
	cmp	x10, x3
	b.ge	.LBB0_5
// %bb.3:                               // %.preheader
                                        //   in Loop: Header=BB0_2 Depth=1
	madd	x12, x10, x6, x1
	mov	x11, xzr
	mov	x14, xzr
	madd	x13, x10, x8, x9
	cmp	x14, x4
	b.ge	.LBB0_1
.LBB0_4:                                //   Parent Loop BB0_2 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	add	x15, x12, x11
	add	x17, x13, x11
	add	x14, x14, #1
	ldrb	w16, [x15, #2]
	ldrb	w18, [x15, #1]
	ldrb	w15, [x15]
	add	x11, x11, #3
	strb	w16, [x17]
	strb	w18, [x17, #1]
	strb	w15, [x17, #2]
	cmp	x14, x4
	b.lt	.LBB0_4
	b	.LBB0_1
.LBB0_5:
	ret
.Lfunc_end0:
	.size	edge_color_smoke_impl, .Lfunc_end0-edge_color_smoke_impl
	.cfi_endproc
                                        // -- End function
	.globl	_mlir_ciface_edge_color_smoke_impl // -- Begin function _mlir_ciface_edge_color_smoke_impl
	.p2align	2
	.type	_mlir_ciface_edge_color_smoke_impl,@function
_mlir_ciface_edge_color_smoke_impl:     // @_mlir_ciface_edge_color_smoke_impl
	.cfi_startproc
// %bb.0:
	sub	sp, sp, #96
	.cfi_def_cfa_offset 96
	.cfi_offset w30, -16
	ldp	x9, x8, [x0]
	ldr	x10, [x0, #64]
	ldp	x6, x7, [x0, #48]
	ldr	x11, [x1, #64]
	ldp	x4, x5, [x0, #32]
	ldp	x2, x3, [x0, #16]
	mov	x0, x9
	ldp	q0, q1, [x1]
	ldp	q2, q3, [x1, #32]
	mov	x1, x8
	stp	x11, x30, [sp, #72]             // 8-byte Folded Spill
	stur	q1, [sp, #24]
	stur	q3, [sp, #56]
	stur	q2, [sp, #40]
	stur	q0, [sp, #8]
	str	x10, [sp]
	bl	edge_color_smoke_impl
	ldr	x30, [sp, #80]                  // 8-byte Folded Reload
	add	sp, sp, #96
	ret
.Lfunc_end1:
	.size	_mlir_ciface_edge_color_smoke_impl, .Lfunc_end1-_mlir_ciface_edge_color_smoke_impl
	.cfi_endproc
                                        // -- End function
	.section	".note.GNU-stack","",@progbits
