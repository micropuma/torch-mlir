module attributes {torch.debug_module_name = "LargeMatrixMultiplication"} {
  memref.global "private" @global_seed : memref<i64> = dense<0>
  func.func @forward(%arg0: memref<2x2xf32, strided<[?, ?], offset: ?>>, %arg1: memref<2x2xf32, strided<[?, ?], offset: ?>>, %arg2: memref<2x2xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %alloca = memref.alloca() {alignment = 64 : i64} : memref<2x2xf32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 2 {
        affine.store %cst, %alloca[%arg3, %arg4] : memref<2x2xf32>
      }
    }
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 2 {
        affine.for %arg5 = 0 to 2 {
          %0 = affine.load %arg0[%arg3, %arg5] : memref<2x2xf32, strided<[?, ?], offset: ?>>
          %1 = affine.load %arg1[%arg5, %arg4] : memref<2x2xf32, strided<[?, ?], offset: ?>>
          %2 = affine.load %alloca[%arg3, %arg4] : memref<2x2xf32>
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          affine.store %4, %alloca[%arg3, %arg4] : memref<2x2xf32>
        }
      }
    }
    memref.copy %alloca, %arg2 : memref<2x2xf32> to memref<2x2xf32>
    return
  }
}

