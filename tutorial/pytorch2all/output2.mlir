module attributes {torch.debug_module_name = "LargeMatrixMultiplication"} {
  memref.global "private" @global_seed : memref<i64> = dense<0>
  func.func @forward(%arg0: memref<2x2xf32, strided<[?, ?], offset: ?>>, %arg1: memref<2x2xf32, strided<[?, ?], offset: ?>>) -> memref<2x2xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x2xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<2x2xf32>)
    linalg.matmul ins(%arg0, %arg1 : memref<2x2xf32, strided<[?, ?], offset: ?>>, memref<2x2xf32, strided<[?, ?], offset: ?>>) outs(%alloc : memref<2x2xf32>)
    %cast = memref.cast %alloc : memref<2x2xf32> to memref<2x2xf32, strided<[?, ?], offset: ?>>
    return %alloc : memref<2x2xf32>
  }
}

