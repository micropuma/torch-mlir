module attributes {torch.debug_module_name = "LargeMatrixMultiplication"} {
  memref.global "private" @global_seed : memref<i64> = dense<0>
  func.func @forward(%arg0: tensor<1024x1024xf32>, %arg1: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
    %0 = bufferization.to_memref %arg1 : memref<1024x1024xf32, strided<[?, ?], offset: ?>>
    %1 = bufferization.to_memref %arg0 : memref<1024x1024xf32, strided<[?, ?], offset: ?>>
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1024x1024xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<1024x1024xf32>)
    linalg.matmul ins(%1, %0 : memref<1024x1024xf32, strided<[?, ?], offset: ?>>, memref<1024x1024xf32, strided<[?, ?], offset: ?>>) outs(%alloc : memref<1024x1024xf32>)
    %2 = bufferization.to_tensor %alloc : memref<1024x1024xf32>
    return %2 : tensor<1024x1024xf32>
  }
}

