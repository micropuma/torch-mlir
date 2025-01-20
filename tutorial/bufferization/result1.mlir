module {
  func.func @test(%arg0: tensor<8xf32>, %arg1: index) -> tensor<8xf32> {
    %0 = bufferization.to_memref %arg0 : memref<8xf32, strided<[?], offset: ?>>
    %cst = arith.constant 5.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<8xf32>
    memref.copy %0, %alloc : memref<8xf32, strided<[?], offset: ?>> to memref<8xf32>
    memref.store %cst, %alloc[%arg1] : memref<8xf32>
    %1 = bufferization.to_tensor %alloc : memref<8xf32>
    return %1 : tensor<8xf32>
  }
}

