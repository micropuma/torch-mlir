module {
  func.func @test(%arg0: memref<8xf32, strided<[?], offset: ?>>, %arg1: index) -> memref<8xf32, strided<[?], offset: ?>> {
    %cst = arith.constant 5.000000e+00 : f32
    memref.store %cst, %arg0[%arg1] : memref<8xf32, strided<[?], offset: ?>>
    return %arg0 : memref<8xf32, strided<[?], offset: ?>>
  }
}

