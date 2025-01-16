func.func @test(%t: tensor<8xf32>, %idx: index)
    -> tensor<8xf32> {
    %f = arith.constant 5.000000e+00 : f32
    %0 = tensor.insert %f into %t[%idx] : tensor<8xf32>
    return %0 : tensor<8xf32>
}
