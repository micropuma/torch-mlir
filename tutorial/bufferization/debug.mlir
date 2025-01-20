func.func @test(%f0: f32, %f1: f32, %idx: index, %idx2: index)
    -> (f32, tensor<3xf32>) {
  // Create a new tensor with [%f0, %f0, %f0].
  %0 = tensor.from_elements %f0, %f0, %f0 : tensor<3xf32>

  // Insert something into the new tensor.
  %1 = tensor.insert %f1 into %0[%idx] : tensor<3xf32>

  // Read from the old tensor.
  %r = tensor.extract %0[%idx2] : tensor<3xf32>

  // Return the extracted value and the result of the insertion.
  func.return %r, %1 : f32, tensor<3xf32>
}
