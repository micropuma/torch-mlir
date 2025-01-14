mlir-opt output-origin.mlir \
  --canonicalize \
  --eliminate-empty-tensors \
  --empty-tensor-to-alloc-tensor \
  -o output1.mlir

mlir-opt output1.mlir \
  --one-shot-bufferize \
  -o output2.mlir

mlir-opt output2.mlir \
  --buffer-hoisting \
  --buffer-loop-hoisting \
  --buffer-results-to-out-params \
  --drop-equivalent-buffer-results \
  --promote-buffers-to-stack \
  --buffer-deallocation-pipeline \
  -o output3.mlir

mlir-opt output3.mlir \
  --convert-linalg-to-affine-loops \
  --convert-linalg-to-parallel-loops \
  --convert-linalg-to-loops \
  --convert-linalg-to-std \
  -o output4.mlir