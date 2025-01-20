mlir-opt buffer.mlir \
    --one-shot-bufferize="bufferize-function-boundaries" \
    --mlir-print-ir-after-all \
    -o result2.mlir 

mlir-opt buffer.mlir \
    --one-shot-bufferize \
    --mlir-print-ir-after-all \
    -o result1.mlir 