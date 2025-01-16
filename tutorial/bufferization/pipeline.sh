mlir-opt buffer.mlir \
    --one-shot-bufferize="bufferize-function-boundaries" \
    --mlir-print-ir-after-all \
    -o result.mlir \
    2>&1 | tee output.txt