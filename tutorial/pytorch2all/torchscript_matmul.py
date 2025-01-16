import numpy as np
import torch
from torch_mlir import torchscript

# 定义矩阵乘法模块
class LargeMatrixMultiplication(torch.nn.Module):
    def __init__(self):
        super(LargeMatrixMultiplication, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)

# 创建大矩阵数据
matrix_size = 2  # 矩阵大小 1024x1024
A = torch.from_numpy(np.random.rand(matrix_size, matrix_size).astype(np.float32))
B = torch.from_numpy(np.random.rand(matrix_size, matrix_size).astype(np.float32))

# 实例化模型并设置为评估模式
matmul_model = LargeMatrixMultiplication()
matmul_model.eval()

# 使用 torch.mlir 将模型编译为 linalg-on-tensors
module = torchscript.compile(
    matmul_model, (A, B), output_type="linalg-on-tensors"
)

# 将生成的 linalg-on-tensors IR 写入文件
with open("output-origin.mlir", "w") as file:
    file.write(module.operation.get_asm(large_elements_limit=10))

print("Linalg-on-Tensors IR 已成功写入到 'output_linalg_on_tensors.txt'")
