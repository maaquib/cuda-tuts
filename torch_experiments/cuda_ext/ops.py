import torch
from torch import Tensor

__all__ = [
    "add",
    "mul",
]

def add(a: Tensor, b: Tensor) -> Tensor:
    return torch.ops.cuda_ext.add.default(a, b)

def mul(a: Tensor, b: Tensor) -> Tensor:
    return torch.ops.cuda_ext.mul.default(a, b)
