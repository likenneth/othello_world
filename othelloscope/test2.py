import torch
import numpy as np

import transformer_lens.utils as utils
from data import get_othello

from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
)


def tensor_variance(x: torch.Tensor):
    return torch.mean((x - torch.mean(x)) ** 2)


x = torch.zeros((3, 4), device="cuda")
y = torch.zeros((3, 4), device="cuda")
for i in range(3):
    for j in range(4):
        x[i, j] = 100 + i * 10 + j
        y[i, j] = 200 + i * 10 + j


z = torch.stack([x, y], dim=0)

for var in z:
    for var in var:
        print(var[0].item())
