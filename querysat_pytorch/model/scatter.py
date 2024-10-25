"""
scatter_reduce functions adopted from `pytorch_scatter`
URL: https://github.com/rusty1s/pytorch_scatter
"""
from typing import Tuple

import torch
from torch_geometric.data import HeteroData


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

def scatter_reduce(
    src: torch.Tensor,
    index: torch.Tensor,
    reduce: str,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None
) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        size[dim] = dim_size or int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return out.scatter_reduce_(dim, index, src, reduce, include_self=False)
