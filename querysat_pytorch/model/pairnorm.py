import torch
from torch import nn
from torch_geometric.data.storage import NodeStorage

from querysat_pytorch.model.scatter import scatter_reduce


class PairNorm(nn.Module):
    def __init__(self, epsilon=1e-6, subtract_mean=False):
        super().__init__()
        self.epsilon = epsilon
        self.subtract_mean = subtract_mean

    def forward(self, x: torch.Tensor, node_store: NodeStorage) -> torch.Tensor:
        if self.subtract_mean:
            mean = scatter_reduce(x, node_store.batch, reduce="mean", dim=0, dim_size=node_store.num_nodes)
            x = x - mean[node_store.batch]
        variance = x.pow(2).mean(dim=1, keepdim=True)
        return x * torch.rsqrt(variance + self.epsilon)