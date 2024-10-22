import torch
from torch import nn


class PairNorm(nn.Module):
    def __init__(self, epsilon=1e-6, subtract_mean=False):
        super().__init__()
        self.epsilon = epsilon
        self.subtract_mean = subtract_mean

    def forward(self, inputs: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        mask = graph.indices()[0] if graph is not None else None

        if self.subtract_mean:
            if graph is not None:
                mean = torch.sparse.mm(graph, inputs)
                inputs = inputs - mean[mask]
            else:
                mean = torch.mean(inputs, dim=0, keepdim=True)
                inputs = inputs - mean
        variance = torch.mean(torch.square(inputs), dim=1, keepdim=True)
        return inputs * torch.rsqrt(variance + self.epsilon)
