import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self, in_nmap: int, layer_count: int, hidden_nmap: int,
        out_nmap: int, activation: nn.Module = nn.LeakyReLU,
    ):
        super().__init__()
        dense_layers = []
        for _ in range(layer_count - 1):
            dense_layers.append(nn.Linear(in_nmap, hidden_nmap))
            dense_layers.append(activation())
            in_nmap = hidden_nmap
        dense_layers.append(nn.Linear(hidden_nmap, out_nmap))
        nn.init.zeros_(dense_layers[-1].bias)
        self.mlp = nn.Sequential(*dense_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
