from argparse import Namespace
from typing import Tuple

import torch
from torch import nn
from torch_geometric.data import HeteroData

from querysat_pytorch.model.loss import softplus_loss


class BaseModel(nn.Module):
    name = "Base"

    def __init__(self, args: Namespace):
        super().__init__()

        # components
        self.encoder = self.build_encoder(args)
        self.core = self.build_core(args)
        self.decoder = self.build_decoder(args)

        # hyperparameters
        self.num_step = args.num_step
        self.grad_alpha = args.grad_alpha

    # builder functions
    def build_encoder(self, args: Namespace) -> nn.Module:
        raise NotImplementedError

    def build_core(self, args: Namespace) -> nn.Module:
        raise NotImplementedError

    def build_decoder(self, args: Namespace) -> nn.Module:
        raise NotImplementedError

    # forward function
    def forward(self, graph: HeteroData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # encode
        self.encoder(graph)

        # core
        model_step, losses, solveds = 0, [], []
        for _ in range(self.num_step):
            self.core(graph)
            model_step += 1

            # reduce gradient
            if self.training:
                for node_store in graph.node_stores:
                    node_store.x = node_store.x * self.grad_alpha + node_store.x.detach() * (1 - self.grad_alpha)

            # decode
            loss, solved = self.decoder(graph)
            losses.append(loss)
            solveds.append(solved)
            if torch.all(solved):
                break

        return model_step, losses, solveds
