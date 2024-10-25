import math

import torch
from torch.nn import functional as F
from torch_geometric.data.storage import EdgeStorage

from querysat_pytorch.model.scatter import scatter_reduce


def softplus_mixed_loss(variable_predictions: torch.Tensor, edge_store: EdgeStorage, eps=1e-8) -> torch.Tensor:
    clauses_val = softplus_loss(variable_predictions, edge_store)
    log_clauses = -(torch.log(1 - clauses_val + eps) - math.log(1 + eps))
    return clauses_val * log_clauses

def softplus_loss(variable_predictions: torch.Tensor, edge_store: EdgeStorage) -> torch.Tensor:
    variable_indices, clause_indices = edge_store.edge_index
    polarity = edge_store.polarity
    variables = F.softplus(variable_predictions[variable_indices] * polarity)
    clauses_val = scatter_reduce(variables, clause_indices, reduce="sum", dim=0)
    return (-clauses_val).exp()
