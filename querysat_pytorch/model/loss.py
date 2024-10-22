import math

import torch
from torch.nn import functional as F


def softplus_mixed_loss(variable_predictions: torch.Tensor, adj_matrix: torch.Tensor, eps=1e-8) -> torch.Tensor:
    clauses_val = softplus_loss(variable_predictions, adj_matrix)
    log_clauses = -(torch.log(1 - clauses_val + eps) - math.log(1 + eps))
    return clauses_val * log_clauses

def softplus_loss(variable_predictions: torch.Tensor, adj_matrix: torch.Tensor,) -> torch.Tensor:
    literals = torch.cat([variable_predictions, -variable_predictions], dim=0)
    literals = F.softplus(literals)
    clauses_val = torch.sparse.mm(adj_matrix, literals)
    clauses_val = torch.exp(-clauses_val)
    return clauses_val
