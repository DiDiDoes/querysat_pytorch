from argparse import Namespace
import math
from typing import Tuple

import torch
from torch import nn
from torch.autograd import grad
from torch_geometric.data import HeteroData
from torch_geometric.utils import degree

from querysat_pytorch.model.base import BaseModel
from querysat_pytorch.model.loss import softplus_loss, softplus_mixed_loss
from querysat_pytorch.model.mlp import MLP
from querysat_pytorch.model.pairnorm import PairNorm
from querysat_pytorch.model.scatter import scatter_reduce


class QuerySATEncoder(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.dim = args.dim

    def forward(self, graph: HeteroData):
        # pre-compute literal_indices and weights
        variable_indices, _ = graph["variable", "in", "clause"].edge_index
        polarity = torch.where(graph["variable", "in", "clause"].polarity > 0, 0, 1)
        literal_indices = variable_indices * 2 + polarity
        graph["variable"].literal_indices = literal_indices
        graph["variable", "in", "clause"].polarity = graph["variable", "in", "clause"].polarity.unsqueeze(1)

        lit_degree = degree(literal_indices, graph["variable"].num_nodes*2).unsqueeze(1)
        graph["variable"].literal_degree_weight = lit_degree.clamp_min(1.0).pow(-0.5)
        graph["variable"].degree_weight = 4 * degree(variable_indices, graph["variable"].num_nodes).clamp_min(1.0).pow(-0.5).unsqueeze(1)

        # initialize hidden states
        device = variable_indices.device
        graph["variable"].x = torch.ones(graph["variable"].num_nodes, self.dim, device=device)
        graph["clause"].x = torch.ones(graph["clause"].num_nodes, self.dim, device=device)


class QuerySATCore(nn.Module):
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.query_maps = args.query_maps

        # normalizations
        self.variables_norm = PairNorm(subtract_mean=True)
        self.clauses_norm = PairNorm(subtract_mean=True)

        # MLPs
        self.variables_query = MLP(args.dim+4, 2, int(1.2*args.query_maps), args.query_maps)
        self.clause_mlp = MLP(args.dim+args.query_maps, 2, int(1.6*args.dim), args.dim+args.query_maps)
        self.update_gate = MLP(args.dim+3*args.query_maps, 3, int(1.8*args.dim), args.dim)

    def forward(self, graph: HeteroData) -> None:
        # make a query for solution, get its value and gradient
        with torch.enable_grad():
            v1 = torch.cat([graph["variable"].x, torch.randn(graph["variable"].num_nodes, 4, device=graph["variable"].x.device)], dim=-1)
            query = self.variables_query(v1)
            clauses_loss = softplus_loss(query, graph["variable", "in", "clause"])
            step_loss = clauses_loss.sum()
        variables_grad = grad(step_loss, query, retain_graph=True)[0]
        variables_grad = variables_grad * graph["variable"].degree_weight

        # calculate new clause state
        clauses_loss = clauses_loss * 4
        clause_unit = torch.cat([graph["clause"].x, clauses_loss], dim=-1)
        clause_data = self.clause_mlp(clause_unit)
        variable_loss_all = clause_data[:, 0:self.query_maps]
        new_clause_value = clause_data[:, self.query_maps:]
        new_clause_value = self.clauses_norm(new_clause_value, graph["clause"])
        graph["clause"].x = new_clause_value + 0.1 * graph["clause"].x

        # Aggregate loss over edges
        clause_indices = graph["variable", "in", "clause"].edge_index[1]
        literal_indices = graph["variable"].literal_indices
        variables_loss = scatter_reduce(variable_loss_all[clause_indices], literal_indices, reduce="sum", dim=0, dim_size=graph["variable"].num_nodes*2)
        variables_loss = variables_loss * graph["variable"].literal_degree_weight
        variables_loss = variables_loss.reshape(graph["variable"].num_nodes, -1)

        # calculate new variable state
        unit = torch.cat([variables_grad, graph["variable"].x, variables_loss], dim=-1)
        new_variables = self.update_gate(unit)
        new_variables = self.variables_norm(new_variables, graph["variable"])
        graph["variable"].x = new_variables + 0.1 * graph["variable"].x


class QuerySATDecoder(nn.Module):
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.logit_maps = args.logit_maps
        self.variables_output = MLP(args.dim, 2, args.dim, self.logit_maps)

    def forward(self, graph: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
        # calculate logits and loss
        logits = self.variables_output(graph["variable"].x)
        per_clause_loss = softplus_mixed_loss(logits, graph["variable", "in", "clause"])
        per_graph_loss = scatter_reduce(per_clause_loss, graph["clause"].batch, reduce="sum", dim=0, dim_size=graph.num_graphs)
        per_graph_loss = torch.sqrt(per_graph_loss + 1e-6) - math.sqrt(1e-6)
        costs = torch.square(torch.arange(1, self.logit_maps + 1, dtype=torch.float32, device=logits.device))
        per_graph_loss_avg = torch.sum(torch.sort(per_graph_loss, dim=-1, descending=True).values * costs) / torch.sum(costs)
        logit_loss = torch.sum(per_graph_loss_avg)

        logits_hard = torch.where(logits > 0, 1.0, -1.0)
        variable_indices, clause_indices = graph["variable", "in", "clause"].edge_index
        polarity = graph["variable", "in", "clause"].polarity
        clauses_hard = scatter_reduce(logits_hard[variable_indices]*polarity, clause_indices, reduce="amax", dim=0, dim_size=graph["clause"].num_nodes)
        graphs_hard = scatter_reduce(clauses_hard, graph["clause"].batch, reduce="amin", dim=0, dim_size=graph.num_graphs)
        solved = (graphs_hard > 0.5).any(dim=1)
        return logit_loss, solved


class QuerySATModel(BaseModel):
    name = "QuerySAT"

    def build_encoder(self, args: Namespace) -> nn.Module:
        return QuerySATEncoder(args)

    def build_core(self, args: Namespace) -> nn.Module:
        return QuerySATCore(args)

    def build_decoder(self, args: Namespace) -> nn.Module:
        return QuerySATDecoder(args)
