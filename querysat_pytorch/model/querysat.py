from argparse import Namespace
import math

import torch
from torch import nn
from torch.autograd import grad
from torch_geometric.data import HeteroData

from querysat_pytorch.model.loss import softplus_loss, softplus_mixed_loss
from querysat_pytorch.model.mlp import MLP
from querysat_pytorch.model.pairnorm import PairNorm


class QuerySATModel(nn.Module):
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        # hyperparameters
        self.num_step = args.num_step
        self.grad_alpha = args.grad_alpha
        self.logit_maps = args.logit_maps
        self.feature_maps = args.dim
        self.query_maps = args.query_maps

        # normalizations
        self.variables_norm = PairNorm(subtract_mean=True)
        self.clauses_norm = PairNorm(subtract_mean=True)

        # MLPs
        self.variables_query = MLP(args.dim+4, 2, int(1.2*args.query_maps), args.query_maps)
        self.clause_mlp = MLP(args.dim+args.query_maps, 2, int(1.6*args.dim), args.dim+args.query_maps)
        self.update_gate = MLP(args.dim+3*args.query_maps, 3, int(1.8*args.dim), args.dim)
        self.variables_output = MLP(args.dim, 2, args.dim, self.logit_maps)

    def forward(self, graph: HeteroData):
        # process the graph
        variable_indices, clause_indices = graph["variable", "in", "clause"].edge_index
        polarity = graph["variable", "in", "clause"].polarity
        offset = torch.where(polarity > 0, 0, graph["variable"].num_nodes)
        literal_indices = variable_indices + offset
        device = literal_indices.device

        # prepare inputs
        adj_matrix = torch.sparse_coo_tensor(
            indices=torch.stack([literal_indices, clause_indices]),
            values=torch.ones_like(literal_indices, dtype=torch.float32),
            size=(graph["variable"].num_nodes * 2, graph["clause"].num_nodes),
        )
        clauses_graph = torch.sparse_coo_tensor(
            indices=torch.stack([graph["clause"].batch, torch.arange(graph["clause"].num_nodes, device=device)]),
            values=torch.ones_like(graph["clause"].batch, dtype=torch.float32),
            size=(graph.num_graphs, graph["clause"].num_nodes)
        )
        variables_graph = torch.sparse_coo_tensor(
            indices=torch.stack([graph["variable"].batch, torch.arange(graph["variable"].num_nodes, device=device)]),
            values=torch.ones_like(graph["variable"].batch, dtype=torch.float32),
            size=(graph.num_graphs, graph["variable"].num_nodes)
        )

        # adopted from original implementation
        shape = adj_matrix.shape
        n_vars = shape[0] // 2
        n_clauses = shape[1]

        variables = torch.ones(n_vars, self.feature_maps, device=device)
        clause_state = torch.ones(n_clauses, self.feature_maps, device=device)

        unsupervised_loss = self.loop(
            adj_matrix,
            clause_state,
            clauses_graph,
            self.num_step,
            variables,
            variables_graph
        )
        return unsupervised_loss

    def loop(self, adj_matrix, clause_state, clauses_graph, rounds, variables, variables_graph):
        step_losses = []
        solveds = []
        cl_adj_matrix = torch.transpose(adj_matrix, 0, 1)
        n_clauses = adj_matrix.shape[1]
        n_vars = adj_matrix.shape[0] // 2
        lit_degree = adj_matrix.sum(dim=1).to_dense().reshape(n_vars * 2, 1)
        degree_weight = torch.rsqrt(torch.clamp(lit_degree, min=1))
        var_degree_weight = 4 * torch.rsqrt(torch.clamp(lit_degree[:n_vars, :] + lit_degree[n_vars, :], min=1))

        # TODO: change this ugly intermediate format
        variables_graph_norm = variables_graph.to_dense()
        variables_graph_norm = variables_graph_norm / torch.sum(variables_graph_norm, axis=-1, keepdims=True)
        variables_graph_norm = variables_graph_norm.to_sparse()
        clauses_graph_norm = clauses_graph.to_dense()
        clauses_graph_norm = clauses_graph_norm / torch.sum(clauses_graph_norm, axis=-1, keepdims=True)
        clauses_graph_norm = clauses_graph_norm.to_sparse()

        for step in range(rounds):
            # make a query for solution, get its value and gradient
            with torch.enable_grad():
                v1 = torch.cat([variables, torch.randn(n_vars, 4, device=variables.device)], dim=-1)
                query = self.variables_query(v1)
                clauses_loss = softplus_loss(query, cl_adj_matrix)
                step_loss = clauses_loss.sum()
            variables_grad = grad(step_loss, query, retain_graph=True)[0]
            variables_grad = variables_grad * var_degree_weight

            # calculate new clause state
            clauses_loss = clauses_loss * 4
            clause_unit = torch.cat([clause_state, clauses_loss], dim=-1)
            clause_data = self.clause_mlp(clause_unit)
            variable_loss_all = clause_data[:, 0:self.query_maps]
            new_clause_value = clause_data[:, self.query_maps:]
            new_clause_value = self.clauses_norm(new_clause_value, clauses_graph_norm)
            clause_state = new_clause_value + 0.1 * clause_state

            # Aggregate loss over edges
            variables_loss = torch.sparse.mm(adj_matrix, variable_loss_all)
            variables_loss = variables_loss * degree_weight
            variables_loss_pos, variables_loss_neg = torch.chunk(variables_loss, 2, dim=0)

            # calculate new variable state
            unit = torch.cat([variables_grad, variables, variables_loss_pos, variables_loss_neg], dim=-1)
            new_variables = self.update_gate(unit)
            new_variables = self.variables_norm(new_variables, variables_graph_norm)
            variables = new_variables + 0.1 * variables

            # calculate logits and loss
            logits = self.variables_output(variables)
            per_clause_loss = softplus_mixed_loss(logits, cl_adj_matrix)
            per_graph_loss = torch.sparse.mm(clauses_graph, per_clause_loss)
            per_graph_loss = torch.sqrt(per_graph_loss + 1e-6) - math.sqrt(1e-6)
            costs = torch.square(torch.arange(1, self.logit_maps + 1, dtype=torch.float32, device=logits.device))
            per_graph_loss_avg = torch.sum(torch.sort(per_graph_loss, dim=-1, descending=True).values * costs) / torch.sum(costs)
            logit_loss = torch.sum(per_graph_loss_avg)

            step_losses.append(logit_loss)

            # test the logits
            logits_hard = (logits > 0).float()
            literals_hard = torch.cat([logits_hard, 1 - logits_hard], dim=0)
            clauses_hard = torch.sparse.mm(cl_adj_matrix, literals_hard).clamp_max(1.0)
            graphs_hard = torch.sparse.mm(clauses_graph_norm, clauses_hard) > 1 - 1e-6
            solved = graphs_hard.any(dim=1)
            solveds.append(solved)

            if self.training:
                variables = variables * self.grad_alpha + variables.detach() * (1 - self.grad_alpha)
                clause_state = clause_state * self.grad_alpha + clause_state.detach() * (1 - self.grad_alpha)

        unsupervised_loss = torch.sum(torch.stack(step_losses)) / rounds
        return unsupervised_loss, solveds
