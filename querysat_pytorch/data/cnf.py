from argparse import Namespace
from cnfgen.formula.cnf import CNF
import numpy as np
import os
from tqdm import tqdm

import torch
from torch_geometric.data import Dataset, HeteroData

from querysat_pytorch.data.solve import kissat_solve


class CNFDataset(Dataset):
    name = "CNF"

    def __init__(self, args: Namespace, split: str) -> None:
        self.split = split
        super().__init__(root=os.path.join(args.data_dir, args.dataset, split))

    # geneartion
    def generate_sat_instance(self) -> CNF:
        while True:
            F = self.generate_one_instance()
            if kissat_solve(F) is True:
                return F

    def generate_one_instance(self) -> CNF:
        raise NotImplementedError

    def download(self):
        for raw_path in tqdm(self.raw_paths, desc=f"Generate {self.name}-{self.split}"):
            F = self.generate_sat_instance()
            F.to_file(raw_path)


    # processing: CNF -> graph
    def process(self):
        for raw_path, processed_path in tqdm(zip(self.raw_paths, self.processed_paths), desc=f"Process {self.name}-{self.split}"):
            F = CNF.from_file(raw_path)
            G = HeteroData()
            G["variable"].num_nodes = F.number_of_variables()
            G["clause"].num_nodes = F.number_of_clauses()
            variable_indices = []
            clause_indices = []
            polarities = []
            for clause_index, clause in enumerate(F.clauses()):
                for literal in clause:
                    variable_indices.append(abs(literal)-1)
                    clause_indices.append(clause_index)
                    polarities.append(np.sign(literal))
            G["variable", "in", "clause"].num_edges = len(variable_indices)
            G["variable", "in", "clause"].edge_index = torch.stack([torch.tensor(variable_indices), torch.tensor(clause_indices)])
            G["variable", "in", "clause"].polarity = torch.tensor(polarities)
            torch.save(G, processed_path)

    # loading
    @property
    def raw_file_names(self):
        return [f"{i}.cnf" for i in range(self.len())]
    
    @property
    def processed_file_names(self):
        return [f"{i}.pt" for i in range(self.len())]

    def len(self) -> int:
        if self.split == "train":
            return 100_000
        elif self.split == "valid":
            return 10_000
        elif self.split == "test":
            return 10_000
        else:
            raise ValueError

    def get(self, idx: int):
        return torch.load(self.processed_paths[idx])
