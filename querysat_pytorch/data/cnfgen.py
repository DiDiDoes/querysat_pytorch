from argparse import Namespace
from cnfgen import RandomKCNF, CliqueFormula, GraphColoringFormula
from cnfgen.formula.cnf import CNF
import math
import networkx as nx
import random

from querysat_pytorch.data.cnf import CNFDataset


class ThreeSATDataset(CNFDataset):
    name = "3-SAT"

    def __init__(self, args: Namespace, split: str) -> None:
        if split in ["train", "valid"]:
            self.variable_min = 5
            self.variable_max = 100
        elif split == "test":
            self.variable_min = 5
            self.variable_max = 405
        else:
            raise ValueError
        super().__init__(args, split)

    def generate_one_instance(self) -> CNF:
        variable_num = random.randint(self.variable_min, self.variable_max)
        clause_num = int(4.258 * variable_num + 58.26 * (variable_num ** (-2/3)))
        return RandomKCNF(3, variable_num, clause_num)


class ThreeCliqueDataset(CNFDataset):
    name = "3-Clique"

    def __init__(self, args: Namespace, split: str) -> None:
        if split in ["train", "valid"]:
            self.vertice_min = 4
            self.vertice_max = 40
        elif split == "test":
            self.vertice_min = 4
            self.vertice_max = 100
        else:
            raise ValueError
        super().__init__(args, split)

    def generate_one_instance(self):
        vertice_num = random.randint(self.vertice_min, self.vertice_max)
        p = 3 ** (1/3) / (vertice_num * (2 - 3*vertice_num + vertice_num**2)) ** (1/3)
        G = nx.generators.erdos_renyi_graph(vertice_num, p=p)
        return CliqueFormula(G, 3)


class KColoringDataset(CNFDataset):
    name = "k-Coloring"

    def __init__(self, args: Namespace, split: str) -> None:
        if split in ["train", "valid"]:
            self.vertice_min = 4
            self.vertice_max = 40
        elif split == "test":
            self.vertice_min = 4
            self.vertice_max = 100
        else:
            raise ValueError
        super().__init__(args, split)

    def generate_one_instance(self):
        vertice_num = random.randint(self.vertice_min, self.vertice_max)
        p = math.log(vertice_num) * (1 + 0.2) / vertice_num + 0.05
        while True:
            G = nx.generators.erdos_renyi_graph(vertice_num, p=p)
            if nx.is_connected(G):
                break
        color_num = random.randint(3, 5)
        return GraphColoringFormula(G, color_num)
