from cnfgen.formula.cnf import CNF
import numpy as np
from pysat.solvers import Cadical195
import random

from querysat_pytorch.data.cnf import CNFDataset


class KSATDataset(CNFDataset):
    name = "k-SAT"

    def __init__(self, args, split):
        if split in ["train", "valid"]:
            self.variable_min = 3
            self.variable_max = 100
        elif split == "test":
            self.variable_min = 3
            self.variable_max = 200
        else:
            raise ValueError
        super().__init__(args, split)

    def generate_one_instance(self) -> CNF:
        variable_num = random.randint(self.variable_min, self.variable_max)
        solver = Cadical195()
        clauses = []
        while True:
            k_base = 1 if random.random()<0.3 else 2
            k = k_base + np.random.geometric(0.4)
            variables = np.random.choice(variable_num, size=min(variable_num, k), replace=False)
            clause = [int(v+1) if random.random()<0.5 else int(-(v+1)) for v in variables]
            solver.add_clause(clause)
            is_sat = solver.solve()
            if is_sat:
                clauses.append(clause)
            else:
                break
        clauses.append([-clause[0]] + clause[1:])
        clauses = list({tuple(sorted(clause)) for clause in clauses})
        return CNF(clauses)
