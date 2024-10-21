from cnfgen.formula.cnf import CNF
import subprocess


def run_external_solver(formula: CNF) -> bool | None:
    output = subprocess.run(["./binary/treengeling_linux"], input=formula.to_dimacs(), capture_output=True, text=True)
    result = [line for line in output.stdout.split("\n") if line.startswith("s")]
    if len(result) != 1:
        return None
    if result[0] == "s SATISFIABLE":
        return True
    elif result[0] == "s UNSATISFIABLE":
        return False
    else:
        return None
