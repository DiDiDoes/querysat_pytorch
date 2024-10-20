from cnfgen.formula.cnf import CNF
import os
import subprocess


def kissat_solve(formula: CNF) -> bool | None:
    # dump CNF to a temp file and solve
    temp_filename = f"{os.getpid()}.cnf"
    formula.to_file(temp_filename)
    output = subprocess.run(["./kissat/build/kissat", temp_filename], capture_output=True, text=True)
    os.remove(temp_filename)

    # parse the result
    result = [line for line in output.stdout.split("\n") if line.startswith("s")]
    if len(result) != 1:
        return None
    if result[0] == "s SATISFIABLE":
        return True
    elif result[0] == "s UNSATISFIABLE":
        return False
    else:
        return None
