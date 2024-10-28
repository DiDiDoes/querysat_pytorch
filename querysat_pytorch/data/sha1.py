from argparse import Namespace
from cnfgen.formula.cnf import CNF
import platform
import os
import random
import subprocess

from querysat_pytorch.data.cnf import CNFDataset


def random_binary_string(n):
    return "".join([str(random.randint(0, 1)) for _ in range(n)])


class SHA1Dataset(CNFDataset):
    name = "SHA-1"

    def __init__(self, args: Namespace, split: str):
        if platform.system() == "Linux":
            self.cgen_executable = "./binary/cgen_linux64"
        elif platform.system() == "Darwin":
            self.cgen_executable == "./binary/cgen_mac"
        else:
            self.cgen_executable = "./binary/cgen"

        self.tmp_file_name = "/tmp/{}.tmp".format(os.getpid())

        #### constraints ####
        # how many free bits; max 512 free bits
        self.bits_from = 2
        self.bits_to = 20

        #### the desired number of variables ####
        self.min_vars = 4
        self.max_vars = 100020

        super().__init__(args, split)

    def generate_one_instance(self) -> CNF:
        while True:
            n_bits = random.randint(self.bits_from, self.bits_to)
            sha_rounds = 17

            bitsstr = random_binary_string(512)
            bitsstr = "0b" + bitsstr

            ok = False
            cmd = "{} encode SHA1 -vM {} except:1..{} -vH compute -r {} {}".format(
                self.cgen_executable, bitsstr, n_bits, sha_rounds, self.tmp_file_name
            )
            try:
                out = subprocess.check_output(cmd, shell=True, universal_newlines=True)
            except:
                out = "" # an unsatisfiable formula or an execution error

            # Searching for the "CNF: <nvars> var" substring;
            # ok will be true iff <nvars> is between MIN_VARS and MAX_VARS;
            # if not ok, we will delete the file.
            j1 = out.find("CNF:")
            j2 = out.find("var", j1 + 1)
            if j1 >= 0 and j2 >= 0:
                nvars = int(out[j1 + 4:j2].strip())
                ok = nvars >= self.min_vars and nvars <= self.max_vars

            if ok:
                F = CNF.from_file(self.tmp_file_name)
                if os.path.exists(self.tmp_file_name):
                    os.remove(self.tmp_file_name)
                ok = self.min_vars <= F.number_of_variables() <= self.max_vars  # checking once again after the removal of unused vars

            if ok:
                return F
