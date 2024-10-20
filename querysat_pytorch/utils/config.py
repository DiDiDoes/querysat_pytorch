import argparse
from querysat_pytorch.data import dataset_regisry

parser = argparse.ArgumentParser()
parser.add_argument("command", type=str, choices=["prepare", "train", "test"])

# dataset
parser.add_argument("--data-dir", type=str, default="./data")
parser.add_argument("--dataset", type=str, choices=dataset_regisry.registered())

# reproducibility
parser.add_argument("--seed", type=int, default=0)

args = parser.parse_args()