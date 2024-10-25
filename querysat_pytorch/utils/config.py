import argparse
from querysat_pytorch.data import dataset_regisry

parser = argparse.ArgumentParser()
parser.add_argument("command", type=str, choices=["prepare", "train", "test"])

# dataset
parser.add_argument("--data-dir", type=str, default="./data")
parser.add_argument("--dataset", type=str, choices=dataset_regisry.registered())

# device
parser.add_argument("--gpu", type=int, default=None)

# model
parser.add_argument("--dim", type=int, default=128)
parser.add_argument("--num-step", type=int, default=32)
parser.add_argument("--query-maps", type=int, default=128)
parser.add_argument("--logit-maps", type=int, default=8)

# optimization
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--learn-step", type=int, default=500_000)
parser.add_argument("--evaluate-interval", type=int, default=10_000)
parser.add_argument("--grad-clip", type=float, default=10.0)
parser.add_argument("--grad-alpha", type=float, default=0.8)

# reproducibility
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--experiment-dir", type=str, default="experiments/dev")
parser.add_argument("--record-interval", type=int, default=100)
parser.add_argument("--checkpoint", type=int, default=None)

args = parser.parse_args()