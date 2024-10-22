from argparse import Namespace
import sys
from tqdm import tqdm

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from querysat_pytorch.utils.meters import AverageMeter, ProgressMeter


class Engine(object):
    def __init__(self, model: torch.nn.Module, device: torch.device, args: Namespace):
        # components
        self.model = model
        self.device = device
        self.optimizer = None
        self.writer = None

        # hyperparameters
        self.grad_clip = args.grad_clip
        self.record_interval = args.record_interval

        # statistics
        self.counter = 0
        self.stats = {
            # "model_step": (AverageMeter, "{:.2f}"),
            "loss": (AverageMeter, "{:.4e}"),
            # "last_layer_loss": (AverageMeter, "{:.4e}"),
            "solved": (AverageMeter, "{:.4f}")
        }

    def build_optimizer(self, args: Namespace):
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

    def build_writer(self, args: Namespace):
        self.writer = SummaryWriter(args.experiment_dir)
        self.writer.add_text("command", " ".join(sys.argv[1:]))

    def state_dict(self) -> dict:
        return {
            "counter": self.counter,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer is not None else None
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.counter = state_dict["counter"]
        self.model.load_state_dict(state_dict["model_state_dict"])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])

    def run(self, data, train: bool = False) -> dict:
        data = data.to(self.device)
        loss, solveds = self.model(data)
        # loss = torch.stack(losses).mean()
        stat_dict = {
            # "model_step": model_step,
            "loss": loss.item(),
            # "last_layer_loss": losses[-1].item(),
            "solved": torch.stack(solveds).any(dim=0).float().mean()
        }
        if train:
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.counter += 1
            if self.counter % self.record_interval == 0:
                self.record(stat_dict, "train")
                self.writer.add_scalar("grad_norm", grad_norm, self.counter)
        return stat_dict

    def record(self, stat_dict: dict, suffix: str) -> None:
        for stat_name in self.stats.keys():
            self.writer.add_scalar(f"{stat_name}/{suffix}", stat_dict[stat_name], self.counter)

    @torch.no_grad()
    def evaluate(self, dataloader) -> None:
        # prepare meters
        meter_dict = {
            stat_name: meter_cls(stat_name, stat_fmt)
            for stat_name, (meter_cls, stat_fmt) in self.stats.items()
        }
        progress_meter = ProgressMeter(f"Evaluate [{self.counter}]", list(meter_dict.values()))

        # evaluate
        tqdm.write(progress_meter.title_str())
        for data in tqdm(dataloader, total=float("inf")):
            stat_dict = self.run(data)
            for stat_name in self.stats.keys():
                meter_dict[stat_name].update(stat_dict[stat_name], len(data))
        tqdm.write(progress_meter.summary_str())

        # record
        if self.writer is not None:
            stat_dict = {k: v.result for k, v in meter_dict.items()}
            self.record(stat_dict, "valid")
