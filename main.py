import numpy as np
import random

from torch_geometric.loader import DataLoader, DynamicBatchSampler

from querysat_pytorch.data import dataset_regisry


def main():
    print("="*20, "querysat_pytorch", "="*20)
    from querysat_pytorch.utils.config import args

    # set seed
    print("[Seed] Use {}".format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)

    # prepare dataset
    dataset_cls = dataset_regisry.get(args.dataset)
    print("[Dataset] Use {}".format(dataset_cls.name))

    if args.command == "prepare":
        train_dataset = dataset_cls(args, "train")
        valid_dataset = dataset_cls(args, "valid")
        test_dataset = dataset_cls(args, "test")
    elif args.command == "train":
        train_dataset = dataset_cls(args, "train")
        valid_dataset = dataset_cls(args, "valid")
        train_sampler = DynamicBatchSampler(train_dataset, max_num=20_000, mode="node", shuffle=True)
        train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler)
        valid_sampler = DynamicBatchSampler(valid_dataset, max_num=20_000, mode="node")
        valid_dataloader = DataLoader(valid_dataset, batch_sampler=valid_sampler)
    elif args.command == "test":
        test_dataset = dataset_cls(args, "test")
        test_sampler = DynamicBatchSampler(test_dataset, max_num=20_000, mode="node")
        test_dataloader = DataLoader(test_dataset, batch_sampler=test_sampler)
    else:
        raise ValueError

if __name__ == "__main__":
    main()