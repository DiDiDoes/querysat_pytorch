import numpy as np
import os
import random
from tqdm import tqdm

import torch
from torch_geometric.loader import DataLoader, DynamicBatchSampler

from querysat_pytorch.data import dataset_regisry
from querysat_pytorch.engine import Engine
from querysat_pytorch.model.querysat import QuerySATModel


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
        return
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

    # set device
    if not torch.cuda.is_available():
        print("[Device] No GPU available, use CPU!")
        device = torch.device("cpu")
    elif args.gpu is None:
        print("[Device] No GPU specified, use CPU!")
        device = torch.device("cpu")
    else:
        print("[Device] Use GPU:{}".format(args.gpu))
        device = torch.device("cuda:{}".format(args.gpu))

    # prepare model
    model = QuerySATModel(args).to(device)
    print("[Model] Use {}".format(model.name))
    num_parameters = 0
    for p in model.parameters():
        num_parameters += p.numel()
    print("[Model] #parameters: {}".format(num_parameters))

    # prepare engine
    engine = Engine(model, device, args)
    if args.command == "train":
        engine.build_optimizer(args)
        engine.build_writer(args)

    # load checkpoint
    if args.checkpoint is not None:
        print("[Engine] Load checkpoint {}".format(args.checkpoint))
        checkpoint = os.path.join(args.experiment_dir, "checkpoint_{}.pth".format(args.checkpoint))
        checkpoint = torch.load(checkpoint, map_location=torch.device("cpu"))
        engine.load_state_dict(checkpoint)
    else:
        print("[Engine] Initialize from scratch")

    if args.command == "train":
        model.train()
        tbar = tqdm(total=args.learn_step, initial=engine.counter)
        while engine.counter < args.learn_step:
            try:
                data = next(train_dataloader_iter)
            except:
                train_dataloader_iter = iter(train_dataloader)
                data = next(train_dataloader_iter)
            engine.run(data, train=True)
            tbar.update()
            if engine.counter % args.evaluate_interval == 0:
                torch.save(engine.state_dict(), os.path.join(args.experiment_dir, f"checkpoint_{engine.counter}.pth"))
                model.eval()
                engine.evaluate(valid_dataloader)
                model.train()
        tbar.close()
    elif args.command == "test":
        model.eval()
        engine.evaluate(test_dataloader)
    else:
        raise ValueError

if __name__ == "__main__":
    main()