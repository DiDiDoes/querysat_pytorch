# querysat_pytorch

A PyTorch implementation of [QuerySAT](https://ieeexplore.ieee.org/document/9892733).

## Setup

```shell
conda create -n querysat python=3.10
conda activate querysat
pip install -r requirements.txt
```

## Prepare data

Each dataset may take several hours to prepare.

```shell
python main.py prepare --dataset <ksat,3sat,3clique,kcoloring,sha1>
```

## Train

```shell
python main.py train --dataset <ksat,3sat,3clique,kcoloring,sha1> --experiment-dir <experiment_dir> --grad-clip <grad_clip> --gpu <gpu_id>
```

The default gradient clipping value is 10.0. I empirically found that using 1.0 for `3clique` and `sha1` works better.

## Test

```shell
python main.py test --dataset <ksat,3sat,3clique,kcoloring,sha1> --experiment-dir <experiment_dir> --checkpoint <checkpoint_id> --num-step <num_step> --gpu <gpu_id>
```

Reproduced results:

| Task | s_test=32 | s_test=512 | s_test=4096 |
|:-|-:|-:|-:|
| k-SAT | 68.78 | 84.10 | 87.18 |
| 3-SAT | 45.83 | 63.85 | 67.50 |
| 3-Clique | | | |
| 3-Coloring | | | |
| SHA-1 | | | |
