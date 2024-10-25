# querysat_pytorch
A PyTorch implementation of QuerySAT

## Setup

```shell
conda create -n querysat python=3.10
conda activate querysat
pip install -r requirements.txt
```

## Prepare data

Each dataset may take several hours to prepare.

```shell
python main.py prepare --dataset ksat
python main.py prepare --dataset 3sat
python main.py prepare --dataset 3clique
python main.py prepare --dataset kcoloring
```

## Training

```shell
python main.py train --dataset <ksat,3sat,3clique,kcoloring> --experiment-dir <experiment_dir> --gpu <gpu_id>
```

## Test

```shell
python main.py test --dataset <ksat,3sat,3clique,kcoloring> --experiment-dir <experiment_dir> --checkpoint <checkpoint_id> --gpu <gpu_id>
```
