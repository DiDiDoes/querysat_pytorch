# querysat_pytorch
A PyTorch implementation of QuerySAT

## Setup

```shell
conda create -n querysat python=3.10
conda activate querysat
pip install -r requirements.txt

git submodule update --init --recursive
cd kissat
./configure && make test
cd ../
```

## Prepare data

```shell
python main.py generate --dataset 3sat
python main.py prepare --dataset 3clique
python main.py prepare --dataset kcoloring
```
