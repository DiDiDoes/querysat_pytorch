from querysat_pytorch.utils.registry import Registry
dataset_regisry = Registry("dataset")

from querysat_pytorch.data.ksat import KSATDataset
dataset_regisry.register("ksat", KSATDataset)

# CNFGen datasets: 3sat, 3clique, kcoloring
from querysat_pytorch.data.cnfgen import ThreeSATDataset, ThreeCliqueDataset, KColoringDataset
dataset_regisry.register("3sat", ThreeSATDataset)
dataset_regisry.register("3clique", ThreeCliqueDataset)
dataset_regisry.register("kcoloring", KColoringDataset)

# SHA-1
from querysat_pytorch.data.sha1 import SHA1Dataset
dataset_regisry.register("sha1", SHA1Dataset)
