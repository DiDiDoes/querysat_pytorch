from querysat_pytorch.utils.registry import Registry
dataset_regisry = Registry("dataset")

from querysat_pytorch.data.cnfgen import ThreeSATDataset, ThreeCliqueDataset, KColoringDataset
dataset_regisry.register("3sat", ThreeSATDataset)
dataset_regisry.register("3clique", ThreeCliqueDataset)
dataset_regisry.register("kcoloring", KColoringDataset)
