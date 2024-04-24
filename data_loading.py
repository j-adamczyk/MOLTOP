from pathlib import Path
from typing import Union

from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import Dataset


DATASETS_DIR = Path("datasets")
DATA_SPLITS_DIR = Path("data_splits")
DATASET_NAMES = [
    "ogbg-molbace",
    "ogbg-molbbbp",
    "ogbg-molhiv",
    "ogbg-molclintox",
    "ogbg-molmuv",
    "ogbg-molsider",
    "ogbg-moltox21",
    "ogbg-moltoxcast",
]

DATASET_TASK_TYPES = {
    "ogbg-molbace": "classification",
    "ogbg-molbbbp": "classification",
    "ogbg-molhiv": "classification",
    "ogbg-molclintox": "multioutput_classification",
    "ogbg-molmuv": "multioutput_classification",
    "ogbg-molsider": "multioutput_classification",
    "ogbg-moltox21": "multioutput_classification",
    "ogbg-moltoxcast": "multioutput_classification",
}


def load_dataset_splits(
    dataset_name: str,
    use_valid_for_testing: bool = False,
    use_full_training_data: bool = False,
    train_valid_test_idxs: bool = False,
) -> Union[tuple[list[int], list[int]], tuple[list[int], list[int], list[int]]]:
    if dataset_name not in DATASET_NAMES:
        raise ValueError(
            f"Dataset {dataset_name} not recognized. It has to be one of: {DATASET_NAMES}"
        )

    if use_valid_for_testing and use_full_training_data:
        raise ValueError("Use validation data either for training or testing!")

    if dataset_name in DATASET_NAMES:
        dataset = PygGraphPropPredDataset(name=dataset_name, root=str(DATASETS_DIR))
        split_idx = dataset.get_idx_split()

        train_idxs = list(split_idx["train"])
        if use_full_training_data:
            train_idxs += list(split_idx["valid"])

        if use_valid_for_testing:
            test_idxs = list(split_idx["valid"])
        else:
            test_idxs = list(split_idx["test"])

        if train_valid_test_idxs:
            return (
                list(split_idx["train"]),
                list(split_idx["valid"]),
                list(split_idx["test"]),
            )
    else:
        raise ValueError(f"Dataset '{dataset_name}' not recognized")

    return train_idxs, test_idxs


def load_dataset(dataset_name: str) -> Dataset:
    if dataset_name in DATASET_NAMES:
        return PygGraphPropPredDataset(root=str(DATASETS_DIR), name=dataset_name)
    else:
        raise ValueError(f"Dataset name '{dataset_name}' not recognized")
