from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import data_loading
from data_loading import DATASET_NAMES, load_dataset


if __name__ == "__main__":
    # make relative to this file
    data_loading.DATASETS_DIR = Path("../datasets/")

    plots_dir = "../plots/dataset_mols_sizes"
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    for dataset_name in DATASET_NAMES:
        dataset = load_dataset(dataset_name)
        sizes = [data.num_nodes for data in dataset]
        median_size = int(np.median(sizes))

        dataset_name = dataset_name[8:]  # remove "ogbg-mol"
        print(f"Median molecule size for {dataset_name}: {median_size}")

        plt.hist(sizes, bins=30)
        plt.xlabel("Molecule size (atoms count)")
        plt.ylabel("Molecules count")
        plt.savefig(f"{plots_dir}/{dataset_name}.svg", bbox_inches="tight")
        plt.clf()
