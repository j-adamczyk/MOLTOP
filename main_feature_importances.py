import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from feature_engine.selection import DropConstantFeatures

from data_loading import (
    DATASET_NAMES,
    load_dataset,
    load_dataset_splits,
    DATASET_TASK_TYPES,
)
from feature_extraction import extract_features
from models import get_model
from utils import ensure_bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        choices=[
            "all",
            "all_fast",
            "ogbg-molbace",
            "ogbg-molbbbp",
            "ogbg-molhiv",
            "ogbg-molmuv",
            "ogbg-molclintox",
            "ogbg-molsider",
            "ogbg-moltox21",
            "ogbg-moltoxcast",
        ],
        default="all",
        help=(
            "Dataset name. You can either provide dataset name from "
            "MoleculeNet (via OGB), or use one of the following options: "
            "'all_fast' to run on MoleculeNet apart from MUV and ToxCast; "
            "'all' to run on the entire MoleculeNet; "
        ),
    )
    parser.add_argument(
        "--verbose",
        type=ensure_bool,
        default=False,
        help="Should print out verbose output?",
    )

    return parser.parse_args()


def perform_experiment(
    dataset_name: str, verbose: bool, plots_dir: Path
) -> pd.DataFrame:
    dataset = load_dataset(dataset_name)
    task_type = DATASET_TASK_TYPES[dataset_name]

    train_idxs, test_idxs = load_dataset_splits(
        dataset_name, use_valid_for_testing=False, use_full_training_data=True
    )

    nodes_nums = [data.num_nodes for data in dataset[train_idxs]]
    n_bins = int(np.median(nodes_nums))
    if verbose:
        print(f"Selected {n_bins} histogram bins")

    X = extract_features(
        dataset,
        degree_features=True,
        edge_betweenness=True,
        rand_index=True,
        scan_structural_score=True,
        atom_types=True,
        bond_types=True,
        n_bins=n_bins,
        verbose=verbose,
    )
    X_train = X[train_idxs, :]

    columns = []
    columns.extend([f"deg {i}" for i in range(11)])
    columns.extend([f"deg_min {i}" for i in range(11)])
    columns.extend([f"deg_max {i}" for i in range(11)])
    columns.extend([f"deg_mean {i}" for i in range(n_bins)])
    columns.extend([f"deg_std {i}" for i in range(n_bins)])
    columns.extend([f"EBC {i}" for i in range(n_bins)])
    columns.extend([f"ARI {i}" for i in range(n_bins)])
    columns.extend([f"SCAN {i}" for i in range(n_bins)])
    columns.extend([f"atom_mean {i}" for i in range(89)])
    columns.extend([f"atom_std {i}" for i in range(89)])
    columns.extend([f"atom_sum {i}" for i in range(89)])
    columns.extend([f"bond_mean {i}" for i in range(5)])
    columns.extend([f"bond_std {i}" for i in range(5)])
    columns.extend([f"bond_sum {i}" for i in range(5)])

    df_train = pd.DataFrame(X_train, columns=columns)

    dropper = DropConstantFeatures()
    X_train = dropper.fit_transform(df_train).values

    if verbose:
        constant_features = X.shape[1] - X_train.shape[1]
        print(f"Eliminated {constant_features} constant features")
        print(f"Final number of features: {X_train.shape[1]}")

    y = np.array(dataset.y)
    if task_type == "classification":
        y = y.ravel()

    y_train = y[train_idxs]

    # fill NaN values with zeros for multioutput classification
    y_train[np.isnan(y_train)] = 0

    # default values, optimized on validation sets of MoleculeNet fast datasets
    hyperparams = {
        "n_estimators": 1000,
        "criterion": "entropy",
        "min_samples_split": 10,
    }

    # aggregate feature importances
    importances = []
    for random_state in range(10):
        if verbose:
            print(f"Starting random seed {random_state}")

        model = get_model(
            dataset_name=dataset_name,
            random_state=random_state,
            hyperparams=hyperparams,
            verbose=verbose,
        )
        model.fit(X_train, y_train)
        importances.append(model.feature_importances_)

    # average feature importances across 10 random seeds
    importances = np.mean(np.row_stack(importances), axis=0).tolist()

    # total importance of each feature group
    columns = dropper.get_feature_names_out()
    columns = [col.split(" ")[0].replace("_", " ") for col in columns]
    df = pd.DataFrame({"column": columns, "value": importances})
    importances = df.groupby("column").sum().transpose()

    # reorder columns and plot
    columns = [
        "deg",
        "deg min",
        "deg max",
        "deg mean",
        "deg std",
        "EBC",
        "ARI",
        "SCAN",
        "atom mean",
        "atom std",
        "atom sum",
        "bond mean",
        "bond std",
        "bond sum",
    ]
    importances = importances[columns]
    importances.columns = columns
    importances.index = [""]

    importances.plot.bar(rot=0)
    plt.tight_layout()

    filename = dataset_name.removeprefix("ogbg-mol")
    plt.savefig(plots_dir / f"{filename}.pdf")

    return importances


if __name__ == "__main__":
    plots_dir = Path("plots") / "feature_importance"
    plots_dir.mkdir(parents=True, exist_ok=True)

    args = parse_args()

    if args.dataset_name == "all":
        datasets = DATASET_NAMES
    elif args.dataset_name == "all_fast":
        datasets = [
            "ogbg-molbace",
            "ogbg-molbbbp",
            "ogbg-molhiv",
            "ogbg-molclintox",
            "ogbg-molsider",
            "ogbg-moltox21",
        ]
    else:
        datasets = [args.dataset_name]

    all_feature_importances = []

    for dataset_name in datasets:
        print(dataset_name)
        importances = perform_experiment(
            dataset_name=dataset_name,
            verbose=args.verbose,
            plots_dir=plots_dir,
        )
        all_feature_importances.append(importances)

    df = pd.concat(all_feature_importances, ignore_index=True)
    df = pd.DataFrame(df.mean(axis=0)).transpose()
    df.index = [""]
    df.plot.bar(rot=0)
    plt.tight_layout()
    plt.savefig(plots_dir / "average.pdf")
