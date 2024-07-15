import argparse
import os

import numpy as np
import pandas as pd
from skfp.fingerprints import ECFPFingerprint
from sklearn.ensemble import RandomForestClassifier

from data_loading import (
    DATASET_NAMES,
    load_dataset,
    DATASET_TASK_TYPES,
    load_dataset_splits,
    DATASETS_DIR,
)
from models import evaluate_model
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
    dataset_name: str,
    verbose: bool,
) -> tuple[float, float]:
    dataset = load_dataset(dataset_name)
    task_type = DATASET_TASK_TYPES[dataset_name]

    dataset_path = os.path.join(
        DATASETS_DIR, dataset_name.replace("-", "_"), "mapping", "mol.csv.gz"
    )
    smiles = pd.read_csv(dataset_path)["smiles"].values

    train_idxs, test_idxs = load_dataset_splits(
        dataset_name, use_valid_for_testing=False, use_full_training_data=True
    )

    smiles_train = smiles[train_idxs]
    smiles_test = smiles[test_idxs]

    fp = ECFPFingerprint()
    X_train = fp.transform(smiles_train)
    X_test = fp.transform(smiles_test)

    y = np.array(dataset.y)
    if task_type == "classification":
        y = y.ravel()

    y_train = y[train_idxs]
    y_test = y[test_idxs]

    # fill NaN values with zeros for multioutput classification
    y_train[np.isnan(y_train)] = 0

    test_metrics = []
    for random_state in range(10):
        if verbose:
            print(f"Starting random seed {random_state}")

        # use less jobs in parallel for ToxCast to avoid OOM
        n_jobs = 4 if dataset_name == "ogbg-moltoxcast" else -1

        # same as in LDP, LTP and D-MPNN papers
        model = RandomForestClassifier(
            n_estimators=500,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
        model.fit(X_train, y_train)

        test_metric = evaluate_model(
            dataset_name=dataset_name,
            task_type=task_type,
            model=model,
            X_test=X_test,
            y_test=y_test,
        )

        test_metrics.append(test_metric)

    test_metrics_mean = np.mean(test_metrics)
    test_metrics_stddev = np.std(test_metrics)

    return test_metrics_mean, test_metrics_stddev


if __name__ == "__main__":
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

    for dataset_name in datasets:
        print(dataset_name)
        test_mean, test_stddev = perform_experiment(
            dataset_name=dataset_name,
            verbose=args.verbose,
        )
        print(f"AUROC: {100 * test_mean:.1f} +- {100 * test_stddev:.1f}")
