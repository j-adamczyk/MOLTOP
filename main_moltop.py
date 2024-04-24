import argparse

import numpy as np
import pandas as pd
from feature_engine.selection import DropConstantFeatures

from data_loading import (
    DATASET_NAMES,
    load_dataset,
    DATASET_TASK_TYPES,
    load_dataset_splits,
)
from feature_extraction import extract_features
from models import tune_hyperparameters, get_model, evaluate_model
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
        "--degree_features",
        type=ensure_bool,
        default=True,
        help="Use degree features based on LDP?",
    )
    parser.add_argument(
        "--edge_betweenness",
        type=ensure_bool,
        default=True,
        help="Add normalized edge betweenness centrality?",
    )
    parser.add_argument(
        "--rand_index",
        type=ensure_bool,
        default=True,
        help="Add normalized Adjusted Rand Index?",
    )
    parser.add_argument(
        "--scan_structural_score",
        type=ensure_bool,
        default=True,
        help="Add SCAN Structural Similarity Score?",
    )
    parser.add_argument(
        "--atom_types",
        type=ensure_bool,
        default=True,
        help="Add atom types features?",
    )
    parser.add_argument(
        "--bond_types",
        type=ensure_bool,
        default=True,
        help="Add bond types features?",
    )
    parser.add_argument(
        "--n_bins",
        type=lambda x: int(x) if x.isnumeric() else x,
        default="median",
        help=(
            "Number of bins for aggregation. Either a number or 'median' "
            "to use median number of atoms in training molecules."
        ),
    )
    parser.add_argument(
        "--model_hyperparams",
        choices=[
            "optimized",
            "LTP_default",
            "tune",
        ],
        default="optimized",
        help=(
            "Which hyperparameters to use for Random Forest: "
            "'optimized' for features tuned on validation sets of MoleculeNet, "
            "'LTP_default' for values suggested in LTP paper, "
            "'tune' to perform hyperparameter tuning."
        ),
    )
    parser.add_argument(
        "--use_valid_for_testing",
        type=ensure_bool,
        default=False,
        help="Use validation split for testing? Only for MoleculeNet datasets!",
    )
    parser.add_argument(
        "--use_full_training_data",
        type=ensure_bool,
        default=True,
        help=(
            "Use both training and validation splits for training? "
            "Only for MoleculeNet datasets!"
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
    degree_features: bool,
    edge_betweenness: bool,
    rand_index: bool,
    scan_structural_score: bool,
    atom_types: bool,
    bond_types: bool,
    n_bins: int | str,
    model_hyperparams: str,
    use_valid_for_testing: bool,
    use_full_training_data: bool,
    verbose: bool,
) -> tuple[float, float, float, float]:
    dataset = load_dataset(dataset_name)
    task_type = DATASET_TASK_TYPES[dataset_name]

    train_idxs, test_idxs = load_dataset_splits(
        dataset_name, use_valid_for_testing, use_full_training_data
    )

    if n_bins == "median":
        nodes_nums = [data.num_nodes for data in dataset[train_idxs]]
        n_bins = int(np.median(nodes_nums))
        if verbose:
            print(f"Selected {n_bins} histogram bins")

    X = extract_features(
        dataset,
        degree_features=degree_features,
        edge_betweenness=edge_betweenness,
        rand_index=rand_index,
        scan_structural_score=scan_structural_score,
        atom_types=atom_types,
        bond_types=bond_types,
        n_bins=n_bins,
        verbose=verbose,
    )
    X_train = X[train_idxs, :]
    X_test = X[test_idxs, :]

    dropper = DropConstantFeatures()
    X_train = dropper.fit_transform(pd.DataFrame(X_train)).values
    X_test = dropper.transform(pd.DataFrame(X_test)).values

    if verbose:
        constant_features = X.shape[1] - X_train.shape[1]
        print(f"Eliminated {constant_features} constant features")

    y = np.array(dataset.y)
    if task_type == "classification":
        y = y.ravel()

    y_train = y[train_idxs]
    y_test = y[test_idxs]

    # fill NaN values with zeros for multioutput classification
    y_train[np.isnan(y_train)] = 0

    if model_hyperparams == "optimized":
        # default values, optimized on validation sets of MoleculeNet fast datasets
        hyperparams = {
            "n_estimators": 1000,
            "criterion": "entropy",
            "min_samples_split": 10,
        }
    elif model_hyperparams == "LTP_default":
        # default values from LTP paper
        hyperparams = {"n_estimators": 500}
    elif model_hyperparams == "tune":
        hyperparams = tune_hyperparameters(
            X_train=X_train, y_train=y_train, verbose=verbose
        )
    else:
        raise ValueError(f"Value '{model_hyperparams}' not recognized")

    test_metrics = []
    params_counts = []
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

        test_metric = evaluate_model(
            dataset_name=dataset_name,
            task_type=task_type,
            model=model,
            X_test=X_test,
            y_test=y_test,
        )
        test_metrics.append(test_metric)

        n_params = sum(tree.tree_.node_count for tree in model.estimators_)
        params_counts.append(n_params)

    test_metrics_mean = np.mean(test_metrics)
    test_metrics_stddev = np.std(test_metrics)

    params_mean = np.mean(params_counts)
    params_stddev = np.std(params_counts)

    return test_metrics_mean, test_metrics_stddev, params_mean, params_stddev


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
        test_mean, test_stddev, params_mean, params_stddev = perform_experiment(
            dataset_name=dataset_name,
            degree_features=args.degree_features,
            edge_betweenness=args.edge_betweenness,
            rand_index=args.rand_index,
            scan_structural_score=args.scan_structural_score,
            atom_types=args.atom_types,
            bond_types=args.bond_types,
            n_bins=args.n_bins,
            model_hyperparams=args.model_hyperparams,
            use_valid_for_testing=args.use_valid_for_testing,
            use_full_training_data=args.use_full_training_data,
            verbose=args.verbose,
        )
        print(f"AUROC: {100 * test_mean:.1f} +- {100 * test_stddev:.1f}")
        print(f"Parameters: {params_mean:.2f} +- {params_stddev:.2f}")
