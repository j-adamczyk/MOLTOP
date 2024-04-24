import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from networkit.linkprediction import JaccardIndex
from networkit.sparsification import LocalDegreeScore
from sklearn.ensemble import RandomForestClassifier
from torch_geometric.data import Dataset, Data
from torch_geometric.nn.aggr.fused import FusedAggregation

from data_loading import (
    DATASET_NAMES,
    load_dataset,
    DATASET_TASK_TYPES,
    load_dataset_splits,
)
from feature_extraction import calculate_edge_betweenness
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
        "--method",
        type=str,
        default="LDP",
        choices=["LDP", "LTP", "molecular_fingerprint"],
        help="Baseline method to use",
    )
    parser.add_argument(
        "--verbose",
        type=ensure_bool,
        default=False,
        help="Should print out verbose output?",
    )

    return parser.parse_args()


def extract_LDP_features(data: Data) -> np.ndarray:
    row, col = data.edge_index
    num_nodes = data.num_nodes

    deg = torch_geometric.utils.degree(row, num_nodes, dtype=torch.float)
    deg = deg.view(-1, 1)
    deg_col = deg[col]

    aggr = FusedAggregation(["min", "max", "mean", "std"])
    ldp_features = [deg] + aggr(deg_col, row, dim_size=num_nodes)
    ldp_features = [feature.numpy().ravel() for feature in ldp_features]

    features = []
    for feature in ldp_features:
        feature = feature.astype(float)
        values, _ = np.histogram(feature, bins=50)
        features.append(values)

    return np.concatenate(features)


def extract_LTP_features(data: Data) -> np.ndarray:
    ldp_features = extract_LDP_features(data)

    graph = torch_geometric.utils.to_networkit(
        edge_index=data.edge_index,
        edge_weight=data.edge_weight,
        num_nodes=data.num_nodes,
        directed=False,
    )
    graph.indexEdges()

    features = []

    ebc = calculate_edge_betweenness(graph)
    features.append(ebc)

    ji = JaccardIndex(graph)
    scores = [ji.run(u, v) for u, v in graph.iterEdges()]
    scores = np.array(scores, dtype=np.float32)
    ji_scores = scores[np.isfinite(scores)]
    features.append(ji_scores)

    lds = LocalDegreeScore(graph)
    lds.run()
    scores = lds.scores()
    lds_scores = np.array(scores, dtype=np.float32)
    features.append(lds_scores)

    topological_features = []
    for feature in features:
        feature = feature.astype(float)
        values, _ = np.histogram(feature, bins=50)
        topological_features.append(values)

    topological_features = np.concatenate(topological_features)

    ltp_features = np.concatenate((ldp_features, topological_features))
    return ltp_features


def extract_atom_counts(data: Data) -> np.ndarray:
    atoms = data.x[:, 0]
    atoms = F.one_hot(atoms, 120).float()
    atoms = atoms[:, :89]
    atoms_counts = torch.sum(atoms, dim=0).numpy()
    return atoms_counts


def extract_features(dataset: Dataset, method: str = "LDP") -> np.ndarray:
    if method == "LDP":
        extract_fun = extract_LDP_features
    elif method == "LTP":
        extract_fun = extract_LTP_features
    elif method == "molecular_fingerprint":
        extract_fun = extract_atom_counts
    else:
        raise ValueError(f"Mode '{method}' not recognized")

    X = []
    for data in dataset:
        features = extract_fun(data)
        X.append(features)

    return np.row_stack(X)


def perform_experiment(
    dataset_name: str,
    method: str,
    verbose: bool,
) -> tuple[float, float]:
    dataset = load_dataset(dataset_name)
    task_type = DATASET_TASK_TYPES[dataset_name]

    train_idxs, test_idxs = load_dataset_splits(
        dataset_name, use_valid_for_testing=False, use_full_training_data=True
    )

    X = extract_features(dataset, method)
    X_train = X[train_idxs, :]
    X_test = X[test_idxs, :]

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
            method=args.method,
            verbose=args.verbose,
        )
        print(f"AUROC: {100 * test_mean:.1f} +- {100 * test_stddev:.1f}")
