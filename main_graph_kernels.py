import argparse

import grakel
import networkx as nx
import numpy as np
from grakel import graph_from_networkx
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from torch_geometric.data import Dataset
from tqdm import tqdm

from data_loading import (
    load_dataset,
    DATASET_TASK_TYPES,
    load_dataset_splits,
)
from models import evaluate_model
from utils import ensure_bool, multioutput_auroc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        choices=[
            "all",
            "ogbg-molbace",
            "ogbg-molbbbp",
            "ogbg-molclintox",
            "ogbg-molsider",
            "ogbg-moltox21",
        ],
        default="all",
        help=(
            "Dataset name. You can either provide dataset name from "
            "MoleculeNet (via OGB), or 'all' to run on MoleculeNet "
            "apart from HIV and MUV"
        ),
    )
    parser.add_argument(
        "--kernel_type",
        type=str,
        default="WL",
        choices=[
            "EH",
            "graphlet",
            "propagation",
            "SP",
            "RW",
            "VH",
            "WL",
            "WL-OA",
        ],
        help="Kernel type to use",
    )
    parser.add_argument(
        "--verbose",
        type=ensure_bool,
        default=True,
        help="Should print out verbose output?",
    )

    return parser.parse_args()


def create_grakel_graphs(
    dataset: Dataset, kernel_type: str, verbose: bool
) -> np.ndarray:
    graphs = []
    dataset = tqdm(dataset) if verbose else dataset

    for data in dataset:
        graph = nx.Graph()

        for idx, atom_features in enumerate(data.x):
            atom_type = atom_features[0].item()
            graph.add_node(idx, atom_type=atom_type)

        for bond_idx, (atom_idx_1, atom_idx_2) in enumerate(
            zip(data.edge_index[0].tolist(), data.edge_index[1].tolist())
        ):
            bond_features = data.edge_attr[bond_idx]
            bond_type = bond_features[0].item() if bond_features.shape[0] > 0 else 0
            graph.add_edge(atom_idx_1, atom_idx_2, bond_type=bond_type)

        graphs.append(graph)

    graphs = list(
        graph_from_networkx(
            graphs, node_labels_tag="atom_type", edge_labels_tag="bond_type"
        )
    )
    return np.array(graphs)


def precomputed_kernel_svm_tuning(
    X_kernel_train: np.ndarray,
    X_kernel_valid: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    task_type: str,
    verbose: bool,
):
    best_model = None
    best_score = 0

    Cs = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    Cs = tqdm(Cs) if verbose else Cs

    # make sure we have no NaN values
    X_kernel_train[~np.isfinite(X_kernel_train)] = 0
    X_kernel_valid[~np.isfinite(X_kernel_valid)] = 0

    for C in Cs:
        svm = SVC(
            C=C,
            kernel="precomputed",
            probability=True,
            cache_size=1024,
            class_weight="balanced",
            random_state=0,
        )
        if task_type == "multioutput_classification":
            svm = MultiOutputClassifier(svm, n_jobs=-1)

        svm.fit(X_kernel_train, y_train)
        y_pred = svm.predict_proba(X_kernel_valid)

        if task_type == "multioutput_classification":
            # extract positive class probability for each task
            y_pred = [y_pred_i[:, 1] for y_pred_i in y_pred]
            y_pred = np.column_stack(y_pred)

            score = multioutput_auroc_score(y_valid, y_pred)
        else:
            score = roc_auc_score(y_valid, y_pred[:, 1])

        if score > best_score:
            best_model = svm
            best_score = score

    return best_model, best_score


def train_kernel_and_svm(
    X_train: list[grakel.Graph],
    X_valid: list[grakel.Graph],
    y_train: np.ndarray,
    y_valid: np.ndarray,
    task_type: str,
    kernel_type: str,
    verbose: bool,
):
    common_kernel_args = dict(n_jobs=-1, normalize=True, verbose=verbose)

    kernels_map = {
        "EH": (grakel.EdgeHistogram, {}),
        "graphlet": (grakel.GraphletSampling, {"k": list(range(3, 6))}),
        "propagation": (grakel.Propagation, {"t_max": list(range(1, 6))}),
        "SP": (grakel.ShortestPath, {}),
        "VH": (grakel.VertexHistogram, {}),
        "WL": (grakel.WeisfeilerLehman, {"n_iter": list(range(1, 6))}),
        "WL-OA": (
            grakel.WeisfeilerLehmanOptimalAssignment,
            {"n_iter": list(range(1, 6))},
        ),
    }

    try:
        kernel_cls, kernel_param_grid = kernels_map[kernel_type]
    except KeyError:
        raise ValueError(f"Kernel type {kernel_type} not recognized")

    best_kernel = None
    best_model = None
    best_score = 0

    kernel_param_grid = ParameterGrid(kernel_param_grid)
    kernel_param_grid = tqdm(kernel_param_grid) if verbose else kernel_param_grid

    for kernel_params in kernel_param_grid:
        kernel = kernel_cls(**kernel_params, **common_kernel_args)
        X_kernel_train = kernel.fit_transform(X_train)
        X_kernel_valid = kernel.transform(X_valid)

        model, score = precomputed_kernel_svm_tuning(
            X_kernel_train, X_kernel_valid, y_train, y_valid, task_type, verbose
        )
        if score > best_score:
            best_kernel = kernel
            best_model = model
            best_score = score

    return best_kernel, best_model


def perform_experiment(
    dataset_name: str,
    kernel_type: str,
    verbose: bool,
) -> float:
    dataset = load_dataset(dataset_name)
    task_type = DATASET_TASK_TYPES[dataset_name]

    train_idxs, valid_idxs, test_idxs = load_dataset_splits(
        dataset_name, train_valid_test_idxs=True
    )

    X = create_grakel_graphs(dataset, kernel_type, verbose)
    X_train = X[train_idxs]
    X_valid = X[valid_idxs]
    X_test = X[test_idxs]

    y = np.array(dataset.y)
    if task_type == "classification":
        y = y.ravel()

    y_train = y[train_idxs]
    y_valid = y[valid_idxs]
    y_test = y[test_idxs]

    # fill NaN values with zeros for multioutput classification
    y_train[np.isnan(y_train)] = 0
    y_valid[np.isnan(y_valid)] = 0

    kernel, model = train_kernel_and_svm(
        X_train, X_valid, y_train, y_valid, task_type, kernel_type, verbose
    )
    X_test = kernel.transform(X_test)

    # make sure we have no NaN values
    X_test[~np.isfinite(X_test)] = 0

    test_metric = evaluate_model(
        dataset_name=dataset_name,
        task_type=task_type,
        model=model,
        X_test=X_test,
        y_test=y_test,
    )

    return test_metric


if __name__ == "__main__":
    args = parse_args()

    if args.dataset_name == "all":
        datasets = [
            "ogbg-molbace",
            "ogbg-molbbbp",
            "ogbg-molclintox",
            "ogbg-molsider",
            "ogbg-moltox21",
        ]
    else:
        datasets = [args.dataset_name]

    for dataset_name in datasets:
        print(dataset_name)
        test_mean = perform_experiment(
            dataset_name=dataset_name,
            kernel_type=args.kernel_type,
            verbose=args.verbose,
        )
        print(f"AUROC: {100 * test_mean:.1f}")
