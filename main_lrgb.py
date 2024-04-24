from time import time

import numpy as np
import pandas as pd
from feature_engine.selection import DropConstantFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from torch_geometric.datasets import LRGBDataset

from data_loading import DATASETS_DIR
from feature_extraction import extract_features


def load_peptides_func_dataset() -> tuple[LRGBDataset, LRGBDataset, LRGBDataset]:
    name = "Peptides-func"
    train = LRGBDataset(root=str(DATASETS_DIR), name=name, split="train")
    valid = LRGBDataset(root=str(DATASETS_DIR), name=name, split="val")
    test = LRGBDataset(root=str(DATASETS_DIR), name=name, split="test")
    return train, valid, test


def extract_MOLTOP_features(dataset: LRGBDataset, n_bins: int) -> np.ndarray:
    return extract_features(
        dataset,
        degree_features=True,
        edge_betweenness=True,
        rand_index=True,
        scan_structural_score=True,
        atom_types=True,
        bond_types=True,
        n_bins=n_bins,
        verbose=True,
    )


def multioutput_average_precision(y_test: np.ndarray, y_pred: np.ndarray) -> float:
    ap_values = []
    for i in range(y_pred.shape[1]):
        mask = ~np.isnan(y_test[:, i])
        y_test_i = y_test[mask, i]
        y_pred_i = y_pred[mask, i]
        ap = average_precision_score(y_test_i, y_pred_i)
        ap_values.append(ap)

    return np.mean(ap_values)


if __name__ == "__main__":
    dataset_train, dataset_valid, dataset_test = load_peptides_func_dataset()

    n_bins = int(np.median([data.num_nodes for data in dataset_train]))

    start_time = time()
    X_train = extract_MOLTOP_features(dataset_train, n_bins)
    X_valid = extract_MOLTOP_features(dataset_valid, n_bins)
    X_train = np.vstack((X_train, X_valid))
    end_time = time()
    print(f"Total feature extraction time: {end_time - start_time:.0f}")
    avg_time_ms = 1000 * (end_time - start_time) / len(X_train)
    print(f"Feature extraction time per graph: {avg_time_ms:.0f} ms")

    X_test = extract_MOLTOP_features(dataset_test, n_bins)

    dropper = DropConstantFeatures()
    X_train = dropper.fit_transform(pd.DataFrame(X_train)).values
    X_test = dropper.transform(pd.DataFrame(X_test)).values

    y_train = np.array(dataset_train.y)
    y_valid = np.array(dataset_valid.y)
    y_train = np.vstack((y_train, y_valid))

    y_test = np.array(dataset_test.y)

    test_metrics = []
    params_counts = []
    train_times = []
    pred_times = []
    for random_state in range(10):
        print(f"Starting random seed {random_state}")

        # default values, optimized on validation sets of MoleculeNet fast datasets
        model = RandomForestClassifier(
            n_estimators=500,
            criterion="entropy",
            min_samples_split=10,
            n_jobs=-1,
            random_state=random_state,
        )
        start_time = time()
        model.fit(X_train, y_train)
        end_time = time()
        train_times.append(end_time - start_time)

        start_time = time()
        y_pred_test = model.predict_proba(X_test)
        end_time = time()
        pred_times.append(end_time - start_time)

        # extract positive class probability for each task
        y_pred_test = [y_pred_i[:, 1] for y_pred_i in y_pred_test]
        y_pred_test = np.column_stack(y_pred_test)

        test_ap = multioutput_average_precision(y_test, y_pred_test)
        test_metrics.append(test_ap)

        n_params = sum(tree.tree_.node_count for tree in model.estimators_)
        params_counts.append(n_params)

    print(f"Test AP: {np.mean(test_metrics):.4f} +- {np.std(test_metrics):.4f}")
    print(f"Parameters: {np.mean(params_counts):.0f} +- {np.std(params_counts):.0f}")
    print(f"Training time: {np.mean(train_times):.2f} +- {np.std(train_times):.2f}")
    avg_pred_time = [(1000 * pred_time) / len(X_test) for pred_time in pred_times]
    print(
        f"Prediction time: {np.mean(pred_times):.4f} ms +- {np.std(pred_times):.4f} ms"
    )
