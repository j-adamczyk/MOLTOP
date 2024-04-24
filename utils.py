import argparse
from typing import Union

import numpy as np
from sklearn.metrics import roc_auc_score


def ensure_bool(data: Union[bool, str]) -> bool:
    if isinstance(data, bool):
        return data
    elif data.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif data.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")


def multioutput_auroc_score(y_test: np.ndarray, y_pred: np.ndarray) -> float:
    aurocs = []
    for i in range(y_pred.shape[1]):
        mask = ~np.isnan(y_test[:, i])
        y_test_i = y_test[mask, i]
        y_pred_i = y_pred[mask, i]
        try:
            aurocs.append(roc_auc_score(y_test_i, y_pred_i))
        except ValueError:
            # only one class, AUROC is undefined for this task
            continue
    return np.mean(aurocs)
