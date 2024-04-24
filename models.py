import numpy as np
from ogb.graphproppred import Evaluator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from utils import multioutput_auroc_score


def get_model(
    dataset_name: str,
    random_state: int,
    hyperparams: dict,
    verbose: bool,
):
    # use less jobs in parallel for ToxCast to avoid OOM
    n_jobs = 4 if dataset_name == "ogbg-moltoxcast" else -1

    model = RandomForestClassifier(
        **hyperparams,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
    )

    return model


def tune_hyperparameters(
    X_train: np.ndarray, y_train: np.ndarray, verbose: bool
) -> dict:
    # Scikit-learn has weird verbosity settings, to get reasonably verbose outputs
    # we need to set 2
    verbose = 2 if verbose else 0

    model = RandomForestClassifier(
        n_jobs=-1,
        random_state=0,
    )
    params_grid = {
        "n_estimators": [500, 750, 1000],
        "criterion": ["gini", "entropy"],
        "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "class_weight": [None, "balanced"],
    }

    cv = GridSearchCV(
        estimator=model,
        param_grid=params_grid,
        n_jobs=1,
        cv=5,
        verbose=verbose,
    )
    cv.fit(X_train, y_train)

    if verbose:
        print(f"Best hyperparameters: {cv.best_params_}")

    return cv.best_params_


def evaluate_model(
    dataset_name: str,
    task_type: str,
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    # use OGB evaluation for MoleculeNet
    if task_type == "classification":
        y_pred = model.predict_proba(X_test)[:, 1]
        y_test = y_test.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    elif task_type == "multioutput_classification":
        # extract positive class probability for each task
        y_pred = model.predict_proba(X_test)
        y_pred = [y_pred_i[:, 1] for y_pred_i in y_pred]
        y_pred = np.column_stack(y_pred)
    else:
        raise ValueError(f"Task type '{task_type}' not recognized")

    # use AUROC for MUV instead of default AP to compare to papers
    if dataset_name == "ogbg-molmuv":
        return multioutput_auroc_score(y_test, y_pred)

    evaluator = Evaluator(dataset_name)
    metrics = evaluator.eval(
        {
            "y_true": y_test,
            "y_pred": y_pred,
        }
    )
    # extract the AUROC
    metric = next(iter(metrics.values()))
    return metric
