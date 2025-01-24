from typing import Callable, List

import numpy as np
from sklearn.metrics import auc, precision_recall_curve

def auprc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Calculate the Area Under the Precision-Recall Curve (AUPRC).

    Args:
        y_true (np.ndarray): True binary labels. Shape (n_samples,).
        y_scores (np.ndarray): Predicted scores or probabilities. Shape (n_samples,).

    Returns:
        auprc_score (float): The AUPRC value.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)

def bootstrap_metric(
        y_test: np.ndarray,
        y_pred: np.ndarray,
        f: Callable[[np.ndarray, np.ndarray], float],
        nsamples: int=100
    ) -> List[float]:
    """
    Compute a metric using bootstrapping to estimate its variability.

    Args:
        y_test (np.ndarray): True labels. Shape (n_samples,).
        y_pred (np.ndarray): Predicted values. Shape (n_samples,).
        f (Callable[[np.ndarray, np.ndarray], float]): A function that calculates the metric.
        nsamples (int, optional): The number of bootstrap samples. Defaults to 100.

    Returns:
        bootstrapped_values (List[float]): A list of metric values computed on each bootstrap sample.
    """
    np.random.seed(0)
    values = []

    for _ in range(nsamples):
        idx = np.random.randint(len(y_test), size=len(y_test))
        pred_sample = y_pred[idx]
        y_test_sample = y_test[idx]
        val = f(y_test_sample.ravel(), pred_sample.ravel())
        values.append(val)
    return values
