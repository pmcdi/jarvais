from typing import Callable

import numpy as np
from sklearn.metrics import auc, precision_recall_curve

def auprc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        return auc(recall, precision)

def bootstrap_metric(
        y_test: np.ndarray,
        y_pred: np.ndarray,
        f: Callable[[np.ndarray, np.ndarray], float],
        nsamples: int=100
    ) -> float:
    np.random.seed(0)
    values = []

    for _ in range(nsamples):
        idx = np.random.randint(len(y_test), size=len(y_test))
        pred_sample = y_pred[idx]
        y_test_sample = y_test[idx]
        val = f(y_test_sample.ravel(), pred_sample.ravel())
        values.append(val)
    return values
