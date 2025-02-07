from typing import Callable, List

import numpy as np
import pandas as pd

from sklearn.metrics import auc, precision_recall_curve
from sksurv.metrics import concordance_index_censored

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

def ci_wrapper(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Wrapper for `sksurv.metrics.concordance_index_censored` to ensure compatibility 
    with `bootstrap_metric`.

    Args:
        y_true (np.ndarray): A 2D NumPy array of shape (n_samples, 2), where:
            - `y_true[:, 0]` represents the observed survival times.
            - `y_true[:, 1]` represents the event indicator 
              (1 if the event occurred, 0 if censored).
        y_pred (np.ndarray): A 1D NumPy array of predicted risk scores or 
            survival times. Higher scores typically indicate higher risk.

    Returns:
        concordance_index (float): The concordance index.
    """
    time = y_true[:, 0]
    event = y_true[:, 1]

    return concordance_index_censored(event.astype(bool), time, y_pred)[0]

def bootstrap_metric(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_func: Callable[[np.ndarray, np.ndarray], float],
        nsamples: int = 100
    ) -> List[float]:
    """
    Compute a metric using bootstrapping to estimate its variability.

    Args:
        y_true (np.ndarray): True labels. Shape (n_samples,).
        y_pred (np.ndarray): Predicted values. Shape (n_samples,).
        metric_func (Callable[[np.ndarray, np.ndarray], float]): A function that calculates the metric.
        nsamples (int, optional): The number of bootstrap samples. Defaults to 100.

    Returns:
        bootstrapped_values (List[float]): A list of metric values computed on each bootstrap sample.
    """
    np.random.seed(0)
    values = []

    for _ in range(nsamples):
        idx = np.random.randint(len(y_true), size=len(y_true))
        pred_sample = y_pred[idx]
        y_true_sample = y_true[idx]
        val = metric_func(y_true_sample, pred_sample)
        values.append(val)

    return values

def undummify(df, prefix_sep="_"):
    """
    Undummifies a DataFrame by collapsing dummy/one-hot encoded columns back into their original categorical column.

    Found here: https://stackoverflow.com/a/62085741

    Args:
        df (pandas.DataFrame): The input DataFrame containing dummy/one-hot encoded columns.
        prefix_sep (str, optional): The separator used to distinguish between the prefix (category) and the column name in the dummy columns. 
            Defaults to "_".

    Returns:
        undummified_df (pandas.DataFrame): A new DataFrame with the undummified (reconstructed) categorical columns.
    """
    dummy_cols = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in dummy_cols.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df

def process_RADCURE_clinical(df):
    """
    Processes RADCURE clinical data.

    Raw data found here: https://www.cancerimagingarchive.net/collection/radcure/
    """
    df_converted = pd.DataFrame({
        'Study ID': df['patient_id'],
        'survival_time': df['Length FU'],
        'death': df['Status'].apply(lambda x: 1 if x == 'Dead' else 0),
        'age at dx': df['Age'],
        'Sex': df['Sex'],
        'T Stage': df['T'],
        'N Stage': df['N'],
        'Stage': df['Stage'],
        'Dose': df['Dose'],
        'Chemotherapy': df['Chemo'].apply(lambda x: 1 if x != 'none' else 0),
        'HPV Combined': df['HPV'].apply(lambda x: 1 if isinstance(x, str) and 'positive' in x.lower() else None),
        'Smoking Status': df['Smoking Status'],
        'Disease Site': df['Ds Site'].str.lower()
    })
    
    return df_converted