"""
Code adapted from https://github.com/epic-open-source/seismometer
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .functional import auprc, bootstrap_metric
import seaborn as sns

def _bin_class_curve(y_true: np.ndarray, y_pred: np.ndarray):
    sort_ix = np.argsort(y_pred, kind="mergesort")[::-1]
    y_true = np.array(y_true)[sort_ix]
    y_pred = np.array(y_pred)[sort_ix]

    # Find where the threshold changes
    distinct_ix = np.where(np.diff(y_pred))[0]
    threshold_idxs = np.r_[distinct_ix, y_true.size - 1]

    # Add up the true positives and infer false ones
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    return fps, tps, y_pred[threshold_idxs]

def plot_epic_copy(
        y_test: np.ndarray,
        y_pred: np.ndarray,
        y_val: np.ndarray,
        y_val_pred: np.ndarray,
        y_train: np.ndarray,
        y_train_pred: np.ndarray,
        output_dir: Path
    ) -> None:

    # Compute test metrics
    fpr_test, tpr_test, thresholds_roc_test = roc_curve(y_test, y_pred)
    roc_auc_test = roc_auc_score(y_test, y_pred)
    precision_test, recall_test, thresholds_pr_test = precision_recall_curve(y_test, y_pred)
    average_precision_test = auprc(y_test, y_pred)
    prob_true_test, prob_pred_test = calibration_curve(y_test, y_pred, n_bins=10, strategy='uniform')

    # Compute validation metrics
    fpr_val, tpr_val, thresholds_roc_val = roc_curve(y_val, y_val_pred)
    roc_auc_val = roc_auc_score(y_val, y_val_pred)
    precision_val, recall_val, thresholds_pr_val = precision_recall_curve(y_val, y_val_pred)
    average_precision_val = auprc(y_val, y_val_pred)
    prob_true_val, prob_pred_val = calibration_curve(y_val, y_val_pred, n_bins=10, strategy='uniform')

    # Compute train metrics
    fpr_train, tpr_train, thresholds_roc_train = roc_curve(y_train, y_train_pred)
    roc_auc_train = roc_auc_score(y_train, y_train_pred)
    precision_train, recall_train, thresholds_pr_train = precision_recall_curve(y_train, y_train_pred)
    average_precision_train = auprc(y_train, y_train_pred)
    prob_true_train, prob_pred_train = calibration_curve(y_train, y_train_pred, n_bins=10, strategy='uniform')

    # Compute confidence intervals
    roc_conf_test = [round(val, 2) for val in np.percentile(bootstrap_metric(y_test, y_pred, roc_auc_score), (2.5, 97.5))]
    roc_conf_val = [round(val, 2) for val in np.percentile(bootstrap_metric(y_val, y_val_pred, roc_auc_score), (2.5, 97.5))]
    roc_conf_train = [round(val, 2) for val in np.percentile(bootstrap_metric(y_train, y_train_pred, roc_auc_score), (2.5, 97.5))]

    precision_conf_test = [round(val, 2) for val in np.percentile(bootstrap_metric(y_test, y_pred, auprc), (2.5, 97.5))]
    precision_conf_val = [round(val, 2) for val in np.percentile(bootstrap_metric(y_val, y_val_pred, auprc), (2.5, 97.5))]
    precision_conf_train = [round(val, 2) for val in np.percentile(bootstrap_metric(y_train, y_train_pred, auprc), (2.5, 97.5))]

    # Set Seaborn style
    sns.set_theme(style="darkgrid")

    plt.figure(figsize=(37.5, 10))

    # 1. ROC Curve
    plt.subplot(2, 5, 1)
    sns.lineplot(x=fpr_test, y=tpr_test, label=f"Test AUROC = {roc_auc_test:.2f} {roc_conf_test}", color="blue")
    sns.lineplot(x=fpr_val, y=tpr_val, label=f"Validation AUROC = {roc_auc_val:.2f} {roc_conf_val}", color="orange")
    sns.lineplot(x=fpr_train, y=tpr_train, label=f"Train AUROC = {roc_auc_train:.2f} {roc_conf_train}", color="green")
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title("ROC Curve")
    plt.legend()

    # 2. Precision-Recall Curve
    plt.subplot(2, 5, 2)
    sns.lineplot(x=recall_test, y=precision_test, label=f"Test AUC-PR = {average_precision_test:.2f} {precision_conf_test}", color="blue")
    sns.lineplot(x=recall_val, y=precision_val, label=f"Validation AUC-PR = {average_precision_val:.2f} {precision_conf_val}", color="orange")
    sns.lineplot(x=recall_train, y=precision_train, label=f"Train AUC-PR = {average_precision_train:.2f} {precision_conf_train}", color="green")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()

    # 3. Calibration Curve
    plt.subplot(2, 5, 6)
    sns.lineplot(x=prob_pred_test, y=prob_true_test, label="Test Calibration Curve", color="blue", marker='o')
    sns.lineplot(x=prob_pred_val, y=prob_true_val, label="Validation Calibration Curve", color="orange", marker='o')
    sns.lineplot(x=prob_pred_train, y=prob_true_train, label="Train Calibration Curve", color="green", marker='o')
    sns.lineplot(x=[0, 1], y=[0, 1], linestyle="--", label="Perfect Calibration", color="gray")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Probability")
    plt.title("Calibration Curve")
    plt.legend()

    # 4. Sensitivity vs Flag Rate
    fps, tps, _ = _bin_class_curve(y_test, y_pred)
    sens_test = tps / sum(y_test)
    flag_rate_test = (tps + fps) / len(y_test)

    fps, tps, _ = _bin_class_curve(y_val, y_val_pred)
    sens_val = tps / sum(y_val)
    flag_rate_val = (tps + fps) / len(y_val)

    fps, tps, _ = _bin_class_curve(y_train, y_train_pred)
    sens_train = tps / sum(y_train)
    flag_rate_train = (tps + fps) / len(y_train)

    plt.subplot(2, 5, 7)
    sns.lineplot(x=flag_rate_test, y=sens_test, label="Test", color="blue")
    sns.lineplot(x=flag_rate_val, y=sens_val, label="Validation", color="orange")
    sns.lineplot(x=flag_rate_train, y=sens_train, label="Train", color="green")
    plt.xlabel('Flag Rate')
    plt.ylabel('Sensitivity')
    plt.title('Sensitivity/Flag Curve')
    plt.legend()

    # 5. Sensitivity, Specificity, PPV by Threshold
    # Test Metrics
    sensitivity_test = tpr_test
    specificity_test = 1 - fpr_test
    ppv_test = precision_test[:-1]

    # Validation Metrics
    sensitivity_val = tpr_val
    specificity_val = 1 - fpr_val
    ppv_val = precision_val[:-1]

    # Train Metrics
    sensitivity_train = tpr_train
    specificity_train = 1 - fpr_train
    ppv_train = precision_train[:-1]

    # Plot Test Metrics
    plt.subplot(2, 5, 3)
    sns.lineplot(x=thresholds_roc_test, y=sensitivity_test, label="Test Sensitivity", color="blue")
    sns.lineplot(x=thresholds_roc_test, y=specificity_test, label="Test Specificity", color="green")
    sns.lineplot(x=thresholds_pr_test, y=ppv_test, label="Test PPV", color="magenta")
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title("Test Metrics by Threshold")
    plt.legend()

    # Plot Validation Metrics
    plt.subplot(2, 5, 4)
    sns.lineplot(x=thresholds_roc_val, y=sensitivity_val, label="Validation Sensitivity", linestyle="--", color="orange")
    sns.lineplot(x=thresholds_roc_val, y=specificity_val, label="Validation Specificity", linestyle="--", color="darkgreen")
    sns.lineplot(x=thresholds_pr_val, y=ppv_val, label="Validation PPV", linestyle="--", color="pink")
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title("Validation Metrics by Threshold")
    plt.legend()

    # Plot Train Metrics
    plt.subplot(2, 5, 5)
    sns.lineplot(x=thresholds_roc_train, y=sensitivity_train, label="Train Sensitivity", linestyle=":", color="purple")
    sns.lineplot(x=thresholds_roc_train, y=specificity_train, label="Train Specificity", linestyle=":", color="brown")
    sns.lineplot(x=thresholds_pr_train, y=ppv_train, label="Train PPV", linestyle=":", color="cyan")
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title("Train Metrics by Threshold")
    plt.legend()

    # 6. Histogram of Predicted Probabilities
    def _get_highest_bin_count(values, bins):
        counts, _ = np.histogram(values, bins=bins)
        return counts.max()

    highest_bin_count = max(
        _get_highest_bin_count(y_pred[y_test == 0], bins=20),
        _get_highest_bin_count(y_pred[y_test == 1], bins=20),
        _get_highest_bin_count(y_val_pred[y_val == 0], bins=20),
        _get_highest_bin_count(y_val_pred[y_val == 1], bins=20),
        _get_highest_bin_count(y_train_pred[y_train == 0], bins=20),
        _get_highest_bin_count(y_train_pred[y_train == 1], bins=20)
    )
    highest_bin_count += highest_bin_count//20

    plt.subplot(2, 5, 8)
    sns.histplot(y_pred[y_test == 0], bins=20, alpha=0.7, label="Test Actual False", color='blue', kde=False)
    sns.histplot(y_pred[y_test == 1], bins=20, alpha=0.7, label="Test Actual True", color='magenta', kde=False)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Histogram of Predicted Probabilities")
    plt.ylim(0, highest_bin_count)
    plt.legend()

    plt.subplot(2, 5, 9)
    sns.histplot(y_val_pred[y_val == 0], bins=20, alpha=0.5, label="Validation Actual False", color='orange', kde=False)
    sns.histplot(y_val_pred[y_val == 1], bins=20, alpha=0.5, label="Validation Actual True", color='pink', kde=False)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Histogram of Predicted Probabilities")
    plt.ylim(0, highest_bin_count)
    plt.legend()

    plt.subplot(2, 5, 10)
    sns.histplot(y_train_pred[y_train == 0], bins=20, alpha=0.5, label="Train Actual False", color='green', kde=False)
    sns.histplot(y_train_pred[y_train == 1], bins=20, alpha=0.5, label="Train Actual True", color='purple', kde=False)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Histogram of Predicted Probabilities")
    plt.ylim(0, highest_bin_count)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'model_evaluation.png')
    plt.close()
