import numpy as np

"""

SEGMENTATION METRICS

"""

def segmentation_metrics(segmentation: np.ndarray, ground_truth: np.ndarray, spacing: None | tuple = None) -> dict:
    """
    Calculate segmentation evaluation metrics.
    
    Parameters:
    - segmentation (ndarray): Segmented volume.
    - ground_truth (ndarray): Ground truth volume.
    - spacing (tuple): Spacing in each dimension (optional).
    
    Returns:
    - metrics (dict): Dictionary containing segmentation evaluation metrics.
    """

    raise NotImplementedError


def _volumetric_dice(segmentation: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculate Volumetric Dice coefficient.
    
    Parameters:
    - segmentation (ndarray): Segmented volume.
    - ground_truth (ndarray): Ground truth volume.
    
    Returns:
    - dice (float): Volumetric Dice coefficient.
    """
    
    raise NotImplementedError

def _surface_dice(segmentation: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculate Surface Dice coefficient.
    
    Parameters:
    - segmentation (ndarray): Segmented volume.
    - ground_truth (ndarray): Ground truth volume.
    
    Returns:
    - dice (float): Surface Dice coefficient.
    """
    
    raise NotImplementedError

def _hd95(segmentation: np.ndarray, ground_truth: np.ndarray, spacing: None | tuple = None) -> float:
    """
    Calculate HD95 (Hausdorff distance 95th percentile).
    
    Parameters:
    - segmentation (ndarray): Segmented volume.
    - ground_truth (ndarray): Ground truth volume.
    - spacing (tuple): Spacing in each dimension (optional).
    
    Returns:
    - hd95_distance (float): HD95 distance.
    """
    
    raise NotImplementedError

def _clinical_acceptability_surrogate(segmentation: np.ndarray, ground_truth: np.ndarray, threshold: None | float = None) -> bool:
    """
    Calculate Clinical Acceptability Surrogate.
    
    Parameters:
    - segmentation (ndarray): Segmented volume.
    - ground_truth (ndarray): Ground truth volume.
    - threshold (float): Threshold value for acceptability (optional).
    
    Returns:
    - acceptability (bool): True if within threshold, False otherwise.
    """
    
    raise NotImplementedError

"""

TIME TO EVENT METRICS

"""

def time_to_event_metrics(time: np.ndarray, event: np.ndarray) -> dict:
    """
    Calculate multiple metrics for time-to-event data.
    
    Parameters:
    - time (array-like): Time variable for each event.
    - event (array-like): Binary variable indicating whether event occurred.
    
    Returns:
    - metrics (dict): Dictionary containing time-to-event metrics.
    """

    raise NotImplementedError

def _calculate_event_rates(time: np.ndarray, event: np.ndarray) -> float:
    """
    Calculate Event Rates for time-to-event data.
    
    Parameters:
    - time (array-like): Time variable for each event.
    - event (array-like): Binary variable indicating whether event occurred.
    
    Returns:
    - event_rates (float): Event rates at different time points.
    """

    raise NotImplementedError

def _calculate_ci_(time: np.ndarray, event: np.ndarray, confidence_level: float = 0.95) -> tuple:
    """
    Calculate Confidence Interval for time-to-event data.
    
    Parameters:
    - time (array-like): Time variable for each event.
    - event (array-like): Binary variable indicating whether event occurred.
    - confidence_level (float): Confidence level for CI (default: 0.95).
    
    Returns:
    - ci (tuple): Tuple containing lower and upper bounds of the CI.
    """

    raise NotImplementedError

"""

MULTI-CLASS METRICS

"""

# CHAT GPT CODE!!!

def multi_class_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Calculate multiple metrics for multi-class classification.
    
    Parameters:
    - y_true (array-like): True labels.
    - y_pred_prob (array-like): Predicted probabilities for each class.
    - threshold (float): Threshold for converting probabilities to binary predictions (default: 0.5).
    
    Returns:
    - metrics (dict): Dictionary containing multi-class metrics.
    """
    metrics = {}
    
    # Convert probabilities to binary predictions
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    # Calculate AUC
    auc = roc_auc_score(y_true, y_pred_prob, multi_class='ovr')
    metrics['AUC'] = auc
    
    # Calculate Precision
    precision = precision_score(y_true, y_pred, average='weighted')
    metrics['Precision'] = precision
    
    # Calculate Recall
    recall = recall_score(y_true, y_pred, average='weighted')
    metrics['Recall'] = recall
    
    # Calculate Sensitivity and Specificity at various thresholds
    sensitivity_specificity = sensitivity_specificity_at_thresholds(y_true, y_pred_prob)
    metrics.update(sensitivity_specificity)
    
    # Calculate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['Confusion Matrix'] = cm
    
    return metrics

def _sensitivity_specificity_at_thresholds(y_true: np.ndarray, y_pred_prob: np.ndarray) -> dict:
    """
    Calculate sensitivity and specificity at various thresholds.
    
    Parameters:
    - y_true (array-like): True labels.
    - y_pred_prob (array-like): Predicted probabilities for each class.
    
    Returns:
    - sensitivity_specificity (dict): Dictionary containing sensitivity and specificity at various thresholds.
    """
    sensitivity_specificity = {}
    
    thresholds = np.linspace(0, 1, num=100)
    for threshold in thresholds:
        y_pred = (y_pred_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        sensitivity_specificity[threshold] = {'Sensitivity': sensitivity, 'Specificity': specificity}
    
    return sensitivity_specificity


