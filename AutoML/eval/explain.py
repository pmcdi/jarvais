
# CHAT GPT CODE!!!

import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from typing import Union, Tuple
import pandas as pd

# Seems redundant abstraction make into one big thing

class ModelExplainability:
    def __init__(self, model: str, X_train: np.ndarray | pd.DataFrame) -> None:
        """
        Initialize the ModelExplainability class with the given model and training data.

        Parameters:
        - model (str): The machine learning model to be explained.
        - X_train (np.ndarray or pd.DataFrame): The training data used to train the model.
        """
        self.model = model
        self.X_train = X_train
        self.explainer = shap.Explainer(model, X_train)
    
    def shap_summary_plot(self, X_test: np.ndarray | pd.DataFrame) -> None:
        """
        Create a SHAP summary plot for the test data.

        Parameters:
        - X_test (np.ndarray or pd.DataFrame): The test data for which to generate the SHAP summary plot.
        """
        shap_values = self.explainer(X_test)
        shap.summary_plot(shap_values, X_test)
    
    def shap_dependence_plot(self, feature: int | str, X_test: np.ndarray | pd.DataFrame) -> None:
        """
        Create a SHAP dependence plot for a specified feature.

        Parameters:
        - feature (int or str): The feature for which to generate the SHAP dependence plot.
        - X_test (np.ndarray or pd.DataFrame): The test data for which to generate the SHAP dependence plot.
        """
        shap_values = self.explainer(X_test)
        shap.dependence_plot(feature, shap_values, X_test)
    
    def shap_force_plot(self, X_test: np.ndarray | pd.DataFrame, instance_idx: int) -> None:
        """
        Create a SHAP force plot for a specific instance in the test data.

        Parameters:
        - X_test (np.ndarray or pd.DataFrame): The test data for which to generate the SHAP force plot.
        - instance_idx (int): The index of the instance in the test data to explain.
        """
        shap_values = self.explainer(X_test)
        shap.force_plot(self.explainer.expected_value, shap_values[instance_idx], X_test[instance_idx])

class ModelCalibration:
    def __init__(self, model: str, X_train: np.ndarray | pd.DataFrame) -> None:
        """
        Initialize the ModelCalibration class with the given model and training data.

        Parameters:
        - model (str): The machine learning model to be calibrated.
        - X_train (np.ndarray or pd.DataFrame): The training data used to train the model.
        """
        self.model = model
        self.X_train = X_train
    
    def plot_calibration_curve(self, X_test: np.ndarray | pd.DataFrame, y_test: np.ndarray, n_bins: int = 10) -> None:
        """
        Plot the calibration curve for the test data.

        Parameters:
        - X_test (np.ndarray or pd.DataFrame): The test data for which to generate the calibration curve.
        - y_test (np.ndarray): The target values for the test data.
        - n_bins (int): The number of bins to use for the calibration curve (default: 10).
        """
        y_prob = self.model.predict_proba(X_test)[:, 1]
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=n_bins)

        plt.figure(figsize=(10, 10))
        plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Calibration curve')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Calibration curve')
        plt.legend()
        plt.show()

    def calibration_error(self, X_test: np.ndarray | pd.DataFrame, y_test: np.ndarray) -> float:
        """
        Calculate the calibration error (Brier score) for the test data.

        Parameters:
        - X_test (np.ndarray or pd.DataFrame): The test data for which to calculate the calibration error.
        - y_test (np.ndarray): The target values for the test data.

        Returns:
        - float: The calibration error (Brier score).
        """
        y_prob = self.model.predict_proba(X_test)[:, 1]
        brier_score = brier_score_loss(y_test, y_prob)
        return brier_score
