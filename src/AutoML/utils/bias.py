import inspect, re
from typing import List

from functools import partial
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from tabulate import tabulate
import fairlearn.metrics as fm

from sklearn.metrics import log_loss
import statsmodels.api as sm

def infer_sensitive_features(data: pd.DataFrame) -> dict:
    """
    Infers potentially sensitive features from a DataFrame.
    """
    sensitive_keywords = [
        "gender", "sex", "age", "race", "ethnicity", "income",
        "religion", "disability", "nationality", "language",
        "marital", "citizen", "veteran", "status", "orientation",
        "disease", "regimen", "disease_site"
    ]

    sensitive_features = {
        col for col in data.columns 
        if any(re.search(rf"\b{keyword}\b", col, re.IGNORECASE) for keyword in sensitive_keywords)
    }

    return {sens_feat: data[sens_feat] for sens_feat in sensitive_features}

def get_metric(metric, sensitive_features=None):
    fn = getattr(fm, metric)
    params = inspect.signature(fn).parameters
    return partial(fn, sensitive_features=sensitive_features) if 'sensitive_features' in params and sensitive_features else fn

class BiasExplainer():
    def __init__(
            self, 
            y_true: pd.DataFrame, 
            y_pred: pd.DataFrame, 
            sensitive_features: dict, 
            metrics: list = ['mean_prediction', 'false_positive_rate', 'true_positive_rate'], 
            **kwargs: dict
        ) -> None:

        self.y_true = y_true
        self.y_pred = y_pred
        self.mapper = {"mean_prediction": "Demographic Parity",
                       "false_positive_rate": "(FPR) Equalized Odds",
                       "true_positive_rate": "(TPR) Equalized Odds or Equal Opportunity"}
        self.metrics = metrics
        self.kwargs = kwargs
        
        # Convert sensitive_features to DataFrame or leave as Series
        if isinstance(sensitive_features, pd.DataFrame) or isinstance(sensitive_features, pd.Series):
            self.sensitive_features = sensitive_features
        elif isinstance(sensitive_features, dict):
            self.sensitive_features = pd.DataFrame.from_dict(sensitive_features)
        elif isinstance(sensitive_features, list):
            if any(isinstance(item, list) for item in sensitive_features):
                self.sensitive_features = pd.DataFrame(sensitive_features, columns=[f'sensitive_feature_{i}' for i in range(len(sensitive_features))])
            else:
                self.sensitive_features = pd.DataFrame(sensitive_features, columns=['sensitive_feature'])
        else:
            raise ValueError("sensitive_features must be a pandas DataFrame, Series, dictionary or list")

    def run(
            self, 
            relative: bool = False, 
            fairness_threshold: float = 1.2
        ) -> List[pd.DataFrame]:

        log_loss_per_patient = self.y_true.index.map(
            lambda idx: log_loss([self.y_true[idx]], [self.y_pred[idx]], labels=self.y_true.unique())
        )

        log_loss_per_patient = np.array(log_loss_per_patient)

        self.y_pred = (self.y_pred >= .5).astype(int)

        self.results = []
        for sensitive_feature in self.sensitive_features.columns:
            plt.figure(figsize=(8, 6))  # Set the figure size for better readability
            sns.set(style="whitegrid")  # Apply a white grid style for a cleaner look

            # Create the plot
            sns.violinplot(
                x=self.sensitive_features[sensitive_feature], 
                y=log_loss_per_patient, 
                palette="muted",  # Use a muted color palette for a refined look
                inner="quart",  # Show quartiles for better data insight
                linewidth=1.25  # Slightly thicker lines for better visibility
            )

            # Customize the labels and title
            plt.title(f'Log Loss Distribution by {sensitive_feature}', fontsize=16, weight='bold')  # Title with bold font
            plt.xlabel(f'{sensitive_feature}', fontsize=14)  # Label for the x-axis
            plt.ylabel('Log Loss per Patient', fontsize=14)  # Label for the y-axis

            # Rotate x-axis labels if necessary (e.g., for long feature names)
            plt.xticks(rotation=45, ha='right')

            # Display the plot
            plt.tight_layout()  # Adjust the layout to prevent clipping of labels
            plt.show()

            one_hot_encoded = pd.get_dummies(self.sensitive_features[sensitive_feature], prefix=sensitive_feature)

            X = one_hot_encoded.values  
            y = log_loss_per_patient  

            model = sm.OLS(y, X).fit()
            print(model.summary())

            largest_feature = self.sensitive_features[sensitive_feature].mode().iloc[0]
            _metrics = {metric: get_metric(metric, sensitive_features=self.sensitive_features[sensitive_feature]) for metric in self.metrics}

            metric_frame = fm.MetricFrame(
                metrics=_metrics, 
                y_true=self.y_true, 
                y_pred=self.y_pred, 
                sensitive_features=self.sensitive_features[sensitive_feature], 
                **self.kwargs
            )
            result = pd.DataFrame(metric_frame.by_group.T, index=_metrics.keys())
            result = result.rename(columns=self.mapper)
            result = result.applymap(lambda x: f"{x:.3f}")

            if relative:
                results_relative = result.T / result[largest_feature]
                results_relative = results_relative.applymap(lambda x: f"{x:.3f} ✅" if x <= fairness_threshold or 1/x <= fairness_threshold else f"{x:.3f} ❌")
                result = pd.concat([result, results_relative.T.rename(index=lambda x: f"Relative {x}")])

            self.results.append(result)
            print(f"Subgroup Analysis({sensitive_feature.title()})")
            print(f'{tabulate(result.iloc[:, :4], headers='keys', tablefmt='fancy_grid')}\n')

        return self.results
