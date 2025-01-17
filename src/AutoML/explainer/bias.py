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
    """
    A class for explaining and analyzing bias in a predictive model's outcomes based on sensitive features.

    This class performs various fairness audits by evaluating predictive outcomes with respect to sensitive features such as
    gender, age, race, and more. It calculates fairness metrics (e.g., demographic parity, equalized odds), generates visualizations
    (e.g., violin plots), and runs statistical analyses (e.g., OLS regression) to assess any bias in model predictions. The
    results are presented for each sensitive feature, with optional relative fairness comparisons.

    Attributes:
        y_true (pd.DataFrame):
            The true target values for the model.
        y_pred (pd.DataFrame):
            The predicted values from the model.
        sensitive_features (dict or pd.DataFrame):
            A dictionary or DataFrame containing sensitive features used for fairness analysis.
        metrics (list):
            A list of metrics to calculate for fairness analysis. Defaults to ['mean_prediction', 'false_positive_rate', 'true_positive_rate'].
        mapper (dict):
            A dictionary mapping internal metric names to user-friendly descriptions.
        kwargs (dict):
            Additional parameters passed to various methods, such as metric calculation and plot generation.
    """
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
        
    def _generate_violin(self, sensitive_feature: str, log_loss_per_patient:np.ndarray) -> None:
        """Generate a violin plot for log loss distribution."""
        plt.figure(figsize=(8, 6)) 
        sns.set_theme(style="whitegrid")  

        sns.violinplot(
            x=self.sensitive_features[sensitive_feature], 
            y=log_loss_per_patient, 
            palette="muted",  
            inner="quart", 
            linewidth=1.25 
        )

        plt.title(f'Log Loss Distribution by {sensitive_feature}', fontsize=16, weight='bold')  
        plt.xlabel(f'{sensitive_feature}', fontsize=14)  
        plt.ylabel('Log Loss per Patient', fontsize=14) 
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()  
        plt.show()

    def _fit_OLS(self, sensitive_feature: str, log_loss_per_patient:np.ndarray) -> None:
        """Fit a statsmodels OLS model to the log loss data."""
        one_hot_encoded = pd.get_dummies(self.sensitive_features[sensitive_feature], prefix=sensitive_feature)

        X = one_hot_encoded.values  
        y = log_loss_per_patient  

        X_columns = one_hot_encoded.columns  
        X = sm.add_constant(X.astype(float), has_constant='add')

        model = sm.OLS(y, X).fit()

        print(model.summary(xname=['const'] + X_columns.tolist()))
    
    def _calculate_fair_metrics(
            self, 
            sensitive_feature: str, 
            fairness_threshold: float, 
            relative: bool
        ) -> pd.DataFrame:
        """Calculate the Fairlearn metrics and return the results in a DataFrame."""
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

        if relative:
            largest_feature = self.sensitive_features[sensitive_feature].mode().iloc[0]
            results_relative = result.T / result[largest_feature]
            results_relative = results_relative.applymap(
                lambda x: f"{x:.3f} ✅" if x <= fairness_threshold or 1/x <= fairness_threshold 
                else f"{x:.3f} ❌")
            result = pd.concat([result, results_relative.T.rename(index=lambda x: f"Relative {x}")])
        
        return result
    
    def run(
            self, 
            relative: bool = False, 
            fairness_threshold: float = 1.2
        ) -> List[pd.DataFrame]:
        """
    Runs the bias explainer analysis on the provided data, calculating fairness metrics and generating plots 
    for each sensitive feature in the dataset.

    Args:
        relative (bool): 
            If True, the metrics will be presented relative to the most frequent value of each sensitive feature.
        fairness_threshold (float): 
            A threshold for determining fairness based on relative metrics. If the relative metric exceeds this threshold, 
            a warning flag will be applied.

    Returns:
        List[pd.DataFrame]: 
            A list of DataFrames, where each DataFrame contains fairness metrics for each sensitive feature.
    """
        log_loss_per_patient = self.y_true.index.map(
            lambda idx: log_loss([self.y_true[idx]], [self.y_pred[idx]], labels=self.y_true.unique())
        )
        log_loss_per_patient = np.array(log_loss_per_patient)
        self.y_pred = (self.y_pred >= .5).astype(int)

        self.results = []
        for sensitive_feature in self.sensitive_features.columns:
            self._generate_violin(sensitive_feature, log_loss_per_patient)
            self._fit_OLS(sensitive_feature, log_loss_per_patient)
            result = self._calculate_fair_metrics(sensitive_feature, fairness_threshold, relative)

            self.results.append(result)
            print(f"Subgroup Analysis({sensitive_feature.title()})")
            print(f"{tabulate(result.iloc[:, :4], headers='keys', tablefmt='fancy_grid')}\n")

        return self.results
