import inspect, re
from typing import List
from pathlib import Path

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
    gender, age, race, and more. It first runs statistical analyses using the OLS regression F-statistic p-value to assess any possibility 
    of bias in the model's predictions based on sensitive features. If the p-value is less than 0.05, indicating potential bias, 
    the class generates visualizations (such as violin plots) and calculates fairness metrics (e.g., demographic parity, equalized odds). 
    The results are presented for each sensitive feature, with optional relative fairness comparisons.

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
            task: str,
            output_dir: Path,
            metrics: list = ['mean_prediction', 'false_positive_rate', 'true_positive_rate'], 
            **kwargs: dict
        ) -> None:
        self.y_true = y_true
        self.y_pred = y_pred
        self.task = task
        self.output_dir = output_dir
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
        
    def _generate_violin(self, sensitive_feature: str, bias_metric:np.ndarray) -> None:
        """Generate a violin plot for the bias metric."""
        plt.figure(figsize=(8, 6)) 
        sns.set_theme(style="whitegrid")  

        sns.violinplot(
            x=self.sensitive_features[sensitive_feature], 
            y=bias_metric, 
            palette="muted",  
            inner="quart", 
            linewidth=1.25 
        )

        bias_metric_name = 'log_loss' if self.task == 'binary' else 'root_mean_squared_error'

        plt.title(f'{bias_metric_name.title()} Distribution by {sensitive_feature}', fontsize=16, weight='bold')  
        plt.xlabel(f'{sensitive_feature}', fontsize=14)  
        plt.ylabel(f'{bias_metric_name.title()} per Patient', fontsize=14) 
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()  
        plt.savefig(self.output_dir / f'{sensitive_feature}_{bias_metric_name}.png') 
        plt.show()

    def _fit_OLS(self, sensitive_feature: str, bias_metric:np.ndarray) -> float:
        """Fit a statsmodels OLS model to the bias metric data."""
        one_hot_encoded = pd.get_dummies(self.sensitive_features[sensitive_feature], prefix=sensitive_feature)

        X = one_hot_encoded.values  
        y = bias_metric  

        X_columns = one_hot_encoded.columns  
        X = sm.add_constant(X.astype(float), has_constant='add')

        model = sm.OLS(y, X).fit()

        if model.f_pvalue < 0.05:
            summary = model.summary(xname=['const'] + X_columns.tolist())
            print(f'Possible Bias in {sensitive_feature.title()}:\n')
            print(summary)

            with (self.output_dir / f'{sensitive_feature}_model_summary.txt').open('w') as f:
                f.write(summary.as_text())

        return model.f_pvalue
    
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
        ) -> None:
        """
        Runs the bias explainer analysis on the provided data. It first evaluates the potential bias in the model's predictions
        using the OLS regression F-statistic p-value. If the p-value is below the threshold of 0.05, indicating 
        potential bias in the sensitive feature, the method proceeds to generate visualizations and calculate fairness metrics.

        Args:
            relative (bool): 
                If True, the metrics will be presented relative to the most frequent value of each sensitive feature.
            fairness_threshold (float): 
                A threshold for determining fairness based on relative metrics. If the relative metric exceeds this threshold, 
                a warning flag will be applied.
        """
        if self.task == 'binary':
            log_loss_per_patient = self.y_true.index.map(
                lambda idx: log_loss([self.y_true[idx]], [self.y_pred[idx]], labels=self.y_true.unique())
            )
            bias_metric = np.array(log_loss_per_patient)
            self.y_pred = (self.y_pred >= .5).astype(int)
        else: # Regression(root mean_squared_error)
            bias_metric = np.sqrt((self.y_true.to_numpy() - self.y_pred.to_numpy()) ** 2)

        self.results = []
        for sensitive_feature in self.sensitive_features.columns:
            f_pvalue = self._fit_OLS(sensitive_feature, bias_metric)
            if f_pvalue < 0.05:
                self._generate_violin(sensitive_feature, bias_metric)
                result = self._calculate_fair_metrics(sensitive_feature, fairness_threshold, relative)

                print(f"Subgroup Analysis({sensitive_feature.title()})")
                print(f"{tabulate(result.iloc[:, :4], headers='keys', tablefmt='fancy_grid')}\n")
                result.to_csv(self.output_dir / f'{sensitive_feature}_fm_metrics.csv')
