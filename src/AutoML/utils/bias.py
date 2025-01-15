import inspect, re
from typing import List

from functools import partial
import pandas as pd
from tabulate import tabulate
import fairlearn.metrics as fm

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

        self.results = []
        for sensitive_feature in self.sensitive_features.columns:
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
            result.rename(columns=self.mapper, inplace=True)

            if relative:
                results_relative = result.T / result[largest_feature]
                results_relative = results_relative.applymap(lambda x: f"{x} ✅" if x <= fairness_threshold or 1/x <= fairness_threshold else f"{x} ❌")
                result = pd.concat([result, results_relative.T.rename(index=lambda x: f"Relative {x}")])

            self.results.append(result)
            print(f"Subgroup Analysis({sensitive_feature.title()})")
            print(f'{tabulate(result.iloc[:, :4], headers='keys', tablefmt='fancy_grid')}\n')

        return self.results
