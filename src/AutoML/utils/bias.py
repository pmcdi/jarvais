import inspect, re
from functools import partial

import pandas as pd
import fairlearn.metrics as fm


def infer_sensitive_features(data: pd.DataFrame) -> list:
    """
    Infers potentially sensitive features from a DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataset.
        
    Returns
    -------
    sensitive_features : list
        List of column names identified as potentially sensitive.
    """
    # Common sensitive keywords
    sensitive_keywords = [
        "gender", "sex", "age", "race", "ethnicity", "income",
        "religion", "disability", "nationality", "language",
        "marital", "citizen", "veteran", "status", "orientation"
        "disease", "regimen", "disease_site"
    ]

    sensitive_features = []

    # Check column names for sensitive keywords
    for col in data.columns:
        if any(re.search(rf"\b{keyword}\b", col, re.IGNORECASE) for keyword in sensitive_keywords):
            sensitive_features.append(col)

    # Remove duplicates
    sensitive_features = list(set(sensitive_features))

    return sensitive_features

def get_metric(metric, sensitive_features=None):
    fn = getattr(fm, metric)
    params = inspect.signature(fn).parameters
    if 'sensitive_features' in params and sensitive_features is not None:
        return partial(fn, sensitive_features=sensitive_features)
    else:
        return fn

class BiasExplainer():
    def __init__(self,
                 y_true, y_pred, sensitive_features: dict, metrics=['mean_prediction', 'false_positive_rate'], **kwargs):
        
        self.y_true = y_true
        self.y_pred = y_pred

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
        
        self.largest_features   = self.sensitive_features.groupby(self.sensitive_features.columns.tolist()).size().idxmax()
        self.metrics            = {metric: get_metric(metric, sensitive_features=sensitive_features) for metric in metrics}
        self.metric_frame       = self.get_metric_frame(**kwargs)

    def get_metric_frame(self, **kwargs):
        return fm.MetricFrame(metrics=self.metrics,
                              y_true=self.y_true,
                              y_pred=self.y_pred,
                              sensitive_features=self.sensitive_features,
                              **kwargs)

    def run(self, relative: bool=False):
        self.results = pd.DataFrame(self.metric_frame.by_group.T, index=self.metrics.keys())
        if relative:
            self.run_relative_largest()

        print("Here are the bias subgroup analysis results:\n")
        print(self.results)

    def run_relative_largest(self):
        print(f"Calculating relative metric to the largest subgroup: {self.largest_features}")
        results_relative = self.results.T/self.results[self.largest_features]
        self.results = pd.concat([self.results, results_relative.T.rename(index=lambda x: f"Relative {x}")])
