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
from lifelines import CoxPHFitter

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
            y_true: pd.Series, 
            y_pred: np.ndarray, 
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

    def _subgroup_analysis_OLS(self, sensitive_feature: str, bias_metric:np.ndarray) -> float:
        """Fit a statsmodels OLS model to the bias metric data, using the sensitive feature and print summary based on p_val."""
        one_hot_encoded = pd.get_dummies(self.sensitive_features[sensitive_feature], prefix=sensitive_feature)
        X_columns = one_hot_encoded.columns

        X = one_hot_encoded.values  
        y = bias_metric  

        X = sm.add_constant(X.astype(float), has_constant='add')
        model = sm.OLS(y, X).fit()

        if model.f_pvalue < 0.05:
            output = []

            print(f"⚠️  **Possible Bias Detected in {sensitive_feature.title()}** ⚠️\n")
            output.append(f"=== Subgroup Analysis for '{sensitive_feature.title()}' Using OLS Regression ===\n")

            output.append("Model Statistics:")
            output.append(f"    R-squared:                  {model.rsquared:.3f}")
            output.append(f"    F-statistic:                {model.fvalue:.3f}")
            output.append(f"    F-statistic p-value:        {model.f_pvalue:.4f}")
            output.append(f"    AIC:                        {model.aic:.2f}")
            output.append(f"    Log-Likelihood:             {model.llf:.2f}")

            summary_df = pd.DataFrame({
                'Feature': ['const'] + X_columns.tolist(),     # Predictor names (includes 'const' if added)
                'Coefficient': model.params,    # Coefficients
                'Standard Error': model.bse     # Standard Errors
            })
            table_output = tabulate(summary_df, headers='keys', tablefmt='grid', showindex=False, floatfmt=".3f")
            output.append("Model Coefficients:")
            output.append('\n'.join(['    ' + line for line in table_output.split('\n')]))

            output_text = '\n'.join(output)
            print(output_text)

            with open(self.output_dir / f'{sensitive_feature}_Cox_model_summary.txt', 'w') as f:
                f.write(output_text)

        return model.f_pvalue

    def _subgroup_analysis_CoxPH(self, sensitive_feature: str) -> None:
        """Fit a CoxPH model using the sensitive feature and print summary based on p_val."""
        one_hot_encoded = pd.get_dummies(self.sensitive_features[sensitive_feature], prefix=sensitive_feature)
        df_encoded = self.y_true.join(one_hot_encoded)

        cph = CoxPHFitter(penalizer=0.0001)
        cph.fit(df_encoded, duration_col='time', event_col='event')            
        
        if cph.log_likelihood_ratio_test().p_value < 0.05:
            output = []

            print(f"⚠️  **Possible Bias Detected in {sensitive_feature.title()}** ⚠️")
            output.append(f"=== Subgroup Analysis for '{sensitive_feature.title()}' Using Cox Proportional Hazards Model ===\n")

            output.append("Model Statistics:")
            output.append(f"    AIC (Partial):               {cph.AIC_partial_:.2f}")
            output.append(f"    Log-Likelihood:              {cph.log_likelihood_:.2f}")
            output.append(f"    Log-Likelihood Ratio p-value: {cph.log_likelihood_ratio_test().p_value:.4f}")
            output.append(f"    Concordance Index (C-index):   {cph.concordance_index_:.2f}")

            summary_df = pd.DataFrame({
                'Feature': cph.summary.index.to_list(),
                'Coefficient': cph.summary['coef'].to_list(),
                'Standard Error': cph.summary['se(coef)'].to_list()
            })
            table_output = tabulate(summary_df, headers='keys', tablefmt='grid', showindex=False, floatfmt=".3f")
            output.append("Model Coefficients:")
            output.append('\n'.join(['    ' + line for line in table_output.split('\n')]))

            output_text = '\n'.join(output)
            print(output_text)

            with open(self.output_dir / f'{sensitive_feature}_OLS_model_summary.txt', 'w') as f:
                f.write(output_text)

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
            y_true_array = self.y_true.to_numpy()
            bias_metric = np.array([
                log_loss([y_true_array[idx]], [self.y_pred[idx]], labels=np.unique(y_true_array))
                for idx in range(len(y_true_array))
            ])
            self.y_pred = (self.y_pred >= .5).astype(int)
        elif self.task == 'regression':
            bias_metric = np.sqrt((self.y_true.to_numpy() - self.y_pred) ** 2)

        self.results = []
        for sensitive_feature in self.sensitive_features.columns:
            if self.task == 'survival':
                self._subgroup_analysis_CoxPH(sensitive_feature)
            else:
                f_pvalue = self._subgroup_analysis_OLS(sensitive_feature, bias_metric)
                if f_pvalue < 0.05:
                    self._generate_violin(sensitive_feature, bias_metric)
                    result = self._calculate_fair_metrics(sensitive_feature, fairness_threshold, relative)

                    print(f"\n=== Subgroup Analysis for '{sensitive_feature.title()}' using FairLearn ===\n")
                    table_output = tabulate(result.iloc[:, :4], headers='keys', tablefmt='grid')
                    print('\n'.join(['    ' + line for line in table_output.split('\n')]), '\n')

                    result.to_csv(self.output_dir / f'{sensitive_feature}_fm_metrics.csv')
