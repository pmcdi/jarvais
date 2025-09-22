from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pydantic import BaseModel, Field
import pandas as pd
from tabulate import tabulate
import fairlearn.metrics as fm
from sklearn.metrics import log_loss
import statsmodels.api as sm
from lifelines import CoxPHFitter

from jarvais.utils.functional import undummify, infer_sensitive_features
from jarvais.loggers import logger

if TYPE_CHECKING:
    from jarvais.trainer import TrainerSupervised

def get_metric(metric: str, sensitive_features : list | None = None): # noqa: ANN201
    fn = getattr(fm, metric)
    params = inspect.signature(fn).parameters
    return partial(fn, sensitive_features=sensitive_features) if 'sensitive_features' in params and sensitive_features else fn


class BiasAuditModule(BaseModel):
    output_dir: Path = Field(
        description="Output directory for bias audit.",
        title="Output Directory",
        examples=["output"]
    )
    sensitive_features: list | None = Field(
        description="Sensitive features.",
        title="Sensitive Features",
        examples=["gender", "race", "ethnicity"]
    )
    fairness_threshold: float = Field(
        description="Fairness threshold.", 
        default=1.2,
        title="Fairness Threshold",
    )
    relative: bool = Field(
        description="Relative.",
        default=False,
        title="Relative",
    )

    def model_post_init(self, __context) -> None: # noqa: ANN001
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, trainer: "TrainerSupervised") -> None:

        logger.info("Running Bias Audit Module...")
        
        if self.sensitive_features is None:
            logger.info("No sensitive features provided, inferring from data...")
            test_data = (
                undummify(trainer.X_test, prefix_sep=trainer.settings.encoding_module.prefix_sep)
                if trainer.settings.encoding_module.enabled
                else trainer.X_test
            )
            self.sensitive_features = infer_sensitive_features(test_data)
        
        y_pred = None if trainer.settings.task == 'survival' else pd.Series(trainer.infer(trainer.X_test) )
        metrics = ['mean_prediction'] if trainer.settings.task == 'regression' else ['mean_prediction', 'false_positive_rate']

        if trainer.settings.task == 'binary':
            y_true_array = trainer.y_test.to_numpy()
            bias_metric = np.array([
                log_loss([y_true_array[idx]], [y_pred[idx]], labels=np.unique(y_true_array))
                for idx in range(len(y_true_array))
            ])
            y_pred = (y_pred >= .5).astype(int)
        elif trainer.settings.task == 'regression':
            bias_metric = np.sqrt((trainer.y_test.to_numpy() - y_pred) ** 2)

        for sensitive_feature_name in self.sensitive_features:
            _sensitive_column = trainer.X_test[sensitive_feature_name]
            if trainer.settings.task == 'survival':
                self._subgroup_analysis_coxph(trainer.y_test, _sensitive_column)
            else:
                f_pvalue = self._subgroup_analysis_ols(_sensitive_column, bias_metric)
                if f_pvalue < 0.05:
                    self._generate_violin(_sensitive_column, bias_metric, trainer.settings.task)
                    result = self._calculate_fair_metrics(trainer.y_test, y_pred, _sensitive_column, self.fairness_threshold, self.relative, metrics)

                    print(f"\n=== Subgroup Analysis for '{sensitive_feature_name.title()}' using FairLearn ===\n") # noqa: T201
                    table_output = tabulate(result.iloc[:, :4], headers='keys', tablefmt='grid')
                    print('\n'.join(['    ' + line for line in table_output.split('\n')]), '\n') # noqa: T201

                    result.to_csv(self.output_dir / f'{sensitive_feature_name}_fm_metrics.csv')
        

    def _generate_violin(self, sensitive_column: pd.Series, bias_metric:np.ndarray, task: str) -> None:
        """Generate a violin plot for the bias metric."""
        plt.figure(figsize=(8, 6)) 
        sns.set_theme(style="whitegrid")  

        sns.violinplot(
            x=sensitive_column, 
            y=bias_metric, 
            palette="muted",  
            inner="quart", 
            linewidth=1.25 
        )

        bias_metric_name = 'log_loss' if task == 'binary' else 'root_mean_squared_error'

        plt.title(f'{bias_metric_name.title()} Distribution by {sensitive_column.name}', fontsize=16, weight='bold')  
        plt.xlabel(f'{sensitive_column.name}', fontsize=14)  
        plt.ylabel(f'{bias_metric_name.title()} per Patient', fontsize=14) 
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()  
        plt.savefig(self.output_dir / f'{sensitive_column.name}_{bias_metric_name}.png') 
        plt.show()

    def _subgroup_analysis_ols(self, sensitive_column: pd.Series, bias_metric: np.ndarray) -> float:
        """Fit a statsmodels OLS model to the bias metric data, using the sensitive feature and print summary based on p_val."""
        one_hot_encoded = pd.get_dummies(sensitive_column, prefix=sensitive_column.name)
        X_columns = one_hot_encoded.columns

        X = one_hot_encoded.values  
        y = bias_metric  

        X = sm.add_constant(X.astype(float), has_constant='add')
        model = sm.OLS(y, X).fit()

        if model.f_pvalue < 0.05:
            output = []

            print(f"⚠️  **Possible Bias Detected in {sensitive_column.name.title()}** ⚠️\n") # noqa: T201
            output.append(f"=== Subgroup Analysis for '{sensitive_column.name.title()}' Using OLS Regression ===\n")

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
            print(output_text) # noqa: T201

            with (self.output_dir / f'{sensitive_column.name}_Cox_model_summary.txt').open('w') as f:
                f.write(output_text)

        return model.f_pvalue

    def _subgroup_analysis_coxph(self, y_true: pd.Series, sensitive_column: pd.Series) -> None:
        """Fit a CoxPH model using the sensitive feature and print summary based on p_val."""
        one_hot_encoded = pd.get_dummies(sensitive_column, prefix=sensitive_column.name)
        df_encoded = y_true.join(one_hot_encoded)

        cph = CoxPHFitter(penalizer=0.0001)
        cph.fit(df_encoded, duration_col='time', event_col='event')            
        
        if cph.log_likelihood_ratio_test().p_value < 0.05:
            output = []

            print(f"⚠️  **Possible Bias Detected in {sensitive_column.name.title()}** ⚠️") # noqa: T201
            output.append(f"=== Subgroup Analysis for '{sensitive_column.name.title()}' Using Cox Proportional Hazards Model ===\n")

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
            print(output_text) # noqa: T201

            with (self.output_dir / f'{sensitive_column.name}_OLS_model_summary.txt').open('w') as f:
                f.write(output_text)

    def _calculate_fair_metrics(
            self, 
            y_true: pd.Series,
            y_pred: pd.Series,
            sensitive_column: pd.Series, 
            fairness_threshold: float, 
            relative: bool,
            metrics: list
        ) -> pd.DataFrame:
        """Calculate the Fairlearn metrics and return the results in a DataFrame."""
        _metrics = {metric: get_metric(metric, sensitive_features=sensitive_column) for metric in metrics}
        metric_frame = fm.MetricFrame(
            metrics=_metrics, 
            y_true=y_true, 
            y_pred=y_pred, 
            sensitive_features=sensitive_column, 
        )
        result = pd.DataFrame(metric_frame.by_group.T, index=_metrics.keys())
        result = result.rename(
                columns={
                    "mean_prediction": "Demographic Parity",
                    "false_positive_rate": "(FPR) Equalized Odds",
                    "true_positive_rate": "(TPR) Equalized Odds or Equal Opportunity"
                }
            )

        if relative:
            largest_feature = sensitive_column.mode().iloc[0]
            results_relative = result.T / result[largest_feature]
            results_relative = results_relative.applymap(
                lambda x: f"{x:.3f} ✅" if x <= fairness_threshold or 1/x <= fairness_threshold 
                else f"{x:.3f} ❌")
            result = pd.concat([result, results_relative.T.rename(index=lambda x: f"Relative {x}")])
        
        return result

