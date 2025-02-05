from pathlib import Path
from typing import List

import pandas as pd
from sklearn.inspection import permutation_importance
from sksurv.util import Surv

from ..utils.pdf import generate_explainer_report_pdf
from ..utils.functional import undummify
from ..utils.plot import (
    plot_classification_diagnostics,
    plot_feature_importance,
    plot_regression_diagnostics,
    plot_shap_values,
    plot_violin_of_bootstrapped_metrics,
)
from .bias import BiasExplainer, infer_sensitive_features

class Explainer():
    """
    A class to generate diagnostic plots and reports for models trained using TrainerSupervised.

    Attributes:
        trainer (TrainerSupervised): The TrainerSupervised object containing the trained model.
        predictor (object): The AutoGluon predictor object used for inference.
        X_train (pd.DataFrame): The training dataset used to train the model.
        X_test (pd.DataFrame): The test dataset for evaluating the model.
        y_test (pd.DataFrame): The true target values for the test dataset.
        output_dir (Path): The directory where plots, reports, and outputs are saved.
        sensitive_features (list, optional): List of features considered sensitive for bias auditing.
    """
    def __init__(
            self,
            trainer,
            X_train: pd.DataFrame,
            X_test: pd.DataFrame,
            y_test: pd.DataFrame,
            output_dir: str | Path | None = None,
            sensitive_features: list | None = None,
        ) -> None:

        self.trainer = trainer
        self.predictor = trainer.predictor
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.sensitive_features = sensitive_features

        self.output_dir = Path.cwd() if output_dir is None else Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'figures').mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """Generate diagnostic plots and reports for the trained model."""

        self._run_bias_audit()

        plot_violin_of_bootstrapped_metrics(
            self.trainer,
            self.X_test,
            self.y_test,
            self.trainer.X_val,
            self.trainer.y_val,
            self.X_train,
            self.trainer.y_train,
            output_dir=self.output_dir / 'figures'
        )            

        if self.trainer.task in ['binary', 'multiclass']:
            plot_classification_diagnostics(
                self.y_test,
                self.predictor.predict_proba(self.X_test).iloc[:, 1],
                self.trainer.y_val,
                self.predictor.predict_proba(self.trainer.X_val).iloc[:, 1],
                self.trainer.y_train,
                self.predictor.predict_proba(self.X_train).iloc[:, 1],
                output_dir=self.output_dir / 'figures'
            )
            plot_shap_values(
                self.predictor,
                self.X_train,
                self.X_test,
                output_dir=self.output_dir / 'figures'
            )

        elif self.trainer.task == 'regression':
            plot_regression_diagnostics(
                self.y_test,
                self.predictor.predict(self.X_test, as_pandas=False),
                output_dir=self.output_dir / 'figures'
            )

        # Plot feature importance
        if self.trainer.task == 'survival': # NEEDS TO BE UPDATED
            model = self.trainer.predictors['CoxPH']
            result = permutation_importance(model, self.X_test,
                                            Surv.from_dataframe('event', 'time', self.y_test),
                                            n_repeats=15)

            importance_df = pd.DataFrame(
                {
                    "importance": result["importances_mean"],
                    "stddev": result["importances_std"],
                },
                index=self.X_test.columns,
            ).sort_values(by="importance", ascending=False)
            model_name = 'CoxPH'
        else:
            importance_df = self.predictor.feature_importance(
                pd.concat([self.X_test, self.y_test], axis=1))
            model_name = self.predictor.model_best

        plot_feature_importance(importance_df, self.output_dir / 'figures', model_name)
        generate_explainer_report_pdf(self.trainer.task, self.output_dir)

    def _run_bias_audit(self) -> List[pd.DataFrame]:

        bias_output_dir = self.output_dir / 'bias'
        bias_output_dir.mkdir(parents=True, exist_ok=True)

        if self.sensitive_features is None:
            if self.trainer.task == 'survival': # Data needs to be not be one hot encoded
                self.sensitive_features = infer_sensitive_features(undummify(self.X_test, prefix_sep='|'))
            else:
                self.sensitive_features = infer_sensitive_features(self.X_test)
        
        y_pred = None if self.trainer.task == 'survival' else pd.Series(self.trainer.infer(self.X_test) )
        metrics = ['mean_prediction'] if self.trainer.task == 'regression' else ['mean_prediction', 'false_positive_rate'] 

        bias = BiasExplainer(
            self.y_test, 
            y_pred, 
            self.sensitive_features,
            self.trainer.task, 
            bias_output_dir,
            metrics
        )
        bias.run(relative=True)

    @classmethod
    def from_trainer(cls, trainer, **kwargs):
        """Create Explainer object from TrainerSupervised object."""
        return cls(trainer, trainer.X_train, trainer.X_test, trainer.y_test, trainer.output_dir, **kwargs)
