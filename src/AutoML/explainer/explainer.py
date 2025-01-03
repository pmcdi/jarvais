from pathlib import Path
from typing import Union

import pandas as pd
from sklearn.inspection import permutation_importance
from sksurv.util import Surv

from ..utils.pdf import generate_explainer_report_pdf
from ..utils.plot import (
    plot_classification_diagnostics,
    plot_feature_importance,
    plot_regression_diagnostics,
    plot_shap_values,
    plot_violin_of_bootsrapped_metrics,
)

class Explainer():
    """A class to generate diagnostic plots and reports for models trained using TrainerSupervised."""
    def __init__(self,
                 trainer,
                 X_train: pd.DataFrame,
                 X_test: pd.DataFrame,
                 y_test: pd.DataFrame,
                 output_dir: Union[str, Path, None] = None):

        self.trainer = trainer
        self.predictor = trainer.predictor
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test

        self.output_dir = Path.cwd() if output_dir is None else Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'figures').mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """Generate diagnostic plots and reports for the trained model."""
        if self.trainer.task != 'time_to_event':
            plot_violin_of_bootsrapped_metrics(
                self.predictor,
                self.X_test,
                self.y_test,
                self.trainer.X_val,
                self.trainer.y_val,
                self.X_train,
                self.trainer.y_train,
                output_dir=self.output_dir / 'figures'
            )

        # Plot diagnostics
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
        if self.trainer.task == 'time_to_event': # NEEDS TO BE UPDATED
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

    @classmethod
    def from_trainer(cls, trainer):
        """Create Explainer object from TrainerSupervised object."""
        return cls(trainer, trainer.X_train, trainer.X_test, trainer.y_test, trainer.output_dir)
