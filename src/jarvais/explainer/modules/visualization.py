from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from pydantic import BaseModel, Field
from jarvais.utils.plot import (
    plot_shap_values, 
    plot_violin_of_bootstrapped_metrics,
    plot_confusion_matrix,
    plot_regression_line,
    plot_residuals,
    plot_residual_histogram,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_calibration_curve,
    plot_sensitivity_flag_curve,
    plot_sensitivity_specificity_ppv_by_threshold,
    plot_histogram_of_predicted_probabilities,
)
from jarvais.loggers import logger

if TYPE_CHECKING:
    from jarvais.trainer import TrainerSupervised


class VisualizationModule(BaseModel):
    output_dir: Path = Field(description="Output directory.")
    shap: bool = Field(
        description="Whether to plot SHAP values. Only available for classification tasks. This flag exists because the SHAP values are computationally expensive to plot.", 
        default=True,
    )

    def model_post_init(self, __context) -> None: # noqa: ANN001
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, trainer: "TrainerSupervised") -> None:

        logger.info("Running Visualization Module...")

        plot_violin_of_bootstrapped_metrics(
            trainer,
            trainer.X_test,
            trainer.y_test,
            trainer.X_val,
            trainer.y_val,
            trainer.X_train,
            trainer.y_train,
            output_dir=self.output_dir,
        ) 

        if trainer.settings.task in ['binary', 'multiclass']:
            self._plot_classification_diagnostics(trainer)
        elif trainer.settings.task == 'regression':
            self._plot_regression_diagnostics(trainer)

    def _plot_regression_diagnostics(self, trainer: "TrainerSupervised") -> None:

        plot_regression_line(
            trainer.y_test,
            trainer.predictor.predict(trainer.X_test, as_pandas=False),
            output_dir=self.output_dir
        )

        plot_residuals(
            trainer.y_test,
            trainer.predictor.predict(trainer.X_test, as_pandas=False),
            output_dir=self.output_dir
        )

        plot_residual_histogram(
            trainer.y_test,
            trainer.predictor.predict(trainer.X_test, as_pandas=False),
            output_dir=self.output_dir
        )

    def _plot_classification_diagnostics(self, trainer: "TrainerSupervised") -> None:

        y_test_pred = trainer.predictor.predict_proba(trainer.X_test).iloc[:, 1]
        y_val_pred = trainer.predictor.predict_proba(trainer.X_val).iloc[:, 1]
        y_train_pred = trainer.predictor.predict_proba(trainer.X_train).iloc[:, 1]

        plot_confusion_matrix(
            trainer.y_test,
            y_test_pred,
            output_dir=self.output_dir,
            tag="(Test)",
        )

        plot_roc_curve(
            y_test=trainer.y_test.to_numpy(),
            y_pred=y_test_pred.to_numpy(),
            output_dir=self.output_dir,
            y_val=trainer.y_val.to_numpy(),
            y_val_pred=y_val_pred.to_numpy(),
            y_train=trainer.y_train.to_numpy(),
            y_train_pred=y_train_pred.to_numpy(),
        )
        
        plot_precision_recall_curve(
            y_test=trainer.y_test.to_numpy(),
            y_pred=y_test_pred.to_numpy(),
            output_dir=self.output_dir,
            y_val=trainer.y_val.to_numpy(),
            y_val_pred=y_val_pred.to_numpy(),
            y_train=trainer.y_train.to_numpy(),
            y_train_pred=y_train_pred.to_numpy(),
        )

        plot_calibration_curve(
            y_test=trainer.y_test.to_numpy(),
            y_pred=y_test_pred.to_numpy(),
            output_dir=self.output_dir,
            y_val=trainer.y_val.to_numpy(),
            y_val_pred=y_val_pred.to_numpy(),
            y_train=trainer.y_train.to_numpy(),
            y_train_pred=y_train_pred.to_numpy(),
        )

        plot_sensitivity_flag_curve(
            y_test=trainer.y_test.to_numpy(),
            y_pred=y_test_pred.to_numpy(),
            output_dir=self.output_dir,
            y_val=trainer.y_val.to_numpy(),
            y_val_pred=y_val_pred.to_numpy(),
            y_train=trainer.y_train.to_numpy(),
            y_train_pred=y_train_pred.to_numpy(),
        )

        plot_sensitivity_specificity_ppv_by_threshold(
            y_test=trainer.y_test.to_numpy(),
            y_pred=y_test_pred.to_numpy(),
            output_dir=self.output_dir,
            tag="(Test)",
        )

        plot_histogram_of_predicted_probabilities(
            y_test=trainer.y_test.to_numpy(),
            y_pred=y_test_pred.to_numpy(),
            output_dir=self.output_dir,
            tag="(Test)",
        )

        if self.shap:
            plot_shap_values(
                trainer.predictor,
                trainer.X_train,
                trainer.X_test,
                output_dir=self.output_dir
            )

            