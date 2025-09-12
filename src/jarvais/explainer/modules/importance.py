from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from pydantic import BaseModel, Field
from sklearn.inspection import permutation_importance
from sksurv.util import Surv
import pandas as pd

from jarvais.utils.plot import plot_feature_importance
from jarvais.loggers import logger

if TYPE_CHECKING:
    from jarvais.trainer import TrainerSupervised


class ImportanceModule(BaseModel):
    output_dir: Path = Field(description="Output directory.")

    def model_post_init(self, __context) -> None: # noqa: ANN001
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, trainer: "TrainerSupervised") -> None:
        
        logger.info("Running Feature Importance Module...")
        
        if trainer.settings.task == 'survival': # NEEDS TO BE UPDATED
            model_name = 'CoxPH'
            model = trainer.predictor.models[model_name]
            result = permutation_importance(
                model, 
                trainer.X_test,
                Surv.from_dataframe('event', 'time', trainer.y_test),
                n_repeats=15
            )

            importance_df = pd.DataFrame(
                {
                    "importance": result["importances_mean"],
                    "stddev": result["importances_std"],
                },
                index=trainer.X_test.columns,
            ).sort_values(by="importance", ascending=False)
        else:
            importance_df = trainer.predictor.feature_importance(
                pd.concat([trainer.X_test, trainer.y_test], axis=1))
            model_name = trainer.predictor.model_best

        plot_feature_importance(importance_df, self.output_dir, model_name)