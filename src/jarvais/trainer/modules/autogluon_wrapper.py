from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr
from pathlib import Path
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import KFold

from jarvais.loggers import logger


class AutogluonTabularWrapper(BaseModel):
    output_dir: Path = Field(
        description="Output directory.",
        title="Output Directory",
        examples=["output"]
    )
    target_variable: str = Field(
        description="Target variable.",
        title="Target Variable",
        examples=["tumor_stage"]
    )
    task: Literal["binary", "multiclass", "regression", "survival"] = Field(
        description="Task to perform.",
        title="Task",
        examples=["binary", "multiclass", "regression", "survival"]
    )
    k_folds: int = Field(
        default=5,
        description="Number of folds.",
        title="Number of Folds"
    )
    eval_metric: str | None = Field(
        default=None,
        description="Evaluation metric.",
        title="Evaluation Metric"
    )
    kwargs: dict = Field(
        default_factory=dict,
        description="Additional arguments to pass to the model.",
        title="Additional Arguments",
        examples={"presets": "best_quality"}
    )

    _predictor: TabularPredictor | None = PrivateAttr(default=None)
    _predictors: list[TabularPredictor] = PrivateAttr(default_factory=list)
    _cv_scores: list = PrivateAttr(default_factory=list)

    @classmethod
    def build(
        cls,
        output_dir: str | Path,
        target_variable: str,
        task: str,
        k_folds: int = 5,
    ):  
        match task:
            case "binary":
                eval_metric = "roc_auc"
            case "regression":
                eval_metric = "r2"

        return cls(
            output_dir=output_dir,
            target_variable=target_variable,
            task=task,
            k_folds=k_folds,
            eval_metric=eval_metric
        )

    def fit(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
        ):

        if self.k_folds > 1:
            self._predictor, X_val, y_val = self._train_autogluon_with_cv(
                X_train, 
                y_train,
            )
        else:
            self._predictor = TabularPredictor(
                label=self.target_variable, 
                problem_type=self.task, 
                eval_metric=self.eval_metric,
                path=(self.output_dir / 'autogluon_models' / 'autogluon_models_best_fold'),
                log_to_file=False,
            ).fit(
                pd.concat([X_train, y_train], axis=1),
                **self.kwargs
            )

            X_val, y_val = self._predictor.load_data_internal(data='val', return_y=True)

        return self._predictor, X_val, y_val
    
    def _train_autogluon_with_cv(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
        ):

        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)

        val_indices = []    

        for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
            X_train_cv, X_val_cv = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]

            val_indices.append(val_index)

            logger.info(f"Training fold {fold+1}/{self.k_folds}...")

            predictor = TabularPredictor(
                label=self.target_variable, 
                problem_type=self.task, 
                eval_metric=self.eval_metric,
                path=(self.output_dir / 'autogluon_models' / f'autogluon_models_fold_{fold}'),
                log_to_file=False,
                verbosity=0
            ).fit(
                pd.concat([X_train_cv, y_train_cv], axis=1),
                **self.kwargs
            )

            self._predictors.append(predictor)

            score = predictor.evaluate(pd.concat([X_val_cv, y_val_cv], axis=1))[self.eval_metric]
            logger.info(f"Fold {fold+1}/{self.k_folds} score: {score} ({self.eval_metric})")
            self._cv_scores.append(score)

            best_fold = self._cv_scores.index(max(self._cv_scores))
            
        return self._predictors[best_fold], X_train.iloc[val_indices[best_fold]], y_train.iloc[val_indices[best_fold]]
        