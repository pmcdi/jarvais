from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr
from pathlib import Path
from autogluon.core.metrics import make_scorer
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import KFold
from tabulate import tabulate

from jarvais.loggers import logger
from jarvais.utils.functional import auprc

from ._leaderboard import aggregate_folds, format_leaderboard
from ._simple_regression_model import SimpleRegressionModel



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
    extra_metrics: list = Field(
        default_factory=list,
        description="List of extra metrics to evaluate.",
        title="Extra Metrics",
        examples=["accuracy"]
    )
    kwargs: dict = Field(
        default_factory=dict,
        description="Additional arguments to pass to the model.",
        title="Additional Arguments",
        examples={"presets": "best_quality"}
    )   

    _cv_scores: list = PrivateAttr(default_factory=list)
    _extra_metrics: list = PrivateAttr(default_factory=list) # Copy of extra_metrics to store the auprc scorer
    _kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)
    _predictor: TabularPredictor | None = PrivateAttr(default=None)
    _predictors: list[TabularPredictor] = PrivateAttr(default_factory=list)

    def model_post_init(self, context):

        self._kwargs = self.kwargs.copy()
        custom_hyperparameters = get_hyperparameter_config('default')
        custom_hyperparameters[SimpleRegressionModel] = {}
        self._kwargs['hyperparameters'] = custom_hyperparameters
        
        if 'auprc' in self.extra_metrics:
            ag_auprc_scorer = make_scorer(
                name='auprc', # Move this to a seperate file?
                score_func=auprc,
                optimum=1,
                greater_is_better=True,
                needs_class=True)
            
            self._extra_metrics = self.extra_metrics.copy()
            self._extra_metrics.remove('auprc')
            self._extra_metrics.append(ag_auprc_scorer)

    @classmethod
    def build(
        cls,
        output_dir: str | Path,
        target_variable: str,
        task: str,
        k_folds: int = 5,
    ):  
        if task == "binary" or task == "multiclass":
            eval_metric = "roc_auc"
            extra_metrics = ['f1', 'auprc']
        elif task == "regression":
            eval_metric = "r2"
            extra_metrics = ['root_mean_squared_error']
        elif task == "survival":
            eval_metric = "c_index"
            extra_metrics = []

        return cls(
            output_dir=output_dir,
            target_variable=target_variable,
            task=task,
            k_folds=k_folds,
            eval_metric=eval_metric,
            extra_metrics=extra_metrics
        )

    def fit(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_test: pd.DataFrame,
            y_test: pd.Series
        ):

        if self.k_folds > 1:
            self._predictor, X_val, y_val = self._train_autogluon_with_cv(
                X_train, 
                y_train,
            )

            train_leaderboards, val_leaderboards, test_leaderboards = [], [], []

            for predictor in self._predictors:
                train_leaderboards.append(predictor.leaderboard(pd.concat([X_train, y_train], axis=1), extra_metrics=self._extra_metrics))
                val_leaderboards.append(predictor.leaderboard(pd.concat([X_val, y_val], axis=1), extra_metrics=self._extra_metrics))
                test_leaderboards.append(predictor.leaderboard(pd.concat([X_test, y_test], axis=1), extra_metrics=self._extra_metrics))
        
            train_leaderboard = aggregate_folds(pd.concat(train_leaderboards, ignore_index=True), self._extra_metrics)
            val_leaderboard = aggregate_folds(pd.concat(val_leaderboards, ignore_index=True), self._extra_metrics)
            test_leaderboard = aggregate_folds(pd.concat(test_leaderboards, ignore_index=True), self._extra_metrics)
        else:
            self._predictor = TabularPredictor(
                label=self.target_variable, 
                problem_type=self.task, 
                eval_metric=self.eval_metric,
                path=(self.output_dir / 'autogluon_models' / 'autogluon_models_best_fold'),
                log_to_file=False,
            ).fit(
                pd.concat([X_train, y_train], axis=1),
                **self._kwargs
            )

            X_val, y_val = self._predictor.load_data_internal(data='val', return_y=True)

            train_leaderboard = self._predictor.leaderboard(
                pd.concat([self.X_train, self.y_train], axis=1),
                extra_metrics=self._extra_metrics).round(2)
            val_leaderboard = self._predictor.leaderboard(
                pd.concat([self.X_val, self.y_val], axis=1),
                extra_metrics=self._extra_metrics).round(2)
            test_leaderboard = self._predictor.leaderboard(
                pd.concat([self.X_test, self.y_test], axis=1),
                extra_metrics=self._extra_metrics).round(2)

        final_leaderboard = pd.merge(
            pd.merge(
                format_leaderboard(train_leaderboard, self.eval_metric, self._extra_metrics, 'score_train'),
                format_leaderboard(val_leaderboard, self.eval_metric, self._extra_metrics, 'score_val'),
                on='model'
            ),
            format_leaderboard(test_leaderboard, self.eval_metric, self._extra_metrics, 'score_test'),
            on='model'
        )

        print('\nModel Leaderboard\n----------------')
        print(tabulate(
            final_leaderboard.sort_values(by='score_test', ascending=False),
            tablefmt = "grid",
            headers="keys",
            showindex=False))

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
                **self._kwargs
            )

            self._predictors.append(predictor)

            score = predictor.evaluate(pd.concat([X_val_cv, y_val_cv], axis=1))[self.eval_metric]
            logger.info(f"Fold {fold+1}/{self.k_folds} score: {score} ({self.eval_metric})")
            self._cv_scores.append(score)

            best_fold = self._cv_scores.index(max(self._cv_scores))
            
        return self._predictors[best_fold], X_train.iloc[val_indices[best_fold]], y_train.iloc[val_indices[best_fold]]
        