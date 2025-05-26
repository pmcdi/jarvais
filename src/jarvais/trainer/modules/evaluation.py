from typing import Literal

from autogluon.core.metrics import make_scorer
import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr
from tabulate import tabulate
from sksurv.metrics import concordance_index_censored

from jarvais.loggers import logger
from jarvais.utils.functional import auprc

from .autogluon_wrapper import AutogluonTabularWrapper
from .survival import SurvivalTrainerModule


def format_leaderboard(leaderboard: pd.DataFrame, eval_metric: str, extra_metrics: list, score_col_name: str) -> pd.DataFrame:

    if score_col_name == 'score_val' and 'score_val' in leaderboard.columns:
        leaderboard = leaderboard.drop(score_col_name, axis=1)
    leaderboard = leaderboard.rename(columns={'score_test': score_col_name})

    def format_scores(row, score_col, extra_metrics):
        """Format scores as a string with AUROC, F1, and AUPRC. Or with R2 and RMSE for regression."""
        if 'f1' in extra_metrics:
            return f"{eval_metric.upper()} {row[score_col]}\nF1: {row['f1']}\nAUPRC: {row['auprc']}"
        elif 'root_mean_squared_error' in extra_metrics:
            return f"{eval_metric.upper()} {row[score_col]}\nRMSE: {row['root_mean_squared_error']}"
        else:
            return f"{eval_metric.upper()} {row[score_col]}"

    leaderboard[score_col_name] = leaderboard.apply(
        lambda row: format_scores(row, score_col_name, extra_metrics),
        axis=1
    )
    return leaderboard[['model', score_col_name]]

def aggregate_folds(consolidated_leaderboard:list, extra_metrics: list) -> pd.DataFrame:
    extra_metrics = [str(item) for item in extra_metrics]

    to_agg = {k: ['mean', 'min', 'max'] for k in ['score_test', *extra_metrics]}

    aggregated_leaderboard = consolidated_leaderboard.groupby('model').agg(to_agg).reset_index()

    final_leaderboard = pd.DataFrame({'model': aggregated_leaderboard['model']})

    for col in to_agg.keys():
        final_leaderboard[col] = [
            f'{round(row[0], 2)} [{round(row[1], 2)}, {round(row[2], 2)}]'
            for row in aggregated_leaderboard[col].values
        ]

    return final_leaderboard


class EvaluationModule(BaseModel):
    output_dir: str = Field(
        description="Output directory.",
        title="Output Directory",
        examples=["output"]
    )
    task: Literal["binary", "multiclass", "regression", "survival"] = Field(
        description="Task to perform.",
        title="Task",
        examples=["binary", "multiclass", "regression", "survival"]
    )
    eval_metric: str | None = Field(
        default="roc_auc",
        description="Evaluation metric.",
        title="Evaluation Metric"
    )
    extra_metrics: list = Field(
        default_factory=list,
        description="List of extra metrics to evaluate.",
        title="Extra Metrics",
        examples=["accuracy"]
    )

    _extra_metrics: list = PrivateAttr(default_factory=list) # Copy of extra_metrics to store the auprc scorer

    def model_post_init(self, context):
        
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
    def build(cls, output_dir: str, task: str):
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
            task=task,
            eval_metric=eval_metric,
            extra_metrics=extra_metrics
        )
    
    def __call__(
            self,
            trainer_module: SurvivalTrainerModule | AutogluonTabularWrapper,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: pd.DataFrame,
            y_val: pd.Series,
            X_test: pd.DataFrame,
            y_test: pd.Series
    ):
        
        match trainer_module:
            case SurvivalTrainerModule():
                self._evaluate_survival(trainer_module, X_train, y_train, X_val, y_val, X_test, y_test)
            case AutogluonTabularWrapper():
                self._evaluate_autogluon(trainer_module, X_train, y_train, X_val, y_val, X_test, y_test)
        
    def _evaluate_autogluon(self, trainer_module: AutogluonTabularWrapper, X_train, y_train, X_val, y_val, X_test, y_test):

        train_leaderboards, val_leaderboards, test_leaderboards = [], [], []
        if trainer_module.k_folds > 1:  
            for predictor in trainer_module._predictors:
                train_leaderboards.append(predictor.leaderboard(pd.concat([X_train, y_train], axis=1), extra_metrics=self._extra_metrics))
                val_leaderboards.append(predictor.leaderboard(pd.concat([X_val, y_val], axis=1), extra_metrics=self._extra_metrics))
                test_leaderboards.append(predictor.leaderboard(pd.concat([X_test, y_test], axis=1), extra_metrics=self._extra_metrics))
        
            train_leaderboard = aggregate_folds(pd.concat(train_leaderboards, ignore_index=True), self._extra_metrics)
            val_leaderboard = aggregate_folds(pd.concat(val_leaderboards, ignore_index=True), self._extra_metrics)
            test_leaderboard = aggregate_folds(pd.concat(test_leaderboards, ignore_index=True), self._extra_metrics)

        else:
            train_leaderboard = trainer_module._predictor.leaderboard(
                pd.concat([self.X_train, self.y_train], axis=1),
                extra_metrics=self._extra_metrics).round(2)
            val_leaderboard = trainer_module._predictor.leaderboard(
                pd.concat([self.X_val, self.y_val], axis=1),
                extra_metrics=trainer_module._extra_metrics).round(2)
            test_leaderboard = trainer_module._predictor.leaderboard(
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
        
    def _evaluate_survival(self, trainer_module: SurvivalTrainerModule, X_train, y_train, X_val, y_val, X_test, y_test):

        leaderboard = []
        for model in trainer_module.classical_models:  
            trained_model = trainer_module._predictor.get(model)
            if not trained_model:
                continue

            train_score = concordance_index_censored(
                y_train['event'].astype(bool), 
                y_train['time'], 
                trainer_module._predictor[model].predict(X_train)
            )[0]
            test_score = concordance_index_censored(
                y_test['event'].astype(bool),
                y_test['time'], 
                trainer_module._predictor[model].predict(X_test)
            )[0]
            leaderboard.append({
                'model': model,
                'test_score': f"{self.eval_metric.upper()}: {test_score}",
                'val_score': 'N/A', # no validation for classical models
                'train_score': f"{self.eval_metric.upper()}: {train_score}",
            })

        for model in trainer_module.deep_models:

            train_score = concordance_index_censored(
                y_train['event'].astype(bool), 
                y_train['time'], 
                trainer_module._predictor[model].predict(X_train)
            )[0]
            # val_score = concordance_index_censored(
            #     y_val['event'].astype(bool), 
            #     y_val['time'], 
            #     trainer_module._predictor[model].predict(X_val)
            # )[0]
            test_score = concordance_index_censored(
                y_test['event'].astype(bool), 
                y_test['time'], 
                trainer_module._predictor[model].predict(X_test)
            )[0]
            leaderboard.append({
                'model': model,
                'test_score': f"{self.eval_metric.upper()}: {test_score}",
                'val_score': 'N/A', # no validation for classical models
                'train_score': f"{self.eval_metric.upper()}: {train_score}",
            })

        print('\nModel Leaderboard\n----------------')
        print(tabulate(
            pd.DataFrame(leaderboard).sort_values(by='test_score', ascending=False),
            tablefmt = "grid",
            headers="keys",
            showindex=False))

        


        