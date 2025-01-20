import json
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from autogluon.core.metrics import make_scorer
from autogluon.tabular import TabularPredictor
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
from sklearn.model_selection import train_test_split
from tabulate import tabulate

from ._feature_reduction import chi2_reduction, kbest_reduction, mrmr_reduction, var_reduction
from ._leaderboard import format_leaderboard
from ._simple_regression_model import SimpleRegressionModel
from ._training import train_autogluon_with_cv, train_survival_models
from ..explainer import Explainer
from ..utils.functional import auprc
from .survival import LitDeepSurv, LitMTLR

class TrainerSupervised:
    """
    TrainerSupervised class for supervised AutoML workflows.

    This class provides functionality for feature reduction, training models (e.g., AutoGluon, survival models), 
    and performing inference. It supports various tasks such as binary/multiclass classification, regression, 
    and time-to-event analysis.

    Attributes:
        task (str, optional): Type of task. Must be one of {'binary', 'multiclass', 'regression', 'time_to_event'}. 
        reduction_method (str | None, optional): Feature reduction method. Supported methods include 
            {'mrmr', 'variance_threshold', 'corr', 'chi2'}.
        keep_k (int, optional): Number of features to retain during reduction.
        output_dir (str | Path, optional): Directory for saving outputs. Defaults to the current working directory.
        
    Example:
        ```python
        from AutoML.trainer import TrainerSupervised

        trainer = TrainerSupervised(
            task="binary",
            reduction_method="mrmr",
            keep_k=10,
            output_dir="./results"
        )
        trainer.run(data=my_data, target_variable="target")
        
        predictions = trainer.infer(new_data)
        ```
    """
    def __init__(
            self,
            task: str=None,
            reduction_method: str | None = None,
            keep_k: int = 2,
            output_dir: str | Path = None
        ) -> None:

        self.task = task
        self.reduction_method = reduction_method
        self.keep_k = keep_k

        if task not in ['binary', 'multiclass', 'regression', 'time_to_event', None]:
            raise ValueError("Invalid task parameter. Choose one of: 'binary', 'multiclass', 'regression', 'time_to_event'. Providing None defaults to Autogluon infering.")

        self.output_dir = Path.cwd() if output_dir is None else Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def _feature_reduction(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series) -> pd.DataFrame:
        """
        Reduce features based on the specified reduction method. 
        
        One-hot encoding applied before reduction and reverted afterward.
        """
        # Step 1: Identify categorical columns
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()

        mappin = {}

        def find_category_mappings(df, variable):
            return {k: i for i, k in enumerate(df[variable].dropna().unique(), 0)}

        def integer_encode(df, variable, ordinal_mapping):
            df[variable] = df[variable].map(ordinal_mapping)

        for variable in categorical_columns:
            mappings = find_category_mappings(X, variable)
            mappin[variable] = mappings

        for variable in categorical_columns:
            integer_encode(X, variable, mappin[variable])

        # Step 3: Perform feature reduction
        if self.reduction_method == 'mrmr':
            X_reduced = mrmr_reduction(self.task, X, y, self.keep_k)
        elif self.reduction_method == 'variance_threshold':
            X_reduced = var_reduction(X, y)
        elif self.reduction_method == 'corr':
            X_reduced = kbest_reduction(self.task, X, y, self.keep_k)
        elif self.reduction_method == 'chi2':
            if self.task not in ['binary', 'multiclass']:
                raise ValueError('chi-squared reduction can only be done with classification tasks')
            X_reduced = chi2_reduction(X, y, self.keep_k)
        else:
            raise ValueError('Unsupported reduction method: {}'.format(self.reduction_method))

        for col in categorical_columns:
            if col in X_reduced.columns:
                inv_map = {v: k for k, v in mappin[col].items()}
                X_reduced[col] = X_reduced[col].map(inv_map)

        return X_reduced

    def _train_autogluon_with_cv(self) -> None:
        self.predictors, leaderboard, self.best_fold, self.X_val, self.y_val = train_autogluon_with_cv(
            pd.concat([self.X_train, self.y_train], axis=1),
            pd.concat([self.X_test, self.y_test], axis=1),
            target_variable=self.target_variable,
            task=self.task,
            extra_metrics=self.extra_metrics,
            eval_metric=self.eval_metric,
            num_folds=self.k_folds,
            output_dir=(self.output_dir / 'autogluon_models'),
            **self.kwargs
        )

        self.predictor = self.predictors[self.best_fold]

        # Update train data to remove validation
        self.X_train = self.X_train[~self.X_train.index.isin(self.X_val.index)]
        self.y_train = self.y_train[~self.y_train.index.isin(self.y_val.index)]

        print('\nModel Leaderboard (Displays values in "mean [min, max]" format across training folds)\n------------------------------------------------------------------------------------')
        print(tabulate(
            leaderboard.sort_values(by='score_test', ascending=False)[self.show_leaderboard],
            tablefmt = "fancy_grid",
            headers="keys",
            showindex=False
        ))

    def _train_autogluon(self) -> None:
        self.predictor = TabularPredictor(
            label=self.target_variable, 
            problem_type=self.task, 
            eval_metric=self.eval_metric,
            path=(self.output_dir / 'autogluon_models' / 'autogluon_models_best_fold'),
            log_to_file=False,
        ).fit(
            pd.concat([self.X_train, self.y_train], axis=1),
            **self.kwargs
        )

        self.X_val, self.y_val = self.predictor.load_data_internal(data='val', return_y=True)
        # Update train data to remove validation
        self.X_train = self.X_train[~self.X_train.index.isin(self.X_val.index)]
        self.y_train = self.y_train[~self.y_train.index.isin(self.y_val.index)]

        train_leaderboard = self.predictor.leaderboard(
            pd.concat([self.X_train, self.y_train], axis=1),
            extra_metrics=self.extra_metrics).round(2)
        val_leaderboard = self.predictor.leaderboard(
            pd.concat([self.X_val, self.y_val], axis=1),
            extra_metrics=self.extra_metrics).round(2)
        test_leaderboard = self.predictor.leaderboard(
            pd.concat([self.X_test, self.y_test], axis=1),
            extra_metrics=self.extra_metrics).round(2)

        leaderboard = pd.merge(
            pd.merge(
                format_leaderboard(train_leaderboard, self.extra_metrics, 'score_train'),
                format_leaderboard(val_leaderboard, self.extra_metrics, 'score_val'),
                on='model'
            ),
            format_leaderboard(test_leaderboard, self.extra_metrics, 'score_test'),
            on='model'
        )

        print('\nModel Leaderboard\n----------------')
        print(tabulate(
            leaderboard.sort_values(by='score_test', ascending=False)[self.show_leaderboard],
            tablefmt = "fancy_grid",
            headers="keys",
            showindex=False))

    def run(
            self,
            data: pd.DataFrame,
            target_variable: str,
            test_size: float = 0.2,
            exclude: List[str] | None = None,
            stratify_on: str | None = None,
            explain: bool = False,
            k_folds: int = 5,
            **kwargs:dict
        ) -> None:
        """
        Execute the AutoML pipeline on the given dataset.

        Args:
            data (pd.DataFrame): The input dataset containing features and target.
            target_variable (str): The name of the target variable in the dataset.
            test_size (float, optional): Proportion of the dataset to include in the test split. 
                Must be between 0 and 1. Default is 0.2.
            exclude (list of str, optional): List of columns to exclude from the feature set. 
                Default is an empty list.
            stratify_on (str, optional): Column to use for stratification, if any. 
                Must be compatible with `target_variable`.
            explain (bool, optional): Whether to generate explainability reports for the model. 
                Default is False.
            k_folds (int, optional): Number of folds for cross-validation. If 1, uses AutoGluon-specific validation. 
                Default is 5.
            kwargs (dict, optional): Additional arguments passed to the AutoGluon predictor's `fit` method.
        """
        self.target_variable = target_variable
        self.k_folds = k_folds
        self.kwargs = kwargs

        # Initialize mutable defaults
        if exclude is None:
            exclude = []

        if isinstance(target_variable, list): # Happens for time_to_event data
            exclude += target_variable
        else:
            exclude.append(target_variable)

        try:
            X = data.drop(columns=exclude)
            y = data[target_variable]
        except KeyError as e:
            raise ValueError(f"Invalid column specified: {e}")

        # Optional feature reduction
        if getattr(self, "reduction_method", None):
            print(f"Applying {self.reduction_method} for feature reduction")
            X = self._feature_reduction(X, y)
            print(f"Features retained: {list(X.columns)}")

            self.feature_names = list(X.columns)

        if self.task in {'binary', 'multiclass'}:
            stratify_col = (
                y.astype(str) + '_' + data[stratify_on].astype(str)
                if stratify_on is not None
                else y
            )
        else:
            stratify_col = None

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, stratify=stratify_col, random_state=42)

        if self.task == 'time_to_event':
            self.predictors, scores, data_train, data_val = train_survival_models(
                self.X_train, self.y_train, self.X_test, self.y_test, self.output_dir)
            self.predictor = self.predictors[max(scores, key=scores.get)]

            self.X_train, self.y_train = data_train.drop(columns=['time', 'event']), data_train[['time', 'event']] 
            self.X_val, self.y_val = data_val.drop(columns=['time', 'event']), data_val[['time', 'event']] 
        else:
            (self.output_dir / 'autogluon_models').mkdir(exist_ok=True, parents=True)

            if self.task in ['binary', 'multiclass']:
                self.eval_metric = 'roc_auc'
            elif self.task == 'regression':
                self.eval_metric = 'r2'

            ag_auprc_scorer = make_scorer(
                name='auprc', # Move this to a seperate file?
                score_func=auprc,
                optimum=1,
                greater_is_better=True,
                needs_class=True)

            # When changing extra_metrics consider where it's used and make updates accordingly
            self.extra_metrics = ['f1', ag_auprc_scorer] if self.task in ['binary', 'multiclass'] else ['root_mean_squared_error']
            self.show_leaderboard = ['model', 'score_test', 'score_val', 'score_train']

            custom_hyperparameters = get_hyperparameter_config('default')
            custom_hyperparameters[SimpleRegressionModel] = {}
            kwargs['hyperparameters'] = custom_hyperparameters

            if k_folds > 1:
                self._train_autogluon_with_cv()
            else:
                self._train_autogluon()

        self.data_dir = self.output_dir / 'data'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.X_train.to_csv((self.data_dir / 'X_train.csv'), index=False)
        self.X_test.to_csv((self.data_dir / 'X_test.csv'), index=False)
        self.X_val.to_csv((self.data_dir / 'X_val.csv'), index=False)
        self.y_train.to_csv((self.data_dir / 'y_train.csv'), index=False)
        self.y_test.to_csv((self.data_dir / 'y_test.csv'), index=False)
        self.y_val.to_csv((self.data_dir / 'y_val.csv'), index=False)

        if explain:
            explainer = Explainer.from_trainer(self)
            explainer.run()

    def infer(self, data: pd.DataFrame, model: str = None) -> pd.DataFrame | pd.Series | np.ndarray:
        """
        Perform inference using the trained predictor on the provided data.

        Args:
            data (pd.DataFrame): The input data for which inference is to be performed.
            model (str, optional): The name of the model to use for inference. If None, the default model is used.

        Returns:
            pd.DataFrame, pd.Series, or np.ndarray: The prediction results from the model.
        """
        if hasattr(self.predictor, 'can_predict_proba'): # Autogluon
            if self.predictor.can_predict_proba:
                inference =  self.predictor.predict_proba(data, model)
            else:
                self.predictor.predict(data, model)
        else: # Survival models
            if model is None:
                inference = self.predictor.predict(data)
            else:
                inference = self.predictors[model].predict(data)

        return inference
    
    @classmethod
    def load_trainer(cls, project_dir: str | Path):
        """
        Load a trained TrainerSupervised from the specified directory.

        Args:
            project_dir (str or Path, optional): The directory where the trainer was run.

        Returns:
            TrainerSupervised: The loaded Trainer.
        """
        project_dir = Path(project_dir)

        trainer = cls()

        if (project_dir / 'survival_models').exists():
            model_dir = (project_dir / 'survival_models')
            with open(model_dir / "model_info.json", "r") as f:
                model_info = json.load(f)

            trainer.predictors = {}
            for model_name, _ in model_info.items():
                if model_name == 'MTLR':
                    trainer.predictors[model_name] = LitMTLR.load_from_checkpoint(model_dir / "MTLR.ckpt")
                elif model_name == 'DeepSurv':
                    trainer.predictors[model_name] = LitDeepSurv.load_from_checkpoint(model_dir / "DeepSurv.ckpt")
                else:
                    with (model_dir / f'{model_name}.pkl').open("rb") as f:
                        pickle.load(f)

            trainer.predictor = trainer.predictors[max(model_info, key=model_info.get)]
            trainer.task = 'time_to_event'
        else:
            model_dir = (project_dir / 'autogluon_models' / 'autogluon_models_best_fold')
            trainer.predictor = TabularPredictor.load(model_dir, verbosity=1)
            trainer.task = trainer.predictor.problem_type
        
        trainer.X_test = pd.read_csv(project_dir / 'data' / 'X_test.csv', index_col=0)
        trainer.X_val = pd.read_csv(project_dir / 'data' / 'X_val.csv', index_col=0)
        trainer.X_train = pd.read_csv(project_dir / 'data' / 'X_train.csv', index_col=0)
        trainer.y_test = pd.read_csv(project_dir / 'data' / 'y_test.csv', index_col=0).squeeze()
        trainer.y_val = pd.read_csv(project_dir / 'data' / 'y_val.csv', index_col=0).squeeze()
        trainer.y_train = pd.read_csv(project_dir / 'data' / 'y_train.csv', index_col=0).squeeze()
  
        return trainer


