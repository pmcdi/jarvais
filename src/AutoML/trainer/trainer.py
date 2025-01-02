from pathlib import Path

import pandas as pd
from tabulate import tabulate
import pickle, json

from typing import Union, List, Optional

from sklearn.model_selection import train_test_split

from autogluon.tabular import TabularPredictor
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
from autogluon.core.metrics import make_scorer

from ._feature_reduction import mrmr_reduction, var_reduction, kbest_reduction, chi2_reduction
from ._simple_regression_model import SimpleRegressionModel
from ._leaderboard import format_leaderboard
from ._training import train_autogluon_with_cv, train_survival_models

from ..explainer import Explainer
from ..utils.functional import auprc

from ..utils.models.survival import LitMTLR, LitDeepSurv

class TrainerSupervised():
    def __init__(self,
                 task: str=None,
                 reduction_method: Union[str, None] = None,
                 keep_k: int = 2,
                 output_dir: str | Path = Path.cwd()):
        """
        Initialize the AutoMLTrainer class with specified configurations.

        Parameters
        ----------
        task : str, default-None
            The type of task to handle. Options are 'binary', 'multiclass', 'regression', 'time_to_event'. Providing None defaults to Autogluon infering.
        reduction_method : str, default=None
            The feature reduction method to apply. Options are 'mrmr', 'variance_threshold', 'corr', 'chi2'.
        keep_k : int, default=2
            Number of features to keep, if a reduction method is defined.
        output_dir : str or pathlib.Path, default=Path.cwd()
            The directory where output files will be saved.
        

        Raises
        ------
        ValueError
            If the task parameter is not one of the specified options.
        """
        self.task = task
        self.output_dir = Path(output_dir)
        self.reduction_method = reduction_method
        self.keep_k = keep_k

        if task not in ['binary', 'multiclass', 'regression', 'time_to_event', None]:
            raise ValueError("Invalid task parameter. Choose one of: 'binary', 'multiclass', 'regression', 'time_to_event'. Providing None defaults to Autogluon infering.")

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def _feature_reduction(self, X, y):
        """
        Reduces features based on the specified reduction method, with one-hot encoding applied before reduction
        and reverted afterward.

        Parameters
        ----------
        X : pd.DataFrame
            The input feature data.
        y : pd.Series
            The target variable.

        Returns
        -------
        pd.DataFrame
            The reduced feature set.
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
    
    def _train_autogluon_with_cv(self):
        self.predictors, leaderboard, self.best_fold, self.X_val, self.y_val = train_autogluon_with_cv(
                    pd.concat([self.X_train, self.y_train], axis=1),
                    pd.concat([self.X_test, self.y_test], axis=1), 
                    target_variable=self.target_variable, 
                    task=self.task,
                    extra_metrics=self.extra_metrics,
                    eval_metric=self.eval_metric,
                    num_folds=self.k_folds,
                    output_dir=(self.output_dir / 'autogluon_models'),
                    **self.kwargs)
                
        self.predictor = self.predictors[self.best_fold]

        # Update train data to remove validation
        self.X_train = self.X_train[~self.X_train.index.isin(self.X_val.index)]
        self.y_train = self.y_train[~self.y_train.index.isin(self.y_val.index)]

        print('\nModel Leaderboard (Displays values in "mean [min, max]" format across training folds)\n------------------------------------------------------------------------------------')
        print(tabulate(
            leaderboard.sort_values(by='score_test', ascending=False)[self.show_leaderboard],
            tablefmt = "fancy_grid", 
            headers="keys",
            showindex=False))
        
    def train_autogluon(self):
        self.predictor = TabularPredictor(
            label=self.target_variable, problem_type=self.task, eval_metric=self.eval_metric,
            path=(self.output_dir / 'autogluon_models' / f'autogluon_models_best_fold'),
            log_to_file=False,
            ).fit(
                pd.concat([self.X_train, self.y_train], axis=1), 
                **self.kwargs)
            
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
    
    def run(self,
            data: pd.DataFrame,
            target_variable: str,
            test_size: float = 0.2,
            exclude: Optional[List[str]] = None,
            stratify_on: Optional[str] = None,
            explain: bool = False,
            save_data: bool = True,
            k_folds: int = 5,
            **kwargs) -> None:
        """
        Executes the AutoML pipeline on the given dataset.

        Parameters
        ----------
        data : pd.DataFrame
            The input dataset containing features and target.
        target_variable : str
            The name of the target variable in the dataset.
        test_size : float, default=0.2
            Proportion of the dataset to include in the test split (0 < test_size < 1).
        exclude : list of str, optional
            List of columns to exclude from the feature set. Default is an empty list.
        stratify_on : str, optional
            Column to use for stratification, if any. Must be compatible with `target_variable`.
        explain : bool, default=False
            Whether to generate explainability reports for the model.
        save_data : bool, default=True
            Whether to save train/test/validation data to disk.
        k_folds : int, default=5
            Number of folds for cross-validation. If 1, uses AutoGluon-specific validation.
        predictor_fit_kwargs : dict, optional
            Additional arguments passed to the AutoGluon predictor's `fit` method.
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
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, stratify=stratify_col, random_state=42)

        if self.task == 'time_to_event':
            self.X_val = pd.DataFrame()
            self.y_val = pd.DataFrame()

            self.predictors, scores = train_survival_models(self.X_train, self.y_train, self.X_test, self.y_test, self.output_dir)
            self.predictor = self.predictors[max(scores, key=scores.get)]
        else:
            (self.output_dir / 'autogluon_models').mkdir(exist_ok=True, parents=True)

            if self.task in ['binary', 'multiclass']:
                self.eval_metric = 'roc_auc'
            elif self.task == 'regression':
                self.eval_metric = 'r2' 
                
            ag_auprc_scorer = make_scorer(name='auprc', # Move this to a seperate file?
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
                self.train_autogluon()

        if save_data:
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

    def infer(self, data: pd.DataFrame,
              model: str = None):
        
        if hasattr(self.predictor, 'can_predict_proba'): # Autogluon
            inference =  self.predictor.predict_proba(data, model) if self.predictor.can_predict_proba else self.predictor.predict(data, model) 
        else: # Survival models
            if model == None:
                inference = self.predictor.predict(data)
            else:
                inference = self.predictors[model].predict(data)

        return inference 
    
    @classmethod
    def load_model(cls, 
                   model_dir: str | Path= None,
                   project_dir: str | Path = None):
        """
        Load a trained model from the specified directory.

        Parameters
        ----------
        model_dir : str
            The directory where the model is saved.
        project_dir : str
            The directory where the trainer was ran.

        Returns
        -------
        TrainerSupervised
            The loaded model.
        """

        if not (model_dir or project_dir):
            raise ValueError('model_dir or project_dir must be provided')

        if model_dir is None and project_dir is not None:
            model_dir = (Path(project_dir) / 'autogluon_models' / 'autogluon_models_best_fold'
             if (Path(project_dir) / 'autogluon_models' / 'autogluon_models_best_fold').exists()
             else Path(project_dir) / 'survival_models')
            
        trainer = cls()
        
        if 'survival' in str(model_dir):
            with open(model_dir / "model_info.json", "r") as f:
                model_info = json.load(f)

            trainer.predictors = dict()
            for model_name, _ in model_info.items():
                if model_name == 'MTLR':
                    trainer.predictors[model_name] = LitMTLR.load_from_checkpoint(checkpoint_path="MTLR.ckpt") 
                elif model_name == 'DeepSurv':
                    trainer.predictors[model_name] = LitDeepSurv.load_from_checkpoint(checkpoint_path="DeepSurv.ckpt") 
                else:
                    with (model_dir / f'{model_name}.pkl').open("rb") as f:
                        pickle.load(f)

            trainer.predictor = trainer.predictors[max(model_info, key=model_info.get)]
        else:
            trainer.predictor = TabularPredictor.load(model_dir, verbosity=1)

        return trainer


