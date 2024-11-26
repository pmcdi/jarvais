import os, shutil

import pandas as pd
from tabulate import tabulate

from typing import Union, List, Any

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from autogluon.tabular import TabularPredictor

from .explainer import Explainer
from .models import SimpleRegressionModel
from .utils import mrmr_reduction, var_reduction, kbest_reduction, chi2_reduction, train_with_cv

class TrainerSupervised():
    def __init__(self,
                 task: str = None,
                 reduction_method: Union[str, None] = None,
                 keep_k: int = 2,
                 output_dir: Union[str, os.PathLike] = '.',
                 predictor_fit_kwargs: dict[str, Any] = {}):
        """
        Initialize the AutoMLTrainer class with specified configurations.

        Parameters
        ----------
        task : str
            The type of task to handle. Options are 'binary', 'multiclass', 'regression', 'quantile'.
        reduction_method : str, default=None
            The feature reduction method to apply. Options are 'mrmr', 'variance_threshold', 'corr', 'chi2'.
        keep_k : int, default=2
            Number of features to keep, if a reduction method is defined.
        output_dir : str or os.PathLike, default='.'
            The directory where output files will be saved.
        predictor_fit_kwargs : dict, default={}
            keyword arguments to be passed to AutoGluon predictor fit method
        

        Raises
        ------
        ValueError
            If the task parameter is not one of the specified options.
        """
        self.task = task
        self.output_dir = output_dir
        self.reduction_method = reduction_method
        self.keep_k = keep_k
        self.predictor_fit_kwargs = predictor_fit_kwargs

        if task not in ['binary', 'multiclass', 'regression', 'quantile', None]:
            raise ValueError("Invalid task parameter. Choose one of: 'binary', 'multiclass', 'regression', 'quantile'. Or provide nothing and let Autogluon infer the task.")

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

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
    
    def run(self,
            data: pd.DataFrame,
            target_variable: str,
            test_size: float = 0.2,
            exclude: List[str] = [],
            stratify_on: Union[str, None] = None,
            explain: bool = False,
            save_data: bool = True,
            k_folds: int = 5):
            """
            Runs the AutoML pipeline with the specified data and target variable.

            Parameters
            ----------
            data : pd.DataFrame
                The input data for training and testing.
            target_variable : str
                The name of the target variable in the dataset.
            test_size : float, default=0.2
                The proportion of the dataset to include in the test split.
            exclude : list, default=[]
                A list of columns to exclude from the feature set.
            stratify_on : str, optional
                The column to use for stratification, if any.
            """
        
            exclude.append(target_variable)
            X = data.drop(columns=exclude) 
            y = data[target_variable]

            self.target_variable = target_variable

            if self.reduction_method is not None:
                print(f'Applying {self.reduction_method} for feature reduction')
                X = self._feature_reduction(X, y)
                print(f'Features kept: {X.columns.values}')

            self.feature_names = list(X.columns)                

            if self.task in ['binary', 'multiclass']:
                if y.value_counts().min() > 1: # Meaning it can be used to stratify, if this condition is not met train_test_split produces - ValueError: The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.
                    if stratify_on is not None:
                        stratify_col = y.astype(str) + '_' + data[stratify_on].astype(str) 
                    else:
                        stratify_col = y                
                else:
                    raise ValueError('Least populated class has only one entry')
            else:
                stratify_col = None
            
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, stratify=stratify_col, random_state=42)

            if self.task in ['binary', 'multiclass']:
                eval_metric = 'roc_auc'
            elif self.task == 'regression':
                eval_metric = 'r2'
                
            self.predictors, leaderboard, self.best_fold, self.X_val, self.y_val = train_with_cv(
                pd.concat([self.X_train, self.y_train], axis=1),
                pd.concat([self.X_test, self.y_test], axis=1), 
                target_variable=target_variable, 
                task=self.task, 
                eval_metric=eval_metric, 
                num_folds=k_folds,
                predictor_fit_kwargs=self.predictor_fit_kwargs,
                output_dir=self.output_dir)
            
            # Update train data to remove validation
            self.X_train = self.X_train[~self.X_train.index.isin(self.X_val.index)]
            self.y_train = self.y_train[~self.y_train.index.isin(self.y_val.index)]
            
            self.predictor = self.predictors[self.best_fold]
         
            extra_metrics = ['f1', 'average_precision'] if self.task in ['binary', 'multiclass'] else ['root_mean_squared_error'] 
            show_leaderboard = ['model', 'score_test', 'score_val', 'score_train', 'eval_metric'] + extra_metrics

            print('\nModel Leaderboard (Displays values in "mean [min, max]" format across training folds)\n----------------')
            print(tabulate(
                leaderboard.sort_values(by='score_val', ascending=False)[show_leaderboard],
                tablefmt = "fancy_grid", 
                headers="keys",
                showindex=False))

            if save_data:
                self.data_dir = os.path.join(self.output_dir, 'data')
                os.makedirs(self.data_dir, exist_ok=True)
                self.X_train.to_csv(os.path.join(self.data_dir, 'X_train.csv'), index=False)
                self.X_test.to_csv(os.path.join(self.data_dir, 'X_test.csv'), index=False)
                self.X_val.to_csv(os.path.join(self.data_dir, 'X_val.csv'), index=False)
                self.y_train.to_csv(os.path.join(self.data_dir, 'y_train.csv'), index=False)
                self.y_test.to_csv(os.path.join(self.data_dir, 'y_test.csv'), index=False)
                self.y_val.to_csv(os.path.join(self.data_dir, 'y_val.csv'), index=False)
            
            if explain:
                explainer = Explainer.from_trainer(self)
                explainer.run()

    def infer(self, data: pd.DataFrame,
              model: str = None):
        
        return self.predictor.predict_proba(data, model) if self.predictor.can_predict_proba else self.predictor.predict(data, model) 
    
    @classmethod
    def load_model(cls, 
                   model_dir: str = None,
                   project_dir: str = None):
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
            model_dir = os.path.join(os.path.join(project_dir, f'autogluon_models_best_fold'))

        trainer = cls()
        trainer.predictor = TabularPredictor.load(model_dir, verbosity=1)

        return trainer


