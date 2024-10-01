import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from typing import Union, List

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, VarianceThreshold, chi2 ,f_classif, f_regression
from mrmr import mrmr_classif, mrmr_regression

from autogluon.tabular import TabularPredictor

from .explainer import AutoMLExplainer

class AutoMLSupervised():
    def __init__(self,
                 task: str,
                 reduction_method: Union[str, None] = None,
                 keep_k: int = 2,
                 output_dir: Union[str, os.PathLike] = '.'):
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

        Raises
        ------
        ValueError
            If the task parameter is not one of the specified options.
        """
        self.task = task
        self.output_dir = output_dir
        self.reduction_method = reduction_method
        self.keep_k = keep_k

        if task not in ['binary', 'multiclass', 'regression', 'quantile', None]:
            raise ValueError("Invalid task parameter. Choose one of: 'binary', 'multiclass', 'regression', 'quantile'. Or provide nothing and let Autogluon infer the task.")

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _plot_feature_importance(self):
        """
        Plots the feature importance with standard deviation and p-value significance.
        """
        df = self.predictor.feature_importance(pd.concat([self.X_test, self.y_test], axis=1))

        # Plotting
        fig, ax = plt.subplots(figsize=(20, 12))

        # Adding bar plot with error bars
        bars = ax.bar(df.index, df['importance'], yerr=df['stddev'], capsize=5, color='skyblue', edgecolor='black')

        # Adding p_value significance indication
        for i, (bar, p_value) in enumerate(zip(bars, df['p_value'])):
            height = bar.get_height()
            significance = '*' if p_value < 0.05 else ''
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, significance, ha='center', va='bottom', fontsize=12, color='red')

        # Labels and title
        ax.set_xlabel('Feature')
        ax.set_ylabel('Importance')
        ax.set_title('Feature Importance with Standard Deviation and p-value Significance')
        ax.axhline(0, color='grey', linewidth=0.8)
        ax.set_xticks(np.arange(len(df.index.values)))
        ax.set_xticklabels(df.index.values, rotation=45)

        # Show plot
        plt.tight_layout()
        plt.show()

    def _feature_reduction(self, X, y):
        """
        Reduces features based on the specified reduction method.

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
        
        if self.reduction_method == 'mrmr':
            mrmr_method = mrmr_classif if self.task in ['binary', 'multiclass'] else mrmr_regression
            selected_features = mrmr_method(X=X, y=y, K=self.keep_k, n_jobs=1)
            return X[selected_features]
        
        if self.reduction_method == 'variance_threshold':
            selector = VarianceThreshold()
        elif self.reduction_method == 'corr':
            f_method = f_classif if self.task in ['binary', 'multiclass'] else f_regression
            selector = SelectKBest(score_func=f_method, k=self.keep_k)
        elif self.reduction_method == 'chi2':
            if self.task in ['binary', 'multiclass']:
                selector = SelectKBest(score_func=chi2, k=self.keep_k)
            else:
                raise ValueError('chi-squared reduction can only be done with classification tasks')

        _ = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support(indices=True)]
        return X[selected_features]

    def run(self,
            data: pd.DataFrame,
            target_variable: str,
            test_size: float = 0.2,
            exclude: List[str] = [],
            stratify_on: Union[str, None] = None):
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

            if self.task == 'binary':
                eval_metric = 'roc_auc'
            else:
                eval_metric = None # Let it infer

            self.predictor = TabularPredictor(label=target_variable,
                                         problem_type=self.task,
                                         eval_metric=eval_metric,
                                         ).fit(pd.concat([self.X_train, self.y_train], axis=1))
            
            extra_metrics = ['f1', 'average_precision'] if self.task in ['binary', 'multiclass'] else None # Need to update for regression
            show_leaderboard = ['model', 'score_test', 'score_val', 'eval_metric', 'f1', 'average_precision'] if self.task in ['binary', 'multiclass'] else ['model', 'score_test', 'score_val', 'eval_metric']

            leaderboard = self.predictor.leaderboard(pd.concat([self.X_test, self.y_test], axis=1), extra_metrics=extra_metrics)
            print('\nModel Leaderbord\n----------------')
            print(tabulate(leaderboard[show_leaderboard], tablefmt = "fancy_grid", headers="keys"))

            print('\nSimple Logistic Model\n---------------------')

            simple_predictor = TabularPredictor(label=target_variable,
                                         problem_type=self.task,
                                         eval_metric=eval_metric,
                                         ).fit(pd.concat([self.X_train, self.y_train], axis=1), hyperparameters={CustomLogisticRegressionModel: {}} )
            
            leaderboard = simple_predictor.leaderboard(pd.concat([self.X_test, self.y_test], axis=1), extra_metrics=extra_metrics)
            print(tabulate(leaderboard.iloc[[0]][show_leaderboard], tablefmt="fancy_grid", headers="keys"))

            explainer = AutoMLExplainer.from_trainer(self)
            explainer.run()

from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator

class CustomLogisticRegressionModel(AbstractModel):
    def __init__(self, **kwargs):
        # Simply pass along kwargs to parent, and init our internal `_feature_generator` variable to None
        super().__init__(**kwargs)
        self._feature_generator = None

    # The `_preprocess` method takes the input data and transforms it to the internal representation usable by the model.
    # `_preprocess` is called by `preprocess` and is used during model fit and model inference.
    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:
        # print(f'Entering the `_preprocess` method: {len(X)} rows of data (is_train={is_train})')
        X = super()._preprocess(X, **kwargs)

        if is_train:
            # X will be the training data.
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            # This converts categorical features to numeric via stateful label encoding.
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        
        # Add a fillna call to handle missing values.
        return X.fillna(0).to_numpy(dtype=np.float32)

    # The `_fit` method takes the input training data (and optionally the validation data) and trains the model.
    def _fit(self,
            X: pd.DataFrame,  # training data
            y: pd.Series,  # training labels
            **kwargs):  
        # print('Entering the `_fit` method')

        # Import the Logistic Regression model from sklearn
        from sklearn.linear_model import LogisticRegression

        # Store the feature names before transforming to numpy
        feature_names = X.columns if isinstance(X, pd.DataFrame) else range(X.shape[1])

        # Make sure to call preprocess on X near the start of `_fit`.
        X = self.preprocess(X, is_train=True)

        # This fetches the user-specified (and default) hyperparameters for the model.
        params = self._get_model_params()

        # Set self.model to Logistic Regression with the desired hyperparameters.
        self.model = LogisticRegression(**params)
        self.model.fit(X, y)

        # Print the coefficients and the intercept after training
        coefs = self.model.coef_.flatten()  # flatten the array if it's multi-dimensional (for binary classification)
        intercept = self.model.intercept_[0]  # assuming binary classification (single intercept)

        # Create a DataFrame to display feature names with their corresponding coefficients
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefs
        })

        # Print the coefficients along with their corresponding features
        print("\nSimple Logistic Model Coefficients:")
        print(coef_df.sort_values(by='Coefficient', key=abs, ascending=False).to_string(index=False))  # sort by coefficient value for better readability
        
        # Print the intercept separately
        print(f"\nSimple Logistic Model Intercept: {intercept}")
        
        # print('Exiting the `_fit` method')

    # The `_set_default_params` method defines the default hyperparameters of the model.
    def _set_default_params(self):
        default_params = {
            'solver': 'lbfgs',  # Solver to use in the optimization problem
            'max_iter': 1000,    # Maximum number of iterations for convergence
            'random_state': 0,   # Set the random seed
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    # The `_get_default_auxiliary_params` method defines model-agnostic parameters such as maximum memory usage and valid input column dtypes.
    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_raw_types=['int', 'float', 'category'],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params
