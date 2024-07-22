import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from typing import Union

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, VarianceThreshold, chi2 ,f_classif, f_regression
from mrmr import mrmr_classif, mrmr_regression

from autogluon.tabular import TabularPredictor

from .eval import plot_classification_diagnostics, plot_regression_diagnostics

class AutoMLSupervised():
    def __init__(self, 
                 task: str,
                 reduction_method: Union[str, None] = None,
                 keep_k: int = 2,
                 output_dir: Union[str, os.PathLike] = '.',
                ):
        """
        Initialize the AutoMLTrainer class with specified configurations.

        Parameters
        ----------
        task : str,
            The type of task to handle. Options are 'binary', 'multiclass', 'regression', 'quantile'.
        reduction_method : str, default=None
            The feature reduction method to apply. Options are 'mrmr', 'variance_threshold', 'corr', 'chi2'. By default no feature reduction applied.
        keep_k : int, default=2
            Number of features to keep, if a reduction method is defined. By default 2.

        Raises
        ------
        ValueError
            If the task parameter is not one of the specified options.
        """
        self.task = task
        self.output_dir = output_dir
        self.reduction_method = reduction_method
        self.keep_k = keep_k

        if self.reduction_method == 'mrmr':
            raise ValueError('The mrmr package is brokie, working on fix')

        if task not in ['binary', 'multiclass', 'regression', 'quantile', None]:
            raise ValueError("Invalid task parameter. Choose one of: 'binary', 'multiclass', 'regression', 'quantile'. Or provide nothing and let Autogluon infer the task.")
        
    def _plot_feature_importance(self):
        
        df = self.predictor.feature_importance(pd.concat([self.X_test, self.y_test], axis=1))

        
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))

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
        
        if self.reduction_method == 'mrmr':
            mrmr_method = mrmr_classif if self.task in ['binary', 'multiclass'] else mrmr_regression
            selected_features = mrmr_method(X=X, y=y, K=self.keep_k)
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

    def run(self, data, target_variable,
            test_size: float = 0.2,
            exclude: list = [], stratify_on=None):
        
            exclude.append(target_variable)
            X = data.drop(columns=exclude) 
            y = data[target_variable]

            if self.reduction_method is not None:
                print(f'Applying {self.reduction_method} for feature reduction')
                X = self._feature_reduction(X, y)
                print(f'Features kept: {X.columns.values}')

            if y.value_counts().min() > 1: # Meaning it can be used to stratify, if this condition is not met train_test_split produces - ValueError: The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.
                if stratify_on is not None:
                    stratify_col = y.astype(str) + '_' + data[stratify_on].astype(str) 
                else:
                    stratify_col = y                
            else:
                    if stratify_on is not None:
                        stratify_col = data[stratify_on]
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

            print('\nModel Leaderbord\n')
            print(tabulate(self.predictor.leaderboard(), tablefmt = "fancy_grid", headers="keys"))

            if self.predictor.problem_type == 'binary':
                plot_classification_diagnostics(self.y_test, self.predictor.predict_proba(self.X_test, as_pandas=False)[:, 1])
            elif self.predictor.problem_type == 'regression':
                plot_regression_diagnostics(self.y_test, self.predictor.predict(self.X_test, as_pandas=False))
   
            self._plot_feature_importance()

    