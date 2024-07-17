import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from typing import Union

from sklearn.model_selection import train_test_split

from autogluon.tabular import TabularPredictor

from .eval import plot_classification_diagnostics, plot_regression_diagnostics

class AutoMLSupervised():
    def __init__(self, 
                 task: Union[str, None] = None,
                 output_dir: Union[str, os.PathLike] = '.'
                ):
        """
        Initialize the AutoMLTrainer class with specified configurations.

        Parameters
        ----------
        task : str, default=None
            The type of task to handle. Options are 'binary', 'multiclass', 'regression', 'quantile'. By default infered by Autogluon

        Raises
        ------
        ValueError
            If the task parameter is not one of the specified options.
        """
        self.output_dir = output_dir
        self.task = task

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

    def run(self, data, target_variable,
            test_size: float = 0.2,
            exclude: list = [], stratify_on=''):
        
            exclude.append(target_variable)
            X = data.drop(columns=exclude) 
            y = data[target_variable]

            self.data_columns = X.columns

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
            print(self.predictor.leaderboard()[['model', 'score_val', 'eval_metric']])

            if self.predictor.problem_type == 'binary':
                plot_classification_diagnostics(self.predictor, self.X_test, self.y_test, self.data_columns)
            elif self.predictor.problem_type == 'regression':
                plot_regression_diagnostics(self.predictor, self.X_test, self.y_test, self.data_columns)
   

            self._plot_feature_importance()

    