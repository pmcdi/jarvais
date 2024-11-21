import os, pickle

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap

from .utils import plot_feature_importance, plot_shap_values, plot_classification_diagnostics, plot_regression_diagnostics

class Explainer():
    def __init__(self, 
                 trainer,
                 X_train,
                 X_test,
                 y_test,
                 output_dir='.'):
        
        self.trainer = trainer
        self.predictor = trainer.predictor
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.output_dir = output_dir

    def run(self):
        # Plot diagnostics
            # try:
            if self.predictor.problem_type == 'binary':
                plot_classification_diagnostics(self.y_test, self.predictor.predict_proba(self.X_test).iloc[:, 1], self.trainer.y_val, self.predictor.predict_proba(self.trainer.X_val).iloc[:, 1], self.output_dir)
                plot_shap_values(self.predictor, self.X_train, self.X_test, output_dir=self.output_dir)
            elif self.predictor.problem_type == 'regression':
                plot_regression_diagnostics(self.y_test, self.predictor.predict(self.X_test, as_pandas=False), self.output_dir)
            # except Exception as e:
            #     print(f"Error in plotting diagnostics: {e}")

            # Plot feature importance
            try:
                plot_feature_importance(self.predictor, self.X_test, self.y_test, output_dir=self.output_dir)
            except Exception as e:
                print(f"Error in plotting feature importance: {e}")

    @classmethod
    def from_trainer(cls, trainer):
        return cls(trainer, trainer.X_train, trainer.X_test, trainer.y_test, trainer.output_dir)