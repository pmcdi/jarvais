import os
import yaml

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap

from .eval import plot_classification_diagnostics, plot_regression_diagnostics

class ModelWrapper:
    def __init__(self, predictor, feature_names, target_variable=None):
        self.ag_model = predictor
        self.feature_names = feature_names
        self.target_variable = target_variable
        if target_variable is None and predictor.problem_type != 'regression':
            print("Since target_class not specified, SHAP will explain predictions for each class")
    
    def predict_proba(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1,-1)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        preds = self.ag_model.predict_proba(X)
        return preds
        # if self.ag_model.problem_type == "regression" or self.target_variable is None:
        #     return preds
        # else:
        #     return preds[self.target_variable]riable]    

class AutoMLExplainer():
    def __init__(self, 
                 trainer,
                 X_train,
                 X_test,
                 y_test,
                 output_dir='.'):
        
        self.trainer = trainer
        self.predictor = ModelWrapper(trainer.predictor, 
                                      trainer.feature_names,
                                      trainer.target_variable)
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.output_dir = output_dir

    def _plot_feature_importance(self):
        """
        Plots the feature importance with standard deviation and p-value significance.
        """
        df = self.predictor.ag_model.feature_importance(pd.concat([self.X_test, self.y_test], axis=1))

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
        ax.set_xticklabels(df.index.values, rotation=90)

        # Show plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'))
        plt.close()

    def shap_values(self):
        # try:

        background_data = shap.sample(self.X_train, 100)  # For instance, use 100 samples

        # Create the SHAP explainer with the summarized background data
        shap_exp = shap.KernelExplainer(self.predictor.predict_proba, background_data)
        
        test_data = shap.sample(self.X_test, 20) 

        # Compute SHAP values for the test set
        self.shap_values = shap_exp.shap_values(test_data)
        
        # Generate and save the SHAP summary plot
        shap.summary_plot(self.shap_values, test_data, show=False)
        plt.savefig(os.path.join(self.output_dir, 'shap_summary.png'))
        plt.close()
            
        # except Exception as e:
        #     # Log or print the error for debugging purposes
        #     print(f"Error in SHAP analysis: {e}")

    def lime_values(self):
        pass

    def run(self):
        # Plot diagnostics
            try:
                if self.predictor.ag_model.problem_type == 'binary':
                    plot_classification_diagnostics(self.y_test, self.predictor.predict_proba(self.X_test).iloc[:, 1], self.output_dir)
                elif self.predictor.ag_model.problem_type == 'regression':
                    plot_regression_diagnostics(self.y_test, self.predictor.predict(self.X_test, as_pandas=False))
            except Exception as e:
                print(f"Error in plotting diagnostics: {e}")

            # Plot feature importance
            try:
                self._plot_feature_importance()
            except Exception as e:
                print(f"Error in plotting feature importance: {e}")

            self.shap_values()

    @classmethod
    def from_trainer(cls, trainer):
        return cls(trainer, trainer.X_train, trainer.X_test, trainer.y_test, trainer.output_dir)