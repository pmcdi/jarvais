import os
import yaml

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap

# class ModelWrapper:
#     def __init__(self, predictor, feature_names, target_class=None):
#         self.ag_model = predictor
#         self.feature_names = feature_names
#         self.target_class = target_class
#         if target_class is None and predictor.problem_type != 'regression':
#             print("Since target_class not specified, SHAP will explain predictions for each class")
    
#     def predict_proba(self, X):
#         if isinstance(X, pd.Series):
#             X = X.values.reshape(1,-1)
#         if not isinstance(X, pd.DataFrame):
#             X = pd.DataFrame(X, columns=self.feature_names)
#         preds = self.ag_model.predict_proba(X)
#         if self.ag_model.problem_type == "regression" or self.target_class is None:
#             return preds
#         else:
#             return preds[self.target_class]    

class AutoMLExplainer():
    def __init__(self, 
                 predictor, 
                 X_test,
                 y_test):
        
        self.predictor = predictor
        self.X_test = X_test
        self.y_test = y_test
        
    def feature_importance(self):
        """
        Returns feature importance of the model
        """
        return self.predictor.feature_importance(pd.concat([self.X_test, self.y_test], axis=1))

    def _plot_feature_importance(self):
        """
        Plots the feature importance with standard deviation and p-value significance.
        """
        df = self.feature_importance()

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

    def shap_values(self):
        # not sure if i can pass median of X_test or if i should be passing X_train
        shap_exp = shap.KernelExplainer(self.predictor.predict_proba, self.X_test.median())
        
        NSHAP_SAMPLES = 100 

        self.shap_values = shap_exp.shap_values(self.X_test, nsamples=NSHAP_SAMPLES)
        shap.summary_plot(shap_values, self.X_test)
        shap.dependence_plot("Education-Num", shap_values, self.X_test)

    def lime_values(self):
        pass

    def explain(self):
        pass

    @classmethod
    def from_model(cls, predictor):
        return cls(predictor, predictor.X_test, predictor.y_test)
    
