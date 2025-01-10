# Explainer Module

The `Explainer` module is designed to evaluate trained models by generating diagnostic plots, auditing bias, and producing comprehensive reports. It supports various supervised learning tasks, including classification, regression, and time-to-event models. 

The module provides an easy-to-use interface for model diagnostics, bias analysis, and feature importance visualization, facilitating deeper insights into the model's performance and fairness.


## Features

- **Diagnostic Plots**: Generates performance diagnostics, including classification metrics, regression plots, and SHAP value visualizations.
- **Bias Audit**: Identifies potential biases in model predictions with respect to sensitive features.
- **Feature Importance**: Calculates and visualizes feature importance using permutation importance or model-specific methods.
- **Comprehensive Reports**: Creates a detailed PDF report summarizing all diagnostic results.

## Example Usage

```python
from AutoML.explainer import Explainer

# Prefered method is to initialize from trainer
exp = Explainer.from_trainer(trainer)
exp.run()
```

## Output Files

The **Explainer** module generates the following files and directories:

- **explainer_report.pdf**: A PDF report summarizing the model diagnostics, bias audit results, and feature importance.
- **bias/**: Contains CSV files with bias metrics for different sensitive features.

### Figures

#### 1. Confusion Matrix(Classification Models)

<img src="../example_images/confusion_matrix.png" alt="Confusion Matrix" width="500"/>

---

#### 2. Feature Importance

<img src="../example_images/feature_importance.png" alt="Feature Importance" width="750"/>

---

#### 3. Model Evaluation

<img src="../example_images/model_evaluation.png" alt="Model Evaluation" width="1000"/>

---


#### 4. Shap Plots

<img src="../example_images/shap_barplot.png" alt="Shap Bar Map" width="750"/><br>
<img src="../example_images/shap_heatmap.png" alt="Shap Heat Map" width="750"/>

#### 5. Bootsrapped Metrics

<img src="../example_images/test_metrics_bootstrap.png" alt="Test Bootstrapped Metrics" width="750"/><br>
<img src="../example_images/validation_metrics_bootstrap.png" alt="Val Bootstrapped Metrics" width="750"/><br>
<img src="../example_images/train_metrics_bootstrap.png" alt="Train Bootstrapped Metrics" width="750"/>


<!-- 
- **figures/**: Contains diagnostic plots for model evaluation and feature importance.
  - `confusion_matrix.png`: Visual representation of the model’s confusion matrix.
  - `feature_importance.png`: A plot visualizing the importance of features used by the model.
  - `model_evaluation.png`: A visual summary of model evaluation.
  - `shap_barplot.png`: SHAP value bar plot for model interpretability.
  - `shap_heatmap.png`: SHAP value heatmap for model interpretability. -->