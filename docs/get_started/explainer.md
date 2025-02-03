# Explainer Module

The `Explainer` module is designed to evaluate trained models by generating diagnostic plots, auditing bias, and producing comprehensive reports. It supports various supervised learning tasks, including classification, regression, and survival models. 

The module provides an easy-to-use interface for model diagnostics, bias analysis, and feature importance visualization, facilitating deeper insights into the model's performance and fairness.


## Features

- **Diagnostic Plots**: Generates performance diagnostics, including classification metrics, regression plots, and SHAP value visualizations.
- **Bias Audit**: Identifies potential biases in model predictions with respect to sensitive features.
- **Feature Importance**: Calculates and visualizes feature importance using permutation importance or model-specific methods.
- **Comprehensive Reports**: Creates a detailed PDF report summarizing all diagnostic results.

## Example Usage

```python
from jarvais.explainer import Explainer

# Prefered method is to initialize from trainer
exp = Explainer.from_trainer(trainer)
exp.run()
```

## Output Files

The **Explainer** module generates the following files and directories:

- **explainer_report.pdf**: A PDF report summarizing the model diagnostics, bias audit results, and feature importance.
- **bias/**: Contains CSV files with bias metrics for different sensitive features.


### Common Figures 

#### Feature Importance

<img src="../example_images/feature_importance.png" alt="Feature Importance" width="750"/>

---

#### Bootsrapped Metrics

<img src="../example_images/test_metrics_bootstrap.png" alt="Test Bootstrapped Metrics" width="750"/><br>
<img src="../example_images/validation_metrics_bootstrap.png" alt="Val Bootstrapped Metrics" width="750"/><br>
<img src="../example_images/train_metrics_bootstrap.png" alt="Train Bootstrapped Metrics" width="750"/>

---

### Classification Figures

#### Confusion Matrix

<img src="../example_images/confusion_matrix.png" alt="Confusion Matrix" width="750"/>

---

#### Model Evaluation

<img src="../example_images/model_evaluation.png" alt="Model Evaluation" width="1000"/>

---

#### Shap Plots

<img src="../example_images/shap_barplot.png" alt="Shap Bar Map" width="750"/><br>
<img src="../example_images/shap_heatmap.png" alt="Shap Heat Map" width="750"/>

### Regression Figures

#### Residual Plots

<img src="../example_images/residual_plot.png" alt="Residual Plot" width="500"/>
<img src="../example_images/residual_hist.png" alt="Residual Histogram" width="500"/>

#### True vs Predicted

<img src="../example_images/true_vs_predicted.png" alt="True vs Predicted" width="750"/>

### Explainer Report

![Explainer Report](<./pdfs/explainer_report.pdf>){ type=application/pdf style="min-height:75vh;width:75%" }