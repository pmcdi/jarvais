# jarvAIs

[![DOI](https://zenodo.org/badge/813671188.svg)](https://doi.org/10.5281/zenodo.14827357)
[![BUILD DOCS](https://github.com/pmcdi/jarvais/actions/workflows/build_docs.yml/badge.svg)](https://github.com/pmcdi/jarvais/actions/workflows/build_docs.yml)
[![CI tests](https://github.com/pmcdi/jarvais/actions/workflows/ci.yml/badge.svg)](https://github.com/pmcdi/jarvais/actions/workflows/ci.yml)

## Overview

jarvAIs is a Python package designed to automate and enhance machine learning workflows. The primary goal of this project is to reduce redundancy in repetitive tasks, improve consistency, and elevate the quality of standardized processes in oncology research.

Follow pixi installation process found [here](https://pixi.sh/latest/)

1. **Clone Repo and Navigate**

    ```bash
    git clone https://github.com/pmcdi/jarvais.git
    cd jarvais
    ```

2. **Install Dependencies**

    ```bash
    pixi install
    ```

## Modules

This package consists of 3 different modules:
-  **Analyzer**: A module that analyzes and processes data, providing valuable insights for downstream tasks.
- **Trainer**: A module for training machine learning models, designed to be flexible and efficient.
- **Explainer**: A module that explains model predictions, offering interpretability and transparency in decision-making.

### Analyzer

The **Analyzer** module is designed for data visualization and exploration. It helps to gain insights into the data, identify patterns, and assess relationships between different features, which is essential for building effective models.

#### Example Usage

```python
from jarvais.analyzer import Analyzer

analyzer = Analyzer(data, target_variable='target', output_dir='.')
analyzer.run()
```
#### Example Output

```bash
Feature Types:
  - Categorical: ['Gender', 'Disease Type', 'Treatment']
  - Continuous: ['Age', 'Tumor Size']

Outlier Detection:
  - Outliers found in Gender: ['Male: 5 out of 1000']
  - Outliers found in Disease Type: ['Lung Cancer: 10 out of 1000']
  - No Outliers found in Treatment
  - Outliers found in Tumor Size: ['12.5: 2 out of 1000']
```

##### TableOne(Data Summary):

| Feature             | Category          | Missing   | Overall     |
|---------------------|-------------------|-----------|-------------|
| n                   |                   |           | 1000        |
| Age, mean (SD)      |                   | 0         | 58.2 (12.3) |
| Tumor Size, mean (SD)|                   | 0         | 4.5 (1.2)   |
| Gender, n (%)       | Female            |           | 520 (52%)   |
|                     | Male              |           | 480 (48%)   |
| Disease Type, n (%) | Breast Cancer     |           | 300 (30%)   |
|                     | Lung Cancer       |           | 150 (15%)   |
|                     | Prostate Cancer   |           | 100 (10%)   |

#### Output Files:

The Analyzer module generates the following files and directories:

- **analysis_report.pdf**: A PDF report summarizing the analysis results.
- **config.yaml**: Configuration file for the analysis setup.

**Figures:**
- **frequency_tables**: Contains visualizations comparing different categorical features.
- **multiplots**: Visualizations showing combinations of features for deeper analysis.
- **Additional Figures**:
  - `pairplot.png`: Pairwise relationships between continuous variables.
  - `pearson_correlation.png`: Pearson correlation matrix.
  - `spearman_correlation.png`: Spearman correlation matrix.
  - `umap_continuous_data.png`: UMAP visualization of continuous data.

- **Data Files:**
  - **tableone.csv**: CSV file containing summary statistics for the dataset.
  - **updated_data.csv**: CSV file with the cleaned and processed data.

### Trainer Module

The **Trainer** module simplifies and automates the process of feature reduction, model training, and evaluation for various machine learning tasks, ensuring flexibility and efficiency.

#### Key Features
1. **Feature Reduction**:
   - Supports methods such as `mrmr`, `variance_threshold`, `corr`, and `chi2` to identify and retain relevant features.
2. **Automated Model Training**:
   - Integrates with AutoGluon for model training, selection, and optimization.
   - Handles tasks such as binary classification, multiclass classification, regression, and survival.

#### Example Usage

```python
from jarvais.trainer import TrainerSupervised

trainer = TrainerSupervised(task='binary', output_dir='./trainer_outputs')
trainer.run(data=data, target_variable='target', save_data=True)
```

#### Example Output

```bash
Training fold 1/5...  
Fold 1 score: `0.8467207586933614`

Training fold 2/5...  
Fold 2 score: `0.8487846136306914`
...
```

##### Model Leaderboard
Displays values in `mean [min, max]` format across training folds.

| **Model**             | **Score Test**               | **Score Val**               | **Score Train**             |
|------------------------|------------------------------|------------------------------|------------------------------|
| **WeightedEnsemble_L2** | AUROC: 0.82 [0.82, 0.83]     | AUROC: 0.85 [0.85, 0.85]     | AUROC: 1.0 [1.0, 1.0]        |
|                        | F1: 0.13 [0.11, 0.14]        | F1: 0.09 [0.07, 0.12]        | F1: 0.95 [0.9, 1.0]          |
|                        | AUPRC: 0.48 [0.45, 0.52]     | AUPRC: 0.47 [0.44, 0.49]     | AUPRC: 0.96 [0.91, 1.0]      |
| **ExtraTreesGini**      | AUROC: 0.82 [0.82, 0.82]     | AUROC: 0.84 [0.84, 0.84]     | AUROC: 1.0 [1.0, 1.0]        |
|                        | F1: 0.21 [0.19, 0.22]        | F1: 0.16 [0.14, 0.18]        | F1: 1.0 [1.0, 1.0]           |
|                        | AUPRC: 0.45 [0.45, 0.45]     | AUPRC: 0.43 [0.41, 0.45]     | AUPRC: 1.0 [1.0, 1.0]        |

### Explainer Module

The **Explainer** module is designed to evaluate trained models by generating diagnostic plots, auditing bias, and producing comprehensive reports. It supports various supervised learning tasks, including classification, regression, and survival models. 

The module provides an easy-to-use interface for model diagnostics, bias analysis, and feature importance visualization, facilitating deeper insights into the model's performance and fairness.


#### Features

- **Diagnostic Plots**: Generates performance diagnostics, including classification metrics, regression plots, and SHAP value visualizations.
- **Bias Audit**: Identifies potential biases in model predictions with respect to sensitive features.
- **Feature Importance**: Calculates and visualizes feature importance using permutation importance or model-specific methods.
- **Comprehensive Reports**: Creates a detailed PDF report summarizing all diagnostic results.

#### Example Usage

```python
from jarvais.explainer import Explainer

# Prefered method is to initialize from trainer
exp = Explainer.from_trainer(trainer)
exp.run()
```

#### Output Files:

The **Explainer** module generates the following files and directories:

- **explainer_report.pdf**: A PDF report summarizing the model diagnostics, bias audit results, and feature importance.
- **bias/**: Contains CSV files with bias metrics for different sensitive features.
- **figures/**: Contains diagnostic plots for model evaluation and feature importance.
  - `confusion_matrix.png`: Visual representation of the model’s confusion matrix.
  - `feature_importance.png`: A plot visualizing the importance of features used by the model.
  - `model_evaluation.png`: A visual summary of model evaluation.
  - `shap_barplot.png`: SHAP value bar plot for model interpretability.
  - `shap_heatmap.png`: SHAP value heatmap for model interpretability.