# Analyzer

The `Analyzer` module is designed for data visualization and exploration. It helps to gain insights into the data, identify patterns, and assess relationships between different features, which is essential for building effective models.

## Example Usage

```python
from jarvais.analyzer import Analyzer

data = pd.DataFrame({
        "stage": ["I", "I", "II", "III", "IV", "IV", "IV", "IV", "IV", "IV"],
        "treatment": ["surgery", "surgery", "chemo", "chemo", "chemo", "chemo", "hormone", "hormone", "hormone", "hormone"],
        "age": [45, 45, 60, 70, 80, 80, 80, 80, 80, 80],
        "tumor_size": [2.1, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5],  
        "death": [True, False, True, False, True, False, True, False, True, False],
    })
    
analyzer = Analyzer(
    data, 
    output_dir="./temp_output/test",
    categorical_columns=["stage", "treatment", "death"], 
    target_variable="death", 
    task="classification"
)

analyzer.run()
```

### Settings

The `Analyzer` module can be configured using settings. The settings are stored in a JSON file and can be loaded using the `Analyzer.from_settings` function. Example settings:

```bash
Analyzer(
    AnalyzerSettings(
        output_dir=PosixPath('temp_output/test'),
        categorical_columns=['stage', 'treatment', 'death'],
        continuous_columns=['tumor_size', 'age'],
        date_columns=[],
        task='classification',
        target_variable='death',
        generate_report=True,
        settings_path=PosixPath('temp_output/test/analyzer_settings.json'),
        settings_schema_path=PosixPath('temp_output/test/analyzer_settings.schema.json'),
        missingness=MissingnessModule(
            categorical_strategy={'stage': 'unknown', 'treatment': 'unknown', 'death': 'unknown'},
            continuous_strategy={'tumor_size': 'median', 'age': 'median'},
            enabled=True
        ),
        outlier=OutlierModule(
            categorical_strategy={'stage': 'frequency', 'treatment': 'frequency', 'death': 'frequency'},
            continuous_strategy={'tumor_size': 'none', 'age': 'none'},
            threshold=0.01,
            enabled=True
        ),
        encoding=OneHotEncodingModule(columns=['stage', 'treatment'], target_variable='death', prefix_sep='|', enabled=True),
        visualization=VisualizationModule(plots=['corr', 'pairplot', 'umap', 'frequency_table', 'multiplot'], enabled=True)
    )
)
```

## Example Output

```bash
Outlier Detection:
  - Outliers found in Gender: ['Male: 5 out of 1000']
  - Outliers found in Disease Type: ['Lung Cancer: 10 out of 1000']
  - No Outliers found in Treatment
  - Outliers found in Tumor Size: ['12.5: 2 out of 1000']
```

### TableOne(Data Summary)

|                      | Category          | Missing   | Overall     |
|----------------------|-------------------|-----------|-------------|
| n                    |                   |           | 1000        |
| Age, mean (SD)       |                   | 0         | 58.2 (12.3) |
| Tumor Size, mean (SD)|                   | 0         | 4.5 (1.2)   |
| Gender, n (%)        | Female            |           | 520 (52%)   |
|                      | Male              |           | 480 (48%)   |
| Disease Type, n (%)  | Breast Cancer     |           | 300 (30%)   |
|                      | Lung Cancer       |           | 150 (15%)   |
|                      | Prostate Cancer   |           | 100 (10%)   |

## Output Files

The Analyzer module generates the following files and directories:

- **analysis_report.pdf**: A PDF report summarizing the analysis results.
- **tableone.csv**: CSV file containing summary statistics for the dataset.
- **updated_data.csv**: CSV file with the cleaned and processed data.
- **analyzer_settings.json**: Configuration file for the analysis setup.
- **analyzer_settings.schema.json**: Schema file for the analyzer settings.

### Figures

#### 1. Frequency Tables
Visualizations comparing different categorical features.

<img src="../example_images/Sex_vs_Chemotherapy.png" alt="Frequency Tables Example" width="600"/>

---

#### 2. Multi-plots
Visualizations showing combinations of features for deeper analysis.

<img src="../example_images/Sex_multiplots.png" alt="Multi-plots Example" width="600"/>

---

### Additional Figures

#### 1. Pairplot
Pairwise relationships between continuous variables.

<img src="../example_images/pairplot.png" alt="Pairplot" width="500"/>

---

#### 2. Pearson Correlation Matrix
A heatmap visualizing Pearson correlations between variables.

<img src="../example_images/pearson_correlation.png" alt="Pearson Correlation" width="500"/>

---

#### 3. Spearman Correlation Matrix
A heatmap visualizing Spearman correlations between variables.

<img src="../example_images/spearman_correlation.png" alt="Spearman Correlation" width="500"/>

---

#### 4. UMAP of Continuous Data
UMAP visualization of continuous data.

<img src="../example_images/umap_continuous_data.png" alt="UMAP Continuous Data" width="500"/>

#### 5. Kaplan Meier Curve (Survival)
Kaplan Meier Curves for all categorical variables. An option exclusive to survival data.

<img src="../example_images/kaplan_meier_Disease Site.png" alt="Kaplan Meier Curve" width="500"/>

### Analysis Report

![Analysis Report](<./pdfs/analysis_report.pdf>){ type=application/pdf style="min-height:75vh;width:75%" }