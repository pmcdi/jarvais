from typing import Tuple, Callable
from pathlib import Path
from itertools import combinations

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
from tabulate import tabulate

import numpy as np
import statsmodels.api as sm
from lifelines import CoxPHFitter

from ..utils.plot import plot_kaplan_meier_by_category

def generate_violin(
        sensitive_features: pd.DataFrame,
        endpoint: pd.Series,
        output_dir: Path | str,
        show_figure: bool = True, 
        save_figure = True, 
        tag: str = '',
    ) -> None:
    """
    Generates violin plots to visualize the distribution of `endpoint` across `sensitive_features`.

    This function creates violin plots for each feature in `sensitive_features` to compare the 
    distribution of `endpoint` across different groups. The plots are saved in `output_dir`.

    Args:
        sensitive_features (pd.DataFrame): DataFrame containing categorical sensitive features.
        endpoint (pd.Series): Series representing the bias metric or outcome variable.
        output_dir (Path | str): Directory where plots should be saved.
        show_figure (bool, optional): Whether to display the figure. Defaults to True.
        save_figure (bool, optional): Whether to save the figure. Defaults to True.
        tag (str, optional): Optional tag for filename customization. Defaults to ''.

    Returns:
        None
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for feature_name, feature_series in sensitive_features.items():
        plt.figure(figsize=(8, 6)) 
        sns.set_theme(style="whitegrid")  

        sns.violinplot(
            x=feature_series, 
            y=endpoint, 
            palette="muted",  
            inner="quart", 
            linewidth=1.25 
        )

        title_tag = f" ({tag})" if tag else ""
        filename_tag = f"_{tag}" if tag else ""

        plt.title(f'{endpoint.name} Distribution by {feature_name} {title_tag}', fontsize=16, weight='bold')  
        plt.xlabel(f'{feature_name}', fontsize=14)  
        plt.ylabel(f'{endpoint.name} per Patient', fontsize=14) 
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        
        if save_figure:
            plt.savefig(output_dir / f'{feature_name}_vs_{endpoint.name}{filename_tag}.png')  
        if show_figure:
            plt.show()

def generate_kaplan_meier(
        sensitive_features: pd.DataFrame,
        endpoint: pd.DataFrame,
        output_dir: Path | str,
    ) -> None:
    """
    Generates Kaplan-Meier survival curves for categorical `sensitive_features`.

    This function plots Kaplan-Meier survival curves for each category in `sensitive_features`, 
    using `time` and `event` columns from the `endpoint` DataFrame.

    Args:
        sensitive_features (pd.DataFrame): DataFrame containing categorical sensitive features.
        endpoint (pd.DataFrame): DataFrame with 'time' (duration) and 'event' (censoring status).
        output_dir (Path | str): Directory to save Kaplan-Meier plots.

    Raises:
        AssertionError: If 'time' or 'event' columns are missing in `endpoint`.

    Returns:
        None
    """
    assert {'time', 'event'}.issubset(set(endpoint.columns)), \
    "Error: 'endpoint' DataFrame must contain 'time' and 'event' columns."

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_kaplan_meier_by_category(
        sensitive_features, 
        endpoint, 
        sensitive_features.columns, 
        output_dir, 
        show_figure=True
    )

def subgroup_analysis_OLS(
        sensitive_features: pd.DataFrame,
        endpoint: pd.Series,
        output_dir: Path | str,
        print_report: bool = True, 
        save_report: bool = True,
    ) -> Tuple[sm.OLS, float]:
    """
    Performs Ordinary Least Squares (OLS) regression to analyze the relationship between 
    `sensitive_features` and `endpoint`.

    This function fits an OLS model using one-hot encoded `sensitive_features` to predict `endpoint`. 
    If the overall model p-value is < 0.05, a detailed report is printed and optionally saved.

    Args:
        sensitive_features (pd.DataFrame): DataFrame containing categorical sensitive features.
        endpoint (pd.Series): Series representing the target variable (bias metric).
        output_dir (Path | str): Directory to save the regression report.
        print_report (bool, optional): Whether to print the report. Defaults to True.
        save_report (bool, optional): Whether to save the report. Defaults to True.

    Returns:
        Tuple[sm.OLS, float]: The fitted OLS model and the model's F-statistic p-value.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = sensitive_features.columns

    one_hot_encoded = pd.get_dummies(
        sensitive_features[feature_names], 
    )
    X_columns = one_hot_encoded.columns

    X = one_hot_encoded.values  
    y = endpoint.values  

    X = sm.add_constant(X.astype(float), has_constant='add')
    model = sm.OLS(y, X).fit()

    p_value = model.f_pvalue

    if p_value < 0.05 and (print_report or save_report):
        output = []

        print(f"⚠️  **Possible Bias Detected in {feature_names}** ⚠️\n")
        output.append(f"=== Subgroup Analysis for '{feature_names}' Using OLS Regression ===\n")

        output.append("Model Statistics:")
        output.append(f"    R-squared:                  {model.rsquared:.3f}")
        output.append(f"    F-statistic:                {model.fvalue:.3f}")
        output.append(f"    F-statistic p-value:        {p_value:.4f}")
        output.append(f"    AIC:                        {model.aic:.2f}")
        output.append(f"    Log-Likelihood:             {model.llf:.2f}")

        summary_df = pd.DataFrame({
            'Feature': ['const'] + X_columns.tolist(),     # Predictor names (includes 'const' if added)
            'Coefficient': model.params,    # Coefficients
            'Standard Error': model.bse     # Standard Errors
        })
        table_output = tabulate(summary_df, headers='keys', tablefmt='fancy_grid', showindex=False, floatfmt=".3f")
        output.append("Model Coefficients:")
        output.append('\n'.join(['    ' + line for line in table_output.split('\n')]))

        output_text = '\n'.join(output)

        if print_report:
            print(output_text)
        
        if save_report:
            with open(output_dir / f'{'_'.join(feature_names)}_OLS_summary.txt', 'w') as f:
                f.write(output_text)

    return model, p_value
    
def subgroup_analysis_CoxPH(
        sensitive_features: pd.DataFrame,
        endpoint: pd.DataFrame,
        output_dir: Path | str,
        print_report: bool = True, 
        save_report: bool = True,
    ) -> Tuple[CoxPHFitter, float]:
    """
    Fits a Cox Proportional Hazards (CoxPH) model using `sensitive_features` and survival data.

    This function models the hazard ratio for different sensitive feature categories 
    using a CoxPH model. If the likelihood ratio test p-value is < 0.05, a detailed 
    report is printed and optionally saved.

    Args:
        sensitive_features (pd.DataFrame): DataFrame containing categorical sensitive features.
        endpoint (pd.DataFrame): DataFrame with 'time' (duration) and 'event' (censoring status).
        output_dir (Path | str): Directory to save the regression report.
        print_report (bool, optional): Whether to print the report. Defaults to True.
        save_report (bool, optional): Whether to save the report. Defaults to True.

    Raises:
        AssertionError: If 'time' or 'event' columns are missing in `endpoint`.

    Returns:
        Tuple[CoxPHFitter, float]: The fitted Cox model and the log-likelihood ratio test p-value.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assert {'time', 'event'}.issubset(set(endpoint.columns)), \
    "Error: 'endpoint' DataFrame must contain 'time' and 'event' columns."

    feature_names = sensitive_features.columns

    one_hot_encoded = pd.get_dummies(
        sensitive_features[feature_names], 
    )
    df_encoded = endpoint.join(one_hot_encoded)

    cph = CoxPHFitter(penalizer=0.0001)
    cph.fit(df_encoded, duration_col='time', event_col='event')  

    p_value = cph.log_likelihood_ratio_test().p_value          
    
    if p_value < 0.05 and (print_report or save_report):
        output = []

        print(f"⚠️  **Possible Bias Detected in {feature_names}** ⚠️")
        output.append(f"=== Subgroup Analysis for '{feature_names}' Using Cox Proportional Hazards Model ===\n")

        output.append("Model Statistics:")
        output.append(f"    AIC (Partial):               {cph.AIC_partial_:.2f}")
        output.append(f"    Log-Likelihood:              {cph.log_likelihood_:.2f}")
        output.append(f"    Log-Likelihood Ratio p-value: {p_value:.4f}")
        output.append(f"    Concordance Index (C-index):   {cph.concordance_index_:.2f}")

        summary_df = pd.DataFrame({
            'Feature': cph.summary.index.to_list(),
            'Coefficient': cph.summary['coef'].to_list(),
            'Standard Error': cph.summary['se(coef)'].to_list()
        })
        table_output = tabulate(summary_df, headers='keys', tablefmt='grid', showindex=False, floatfmt=".3f")
        output.append("Model Coefficients:")
        output.append('\n'.join(['    ' + line for line in table_output.split('\n')]))

        output_text = '\n'.join(output)
        
        if print_report:
            print(output_text)

        if save_report:
            with open(output_dir / f'{'_'.join(feature_names)}_CoxPH_summary.txt', 'w') as f:
                f.write(output_text)
    
    return cph, p_value

def intersectional_analysis(
        sensitive_features: pd.DataFrame,
        endpoint: pd.DataFrame,
        output_dir: Path | str,
        show_figure: bool = True, 
        tag: str = '',
        analysis_func: Callable = subgroup_analysis_OLS,
    ) -> None:
    """
    Performs intersectional analysis on `sensitive_features` and generates a p-value heatmap.

    This function evaluates pairwise interactions between sensitive features using 
    a specified analysis function (default: OLS). A heatmap of p-values is generated 
    to visualize statistical significance.

    Args:
        sensitive_features (pd.DataFrame): DataFrame containing categorical sensitive features.
        endpoint (pd.DataFrame): Target variable or survival data (depends on `analysis_func`).
        output_dir (Path | str): Directory to save the heatmap.
        show_figure (bool, optional): Whether to display the heatmap. Defaults to True.
        tag (str, optional): Optional tag for filename customization. Defaults to ''.
        analysis_func (Callable, optional): Function to compute p-values (default: `subgroup_analysis_OLS`).

    Raises:
        AssertionError: If fewer than two features are provided.

    Returns:
        None
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = sensitive_features.columns
    assert len(feature_names) > 1, "This requires more than sensitive features"

    feat_pairs = list(combinations(feature_names, 2))
    pval_matrix = pd.DataFrame(index=feature_names, columns=feature_names, dtype=float)

    for col1, col2 in feat_pairs:
        _, pval = analysis_func(
            sensitive_features=sensitive_features[[col1, col2]], 
            endpoint=endpoint, 
            output_dir=output_dir, 
            print_report=False, 
            save_report=False
        )
        pval_matrix.loc[col1, col2] = pval
        pval_matrix.loc[col2, col1] = pval  

    pval_matrix = pval_matrix.apply(pd.to_numeric)
    mask = np.triu(np.ones_like(pval_matrix, dtype=bool)) # Keep only lower triangle
    np.fill_diagonal(mask, False)

    plt.figure(figsize=(10, 8))
    sns.heatmap(pval_matrix, annot=True, cmap="coolwarm_r", fmt=".3f", linewidths=0.5, cbar=True, mask=mask)

    title_tag = f" ({tag})" if tag else ""
    filename_tag = f"_{tag}" if tag else ""

    plt.title(f"P-value Heatmap OLS{title_tag}", fontsize=14, weight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(output_dir / f"pval_heatmap_OLS{filename_tag}.png")
    
    if show_figure:
        plt.show()

