from typing import Dict, List, Tuple
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

class SubgroupAnalysis():
    """Performs subgroup analysis on sensitive features and endpoints.

    Attributes:
        sensitive_features (pd.DataFrame): DataFrame containing the sensitive features.
        endpoint (pd.Series): Series representing the endpoint.
        output_dir (Path): Directory to save generated plots and reports.
        feature_names (Index): List of feature names from the `sensitive_features` DataFrame.

    Methods:
        generate_violin(): Generates violin plots for each sensitive feature vs the bias metric.
        subgroup_analysis_OLS(print_report=True, save_report=True, feature_names=None): Fits a statsmodels OLS 
            model and prints the summary based on p-value, optionally saving it.
        intersectional_analysis(): Performs pairwise OLS regressions between sensitive features and generates a 
            p-value heatmap.
    """
    def __init__(
            self,
            sensitive_features: Dict[str, list] | pd.DataFrame,
            endpoint: Dict[str, list] | pd.Series,
            output_dir: str | Path,
        ) -> None:
        """Initializes the SubgroupAnalysis class for analysis on sensitive features and endpoint data.

        Args:
            sensitive_features (Dict[str, list] | pd.DataFrame): Sensitive feature data, either as a dictionary 
                of feature names to lists or as a pandas DataFrame containing sensitive features.
            endpoint (Dict[str, list] | pd.Series): Endpoint data, either as a dictionary of values or as a 
                pandas Series containing the endpoint measurements.
            output_dir (str | Path): Directory where the output plots and reports will be saved.
        """
        self.sensitive_features = pd.DataFrame(sensitive_features) if isinstance(sensitive_features, dict) else sensitive_features
        self.endpoint = pd.Series(endpoint) if isinstance(endpoint, dict) else endpoint
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.feature_names = self.sensitive_features.columns

    def generate_violin(self) -> None:
        """Generates violin plots for each sensitive feature vs the bias metric.

        This method creates a violin plot for each feature in the `sensitive_features` DataFrame,
        visualizing the distribution of the bias metric for each feature. The plot is saved to the output directory.
        """
        for feature_name, feature_series in self.sensitive_features.items():
            plt.figure(figsize=(8, 6)) 
            sns.set_theme(style="whitegrid")  

            sns.violinplot(
                x=feature_series, 
                y=self.endpoint, 
                palette="muted",  
                inner="quart", 
                linewidth=1.25 
            )

            plt.title(f'{self.endpoint.name} Distribution by {feature_name}', fontsize=16, weight='bold')  
            plt.xlabel(f'{feature_name}', fontsize=14)  
            plt.ylabel(f'{self.endpoint.name} per Patient', fontsize=14) 
            plt.xticks(rotation=45, ha='right')

            plt.tight_layout()
            plt.savefig(self.output_dir / f'{feature_name}_vs_{self.endpoint.name}.png')  
            plt.show()

    def generate_kaplan_meier(self) -> None:
        """Generates Kaplan-Meier plots by category.

        This method plots Kaplan-Meier survival curves for each category in the `sensitive_features` DataFrame.
        The Kaplan-Meier plots are saved in the provided `output_dir`.

        Ensures that the 'time' and 'event' columns are available in the `endpoint` DataFrame.
        """
        assert {'time', 'event'}.issubset(set(self.endpoint.index)), "The 'endpoint' must contain the 'time' and 'event'."

        plot_kaplan_meier_by_category(
            self.sensitive_features, 
            self.endpoint, 
            self.feature_names, 
            self.output_dir, 
            show_figure=True
        )   

    def subgroup_analysis_OLS(
            self, 
            print_report: bool = True, 
            save_report: bool = True,
            feature_names: List[str] | None = None,
        ) -> Tuple[sm.OLS, float]:
        """Fits a statsmodels OLS model to the bias metric data, using the sensitive feature and prints the summary based on p-value.

        This method performs an Ordinary Least Squares (OLS) regression between the bias metric and the
        specified sensitive feature(s), and prints a detailed report if the p-value is below 0.05. The report
        is also saved if the `save_report` argument is True.

        Args:
            print_report (bool): Whether to print the report to the console (default is True).
            save_report (bool): Whether to save the report to a text file (default is True).
            feature_names (List[str] | None): List of sensitive feature names to include in the analysis. If None, uses all features.

        Returns:
            sm.OLS: The fitted OLS model.
            float: The F pvalue of the OLS model.
        """
        feature_names = self.feature_names if feature_names is None else feature_names
        one_hot_encoded = pd.get_dummies(
            self.sensitive_features[feature_names], 
            prefix=feature_names
        )
        X_columns = one_hot_encoded.columns

        X = one_hot_encoded.values  
        y = self.endpoint.values  

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
                with open(self.output_dir / f'{'_'.join(feature_names)}_OLS_summary.txt', 'w') as f:
                    f.write(output_text)

        return model, p_value
    
    def subgroup_analysis_CoxPH(
            self, 
            print_report: bool = True, 
            save_report: bool = True,
            feature_names: List[str] | None = None,
        ) -> Tuple[CoxPHFitter, float]:
        """Fit a CoxPH model using the sensitive feature and print summary based on p_val.

        Args:
            print_report (bool): Whether to print the report to the console (default is True).
            save_report (bool): Whether to save the report to a text file (default is True).
            feature_names (List[str] | None): List of sensitive feature names to include in the analysis. If None, uses all features.

        Returns:
            CoxPHFitter: The fitted Cox Proportional Hazards model.
            float: The p-value of the likelihood ratio test.
        """
        assert {'time', 'event'}.issubset(set(self.endpoint.index)), "The 'endpoint' must contain the 'time' and 'event'."

        feature_names = self.feature_names if feature_names is None else feature_names

        one_hot_encoded = pd.get_dummies(
            self.sensitive_features[feature_names], 
            prefix=feature_names
        )
        df_encoded = self.endpoint.join(one_hot_encoded)

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
                with open(self.output_dir / f'{'_'.join(feature_names)}_CoxPH_summary.txt', 'w') as f:
                    f.write(output_text)
        
        return cph, p_value

    def intersectional_analysis(self, show_figure: bool = True, tag: str = '') -> None:
        """Performs an intersectional analysis on sensitive features.

        This method performs pairwise OLS regression between sensitive features and generates a p-value matrix,
        which is then visualized using a heatmap. The heatmap is saved to the output directory.
         
        Raises:
            AssertionError: If there are fewer than two sensitive features.
        """
        assert len(self.feature_names) > 1, "This requires more than sensitive features"

        if {'time', 'event'}.issubset(set(self.endpoint.index)):
            analysis_func = self.subgroup_analysis_CoxPH
        else:
            analysis_func = self.subgroup_analysis_OLS

        feat_pairs = list(combinations(self.feature_names, 2))
        pval_matrix = pd.DataFrame(index=self.feature_names, columns=self.feature_names, dtype=float)

        for col1, col2 in feat_pairs:
            _, pval = analysis_func(feature_names=[col1, col2], print_report=False, save_report=False)
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

        plt.savefig(self.output_dir / f"pval_heatmap_OLS{filename_tag}.png")
        
        if show_figure:
            plt.show()

