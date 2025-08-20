"""
Statistical ranking utilities for analyzer results.
"""

from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, ttest_ind

from jarvais.loggers import logger


def find_top_multiplots(
    data: pd.DataFrame,
    categorical_columns: List[str],
    continuous_columns: List[str],
    output_dir: Union[str, Path],
    n_top: int = 10,
    significance_threshold: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Find the most statistically significant multiplots from analyzer results.
    
    This function calculates statistical significance for each categorical-continuous
    variable combination and returns the top N most significant ones.
    
    Args:
        data (pd.DataFrame): The dataset used for analysis
        categorical_columns (List[str]): List of categorical column names
        continuous_columns (List[str]): List of continuous column names
        output_dir (Union[str, Path]): Output directory where multiplots are saved
        n_top (int): Number of top significant plots to return (default: 10)
        significance_threshold (float): P-value threshold for significance (default: 0.05)
    
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing:
            - 'categorical_var': Name of categorical variable
            - 'continuous_var': Name of continuous variable
            - 'p_value': Statistical significance p-value
            - 'test_type': Type of statistical test used ('t-test' or 'anova')
            - 'effect_size': Cohen's d for t-test or eta-squared for ANOVA
            - 'plot_path': Path to the corresponding multiplot image
            - 'significant': Boolean indicating if p < significance_threshold
    
    Example:
        ```python
        from jarvais.utils.statistical_ranking import find_top_multiplots
        import pandas as pd
        
        # Assuming you have analyzer results
        results = find_top_multiplots(
            data=df,
            categorical_columns=['sex', 'tumor_stage'],
            continuous_columns=['age', 'tumor_size'],
            output_dir='./output',
            n_top=10
        )
        
        for result in results:
            print(f"{result['categorical_var']} vs {result['continuous_var']}: "
                  f"p={result['p_value']:.4f}")
        ```
    """
    output_dir = Path(output_dir)
    multiplots_dir = output_dir / "figures" / "multiplots"
    
    if not multiplots_dir.exists():
        logger.warning(f"Multiplots directory not found: {multiplots_dir}")
        return []
    
    logger.info(f"Analyzing statistical significance for {len(categorical_columns)} "
                f"categorical × {len(continuous_columns)} continuous variables")
    
    all_results = []
    
    for cat_var in categorical_columns:
        # Check if the categorical variable exists in the data
        if cat_var not in data.columns:
            logger.warning(f"Categorical variable '{cat_var}' not found in data columns. Available columns: {list(data.columns)}")
            continue
            
        # Check if the multiplot file exists
        plot_path = multiplots_dir / f"{cat_var}_multiplots.png"
        
        if not plot_path.exists():
            logger.warning(f"Plot not found for {cat_var}: {plot_path}")
            continue
            
        # Get unique values for the categorical variable
        unique_values = data[cat_var].dropna().unique()
        
        if len(unique_values) < 2:
            logger.warning(f"Insufficient groups for {cat_var} (only {len(unique_values)} unique values)")
            continue
            
        for cont_var in continuous_columns:
            # Check if the continuous variable exists in the data
            if cont_var not in data.columns:
                logger.warning(f"Continuous variable '{cont_var}' not found in data columns. Available columns: {list(data.columns)}")
                continue
                
            try:
                # Calculate statistical significance
                result = _calculate_significance(data, cat_var, cont_var, unique_values)
                
                if result is not None:
                    result.update({
                        'categorical_var': cat_var,
                        'continuous_var': cont_var,
                        'plot_path': str(plot_path),
                        'significant': result['p_value'] < significance_threshold
                    })
                    all_results.append(result)
                    
            except Exception as e:
                logger.warning(f"Error calculating significance for {cat_var} vs {cont_var}: {e}")
                continue
    
    # Sort by p-value (most significant first)
    all_results.sort(key=lambda x: x['p_value'])
    
    # Log summary statistics
    significant_count = sum(1 for r in all_results if r['significant'])
    logger.info(f"Found {len(all_results)} total comparisons, "
                f"{significant_count} significant (p < {significance_threshold})")
    
    if len(all_results) > 0:
        min_p = min(r['p_value'] for r in all_results)
        logger.info(f"Most significant p-value: {min_p:.2e}")
    
    return all_results[:n_top]


def _calculate_significance(
    data: pd.DataFrame, 
    cat_var: str, 
    cont_var: str, 
    unique_values: np.ndarray
) -> Dict[str, Any]:
    """
    Calculate statistical significance between a categorical and continuous variable.
    
    Args:
        data (pd.DataFrame): The dataset
        cat_var (str): Name of categorical variable
        cont_var (str): Name of continuous variable
        unique_values (np.ndarray): Unique values of the categorical variable
    
    Returns:
        Dict[str, Any]: Dictionary with p_value, test_type, and effect_size
    """
    # Remove missing values
    clean_data = data[[cat_var, cont_var]].dropna()
    
    if len(clean_data) < 3:  # Need at least 3 observations
        return None
        
    if len(unique_values) == 2:
        # Two groups: use t-test
        group1 = clean_data[clean_data[cat_var] == unique_values[0]][cont_var]
        group2 = clean_data[clean_data[cat_var] == unique_values[1]][cont_var]
        
        if len(group1) < 2 or len(group2) < 2:
            return None
            
        # Welch's t-test (unequal variances)
        _, p_value = ttest_ind(group1, group2, equal_var=False)
        
        # Calculate Cohen's d (effect size)
        pooled_std = np.sqrt(((len(group1) - 1) * group1.var() + 
                             (len(group2) - 1) * group2.var()) / 
                            (len(group1) + len(group2) - 2))
        effect_size = abs(group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0
        
        return {
            'p_value': p_value,
            'test_type': 't-test',
            'effect_size': effect_size
        }
    
    elif len(unique_values) > 2:
        # Multiple groups: use ANOVA
        groups = [clean_data[clean_data[cat_var] == value][cont_var] 
                 for value in unique_values]
        
        # Filter out empty groups
        groups = [group for group in groups if len(group) >= 2]
        
        if len(groups) < 2:
            return None
            
        # One-way ANOVA
        _, p_value = f_oneway(*groups)
        
        # Calculate eta-squared (effect size for ANOVA)
        # SS_between / SS_total
        overall_mean = clean_data[cont_var].mean()
        ss_between = sum(len(group) * (group.mean() - overall_mean)**2 for group in groups)
        ss_total = ((clean_data[cont_var] - overall_mean)**2).sum()
        
        effect_size = ss_between / ss_total if ss_total > 0 else 0
        
        return {
            'p_value': p_value,
            'test_type': 'anova',
            'effect_size': effect_size
        }
    
    return None


def summarize_significant_results(
    results: List[Dict[str, Any]], 
    output_file: Union[str, Path] = None
) -> pd.DataFrame:
    """
    Create a summary DataFrame of significant multiplot results.
    
    Args:
        results (List[Dict[str, Any]]): Results from find_most_significant_multiplots
        output_file (Union[str, Path], optional): Path to save summary CSV
    
    Returns:
        pd.DataFrame: Summary table with statistical results
    """
    if not results:
        logger.warning("No results to summarize")
        return pd.DataFrame()
    
    summary_df = pd.DataFrame(results)
    
    # Add interpretation columns
    summary_df['significance_level'] = summary_df['p_value'].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    )
    
    summary_df['effect_interpretation'] = summary_df.apply(
        lambda row: _interpret_effect_size(row['effect_size'], row['test_type']), axis=1
    )
    
    # Reorder columns for better readability
    columns_order = [
        'categorical_var', 'continuous_var', 'p_value', 'significance_level',
        'test_type', 'effect_size', 'effect_interpretation', 'significant', 'plot_path'
    ]
    summary_df = summary_df[columns_order]
    
    if output_file:
        summary_df.to_csv(output_file, index=False)
        logger.info(f"Summary saved to {output_file}")
    
    return summary_df


def _interpret_effect_size(effect_size: float, test_type: str) -> str:
    """
    Interpret effect size magnitude based on common conventions.
    
    Args:
        effect_size (float): The calculated effect size
        test_type (str): Type of test ('t-test' or 'anova')
    
    Returns:
        str: Interpretation of effect size magnitude
    """
    if test_type == 't-test':
        # Cohen's d interpretation
        if effect_size < 0.2:
            return 'negligible'
        elif effect_size < 0.5:
            return 'small'
        elif effect_size < 0.8:
            return 'medium'
        else:
            return 'large'
    
    else:  # ANOVA (eta-squared)
        if effect_size < 0.01:
            return 'negligible'
        elif effect_size < 0.06:
            return 'small'
        elif effect_size < 0.14:
            return 'medium'
        else:
            return 'large'