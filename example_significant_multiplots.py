#!/usr/bin/env python3
"""
Example script demonstrating how to find the most statistically significant multiplots
from the JARVAIS analyzer.

This script shows two ways to use the new functionality:
1. Using the analyzer's built-in method
2. Using the standalone function directly
"""

import pandas as pd
from pathlib import Path

# Import the analyzer and utility functions
from src.jarvais.analyzer import Analyzer
from src.jarvais.utils.statistical_ranking import find_top_multiplots, summarize_significant_results
from src.jarvais.analyzer._utils import infer_types

def main():
    """Demonstrate finding most significant multiplots."""
    
    # Example dataset (you can replace this with your actual data)
    # print("Creating example dataset...")
    # np.random.seed(42)  # For reproducibility
    
    # Create synthetic medical data with varying levels of statistical significance
    # n_samples = 1000
    data = pd.read_csv("./data/RADCURE_processed_clinical.csv", index_col=0)
    # print(infer_types(data))
    # return
    # data = pd.DataFrame({
    #     # Categorical variables
    #     'sex': np.random.choice(['Male', 'Female'], n_samples),
    #     'tumor_stage': np.random.choice(['I', 'II', 'III', 'IV'], n_samples),
    #     'treatment_type': np.random.choice(['Surgery', 'Chemotherapy', 'Radiation'], n_samples),
    #     'disease_site': np.random.choice(['Lung', 'Breast', 'Prostate', 'Colon'], n_samples),
        
    #     # Continuous variables with different levels of correlation to categorical vars
    #     'age': np.random.normal(65, 15, n_samples),
    #     'tumor_size': np.random.exponential(3, n_samples),
    #     'survival_time': np.random.exponential(24, n_samples),
    #     'bmi': np.random.normal(25, 5, n_samples),
    # })
    
    # Add some realistic correlations to make certain relationships significant
    # Make tumor size correlate with stage
    # stage_multipliers = {'I': 0.5, 'II': 1.0, 'III': 1.5, 'IV': 2.0}
    # data['tumor_size'] = data.apply(
    #     lambda row: row['tumor_size'] * stage_multipliers[row['tumor_stage']], axis=1
    # )
    
    # # Make age correlate with sex (slight difference)
    # data.loc[data['sex'] == 'Male', 'age'] += 3
    
    # # Make survival time correlate with treatment
    # treatment_effects = {'Surgery': 1.2, 'Chemotherapy': 0.9, 'Radiation': 1.0}
    # data['survival_time'] = data.apply(
    #     lambda row: row['survival_time'] * treatment_effects[row['treatment_type']], axis=1
    # )
    
    # print(f"Created dataset with {len(data)} samples")
    # print(f"Categorical columns: {['sex', 'tumor_stage', 'treatment_type', 'disease_site']}")
    # print(f"Continuous columns: {['age', 'tumor_size', 'survival_time', 'bmi']}")
    
    # Set up output directory
    output_dir = Path("./example_output")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nRunning analyzer with output to: {output_dir}")
    
    # Method 1: Using the Analyzer class with built-in method
    print("\n" + "="*60)
    print("METHOD 1: Using Analyzer.get_top_multiplots()")
    print("="*60)
    
    # try:
    # Create and run analyzer
    analyzer = Analyzer(
        data=data,
        output_dir=str(output_dir),
        # categorical_columns=['sex', 'tumor_stage', 'treatment_type', 'disease_site'],
        # continuous_columns=['age', 'tumor_size', 'survival_time', 'bmi'],
        target_variable='death',  # Example target
        task='classification'
    )
    
    # Run the analyzer to generate multiplots
    analyzer.run()
    
    # print(data.Dose.dtype)
    # print(data.Dose.unique())

    # Get the most significant multiplots
    significant_results = analyzer.get_top_multiplots(n_top=10)
    
    print(f"\nFound {len(significant_results)} significant relationships:")
    print("-" * 80)
    print(f"{'Categorical':<15} {'Continuous':<15} {'P-value':<12} {'Test':<8} {'Effect':<10} {'Significant'}")
    print("-" * 80)
    
    for result in significant_results:
        significance_mark = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else ""
        print(f"{result['categorical_var']:<15} {result['continuous_var']:<15} "
                f"{result['p_value']:<12.4f} {result['test_type']:<8} "
                f"{result['effect_size']:<10.3f} {significance_mark}")
    
    # except Exception as e:
        # print(f"Error with Method 1: {e}")
        # print("This might be due to missing dependencies or environment issues.")
    
    # # Method 2: Using the standalone function
    # print("\n" + "="*60)
    # print("METHOD 2: Using standalone find_top_multiplots()")
    # print("="*60)
    
    # try:
    #     # Use the standalone function directly
    #     results = find_top_multiplots(
    #         data=data,
    #         # categorical_columns=['sex', 'tumor_stage', 'treatment_type', 'disease_site'],
    #         # continuous_columns=['age', 'tumor_size', 'survival_time', 'bmi'],
    #         output_dir=output_dir,
    #         n_top=10,
    #         significance_threshold=0.05
    #     )
        
    #     print(f"\nFound {len(results)} relationships:")
        
    #     # Create and display summary
    #     summary_df = summarize_significant_results(results, output_dir / "significance_summary.csv")
    #     print("\nSummary DataFrame:")
    #     print(summary_df.to_string(index=False))
        
    #     # Show the most significant finding
    #     if results:
    #         top_result = results[0]
    #         print(f"\nMost significant relationship:")
    #         print(f"  {top_result['categorical_var']} vs {top_result['continuous_var']}")
    #         print(f"  P-value: {top_result['p_value']:.2e}")
    #         print(f"  Test type: {top_result['test_type']}")
    #         print(f"  Effect size: {top_result['effect_size']:.3f}")
    #         print(f"  Plot location: {top_result['plot_path']}")
        
    # except Exception as e:
    #     print(f"Error with Method 2: {e}")
    #     print("This might be due to missing plot files if analyzer didn't run successfully.")
    
    # print(f"\nExample completed! Check {output_dir} for generated files.")


if __name__ == "__main__":
    import numpy as np  # Import here to avoid issues if not available
    main()