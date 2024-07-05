import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

from scipy.stats import pearsonr, spearmanr
import numpy as np

"""

CONTINOUS VARIABLES

"""

# Replace plots with seaborn.pairplot add correlation values to plots
# Save all to one big markdown and figures to folder
# Detect and drop continous/discrete
# Have one big function that has a param to decide cont vs disc


def analyze_continuous_variables(data: pd.DataFrame, target_variable: str):
    """
    Function to analyze continuous variables.

    - Performs descriptive statistics
    - Creates scatter plots, violin plots
    - Calculates Pearson and Spearman correlation
    - (Optional) Creates Kaplan Meier curves (if data has survival information)

    Parameters:
    - data (DataFrame): The input data containing continuous variables.
    - target_variable (str): The name of the target variable in the data.
    """

    # Descriptive statistics
    print(data.describe(include='all'))

    # Scatter plots
    _scatter_plots(data.copy(), target_variable)

    # Violin plots
    # _violin_plots(data.copy(), target_variable)

    # Correlation analysis
    _pearson_correlation(data.copy(), target_variable)
    _spearman_correlation(data.copy(), target_variable)

    # Check if data has event (death) and time columns for Kaplan Meier
    if 'event' in data.columns and 'time' in data.columns:
        _kaplan_meier_curves(data.copy(), target_variable)


def _scatter_plots(data: pd.DataFrame, target_variable: str):
    """
    Function to create scatter plots of each continuous variable vs target variable.

    Parameters:
    - data (DataFrame): The input data containing continuous variables.
    - target_variable (str): The name of the target variable in the data.
    """

    for col in data.columns:
        if col != target_variable:
            plt.scatter(data[col], data[target_variable])
            plt.xlabel(col)
            plt.ylabel(target_variable)
            plt.title(f"Scatter Plot: {col} vs {target_variable}")
            plt.show()
            plt.close()


def _violin_plots(data: pd.DataFrame, target_variable: str):
    """
    Function to create violin plots of each continuous variable vs target variable.

    Parameters:
    - data (DataFrame): The input data containing continuous variables.
    - target_variable (str): The name of the target variable in the data.
    """

    for col in data.columns:
        if col != target_variable:
            sns.violinplot(x=col, y=target_variable, showmeans=True, data=data)
            plt.title(f"Violin Plot: {col} vs {target_variable}")
            plt.show()
            plt.close()  # Clear plot for next iteration


def _pearson_correlation(data: pd.DataFrame, target_variable: str):
    """
    Function to calculate Pearson correlation between each continuous variable and target.

    Parameters:
    - data (DataFrame): The input data containing continuous variables.
    - target_variable (str): The name of the target variable in the data.
    """

    correlation_matrix = data.corr('pearson')

    # Plotting the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Pearson Correlation of Performance Metrics with Acceptability')
    plt.show()

def _spearman_correlation(data: pd.DataFrame, target_variable: str):
    """
    Function to calculate Spearman correlation between each continuous variable and target.

    Parameters:
    - data (DataFrame): The input data containing continuous variables.
    - target_variable (str): The name of the target variable in the data.
    """

    correlation_matrix = data.corr('spearman')

    # Plotting the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Spearman Correlation of Performance Metrics with Acceptability')
    plt.show()

def _kaplan_meier_curves(data: pd.DataFrame, target_variable: str):
    """
    Function to produce Kaplan Meier curves.
    
    Parameters:
    - data (DataFrame): The input data containing continuous variables.
    - target_variable (str): The name of the target variable in the data.
    """

    raise NotImplementedError

"""

DISCRETE VARIABLES

"""

def analyze_discrete_variables(data: pd.DataFrame, target_variable: str):
    """
    Function to analyze discrete variables.
    
    Parameters:
    - data (DataFrame): The input data containing discrete variables.
    - target_variable (str): The name of the target variable in the data.
    """

    raise NotImplementedError

def _histogram_plots(data: pd.DataFrame, target_variable: str):
    """
    Function to create histogram plots.
    
    Parameters:
    - data (DataFrame): The input data containing discrete variables.
    - target_variable (str): The name of the target variable in the data.
    """

    raise NotImplementedError