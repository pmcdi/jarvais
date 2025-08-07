from typing import List, Tuple
import re

import pandas as pd
from pandas.api.types import is_numeric_dtype
from jarvais.loggers import logger

def infer_types(data: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Infer and categorize column data types in the dataset.

    Adapted from https://github.com/tompollard/tableone/blob/main/tableone/preprocessors.py

    This method analyzes the dataset to categorize columns as either
    continuous or categorical based on their data types and unique value proportions.

    Assumptions:
        - All non-numerical and non-date columns are considered categorical.
        - Boolean columns are not considered numerical but categorical.
        - Numerical columns with a unique value proportion below a threshold are
          considered categorical.

    The method also applies a heuristic to detect and classify ID columns
    as categorical if they have a low proportion of unique values.
    """
    date_columns = [
        col for col in data.select_dtypes(include=['object']).columns
        if pd.to_datetime(data[col], format='mixed', errors='coerce').notna().any()
    ]

    # Assume all non-numerical and date columns are categorical
    numeric_cols = {col for col in data.columns if is_numeric_dtype(data[col])}
    numeric_cols = {col for col in numeric_cols if data[col].dtype != bool}
    likely_cat = set(data.columns) - numeric_cols
    likely_cat = list(likely_cat - set(date_columns))
    

    # Check for columns with numeric values with auxiliary symbols (e.g. "50+", "<20", ">50")
    auxiliary_symbols_pattern = r'[<>≤≥±~+\-]+'
    for col in likely_cat.copy():  # Create a copy to avoid modifying list during iteration
        # Convert to string and filter out common string representations of missing values
        string_values = data[col].dropna().astype(str).str.lower()
        
        # Remove rows with string representations of missing values
        na_patterns = ['na', 'none', 'null', 'n/a', 'nan', 'missing', '']
        filtered_values = string_values[~string_values.isin(na_patterns)]
        
        # Remove auxiliary symbols and try to convert to numeric
        cleaned_values = filtered_values.str.replace(auxiliary_symbols_pattern, '', regex=True)
        
        # Remove empty strings that might result from cleaning
        cleaned_values = cleaned_values[cleaned_values.str.strip() != '']
        
        # Check if the cleaned values can be converted to numeric
        if len(cleaned_values) > 0 and pd.Series(cleaned_values).is_numeric_dtype():  # Only proceed if we have values left
            # If successful, this column should be treated as numeric
            likely_cat.remove(col)
            numeric_cols.add(col)
            logger.info(f"Column '{col}' detected as numeric with auxiliary symbols. Sample values: {data[col].dropna().unique()[:10]}")
        
    # Check proportion of unique values if numerical
    for var in numeric_cols:
        likely_flag = 1.0 * data[var].nunique()/data[var].count() < 0.025
        if likely_flag:
            ### Aug 6 2025 DEVNOTE:
            # We are deciding to NOT auto-switch to categorical to avoid false positives. 
            # The user MUST manually move the column to the categorical_columns list in `analyzer_settings.json` if it should be considered categorical.
            logger.warning(f"ATTN: Column {var} is potentially categorical because it has a low proportion of unique values.\n"
                           f"If the variable should be considered categorical, move it to the categorical_columns list in `analyzer_settings.json`.")

    # Heuristic targeted at detecting ID columns
    categorical_columns = []
    for cat_var in likely_cat:
        if data[cat_var].nunique()/data[cat_var].count() < 0.2:
            categorical_columns.append(cat_var)
        else:
            logger.warning(f"ATTN: Column {cat_var} is not considered categorical because it has a high proportion of unique values.\n"
                           f"This variable is likely an ID column.")

    continuous_columns = list(set(data.columns) - set(likely_cat) - set(date_columns))

    return categorical_columns, continuous_columns, date_columns

