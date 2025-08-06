from typing import List, Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype


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

    # assume all non-numerical and date columns are categorical
    numeric_cols = {col for col in data.columns if is_numeric_dtype(data[col])}
    numeric_cols = {col for col in numeric_cols if data[col].dtype != bool}
    likely_cat = set(data.columns) - numeric_cols
    likely_cat = list(likely_cat - set(date_columns))
    
    # check proportion of unique values if numerical
    for var in numeric_cols:
        likely_flag = 1.0 * data[var].nunique()/data[var].count() < 0.025
        if likely_flag:
            ### DEVNOTE Aug 6 2025
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

