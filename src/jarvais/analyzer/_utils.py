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
            likely_cat.append(var)

    # Heuristic targeted at detecting ID columns
    likely_cat = [cat for cat in likely_cat if data[cat].nunique()/data[cat].count() < 0.2]

    categorical_columns = likely_cat
    continuous_columns = list(set(data.columns) - set(likely_cat) - set(date_columns))

    return categorical_columns, continuous_columns, date_columns

