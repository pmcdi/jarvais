from typing import List, Tuple
import re

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype
from jarvais.loggers import logger

# Shared truthy/falsy tokens and regex for boolean-like detection
TRUTHY_TOKENS: set[str] = {"true", "t", "yes", "y", "1", "on", "positive", "pos"}
FALSY_TOKENS: set[str] = {"false", "f", "no", "n", "0", "off", "negative", "neg"}
TRUTHY_REGEX = re.compile(rf'^({"|".join(map(re.escape, TRUTHY_TOKENS))})$', re.IGNORECASE)
FALSY_REGEX = re.compile(rf'^({"|".join(map(re.escape, FALSY_TOKENS))})$', re.IGNORECASE)

def infer_types(data: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Infer and categorize column data types in the dataset.

    Adapted from https://github.com/tompollard/tableone/blob/main/tableone/preprocessors.py

    This method analyzes the dataset to categorize columns as either
    continuous, categorical, boolean, or date based on their data types and unique value proportions.

    Assumptions:
        - All non-numerical and non-date columns are considered categorical.
        - Boolean columns are detected and returned separately.
        - Numerical columns with a unique value proportion below a threshold are
          considered categorical.

    The method also applies a heuristic to detect and classify ID columns
    as categorical if they have a low proportion of unique values.
    
    Returns:
        Tuple[List[str], List[str], List[str], List[str]]: 
        (categorical_columns, continuous_columns, date_columns, boolean_columns)
    """
    # One-pass classification over all columns
    categorical_columns: List[str] = []
    continuous_columns: List[str] = []
    date_columns: List[str] = []
    boolean_columns: List[str] = []

    # Helpers and patterns
    na_tokens = {'na', 'none', 'null', 'n/a', 'nan', 'missing', ''}
    auxiliary_symbols_pattern = r'[<>≤≥±~+\-]+'
    name_hint_re = re.compile(r'(?:^is_|^has_|^was_|^can_|^should_|^did_|^do_|^does_|^will_|_flag$|flag$|indicator$|binary$)', re.IGNORECASE)

    for col in data.columns:
        series = data[col]

        # 1) Date detection
        if pd.api.types.is_datetime64_any_dtype(series) or \
            (series.dtype == object and \
                pd.to_datetime(series, format='mixed', errors='coerce').notna().any()):
            date_columns.append(col)
            continue

        # 2) Native boolean dtype
        elif is_bool_dtype(series):
            boolean_columns.append(col)
            continue

        # 3) Native numeric dtype
        elif is_numeric_dtype(series):
            continuous_columns.append(col)
            ratio_unique = (series.nunique() / max(series.count(), 1)) if series.count() else 0
            if ratio_unique < 0.025:
                logger.warning(
                    f"ATTN: Column {col} is potentially categorical because it has a low proportion of unique values.\n"
                    f"If the variable should be considered categorical, move it to the categorical_columns list in `analyzer_settings.json`."
                )
            continue

        # 4) Start cleaning/coercing text/object columns
        non_null = series.dropna().astype(str).str.strip()      
        lower = non_null.str.lower()
        lower = lower[~lower.isin(na_tokens)]

        # 4a) Remove auxiliary numeric symbols and coerce to numeric
        cleaned = lower.str.replace(auxiliary_symbols_pattern, '', regex=True)
        cleaned = cleaned[cleaned.str.strip() != '']
        if not cleaned.empty:
            coerced = pd.to_numeric(cleaned, errors='coerce')
            if coerced.notna().all():
                unique_vals = set(coerced.unique().tolist())
                if unique_vals and unique_vals.issubset({0, 1}):
                    boolean_columns.append(col)
                else:
                    continuous_columns.append(col)
                    ratio_unique = (coerced.nunique() / max(coerced.count(), 1)) if coerced.count() else 0
                    if ratio_unique < 0.025:
                        logger.warning(
                            f"ATTN: Column {col} is potentially categorical because it has a low proportion of unique values.\n"
                            f"If the variable should be considered categorical, move it to the categorical_columns list in `analyzer_settings.json`."
                        )
                continue

        # 4b) Truthy/Falsy token mapping
        mapped = lower.map(lambda x: True if TRUTHY_REGEX.match(x) else (False if FALSY_REGEX.match(x) else None))
        coverage = mapped.notna().mean() if len(mapped) else 0.0
        threshold = 0.95 if not name_hint_re.search(col) else 0.85
        if coverage >= threshold:
            boolean_columns.append(col)
            continue

        # 4c) Treat as categorical with ID heuristic
        ratio_unique = (lower.nunique() / max(lower.count(), 1)) if lower.count() else 0
        if ratio_unique < 0.2:
            categorical_columns.append(col)
        else:
            logger.warning(
                f"ATTN: Column {col} is not considered categorical because it has a high proportion of unique values.\n"
                f"This variable is likely an ID column."
            )

    return categorical_columns, continuous_columns, date_columns, boolean_columns

