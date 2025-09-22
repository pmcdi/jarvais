import pandas as pd
from pydantic import Field

from jarvais.loggers import logger
from .base import AnalyzerModule
from jarvais.analyzer._utils import (
    TRUTHY_TOKENS,
    FALSY_TOKENS,
)


class BooleanEncodingModule(AnalyzerModule):
    columns: list[str] = Field(
        default_factory=list,
        description="List of boolean-like columns to encode as 1/0."
    )
    enabled: bool = Field(
        default=True,
        description="Whether to perform boolean encoding."
    )

    @classmethod
    def build(
        cls,
        boolean_columns: list[str] | None = None,
    ) -> "BooleanEncodingModule":
        return cls(columns=boolean_columns or [])

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled or not self.columns:
            return df

        df = df.copy()

        for col in self.columns:
            if col not in df.columns:
                logger.warning(f"BooleanEncodingModule: column '{col}' not found; skipping.")
                continue

            series = df[col]

            # If already numeric boolean-like, pass through
            if pd.api.types.is_numeric_dtype(series):
                encoded = series.fillna(0).astype(float)
                positive_value = 1
            else:
                # Normalize tokens
                tokens = series.astype(str).str.strip().str.lower()
                mapped = tokens.map(lambda x: 1 if x in TRUTHY_TOKENS else (0 if x in FALSY_TOKENS else None))

                # If mapping failed widely, try strict equality to the most frequent non-null token
                if mapped.isna().mean() > 0.5:
                    top = tokens[tokens != ""].mode().iloc[0] if not tokens[tokens != ""].mode().empty else "true"
                    mapped = tokens.eq(top).astype(float)
                    positive_value = top
                else:
                    # Determine positive_value for naming
                    # Prefer the most frequent truthy token present, else '1'
                    positive_value = None
                    for v in TRUTHY_TOKENS:
                        if (tokens == v).any():
                            positive_value = v
                            break
                    if positive_value is None:
                        positive_value = "1"

                encoded = mapped.fillna(0).astype(float)

            new_col_name = f"{col}_{positive_value}"
            df[new_col_name] = encoded

        return df
