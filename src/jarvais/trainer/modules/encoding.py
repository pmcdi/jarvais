import pandas as pd
from pydantic import Field, BaseModel

from jarvais.loggers import logger


class OneHotEncodingModule(BaseModel):
    columns: list[str] | None = Field(
        default=None,
        description="List of categorical columns to one-hot encode. If None, all columns are used."
    )
    prefix_sep: str = Field(
        default="|",
        description="Prefix separator used in encoded feature names."
    )
    enabled: bool = Field(
        default=True,
        description="Whether to perform one-hot encoding."
    )

    @classmethod
    def build(
        cls,
        categorical_columns: list[str] | None = None,
        prefix_sep: str = "|",
    ) -> "OneHotEncodingModule":
        return cls(
            columns=categorical_columns,
            prefix_sep=prefix_sep
        )

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            logger.warning("One-hot encoding is disabled.")
            return df

        df = df.copy()
        return pd.get_dummies(
            df,
            columns=self.columns,
            dtype=float,
            prefix_sep=self.prefix_sep
        )