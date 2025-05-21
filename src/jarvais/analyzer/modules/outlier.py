from typing import Dict, Literal

import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr

from jarvais.loggers import logger


class OutlierModule(BaseModel):

    categorical_strategy: Dict[str, Literal['frequency']] = Field(
        description="Outlier strategy for categorical columns.",
        title="Categorical Strategy",
        examples=[{"treatment_type": "frequency"}]
    )
    continuous_strategy: Dict[str, Literal['none']] = Field(
        description="Outlier strategy for continuous columns (currently unsupported).",
        title="Continuous Strategy",
        examples=[{"age": "none"}]
    )
    threshold: float = Field(
        default=0.01,
        description="Frequency threshold below which a category is considered an outlier.",
        title="Threshold",
    )
    enabled: bool = Field(
        default=True,
        description="Whether to perform outlier analysis."
    )

    categorical_mapping: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description="Mapping from categorical column names to outlier handling details."
        "Generated after running outlier analysis. If a mapping is already provided, it will be used directly.",
        title="Categorical Outlier Mapping"
    )

    _outlier_report: str = PrivateAttr(default="")

    @classmethod
    def build(
            cls, 
            categorical_columns: list[str],
            continuous_columns: list[str] | None = None, 
        ) -> "OutlierModule":
        return cls(
            categorical_strategy={col: "frequency" for col in categorical_columns},
            continuous_strategy={col: "none" for col in continuous_columns} if continuous_columns is not None else {},
        )
    
    @property
    def report(self) -> str:
        return self._outlier_report

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            logger.warning("Outlier analysis is disabled.")
            return df

        logger.info("Performing outlier analysis...")

        df = df.copy()

        # Handle continuous outliers
        # for col, strategy in self.continuous_strategy.items(): 
        #     continue

        # Handle categorical outliers
        for col, strategy in self.categorical_strategy.items():
            if col not in df.columns or strategy != 'frequency':
                continue

            # If a mapping is already provided, use it directly
            if col in self.categorical_mapping and self.categorical_mapping[col]:
                logger.warning(f"Using provided categorical mapping for column: {col}")
                mapping = self.categorical_mapping[col]
            else:
                # Otherwise, compute the mapping based on frequency threshold
                value_counts = df[col].value_counts()
                threshold = int(len(df) * self.threshold)
                outliers = value_counts[value_counts < threshold].index

                mapping = {
                    val: ("Other" if val in outliers else val)
                    for val in value_counts.index
                }

                self.categorical_mapping[col] = dict(mapping)

                if len(outliers) > 0:
                    outliers_msg = [f'{o}: {value_counts[o]} out of {df[col].count()}' for o in outliers]
                    self._outlier_report += f'  - Outliers found in {col}: {outliers_msg}\n'
                else:
                    self._outlier_report += f'  - No Outliers found in {col}\n'

            # Apply the mapping (whether passed or computed)
            df[col] = df[col].map(mapping).astype("category")

        if self._outlier_report:
            print(f"\nOutlier Report:\n{self._outlier_report}")

        return df


if __name__ == "__main__":
    from rich import print  # noqa: A004

    outlier_module = OutlierModule(
        categorical_strategy={"treatment_type": "frequency"},
        continuous_strategy={"age": "none"}
    )

    print(outlier_module)
