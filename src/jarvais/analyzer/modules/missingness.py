import pandas as pd
from sklearn.impute import KNNImputer
from pydantic import BaseModel, Field
from typing import Dict, Literal

from jarvais.loggers import logger


class MissingnessModule(BaseModel):

    categorical_strategy: Dict[str, Literal['unknown', 'knn', 'mode']] = Field(
        description="Missingness strategy for categorical columns.",
        title="Categorical Strategy",
        examples={"gender": "Unknown", "treatment_type": "knn", "tumor_stage": "mode"}
    )
    continuous_strategy: Dict[str, Literal['mean', 'median', 'mode']] = Field(
        description="Missingness strategy for continuous columns.",
        title="Continuous Strategy",
        examples={"age": "median", "tumor_size": "mean", "survival_rate": "median"}
    )
    enabled: bool = Field(
        default=True,
        description="Whether to perform missingness analysis."
    )

    @classmethod
    def build(
            cls, 
            continuous_columns: list[str], 
            categorical_columns: list[str],
        ) -> "MissingnessModule":
        return cls(
            continuous_strategy={col: 'median' for col in continuous_columns},
            categorical_strategy={col: 'unknown' for col in categorical_columns}
        )

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            logger.warning("Missingness analysis is disabled.")
            return df
        
        logger.info("Performing missingness analysis...")
        
        df = df.copy()

        # Handle continuous columns
        for col, strategy in self.continuous_strategy.items():
            if col not in df.columns:
                continue
            if strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "mode":
                df[col] = df[col].fillna(df[col].mode().iloc[0])
            else:
                raise ValueError(f"Unsupported strategy for continuous column: {strategy}")

        # Handle categorical columns
        for col, strategy in self.categorical_strategy.items():
            if col not in df.columns:
                continue
            if strategy == "unknown":
                df[col] = df[col].astype(str).fillna("Unknown").astype("category")
            elif strategy == "mode":
                df[col] = df[col].fillna(df[col].mode().iloc[0])
            elif strategy == "knn":
                df = self._knn_impute(df, col)
            else:
                df[col] = df[col].fillna(strategy)

        return df

    def _knn_impute(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        df = df.copy()
        df_encoded = df.copy()

        # Encode categorical columns for KNN
        cat_cols = df_encoded.select_dtypes(include="category").columns
        encoders = {col: {k: v for v, k in enumerate(df_encoded[col].dropna().unique())} for col in cat_cols}
        for col in cat_cols:
            df_encoded[col] = df_encoded[col].map(encoders[col])

        df_imputed = pd.DataFrame(
            KNNImputer(n_neighbors=3).fit_transform(df_encoded),
            columns=df.columns,
            index=df.index
        )

        # Decode imputed categorical column
        if target_col in encoders:
            inverse = {v: k for k, v in encoders[target_col].items()}
            df[target_col] = (
                df_imputed[target_col]
                .round()
                .astype(int)
                .map(inverse)
                .astype("category")
            )
        else:
            df[target_col] = df_imputed[target_col]

        return df

if __name__ == "__main__":
    from rich import print
    
    missingness = MissingnessModule(
        continuous_strategy = {
            'age': 'median',  
            'tumor_size': 'mean',  
            'survival_rate': 'median',  
        },
        categorical_strategy = {
            'gender': 'unknown',  
            'treatment_type': 'knn', 
            'tumor_stage': 'mode', 
        }
    )

    print(missingness)

    missingness = MissingnessModule.build(['age'], ['gender'])

    print(missingness)
