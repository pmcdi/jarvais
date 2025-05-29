from typing import Literal, Optional
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.feature_selection import SelectKBest, VarianceThreshold, chi2, f_classif, f_regression
from mrmr import mrmr_classif, mrmr_regression

from jarvais.loggers import logger


class FeatureReductionModule(BaseModel):
    method: Literal['mrmr', 'variance_threshold', 'corr', 'chi2', None] = Field(
        description="Feature reduction strategy to apply."
    )
    task: Literal['binary', 'multiclass', 'regression', 'survival'] = Field(
        description="Supervised learning task type."
    )
    keep_k: int = Field(
        default=10,
        description="Number of features to retain for relevant methods."
    )
    enabled: bool = Field(
        default=True,
        description="Whether to perform feature reduction."
    )

    @classmethod
    def build(cls, method: str, task: str, keep_k: int = 10) -> "FeatureReductionModule":
        return cls(method=method, task=task, keep_k=keep_k)

    def __call__(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        if not self.enabled or self.method is None:
            logger.info("Skipping feature reduction.")
            return X, y

        logger.info(f"Applying feature reduction: {self.method}")

        X = X.copy()
        y = y.copy()

        # Step 1: Ordinal encode categoricals
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        mappings = {}

        for col in categorical_columns:
            mapping = {k: i for i, k in enumerate(X[col].dropna().unique())}
            X[col] = X[col].map(mapping)
            mappings[col] = mapping

        # Step 2: Perform reduction
        if self.method == "variance_threshold":
            X_reduced = self._variance_threshold(X, y)
        elif self.method == "chi2":
            if self.task not in ['binary', 'multiclass']:
                raise ValueError("Chi2 only supports classification tasks.")
            X_reduced = self._chi2(X, y)
        elif self.method == "corr":
            X_reduced = self._kbest(X, y)
        elif self.method == "mrmr":
            X_reduced = self._mrmr(X, y)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

        # Step 3: Reverse mappings for categorical columns
        for col in categorical_columns:
            if col in X_reduced.columns:
                inverse_map = {v: k for k, v in mappings[col].items()}
                X_reduced[col] = X_reduced[col].round().astype(int).map(inverse_map).astype("category")

        logger.info(f"Feature reduction complete. Remaining features: {X_reduced.columns}")

        return X_reduced, y

    def _variance_threshold(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        selector = VarianceThreshold()
        _ = selector.fit_transform(X, y)
        return X[X.columns[selector.get_support(indices=True)]]

    def _chi2(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        selector = SelectKBest(score_func=chi2, k=self.keep_k)
        _ = selector.fit_transform(X, y)
        return X[X.columns[selector.get_support(indices=True)]]

    def _kbest(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        score_func = f_classif if self.task in ['binary', 'multiclass'] else f_regression
        selector = SelectKBest(score_func=score_func, k=self.keep_k)
        _ = selector.fit_transform(X, y)
        return X[X.columns[selector.get_support(indices=True)]]

    def _mrmr(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        mrmr_fn = mrmr_classif if self.task in ['binary', 'multiclass'] else mrmr_regression
        selected_features = mrmr_fn(X=X, y=y, K=self.keep_k, n_jobs=1)
        return X[selected_features]


   
