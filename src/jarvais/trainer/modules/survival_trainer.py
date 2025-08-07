import json
import pickle
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from sksurv.ensemble import (
    GradientBoostingSurvivalAnalysis,
    RandomSurvivalForest,
)
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.svm import FastSurvivalSVM
from sksurv.util import Surv
from tabulate import tabulate

from jarvais.loggers import logger

from .deep_survival import LitDeepSurv, LitMTLR, train_deepsurv, train_mtlr


MODELType: TypeAlias = CoxnetSurvivalAnalysis | RandomSurvivalForest | GradientBoostingSurvivalAnalysis | FastSurvivalSVM | LitDeepSurv | LitMTLR


class SurvivalPredictor:

    def __init__(
            self,
            models: dict[str, MODELType],
            model_scores: dict[str, float],
            best_model: str
        ) -> None:
            self.models = models
            self.model_scores = model_scores
            self.best_model = best_model
        
    def predict(self, X: pd.DataFrame, model: str | None = None) -> np.ndarray:

        if model:
            return self.models[model].predict(X)

        return self.models[self.best_model].predict(X)
    
    def model_names(self) -> list[str]:
        return list(self.models.keys())
    
    @classmethod
    def load(cls, model_dir: str | Path) -> "SurvivalPredictor":

        model_dir = Path(model_dir)

        with (model_dir / 'survival_metadata.json').open('r') as f:
                model_info = json.load(f)
            
        models: dict[str, MODELType] = {}
        for model_name in model_info["model_scores"]:
            if model_name == 'MTLR':
                models[model_name] = LitMTLR.load_from_checkpoint(model_dir / "MTLR.ckpt")
            elif model_name == 'DeepSurv':
                models[model_name] = LitDeepSurv.load_from_checkpoint(model_dir / "DeepSurv.ckpt")
            else:
                with (model_dir / f'{model_name}.pkl').open("rb") as f:
                    models[model_name] = pickle.load(f)

        return cls(
            models, 
            model_info["model_scores"], 
            model_info["best_model"]
        )


class SurvivalTrainerModule(BaseModel):
    output_dir: Path = Field(
        description="Output directory.",
        title="Output Directory",
        examples=["output"]
    )
    classical_models: list[str] = Field(
        description="List of classical machine learning models to train.",
        title="Classical Machine Learning Models",
        examples=["CoxPH", "RandomForest", "GradientBoosting", "SVM"]
    )
    deep_models: list[str] = Field(
        description="List of deep learning models to train.",
        title="Deep Learning Models",
        examples=["MTLR", "DeepSurv"]
    )
    eval_metric: Literal["c_index"] = Field(
        default="c_index",
        description="Evaluation metric.",
        title="Evaluation Metric"
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for reproducibility.",
        title="Random Seed"
    )     

    @classmethod
    def validate_classical_models(cls, models: list[str]) -> list[str]:
        model_registry = ["CoxPH", "RandomForest", "GradientBoosting", "SVM"]
        invalid = [m for m in models if m not in model_registry]
        if invalid:
            msg = f"Invalid models: {invalid}. Available: {model_registry}"
            logger.error(msg)
            raise ValueError(msg)
        return models
    
    @classmethod
    def validate_deep_models(cls, models: list[str]) -> list[str]:
        model_registry = ["MTLR", "DeepSurv"]
        invalid = [m for m in models if m not in model_registry]
        if invalid:
            msg = f"Invalid models: {invalid}. Available: {model_registry}"
            logger.error(msg)
            raise ValueError(msg)
        return models
    
    @classmethod
    def build(
        cls,
        output_dir: str | Path,
    ) -> "SurvivalTrainerModule":
        return cls(
            output_dir=Path(output_dir),
            deep_models=["MTLR", "DeepSurv"],
            classical_models=["CoxPH", "RandomForest", "GradientBoosting", "SVM"]
        )

    def fit(
            self,
            X_train: pd.DataFrame, 
            y_train: pd.DataFrame,
            X_test: pd.DataFrame,
            y_test: pd.DataFrame
        ) -> tuple[SurvivalPredictor, pd.DataFrame, pd.DataFrame]:

        """Train both deep and traditional survival models, consolidate fitted models and C-index scores."""
        (self.output_dir / 'survival_models').mkdir(exist_ok=True, parents=True)

        trained_models = {}

        # Deep Models

        data_train, data_val = train_test_split(
            pd.concat([X_train, y_train], axis=1), 
            test_size=0.1, 
            stratify=y_train['event'], 
            random_state=self.random_seed
        )

        if "MTLR" in self.deep_models:
            try:
                trained_models['MTLR'] = train_mtlr(
                    data_train,
                    data_val,
                    self.output_dir / 'survival_models',
                    self.random_seed)
            except Exception as e:
                logger.error(f"Error training MTLR model: {e}")
        else:
            logger.info("Skipping MTLR model training.")

        if "DeepSurv" in self.deep_models:
            try:
                trained_models['DeepSurv'] = train_deepsurv(
                    data_train,
                    data_val,
                    self.output_dir / 'survival_models',
                    self.random_seed)
            except Exception as e:
                logger.error(f"Error training DeepSurv model: {e}")
        else:
            logger.info("Skipping DeepSurv model training.")

        # Basic Models

        models = {
            "CoxPH": CoxnetSurvivalAnalysis(fit_baseline_model=True),
            "GradientBoosting": GradientBoostingSurvivalAnalysis(),
            "RandomForest": RandomSurvivalForest(n_estimators=100, random_state=self.random_seed),
            "SVM": FastSurvivalSVM(max_iter=1000, tol=1e-5, random_state=self.random_seed),
        }

        y_train_surv = Surv.from_dataframe('event', 'time', y_train)
        for name, model in models.items():
            if name not in self.classical_models:
                logger.info(f"Skipping {name} model training.")
                continue
            
            try:
                logger.info(f"Training {name} model...")
                model.fit(X_train.astype(float), y_train_surv)
                trained_models[name] = model

                model_path = self.output_dir / 'survival_models' / f"{name}.pkl"
                with model_path.open("wb") as f:
                    pickle.dump(model, f)

            except Exception as e:
                logger.error(f"Error training {name} model: {e}")

        X_val = data_val.drop(["time", "event"], axis=1)
        y_val = data_val[["time", "event"]]

        predictor = self._evaluate(trained_models, X_train, y_train, X_val, y_val, X_test, y_test)
        
        survival_metadata = {
            "model_scores": predictor.model_scores,
            "best_model": predictor.best_model
        }
        with (self.output_dir / "survival_models" / "survival_metadata.json").open("w") as f:
            json.dump(survival_metadata, f)

        return predictor, X_val, y_val
    
    def _evaluate(
            self, 
            trained_models: dict[str, MODELType], 
            X_train: pd.DataFrame, 
            y_train: pd.DataFrame, 
            X_val: pd.DataFrame, 
            y_val: pd.DataFrame, 
            X_test: pd.DataFrame, 
            y_test: pd.DataFrame
        ) -> SurvivalPredictor:

        leaderboard = []
        test_scores = {}
        for model in self.classical_models:  
            trained_model = trained_models.get(model)
            if not trained_model:
                continue

            train_score = concordance_index_censored( # Classical models don't have a validation set
                pd.concat([y_train['event'], y_val['event']], axis=0).astype(bool),
                pd.concat([y_train['time'], y_val['time']], axis=0),
                trained_models[model].predict(pd.concat([X_train, X_val], axis=0))
            )[0]
            test_score = concordance_index_censored(
                y_test['event'].astype(bool),
                y_test['time'], 
                trained_models[model].predict(X_test)
            )[0]
            leaderboard.append({
                'model': model,
                'test_score': f"{self.eval_metric.upper()}: {round(test_score, 3)}",
                'val_score': 'N/A', # no validation for classical models
                'train_score': f"{self.eval_metric.upper()}: {round(train_score, 3)}",
            })

            test_scores[model] = test_score

        for model in self.deep_models:
            trained_model = trained_models.get(model)
            if not trained_model:
                continue

            train_score = concordance_index_censored(
                y_train['event'].astype(bool), 
                y_train['time'], 
                trained_model.predict(X_train)
            )[0]
            val_score = concordance_index_censored(
                y_val['event'].astype(bool), 
                y_val['time'], 
                trained_model.predict(X_val)
            )[0]
            test_score = concordance_index_censored(
                y_test['event'].astype(bool), 
                y_test['time'], 
                trained_model.predict(X_test)
            )[0]
            leaderboard.append({
                'model': model,
                'test_score': f"{self.eval_metric.upper()}: {round(test_score, 3)}",
                'val_score': f"{self.eval_metric.upper()}: {round(val_score, 3)}",
                'train_score': f"{self.eval_metric.upper()}: {round(train_score, 3)}",
            })

            test_scores[model] = test_score

        leaderboard_df = pd.DataFrame(leaderboard).sort_values(by='test_score', ascending=False)
        leaderboard_df.to_csv(self.output_dir / 'leaderboard.csv', index=False)

        print('\nModel Leaderboard\n----------------') # noqa: T201
        print(tabulate( # noqa: T201
            leaderboard_df,
            tablefmt = "grid",
            headers="keys",
            showindex=False))
        
        return SurvivalPredictor(
                models=trained_models, 
                model_scores=test_scores,
                best_model=max(test_scores, key=test_scores.get) # type: ignore
            )
