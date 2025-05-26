import pickle

import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field, PrivateAttr
from sklearn.model_selection import train_test_split
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.util import Surv

from jarvais.loggers import logger
from jarvais.trainer.survival import train_deepsurv, train_mtlr


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
    random_seed: int = Field(
        default=42,
        description="Random seed for reproducibility.",
        title="Random Seed"
    )     

    _predictor: dict = PrivateAttr(default_factory=dict)  

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
        output_dir: str,
    ):
        return cls(
            output_dir=output_dir,
            deep_models=["MTLR", "DeepSurv"],
            classical_models=["CoxPH", "RandomForest", "GradientBoosting", "SVM"]
        )

    def fit(
            self,
            X_train: pd.DataFrame, 
            y_train: pd.DataFrame,
        ):

        """Train both deep and traditional survival models, consolidate fitted models and C-index scores."""
        (self.output_dir / 'survival_models').mkdir(exist_ok=True, parents=True)

        self._predictor = {}
        # cindex_scores = {}

        # Deep Models

        data_train, data_val = train_test_split(
            pd.concat([X_train, y_train], axis=1), 
            test_size=0.1, 
            stratify=y_train['event'], 
            random_state=self.random_seed
        )

        if "MTLR" in self.deep_models:
            try:
                self._predictor['MTLR'] = train_mtlr(
                    data_train,
                    data_val,
                    self.output_dir / 'survival_models')
            except Exception as e:
                logger.error(f"Error training MTLR model: {e}")
        else:
            logger.info("Skipping MTLR model training.")

        if "DeepSurv" in self.deep_models:
            try:
                self._predictor['DeepSurv'] = train_deepsurv(
                    data_train,
                    data_val,
                    self.output_dir / 'survival_models')
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
                self._predictor[name] = model

                model_path = self.output_dir / 'survival_models' / f"{name}.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)

                # predictions = model.predict(X_test)
                # cindex_scores[name] = concordance_index_censored(
                #     y_test["event"], y_test["time"], predictions
                # )[0]
            except Exception as e:
                logger.error(f"Error training {name} model: {e}")

        # For later saving to yaml
        # cindex_scores = {key: float(value) for key, value in cindex_scores.items()}

        return self._predictor, data_train, data_val
