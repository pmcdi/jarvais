import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Tuple

import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import KFold, train_test_split
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.svm import FastSurvivalSVM
from sksurv.util import Surv

from ._leaderboard import aggregate_folds, format_leaderboard
from .survival import train_deepsurv, train_mtlr

def train_autogluon_with_cv(
        data_train: pd.DataFrame, data_test: pd.DataFrame,
        target_variable: str, task: str,
        output_dir: Path, extra_metrics: list,
        eval_metric: str='accuracy', num_folds: int=5, **kwargs: dict):
    """
    Trains a TabularPredictor using manual cross-validation without bagging and consolidates the leaderboards.

    Parameters
    ----------
    - data_train (DataFrame): Combined training data (features + target).
    - data_test (DataFrame): Combined test data (features + target).
    - target_variable (str): Name of the target column.
    - task (str): Problem type (e.g., 'binary', 'multiclass', 'regression').
    - output_dir (Path): Directory to save model files.
    - eval_metric (str): Evaluation metric to optimize (default: 'accuracy').
    - num_folds (int): Number of cross-validation folds (default: 5).
    - kwargs (dict): Additional arguments to pass to TabularPredictor's fit method.

    Returns
    -------
    - predictors: A list of trained predictors (one per fold).
    - final_leaderboard: A single DataFrame containing all models across folds.
    - best_fold: The index of best training fold
    - X_val: The validation features 
    - y_val: The validation target
    """
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    predictors, cv_scores, val_indices = [], [], []
    train_leaderboards, val_leaderboards, test_leaderboards = [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(data_train)):
        print(f"Training fold {fold + 1}/{num_folds}...")

        train_data, val_data = data_train.iloc[train_idx], data_train.iloc[val_idx]
        val_indices.append(val_idx)

        predictor = TabularPredictor(
            label=target_variable, problem_type=task, eval_metric=eval_metric,
            path=os.path.join(output_dir, f'autogluon_models_fold_{fold + 1}'),
            verbosity=0, log_to_file=False,
        ).fit(
            train_data,
            tuning_data=val_data,
            **kwargs)

        score = predictor.evaluate(val_data)[eval_metric]
        print(f"Fold {fold + 1} score: {score}")

        predictors.append(predictor)
        cv_scores.append(score)

        train_leaderboards.append(predictor.leaderboard(train_data, extra_metrics=extra_metrics))
        val_leaderboards.append(predictor.leaderboard(val_data, extra_metrics=extra_metrics))
        test_leaderboards.append(predictor.leaderboard(data_test, extra_metrics=extra_metrics))

    train_leaderboard = aggregate_folds(pd.concat(train_leaderboards, ignore_index=True), extra_metrics)
    val_leaderboard = aggregate_folds(pd.concat(val_leaderboards, ignore_index=True), extra_metrics)
    test_leaderboard = aggregate_folds(pd.concat(test_leaderboards, ignore_index=True), extra_metrics)

    final_leaderboard = pd.merge(
        pd.merge(
            format_leaderboard(train_leaderboard, extra_metrics, 'score_train'),
            format_leaderboard(val_leaderboard, extra_metrics, 'score_val'),
            on='model'
        ),
        format_leaderboard(test_leaderboard, extra_metrics, 'score_test'),
        on='model'
    )

    best_fold = cv_scores.index(max(cv_scores))
    val_indices_best = val_indices[best_fold]
    X_val = data_train.iloc[val_indices_best].drop(columns=target_variable)
    y_val = data_train.iloc[val_indices_best][target_variable]

    shutil.copytree(os.path.join(output_dir, f'autogluon_models_fold_{best_fold + 1}'),
                    os.path.join(output_dir, 'autogluon_models_best_fold'), dirs_exist_ok=True)

    return predictors, final_leaderboard, best_fold, X_val, y_val

def train_survival_models(
        X_train: pd.DataFrame, 
        y_train: pd.DataFrame,
        X_test: pd.DataFrame, 
        y_test: pd.DataFrame, 
        output_dir: Path
    ):
    """Train both deep and traditional survival models, consolidate fitted models and C-index scores."""
    (output_dir / 'survival_models').mkdir(exist_ok=True, parents=True)

    fitted_models = {}
    cindex_scores = {}

    # Deep Models

    data_train, data_val = train_test_split(pd.concat([X_train, y_train], axis=1), test_size=0.1, stratify=y_train['event'], random_state=42)

    fitted_models['MTLR'], cindex_scores['MTLR'] = train_mtlr(
        data_train,
        data_val,
        pd.concat([X_test, y_test], axis=1),
        output_dir / 'survival_models')
    fitted_models['DeepSurv'], cindex_scores['DeepSurv'] = train_deepsurv(
        data_train,
        data_val,
        pd.concat([X_test, y_test], axis=1),
        output_dir / 'survival_models')

    # Basic Models
    y_train = Surv.from_dataframe('event', 'time', y_train)
    y_test = Surv.from_dataframe('event', 'time', y_test)

    models = {
        "CoxPH": CoxnetSurvivalAnalysis(fit_baseline_model=True),
        "GradientBoosting": GradientBoostingSurvivalAnalysis(),
        "RandomForest": RandomSurvivalForest(n_estimators=100, random_state=42),
        "SVM": FastSurvivalSVM(max_iter=1000, tol=1e-5, random_state=42),
    }

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        fitted_models[name] = model

        model_path = output_dir / 'survival_models' / f"{name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        predictions = model.predict(X_test)
        cindex_scores[name] = concordance_index_censored(
            y_test["event"], y_test["time"], predictions
        )[0]

    print("\nConsolidated C-index Scores:")
    for model_name, cindex in cindex_scores.items():
        print(f"{model_name}: {cindex:.4f}")

    # For later saving to yaml
    cindex_scores = {key: float(value) for key, value in cindex_scores.items()}

    return fitted_models, cindex_scores, data_train, data_val
