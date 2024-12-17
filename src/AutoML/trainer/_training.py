import os, pickle
import shutil
import pandas as pd

from autogluon.tabular import TabularPredictor

from sklearn.model_selection import KFold

from sksurv.metrics import concordance_index_censored
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.svm import FastSurvivalSVM
from sksurv.util import Surv

from ._leaderboard import aggregate_folds, format_leaderboard

def train_autogluon_with_cv(data_train, data_test, target_variable, task, 
                  output_dir, extra_metrics, eval_metric='accuracy', num_folds=5, **kwargs):
    """
    Trains a TabularPredictor using manual cross-validation without bagging and consolidates the leaderboards.

    Parameters:
    - data_train (DataFrame): Combined training data (features + target).
    - data_test (DataFrame): Combined test data (features + target).
    - target_variable (str): Name of the target column.
    - task (str): Problem type (e.g., 'binary', 'multiclass', 'regression').
    - predictor_fit_kwargs (dict): Additional arguments to pass to TabularPredictor's fit method.
    - output_dir (str): Directory to save model files.
    - eval_metric (str): Evaluation metric to optimize (default: 'accuracy').
    - num_folds (int): Number of cross-validation folds (default: 5).

    Returns:
    - predictors: A list of trained predictors (one per fold).
    - final_leaderboard: A single DataFrame containing all models across folds.
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
    X_val, y_val = data_train.iloc[val_indices_best].drop(columns=target_variable), data_train.iloc[val_indices_best][target_variable]

    shutil.copytree(os.path.join(output_dir, f'autogluon_models_fold_{best_fold + 1}'), 
                    os.path.join(output_dir, f'autogluon_models_best_fold'), dirs_exist_ok=True)

    return predictors, final_leaderboard, best_fold, X_val, y_val

def train_survival_models(X_train, y_train, X_test, y_test, output_dir):

    y_train = Surv.from_dataframe('event', 'time', y_train)
    y_test = Surv.from_dataframe('event', 'time', y_test)

    (output_dir / 'survival_models').mkdir(exist_ok=True, parents=True)

    models = {
        "CoxPH": CoxnetSurvivalAnalysis(fit_baseline_model=True),  # fit_baseline_model=True to enable cumulative hazard prediction
        "GradientBoosting": GradientBoostingSurvivalAnalysis(),
        "RandomForest": RandomSurvivalForest(n_estimators=100, random_state=42),
        "SVM": FastSurvivalSVM(max_iter=1000, tol=1e-5, random_state=0),
    }

    fitted_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        fitted_models[name] = model

        model_path = output_dir / 'survival_models' / f"{name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    def evaluate_model(model, X_test, y_test):
        predictions = model.predict(X_test)
        return concordance_index_censored(y_test["event"], y_test["time"], predictions)[0]

    cindex_scores = {}
    for name, model in fitted_models.items():
        cindex_scores[name] = evaluate_model(model, X_test, y_test)

    # Output C-index scores
    print("\nC-index Scores:")
    for model_name, cindex in cindex_scores.items():
        print(f"{model_name}: {cindex:.4f}")

    return fitted_models, cindex_scores

