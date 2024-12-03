
import os
import shutil
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

from autogluon.tabular import TabularPredictor
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config

from ..models import SimpleRegressionModel

def knn_impute_categorical(data, columns):
    """
    Perform KNN imputation on categorical columns. 
    From https://www.kaggle.com/discussions/questions-and-answers/153147

    Args:
        data (DataFrame): The data containing categorical columns.
        columns (list): List of categorical column names.

    Returns:
        DataFrame: Data with imputed categorical columns.
    """
    mm = MinMaxScaler()
    mappin = {}

    def find_category_mappings(df, variable):
        return {k: i for i, k in enumerate(df[variable].dropna().unique(), 0)}

    def integer_encode(df, variable, ordinal_mapping):
        df[variable] = df[variable].map(ordinal_mapping)

    df = data.copy()
    for variable in columns:
        mappings = find_category_mappings(df, variable)
        mappin[variable] = mappings

    for variable in columns:
        integer_encode(df, variable, mappin[variable])

    scaled_data = mm.fit_transform(df)
    knn_imputer = KNNImputer()
    knn_imputed = knn_imputer.fit_transform(scaled_data)
    df.iloc[:, :] = mm.inverse_transform(knn_imputed)
    for col in df.columns:
        df[col] = round(df[col]).astype('int')

    for col in columns:
        inv_map = {v: k for k, v in mappin[col].items()}
        df[col] = df[col].map(inv_map)

    return df

def get_outliers(data, categorical_columns):
    mapping = {}
    
    for cat in categorical_columns:
        category_counts = data[cat].value_counts()
        threshold = int(len(data)*.01)
        outliers = category_counts[category_counts < threshold].index.tolist()

        mapping[cat] = {}

        for _cat in data[cat].unique():
            if _cat in outliers:
                mapping[cat][f'{_cat}'] = 'Other'
            else:
                mapping[cat][f'{_cat}'] = f'{_cat}'

        if len(outliers) > 0:
            outliers = [f'{o}: {category_counts[o]} out of {data[cat].count()}' for o in outliers]
            print(f'  - Outliers found in {cat}: {outliers}')
            return f'  - Outliers found in {cat}: {outliers}\n', mapping
        else:
            print(f'  - No Outliers found in {cat}')
            return f'  - No Outliers found in {cat}\n', mapping
    
    return "No outliers found\n", mapping

def mrmr_reduction(task, X, y, k):
    from mrmr import mrmr_classif, mrmr_regression

    mrmr_method = mrmr_classif if task in ['binary', 'multiclass'] else mrmr_regression
    selected_features = mrmr_method(X=X, y=y, K=k, n_jobs=1)
    return X[selected_features]

def var_reduction(X, y):
    from sklearn.feature_selection import VarianceThreshold

    selector = VarianceThreshold()
    _ = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support(indices=True)]
    return X[selected_features]

def kbest_reduction(task, X, y, k):
    from sklearn.feature_selection import SelectKBest, f_classif, f_regression

    if task in ['binary', 'multiclass']:
        f_method = f_classif
    else:
        f_method = f_regression

    selector = SelectKBest(score_func=f_method, k=k)
    _ = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support(indices=True)]
    return X[selected_features]

def chi2_reduction(X, y, k):
    from sklearn.feature_selection import SelectKBest, chi2

    selector = SelectKBest(score_func=chi2, k=k)
    _ = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support(indices=True)]
    return X[selected_features]

def train_with_cv(data_train, data_test, target_variable, task, 
                  output_dir, eval_metric='accuracy', num_folds=5, **kwargs):
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

    predictors, cv_scores, leaderboards, val_indices = [], [], [], []

    custom_hyperparameters = get_hyperparameter_config('default')
    custom_hyperparameters[SimpleRegressionModel] = {}

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
            hyperparameters=custom_hyperparameters, 
            **kwargs)

        score = predictor.evaluate(val_data)[eval_metric]
        print(f"Fold {fold + 1} score: {score}")

        predictors.append(predictor)
        cv_scores.append(score)

        extra_metrics = ['f1', 'average_precision'] if task in ['binary', 'multiclass'] else ['root_mean_squared_error']
        leaderboard = predictor.leaderboard(data_test, extra_metrics=extra_metrics)
        train_metrics = predictor.leaderboard(train_data)[['model', 'score_test']].rename(columns={'score_test': 'score_train'})
        leaderboard = leaderboard.merge(train_metrics, on='model')
        leaderboards.append(leaderboard)

    consolidated_leaderboard = pd.concat(leaderboards, ignore_index=True)

    # Specify metrics for aggregation
    to_agg = {k: ['mean', 'min', 'max'] for k in ['score_test', 'score_val', 'score_train'] + extra_metrics}

    # Group by 'model' and aggregate
    aggregated_leaderboard = consolidated_leaderboard.groupby('model').agg(to_agg).reset_index()

    # Create the final leaderboard dataframe with unique models
    final_leaderboard = pd.DataFrame({'model': aggregated_leaderboard['model']})

    # Populate the leaderboard with formatted metrics
    for col in to_agg.keys():
        final_leaderboard[col] = [
            f'{round(row[0], 2)} [{round(row[1], 2)}, {round(row[2], 2)}]'
            for row in aggregated_leaderboard[col].values
        ]
    
    final_leaderboard['eval_metric'] = eval_metric

    best_fold = cv_scores.index(max(cv_scores))
    val_indices_best = val_indices[best_fold]
    X_val, y_val = data_train.iloc[val_indices_best].drop(columns=target_variable), data_train.iloc[val_indices_best][target_variable]

    shutil.copytree(os.path.join(output_dir, f'autogluon_models_fold_{best_fold + 1}'), 
                    os.path.join(output_dir, f'autogluon_models_best_fold'), dirs_exist_ok=True)

    return predictors, final_leaderboard, best_fold, X_val, y_val

