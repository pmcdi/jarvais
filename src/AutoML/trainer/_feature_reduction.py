from mrmr import mrmr_classif, mrmr_regression
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, f_regression, chi2

def var_reduction(X, y):

    selector = VarianceThreshold()
    _ = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support(indices=True)]
    return X[selected_features]

def chi2_reduction(X, y, k):
    
    selector = SelectKBest(score_func=chi2, k=k)
    _ = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support(indices=True)]
    return X[selected_features]

def kbest_reduction(task, X, y, k):

    if task in ['binary', 'multiclass']:
        f_method = f_classif
    else:
        f_method = f_regression

    selector = SelectKBest(score_func=f_method, k=k)
    _ = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support(indices=True)]
    return X[selected_features]

def mrmr_reduction(task, X, y, k):

    mrmr_method = mrmr_classif if task in ['binary', 'multiclass'] else mrmr_regression
    selected_features = mrmr_method(X=X, y=y, K=k, n_jobs=1)
    return X[selected_features]
