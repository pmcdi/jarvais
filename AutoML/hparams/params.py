import numpy as np

def get_hyperparameters(data, task):
    """
    Function to generate hyperparameter space dynamically based on the data and task type.
    """
    if task == 'classifier':
        return {
            'logr': {
                'C': np.logspace(-4, 0, 5).tolist(),
                'class_weight': [None, 'balanced'],
                'solver': ['newton-cg', 'lbfgs', 'saga']
            },
            'logr_pen': [
                {
                    'penalty': ['l1'],
                    'C': np.logspace(-4, 0, 5).tolist(),
                    'solver': ['saga']
                },
                {
                    'penalty': ['l2'],
                    'C': np.logspace(-4, 0, 5).tolist(),
                    'solver': ['newton-cg', 'lbfgs', 'saga']
                }
            ],
            'boost_c': {
                'loss': ['exponential'],
                'learning_rate': [0.001, 0.01, 0.1],
                'n_estimators': [5, 25, 125, 625, 3125],
                'subsample': [0.5, 0.75, 1.0],
                'criterion': ['friedman_mse', 'squared_error'],
                'min_samples_split': [2, 4, 6, 8],
                'min_samples_leaf': [2, 3, 4, 5],
                'max_depth': [3, 5, 7],
                'max_features': ['sqrt', 'log2']
            },
            'rf_c': {
                'n_estimators': [24, 48, 96, 192, 384, 768, 1536],
                'criterion': ['gini', 'entropy'],
                'min_samples_split': [2, 4, 8],
                'min_samples_leaf': [2, 3, 4, 5, 6],
                'max_depth': [3, 5, 7],
                'max_features': ['sqrt', 'log2']
            },
            'tree_c': {
                'criterion': ['gini', 'entropy'],
                'min_samples_split': [2, 4, 6, 8, 10],
                'min_samples_leaf': [1, 2, 3, 4, 5, 6],
                'max_depth': [3, 5, 7],
                'max_features': ['sqrt', 'log2'],
                'class_weight': ['balanced', None]
            },
            'svc': {
                'C': np.logspace(-4, 0, 5).tolist(),
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto'],
                'probability': [True],
                'class_weight': ['balanced', None]
            },
            'mlp_c': {
                'hidden_layer_sizes': [[128], [128, 64]],
                'activation': ['relu', 'tanh'],
                'solver': ['sgd', 'adam'],
                'alpha': [0.1, 0.01, 0.001],
                'learning_rate': ['constant', 'adaptive'],
                'learning_rate_init': [0.01, 0.001]
            },
            'knn_c': {
                'n_neighbors': [3, 5, 9, 17, 31],
                'weights': ['uniform', 'distance'],
                'p': [1, 2, 3]
            }
        }
    elif task == 'regression':
        return {
            'linr': {
                'fit_intercept': [True, False]
            },
            'linr_pen': {
                'fit_intercept': [True, False]
            },
            'boost_r': {
                'learning_rate': [0.001, 0.01, 0.1],
                'n_estimators': [5, 25, 125, 625, 3125],
                'subsample': [0.5, 0.75, 1.0],
                'criterion': ['friedman_mse', 'squared_error'],
                'min_samples_split': [2, 4, 6, 8],
                'min_samples_leaf': [1, 2, 3, 4],
                'max_depth': [3, 5, 7],
                'max_features': ['sqrt', 'log2']
            },
            'rf_r': {
                'n_estimators': [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
                'criterion': ['friedman_mse', 'squared_error'],
                'min_samples_split': [2, 4, 8],
                'min_samples_leaf': [1, 2, 3, 4, 5],
                'max_depth': [3, 5, 7],
                'max_features': ['sqrt', 'log2'],
                'bootstrap': [True],
                'oob_score': [True]
            },
            'tree_r': {
                'criterion': ['friedman_mse', 'squared_error'],
                'min_samples_split': [2, 4, 6, 8, 10],
                'min_samples_leaf': [1, 2, 3, 4, 5, 6],
                'max_depth': [2, 3, 4, 5, 6, 7, 8, 9],
                'max_features': ['sqrt', 'log2']
            },
            'svr': {
                'C': np.logspace(-4, 0, 5).tolist(),
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto']
            },
            'knn_r': {
                'n_neighbors': [2, 4, 8, 16, 32],
                'weights': ['uniform', 'distance'],
                'p': [1, 2, 3]
            }
        }
    elif task == 'cluster':
        return {
            'km': {
                'n_clusters': list(range(2, 11)),
                'init': ['k-means++', 'random'],
                'n_init': [10, 20, 30],
                'max_iter': [300, 600, 900],
                'tol': [0.0001, 0.001, 0.01]
            },
            'ap': {
                'damping': np.linspace(0.5, 0.9, 5).tolist(),
                'max_iter': [200, 400, 600],
                'convergence_iter': [15, 30, 45],
                'preference': [-50, -10, 0, 10, 50]
            },
            'sc': {
                'n_clusters': list(range(2, 11)),
                'eigen_solver': ['arpack', 'lobpcg', 'amg'],
                'affinity': ['nearest_neighbors', 'rbf']
            }
        }