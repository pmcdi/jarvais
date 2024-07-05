
# CHAT GPT CODE!!!

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import HalvingGridSearchCV
from skopt import BayesSearchCV
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

class ModelHyperparams:
    def __init__(self, model_type='classifier', model_name=None, params=None):
        self.model_type = model_type
        self.model_name = model_name
        self.params = params

    def get_params(self):
        if self.model_type == 'classifier':
            return self._get_classifier_params()
        elif self.model_type == 'regressor':
            return self._get_regressor_params()
        elif self.model_type == 'survival':
            return self._get_survival_params()
        else:
            raise ValueError("Unknown model type. Choose 'classifier', 'regressor', or 'survival'.")

    def _get_classifier_params(self):
        if self.model_name == 'LogisticRegression':
            return {'C': [0.1, 1, 10]}
        elif self.model_name == 'kNN':
            return {'n_neighbors': [3, 5, 7]}
        elif self.model_name == 'DecisionTree':
            return {'max_depth': [None, 10, 20]}
        elif self.model_name == 'SVM':
            return {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        elif self.model_name == 'GradientBoostingClassifier':
            return {'n_estimators': [50, 100, 200]}
        elif self.model_name == 'RandomForest':
            return {'n_estimators': [50, 100, 200], 'max_features': ['auto', 'sqrt']}
        elif self.model_name == 'MLP':
            return {'hidden_layer_sizes': [(100,), (50, 50)], 'alpha': [0.0001, 0.001]}
        elif self.model_name == 'XGBoostClassifier':
            return {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1]}
        else:
            raise ValueError("Unknown classifier model name.")

    def _get_regressor_params(self):
        if self.model_name == 'LinearRegression':
            return {}
        elif self.model_name == 'RidgeRegression':
            return {'alpha': [0.1, 1, 10]}
        elif self.model_name == 'DecisionTreeRegressor':
            return {'max_depth': [None, 10, 20]}
        elif self.model_name == 'GradientBoostingRegressor':
            return {'n_estimators': [50, 100, 200]}
        elif self.model_name == 'RandomForestRegressor':
            return {'n_estimators': [50, 100, 200], 'max_features': ['auto', 'sqrt']}
        elif self.model_name == 'MLPRegressor':
            return {'hidden_layer_sizes': [(100,), (50, 50)], 'alpha': [0.0001, 0.001]}
        else:
            raise ValueError("Unknown regressor model name.")

    def _get_survival_params(self):
        if self.model_name == 'CoxPH':
            return {}
        elif self.model_name == 'MTLR':
            return {'alpha': [0.1, 1, 10]}
        elif self.model_name == 'NeuralMTLR':
            return {'num_durations': [5, 10, 20]}
        elif self.model_name == 'SurvivalSVM':
            return {'alpha': [0.1, 1, 10]}
        elif self.model_name == 'SurvivalForest':
            return {'n_estimators': [50, 100, 200], 'max_features': ['auto', 'sqrt']}
        else:
            raise ValueError("Unknown survival model name.")

class HyperparamSearch:
    def __init__(self, model, param_grid, strategy='grid', cv_strategy='random', n_splits=5):
        self.model = model
        self.param_grid = param_grid
        self.strategy = strategy
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits

    def _get_cv_strategy(self):
        if self.cv_strategy == 'random':
            return KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        elif self.cv_strategy == 'stratified':
            return StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        elif self.cv_strategy == 'time_based':
            return TimeSeriesSplit(n_splits=self.n_splits)
        else:
            raise ValueError("Unknown CV strategy. Choose 'random', 'stratified', or 'time_based'.")

    def search(self, X_train, y_train):
        cv = self._get_cv_strategy()
        if self.strategy == 'grid':
            search = GridSearchCV(self.model, self.param_grid, cv=cv)
        elif self.strategy == 'halving_grid':
            search = HalvingGridSearchCV(self.model, self.param_grid, cv=cv)
        elif self.strategy == 'random':
            search = RandomizedSearchCV(self.model, self.param_grid, n_iter=10, cv=cv)
        elif self.strategy == 'bayesian':
            search = BayesSearchCV(self.model, self.param_grid, cv=cv)
        else:
            raise ValueError("Unknown search strategy. Choose 'grid', 'halving_grid', 'random', or 'bayesian'.")
        
        search.fit(X_train, y_train)
        return search.best_params_, search.best_score_

class DataScaler:
    def __init__(self, method='minmax'):
        if method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'meanstd':
            self.scaler = StandardScaler()
        else:
            raise ValueError("Unknown scaling method. Choose 'minmax' or 'meanstd'.")

    def fit_transform(self, X):
        return self.scaler.fit_transform(X)

    def transform(self, X):
        return self.scaler.transform(X)


# Usage Example
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    # Load data
    data = load_iris()
    X = data.data
    y = data.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale data
    scaler = DataScaler(method='minmax')
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Get model params
    params = ModelHyperparams(model_type='classifier', model_name='LogisticRegression').get_params()

    # Perform hyperparameter search with stratified CV strategy
    search = HyperparamSearch(LogisticRegression(), params, strategy='grid', cv_strategy='stratified')
    best_params, best_score = search.search(X_train_scaled, y_train)

    print(f"Best Params: {best_params}")
    print(f"Best Score: {best_score}")