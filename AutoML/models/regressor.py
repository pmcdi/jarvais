
# CHAT GPT CODE!!!

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor

class LinearRegressionModel:
    def __init__(self, **kwargs):
        self.model = LinearRegression(**kwargs)
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
        
    def evaluate(self, X_test, y_test):
        from sklearn.metrics import mean_squared_error, r2_score
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {'mse': mse, 'r2': r2}

class RidgeRegressionModel:
    def __init__(self, **kwargs):
        self.model = Ridge(**kwargs)
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
        
    def evaluate(self, X_test, y_test):
        from sklearn.metrics import mean_squared_error, r2_score
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {'mse': mse, 'r2': r2}

class DecisionTreeRegressorModel:
    def __init__(self, **kwargs):
        self.model = DecisionTreeRegressor(**kwargs)
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
        
    def evaluate(self, X_test, y_test):
        from sklearn.metrics import mean_squared_error, r2_score
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {'mse': mse, 'r2': r2}

class GradientBoostingRegressorModel:
    def __init__(self, **kwargs):
        self.model = GradientBoostingRegressor(**kwargs)
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
        
    def evaluate(self, X_test, y_test):
        from sklearn.metrics import mean_squared_error, r2_score
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {'mse': mse, 'r2': r2}

class RandomForestRegressorModel:
    def __init__(self, **kwargs):
        self.model = RandomForestRegressor(**kwargs)
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
        
    def evaluate(self, X_test, y_test):
        from sklearn.metrics import mean_squared_error, r2_score
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {'mse': mse, 'r2': r2}

class MLPRegressorModel:
    def __init__(self, **kwargs):
        self.model = MLPRegressor(**kwargs)
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
        
    def evaluate(self, X_test, y_test):
        from sklearn.metrics import mean_squared_error, r2_score
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {'mse': mse, 'r2': r2}
