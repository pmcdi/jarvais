
# CHAT GPT CODE!!!

from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

class LogisticRegressionModel:
    def __init__(self, **kwargs):
        self.model = LogisticRegression(**kwargs)
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
        
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)
        
    def evaluate(self, X_test, y_test):
        from sklearn.metrics import accuracy_score, roc_auc_score
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, self.predict_proba(X_test)[:, 1])
        return {'accuracy': accuracy, 'auc': auc}

class ElasticNetModel:
    def __init__(self, **kwargs):
        self.model = ElasticNet(**kwargs)
        
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

class KNeighborsModel:
    def __init__(self, **kwargs):
        self.model = KNeighborsClassifier(**kwargs)
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
        
    def evaluate(self, X_test, y_test):
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return {'accuracy': accuracy}

class DecisionTreeModel:
    def __init__(self, **kwargs):
        self.model = DecisionTreeClassifier(**kwargs)
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
        
    def evaluate(self, X_test, y_test):
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return {'accuracy': accuracy}

class SVMModel:
    def __init__(self, **kwargs):
        self.model = SVC(**kwargs)
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
        
    def evaluate(self, X_test, y_test):
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return {'accuracy': accuracy}

class GradientBoostingModel:
    def __init__(self, **kwargs):
        self.model = GradientBoostingClassifier(**kwargs)
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
        
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)
        
    def evaluate(self, X_test, y_test):
        from sklearn.metrics import accuracy_score, roc_auc_score
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, self.predict_proba(X_test)[:, 1])
        return {'accuracy': accuracy, 'auc': auc}

class RandomForestModel:
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
        
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)
        
    def evaluate(self, X_test, y_test):
        from sklearn.metrics import accuracy_score, roc_auc_score
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, self.predict_proba(X_test)[:, 1])
        return {'accuracy': accuracy, 'auc': auc}

class MLPModel:
    def __init__(self, **kwargs):
        self.model = MLPClassifier(**kwargs)
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
        
    def evaluate(self, X_test, y_test):
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return {'accuracy': accuracy}

class XGBoostModel:
    def __init__(self, **kwargs):
        self.model = XGBClassifier(**kwargs)
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
        
    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)
        
    def evaluate(self, X_test, y_test):
        from sklearn.metrics import accuracy_score, roc_auc_score
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, self.predict_proba(X_test)[:, 1])
        return {'accuracy': accuracy, 'auc': auc}
