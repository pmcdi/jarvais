
# CHAT GPT CODE!!!

from lifelines import CoxPHFitter
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.ensemble import RandomSurvivalForest
from pycox.models import MTLR, CoxPH, LogisticHazard
from pycox.models.data import make_dataset
import torch

class CoxPHModel:
    def __init__(self):
        self.model = CoxPHSurvivalAnalysis()
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
        
    def evaluate(self, X_test, y_test):
        from sksurv.metrics import concordance_index_censored
        y_pred = self.predict(X_test)
        ci = concordance_index_censored(y_test["event"], y_test["time"], y_pred)[0]
        return {'concordance_index': ci}

class MTLRModel:
    def __init__(self):
        self.model = MTLR()
        
    def train(self, X_train, y_train):
        dataset = make_dataset(X_train, y_train)
        self.model.fit(dataset)
        
    def predict(self, X_test):
        return self.model.predict_surv(X_test)
        
    def evaluate(self, X_test, y_test):
        from sksurv.metrics import concordance_index_censored
        y_pred = self.predict(X_test)
        ci = concordance_index_censored(y_test["event"], y_test["time"], y_pred)[0]
        return {'concordance_index': ci}

class NeuralMTLRModel:
    def __init__(self, in_features):
        self.model = LogisticHazard(in_features, num_durations=10)
        
    def train(self, X_train, y_train, epochs=100, batch_size=256, learning_rate=0.01):
        dataset = make_dataset(X_train, y_train)
        self.model.fit(dataset, epochs, batch_size, learning_rate)
        
    def predict(self, X_test):
        return self.model.predict_surv(X_test)
        
    def evaluate(self, X_test, y_test):
        from sksurv.metrics import concordance_index_censored
        y_pred = self.predict(X_test)
        ci = concordance_index_censored(y_test["event"], y_test["time"], y_pred)[0]
        return {'concordance_index': ci}

class SurvivalSVMModel:
    def __init__(self, **kwargs):
        self.model = FastSurvivalSVM(**kwargs)
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
        
    def evaluate(self, X_test, y_test):
        from sksurv.metrics import concordance_index_censored
        y_pred = self.predict(X_test)
        ci = concordance_index_censored(y_test["event"], y_test["time"], y_pred)[0]
        return {'concordance_index': ci}

class SurvivalForestModel:
    def __init__(self, **kwargs):
        self.model = RandomSurvivalForest(**kwargs)
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
        
    def evaluate(self, X_test, y_test):
        from sksurv.metrics import concordance_index_censored
        y_pred = self.predict(X_test)
        ci = concordance_index_censored(y_test["event"], y_test["time"], y_pred)[0]
        return {'concordance_index': ci}
