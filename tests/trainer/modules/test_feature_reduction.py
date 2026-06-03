import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

from jarvais.trainer.modules.feature_reduction import FeatureReductionModule 

@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y = pd.Series(y)
    return X.abs(), y  # chi2 requires non-negative

@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y = pd.Series(y)
    return X, y

def test_method_none(classification_data):
    X, y = classification_data
    module = FeatureReductionModule(method=None, task="binary", keep_k=10)
    X_out, y_out = module(X, y)
    assert X_out.equals(X)
    assert y_out.equals(y)

def test_variance_threshold_removes_constant(classification_data):
    X, y = classification_data
    X["constant"] = 1.0
    module = FeatureReductionModule(method="variance_threshold", task="binary")
    X_out, _ = module(X, y)
    assert "constant" not in X_out.columns
    assert X_out.shape[1] < X.shape[1]

def test_chi2_classification(classification_data):
    X, y = classification_data
    module = FeatureReductionModule(method="chi2", task="binary", keep_k=5)
    X_out, y_out = module(X, y)
    assert X_out.shape[1] == 5
    assert y_out.equals(y)

def test_chi2_invalid_task(regression_data):
    X, y = regression_data
    module = FeatureReductionModule(method="chi2", task="regression", keep_k=5)
    with pytest.raises(ValueError, match="Chi2 only supports classification tasks."):
        module(X, y)

def test_corr_kbest_classification(classification_data):
    X, y = classification_data
    module = FeatureReductionModule(method="corr", task="binary", keep_k=7)
    X_out, y_out = module(X, y)
    assert X_out.shape[1] == 7
    assert y_out.equals(y)

def test_corr_kbest_regression(regression_data):
    X, y = regression_data
    module = FeatureReductionModule(method="corr", task="regression", keep_k=4)
    X_out, _ = module(X, y)
    assert X_out.shape[1] == 4

def test_mrmr_classification(classification_data):
    X, y = classification_data
    module = FeatureReductionModule(method="mrmr", task="binary", keep_k=6)
    X_out, y_out = module(X, y)
    assert X_out.shape[1] == 6
    assert y_out.equals(y)

def test_mrmr_regression(regression_data):
    X, y = regression_data
    module = FeatureReductionModule(method="mrmr", task="regression", keep_k=3)
    X_out, _ = module(X, y)
    assert X_out.shape[1] == 3


@pytest.fixture
def survival_data(classification_data):
    X, y_cls = classification_data
    rng = np.random.default_rng(42)
    time = np.abs(rng.standard_normal(len(y_cls))) * 10 + 1.0
    y = pd.DataFrame({"time": time, "event": y_cls.values}, index=y_cls.index)
    return X, y


def test_corr_kbest_survival_event_column(survival_data):
    X, y = survival_data
    module = FeatureReductionModule(method="corr", task="survival", keep_k=5)
    X_out, y_out = module(X, y)
    assert X_out.shape[1] == 5
    pd.testing.assert_frame_equal(y_out, y)


def test_mrmr_survival_event_column(survival_data):
    X, y = survival_data
    module = FeatureReductionModule(method="mrmr", task="survival", keep_k=4)
    X_out, y_out = module(X, y)
    assert X_out.shape[1] == 4
    pd.testing.assert_frame_equal(y_out, y)


def test_variance_threshold_ignores_survival_y_shape(survival_data):
    X, y = survival_data
    X["constant"] = 1.0
    module = FeatureReductionModule(method="variance_threshold", task="survival")
    X_out, y_out = module(X, y)
    assert "constant" not in X_out.columns
    pd.testing.assert_frame_equal(y_out, y)


def test_chi2_rejects_survival_task(survival_data):
    X, y = survival_data
    module = FeatureReductionModule(method="chi2", task="survival", keep_k=5)
    with pytest.raises(ValueError, match="Chi2 only supports classification tasks."):
        module(X, y)

