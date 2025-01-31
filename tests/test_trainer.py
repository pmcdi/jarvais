import pytest
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from sklearn.datasets import make_classification, make_regression
from jarvais.trainer import TrainerSupervised

@pytest.fixture
def classification_data():
    X, y = make_classification(
        n_samples=50, n_features=5, n_informative=3, n_redundant=1, random_state=42
    )
    X = pd.DataFrame(X, columns=[f"feature{i}" for i in range(1, X.shape[1] + 1)])
    y = pd.Series(y, name="target")
    return X, y

@pytest.fixture
def regression_data():
    X, y = make_regression(
        n_samples=50, n_features=5, noise=0.1, random_state=42
    )
    X = pd.DataFrame(X, columns=[f"feature{i}" for i in range(1, X.shape[1] + 1)])
    y = pd.Series(y, name="target")
    return X, y

@pytest.fixture
def survival_data():
    """Synthetic survival data generation."""
    n_samples = 50
    np.random.seed(42)
    X = np.random.normal(size=(n_samples, 5))  # Covariates
    event_time = np.random.exponential(scale=10, size=n_samples)  # Simulated survival times
    censor_time = np.random.uniform(0, 15, size=n_samples)  # Censoring times
    observed_time = np.minimum(event_time, censor_time)  # Observed time
    event_observed = (event_time <= censor_time).astype(int)  # Event indicator
    X_df = pd.DataFrame(X, columns=[f"feature{i}" for i in range(1, 6)])
    data = pd.concat([X_df, pd.Series(observed_time, name="time"), pd.Series(event_observed, name="event")], axis=1)
    return data

@pytest.fixture
def tmpdir():
    temp_path = Path("./tests/tmp")
    temp_path.mkdir(parents=True, exist_ok=True)

    for file in temp_path.iterdir():
        file_path = temp_path / file
        if file_path.is_file() or file_path.is_symlink():
            file_path.unlink()
        elif file_path.is_dir():
            shutil.rmtree(file_path)

    yield temp_path

def test_trainer_initialization():
    trainer = TrainerSupervised(task='binary')
    assert trainer.task == 'binary'
    assert trainer.reduction_method is None
    assert trainer.keep_k == 2
    with pytest.raises(ValueError):
        TrainerSupervised(task='invalid_task')

def test_feature_reduction(classification_data):
    X, y = classification_data
    trainer = TrainerSupervised(task='binary', reduction_method='variance_threshold')
    X_reduced = trainer._feature_reduction(X, y)
    assert X_reduced.shape[1] <= X.shape[1]

def test_run_method_classification(classification_data, tmpdir):
    X, y = classification_data
    data = pd.concat([X, y], axis=1)
    trainer = TrainerSupervised(task='binary', output_dir=str(tmpdir))
    trainer.run(data=data, target_variable='target')
    data_dir = tmpdir / 'data'
    assert (data_dir / 'X_train.csv').exists()
    assert (data_dir / 'X_test.csv').exists()
    assert hasattr(trainer, 'predictor')
    assert hasattr(trainer, 'X_train')
    assert hasattr(trainer, 'X_test')

def test_run_method_regression(regression_data, tmpdir):
    X, y = regression_data
    data = pd.concat([X, y], axis=1)
    trainer = TrainerSupervised(task='regression', output_dir=str(tmpdir))
    trainer.run(data=data, target_variable='target')
    data_dir = tmpdir / 'data'
    assert (data_dir / 'X_train.csv').exists()
    assert (data_dir / 'X_test.csv').exists()
    assert hasattr(trainer, 'predictor')
    assert hasattr(trainer, 'X_train')
    assert hasattr(trainer, 'X_test')

def test_run_method_survival(survival_data, tmpdir):
    data = survival_data
    trainer = TrainerSupervised(task='survival', output_dir=str(tmpdir))
    trainer.run(data=data, target_variable=["time", "event"])
    data_dir = tmpdir / 'data'
    assert (data_dir / 'X_train.csv').exists()
    assert (data_dir / 'X_test.csv').exists()
    assert hasattr(trainer, 'predictor')
    assert hasattr(trainer, 'X_train')
    assert hasattr(trainer, 'X_test')

def test_load_trainer_autogluon(classification_data, tmpdir):
    X, y = classification_data
    data = pd.concat([X, y], axis=1)
    trainer = TrainerSupervised(task='binary', output_dir=str(tmpdir))
    trainer.run(data=data, target_variable='target')
    loaded_trainer = TrainerSupervised.load_trainer(project_dir=str(tmpdir))
    assert loaded_trainer.predictor is not None

def test_load_trainer_survival(survival_data, tmpdir):
    data = survival_data
    trainer = TrainerSupervised(task='survival', output_dir=str(tmpdir))
    trainer.run(data=data, target_variable=["time", "event"])
    loaded_trainer = TrainerSupervised.load_trainer(project_dir=str(tmpdir))
    assert loaded_trainer.predictor is not None