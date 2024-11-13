import pytest
import pandas as pd
import numpy as np
import os, shutil
from AutoML.trainer import TrainerSupervised

@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.randint(0, 2, 100),
        'feature3': np.random.randn(100)
    })
    y = pd.Series(np.random.randint(0, 2, 100), name='target')
    return X, y

@pytest.fixture
def tmpdir():
    temp_path = "./tests/tmp"
    if not os.path.exists(temp_path):
        os.makedirs(temp_path, exist_ok=True)
    else:
        for file in os.listdir(temp_path):
            file_path = os.path.join(temp_path, file)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
            
    yield temp_path

def test_trainer_initialization():
    trainer = TrainerSupervised(task='binary')    
    assert trainer.task == 'binary'
    assert trainer.reduction_method is None
    assert trainer.keep_k == 2
    with pytest.raises(ValueError):
        TrainerSupervised(task='invalid_task')

def test_feature_reduction(sample_data):
    X, y = sample_data
    trainer = TrainerSupervised(task='binary', reduction_method='variance_threshold')
    X_reduced = trainer._feature_reduction(X, y)
    assert X_reduced.shape[1] <= X.shape[1]

def test_run_method(sample_data, tmpdir):
    X, y = sample_data
    data = pd.concat([X, y], axis=1)
    trainer = TrainerSupervised(task='binary', output_dir=str(tmpdir))
    trainer.run(data=data, target_variable='target', save_data=True)
    data_dir = os.path.join(str(tmpdir), 'data')
    assert os.path.exists(os.path.join(data_dir, 'X_train.csv'))
    assert os.path.exists(os.path.join(data_dir, 'X_test.csv'))
    assert hasattr(trainer, 'predictor')
    assert hasattr(trainer, 'X_train')
    assert hasattr(trainer, 'X_test')

def test_load_model(tmpdir):
    X = pd.DataFrame({'feature': np.random.rand(10)})
    y = pd.Series(np.random.randint(0, 2, 10), name='target')
    data = pd.concat([X, y], axis=1)
    trainer = TrainerSupervised(task='binary', output_dir=str(tmpdir))
    trainer.run(data=data, target_variable='target', save_data=True)
    loaded_trainer = TrainerSupervised.load_model(model_dir=str(tmpdir))
    assert loaded_trainer.predictor is not None