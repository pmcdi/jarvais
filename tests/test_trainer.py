import pytest
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
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
    data_dir = tmpdir / 'data'
    assert (data_dir / 'X_train.csv').exists()
    assert (data_dir / 'X_test.csv').exists()
    assert hasattr(trainer, 'predictor')
    assert hasattr(trainer, 'X_train')
    assert hasattr(trainer, 'X_test')

def test_load_model(sample_data, tmpdir):
    X, y = sample_data
    data = pd.concat([X, y], axis=1)
    trainer = TrainerSupervised(task='binary', output_dir=str(tmpdir))
    trainer.run(data=data, target_variable='target', save_data=True)
    loaded_trainer = TrainerSupervised.load_model(project_dir=str(tmpdir))
    assert loaded_trainer.predictor is not None