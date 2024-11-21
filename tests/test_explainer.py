import pytest
import os, shutil
import pandas as pd
import numpy as np
from AutoML.explainer import Explainer
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

@pytest.fixture
def trained_binary_model(sample_data, tmpdir):
    X, y = sample_data
    data = pd.concat([X, y], axis=1)
    trainer = TrainerSupervised(task='binary', output_dir=str(tmpdir))
    trainer.run(data=data, target_variable='target')
    return trainer

@pytest.fixture
def explainer_instance(trained_binary_model, tmpdir):
    trainer = trained_binary_model
    X_train, X_test = trainer.X_train, trainer.X_test
    y_test = trainer.y_test
    return Explainer(trainer, X_train, X_test, y_test, output_dir=tmpdir)

def test_explainer_initialization(explainer_instance, tmpdir):
    explainer = explainer_instance
    assert explainer.trainer is not None
    assert explainer.predictor is not None
    assert explainer.output_dir == tmpdir
    assert hasattr(explainer, 'X_train')
    assert hasattr(explainer, 'X_test')
    assert hasattr(explainer, 'y_test')

def test_explainer_run_binary_classification(explainer_instance):
    explainer = explainer_instance
    explainer.run()
    # Check if diagnostic plots are saved
    assert os.path.exists(os.path.join(explainer.output_dir, 'confusion_matrix.png'))
    assert os.path.exists(os.path.join(explainer.output_dir, 'feature_importance.png'))
    assert os.path.exists(os.path.join(explainer.output_dir, 'model_evaluation.png'))
    assert os.path.exists(os.path.join(explainer.output_dir, 'shap_heatmap.png'))
    assert os.path.exists(os.path.join(explainer.output_dir, 'shap_barplot.png'))

def test_explainer_from_trainer(trained_binary_model, tmpdir):
    trainer = trained_binary_model
    explainer = Explainer.from_trainer(trainer)
    assert explainer.trainer is trainer
    assert explainer.output_dir == trainer.output_dir
    assert explainer.X_train is trainer.X_train
    assert explainer.X_test is trainer.X_test
    assert explainer.y_test is trainer.y_test

# If a regression trainer is added, add regression-specific tests:
@pytest.fixture
def trained_regression_model(tmpdir):
    X = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'feature3': np.random.rand(100)
    })
    y = pd.Series(np.random.rand(100), name='target')
    data = pd.concat([X, y], axis=1)
    trainer = TrainerSupervised(task='regression', output_dir=str(tmpdir))
    trainer.run(data=data, target_variable='target', save_data=True)
    return trainer 

def test_explainer_run_regression(trained_regression_model, tmpdir):
    trainer = trained_regression_model
    explainer = Explainer.from_trainer(trainer)
    explainer.run()
    # Check if regression diagnostic plots are saved
    assert os.path.exists(os.path.join(explainer.output_dir, 'residual_plot.png'))
    assert os.path.exists(os.path.join(explainer.output_dir, 'true_vs_predicted.png'))
    assert os.path.exists(os.path.join(explainer.output_dir, 'feature_importance.png'))
