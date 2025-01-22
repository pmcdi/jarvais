import pytest
import shutil
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification, make_regression
from jarvais.explainer import Explainer
from jarvais.trainer import TrainerSupervised

@pytest.fixture
def classification_data():
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=3, n_redundant=1, random_state=42,
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

@pytest.fixture
def trained_binary_model(classification_data, tmpdir):
    X, y = classification_data
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
    assert (explainer.output_dir / 'figures' / 'confusion_matrix.png').exists()
    assert (explainer.output_dir / 'figures' / 'feature_importance.png').exists()
    assert (explainer.output_dir / 'figures' / 'model_evaluation.png').exists()
    assert (explainer.output_dir / 'figures' / 'shap_heatmap.png').exists()
    assert (explainer.output_dir / 'figures' / 'shap_barplot.png').exists()

def test_explainer_from_trainer(trained_binary_model, tmpdir):
    trainer = trained_binary_model
    explainer = Explainer.from_trainer(trainer)
    assert explainer.trainer is trainer
    assert explainer.output_dir == trainer.output_dir
    assert explainer.X_train is trainer.X_train
    assert explainer.X_test is trainer.X_test
    assert explainer.y_test is trainer.y_test

@pytest.fixture
def trained_regression_model(regression_data, tmpdir):
    X, y = regression_data
    data = pd.concat([X, y], axis=1)
    trainer = TrainerSupervised(task='regression', output_dir=str(tmpdir))
    trainer.run(data=data, target_variable='target')
    return trainer 

def test_explainer_run_regression(trained_regression_model, tmpdir):
    trainer = trained_regression_model
    explainer = Explainer.from_trainer(trainer)
    explainer.run()
    # Check if regression diagnostic plots are saved
    assert (explainer.output_dir / 'figures' / 'residual_plot.png').exists()
    assert (explainer.output_dir / 'figures' / 'true_vs_predicted.png').exists()
    assert (explainer.output_dir / 'figures' / 'feature_importance.png').exists()

