import pytest
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from jarvais.explainer import Explainer
from jarvais.trainer import TrainerSupervised

@pytest.fixture
def classification_trainer(tmp_path):
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=3, n_redundant=1, random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"feature{i}" for i in range(1, X.shape[1] + 1)])
    y = pd.Series(y, name="target")

    data = pd.concat([X, y], axis=1)

    trainer = TrainerSupervised(
        output_dir=tmp_path / 'classification_trainer',
        target_variable='target',
        task='binary'
    )
    trainer.run(data)
    return trainer

@pytest.fixture
def regression_trainer(tmp_path):
    X, y = make_regression(
        n_samples=50, n_features=5, noise=0.1, random_state=42
    )
    X = pd.DataFrame(X, columns=[f"feature{i}" for i in range(1, X.shape[1] + 1)])
    y = pd.Series(y, name="target")
    
    data = pd.concat([X, y], axis=1)

    trainer = TrainerSupervised(
        output_dir=tmp_path / 'regression_trainer',
        target_variable='target',
        task='regression'
    )
    trainer.run(data)
    return trainer

def test_classification_explainer(classification_trainer, tmp_path):

    explainer_dir = tmp_path / 'classification_explainer'
    
    explainer = Explainer(
        output_dir=explainer_dir,
    )

    explainer.run(classification_trainer)

    assert (explainer_dir / 'figures' / 'test_metrics_bootstrap.png').exists()
    assert (explainer_dir / 'figures' / 'validation_metrics_bootstrap.png').exists()
    assert (explainer_dir / 'figures' / 'train_metrics_bootstrap.png').exists()

    assert (explainer_dir / 'figures' / 'confusion_matrix.png').exists()
    assert (explainer_dir / 'figures' / 'feature_importance.png').exists()
    assert (explainer_dir / 'figures' / 'shap_heatmap.png').exists()
    assert (explainer_dir / 'figures' / 'shap_barplot.png').exists()

    assert (explainer_dir / 'figures' / 'calibration_curve.png').exists()
    assert (explainer_dir / 'figures' / 'roc_curve.png').exists()
    assert (explainer_dir / 'figures' / 'precision_recall_curve.png').exists()
    assert (explainer_dir / 'figures' / 'sensitivity_flag_curve.png').exists()
    assert (explainer_dir / 'figures' / 'sensitivity_specificity_ppv_by_threshold.png').exists()
    assert (explainer_dir / 'figures' / 'histogram_of_predicted_probabilities.png').exists()

def test_regression_explainer(regression_trainer, tmp_path):

    explainer_dir = tmp_path / 'regression_explainer'

    explainer = Explainer(
        output_dir=explainer_dir,
    )
    explainer.run(regression_trainer)
    
    assert (explainer_dir / 'figures' / 'test_metrics_bootstrap.png').exists()
    assert (explainer_dir / 'figures' / 'validation_metrics_bootstrap.png').exists()
    assert (explainer_dir / 'figures' / 'train_metrics_bootstrap.png').exists()

    assert (explainer_dir / 'figures' / 'feature_importance.png').exists()
    assert (explainer_dir / 'figures' / 'residual_plot.png').exists()
    assert (explainer_dir / 'figures' / 'true_vs_predicted.png').exists()
    assert (explainer_dir / 'figures' / 'residual_hist.png').exists()
    

