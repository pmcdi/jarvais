"""Unit tests for TrainerSupervised.run pre-split vs full ``data``."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from jarvais.trainer import TrainerSupervised
from jarvais.trainer.modules.autogluon_trainer import AutogluonTabularWrapper


def _regression_trainer(tmp_path) -> TrainerSupervised:
    return TrainerSupervised(
        output_dir=tmp_path,
        target_variable="y",
        task="regression",
        k_folds=2,
        reduction_method=None,
    )


def test_run_requires_data_or_pre_split(tmp_path):
    trainer = _regression_trainer(tmp_path)
    with pytest.raises(ValueError, match="Pass `data`, or both"):
        trainer.run()


def test_run_rejects_data_with_train_data(tmp_path):
    trainer = _regression_trainer(tmp_path)
    df = pd.DataFrame({"a": [1], "y": [0.0]})
    with pytest.raises(ValueError, match="either `data` or both"):
        trainer.run(data=df, train_data=df, test_data=df)


def test_run_rejects_data_with_test_data_only(tmp_path):
    trainer = _regression_trainer(tmp_path)
    df = pd.DataFrame({"a": [1], "y": [0.0]})
    t = pd.DataFrame({"a": [2], "y": [1.0]})
    with pytest.raises(ValueError, match="either `data` or both"):
        trainer.run(data=df, test_data=t)


def test_run_rejects_incomplete_pre_split(tmp_path):
    trainer = _regression_trainer(tmp_path)
    train = pd.DataFrame({"a": [1], "y": [0.0]})
    with pytest.raises(ValueError, match="both `train_data` and `test_data`"):
        trainer.run(train_data=train, test_data=None)
    with pytest.raises(ValueError, match="both `train_data` and `test_data`"):
        trainer.run(train_data=None, test_data=train)


def test_run_rejects_mismatched_columns(tmp_path):
    trainer = _regression_trainer(tmp_path)
    train = pd.DataFrame({"a": [1], "y": [0.0]})
    test = pd.DataFrame({"b": [2], "y": [1.0]})
    with pytest.raises(ValueError, match="same columns"):
        trainer.run(train_data=train, test_data=test)

    test_reordered = pd.DataFrame({"y": [1.0], "a": [2.0]})
    with pytest.raises(ValueError, match="same column"):
        trainer.run(train_data=train, test_data=test_reordered)


def test_pre_split_row_counts_match_concat_order(tmp_path):
    """After concat, first n_train rows are train; empty val leaves train size n_train."""
    train = pd.DataFrame({"a": [1.0, 2.0], "y": [0.0, 1.0]})
    test = pd.DataFrame({"a": [3.0], "y": [2.0]})
    trainer = _regression_trainer(tmp_path)
    # Pydantic models do not allow assigning a mock to ``trainer.trainer_module.fit``; patch the class method.
    with patch.object(
        AutogluonTabularWrapper,
        "fit",
        return_value=(MagicMock(), pd.DataFrame(), pd.Series(dtype="float64")),
    ):
        trainer.run(train_data=train, test_data=test)
    assert len(trainer.input_data) == 3
    assert trainer.X_test.shape[0] == 1
    assert trainer.y_test.shape[0] == 1
    # No val holdout: training set stays two rows
    assert trainer.X_train.shape[0] == 2
    assert trainer.y_train.shape[0] == 2
