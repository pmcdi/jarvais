import pandas as pd
import pytest
from pathlib import Path

from jarvais.analyzer import Analyzer
from jarvais.trainer import TrainerSupervised


def get_analyzed_data(df, target, task, output_dir, encode):
    analyzer = Analyzer(
        df,
        output_dir=output_dir,
        categorical_columns= [
            "Sex",
            "T Stage",
            "N Stage",
            "Stage",
            "Smoking Status",
            "Disease Site",
            "HPV Combined",
            "Chemotherapy"
        ],
        continuous_columns=["time", "age at dx", "Dose"],
        target_variable=target,
        task=task,
    )
    analyzer.visualization_module.enabled = False
    analyzer.encoding_module.enabled = encode
    analyzer.run()
    return analyzer.data


def test_binary_trainer_runs(radcure_clinical, tmp_path):
    radcure_clinical.rename(columns={"survival_time": "time", "death": "event"}, inplace=True)

    output_dir = tmp_path / "binary"
    data = get_analyzed_data(
        radcure_clinical,
        target="event",
        task="binary",
        output_dir=output_dir / "analyzer",
        encode=False,
    )

    trainer = TrainerSupervised(
        output_dir=output_dir / "trainer",
        target_variable="event",
        task="binary",
        k_folds=2,
    )
    trainer.run(data)

    # Assertions
    assert trainer.X_train.shape[0] > 0
    assert trainer.X_test.shape[0] > 0
    assert trainer.X_val.shape[0] > 0
    assert set(trainer.model_names())

    # Load trainer and predict
    loaded_trainer = TrainerSupervised.load_trainer(output_dir / "trainer")
    preds = loaded_trainer.infer(loaded_trainer.X_val)
    assert len(preds) == len(loaded_trainer.X_val)


def test_regression_trainer_runs(radcure_clinical, tmp_path):
    radcure_clinical.rename(columns={"survival_time": "time", "death": "event"}, inplace=True)

    output_dir = tmp_path / "regression"
    data = get_analyzed_data(
        radcure_clinical,
        target="time",
        task="regression",
        output_dir=output_dir / "analyzer",
        encode=False,
    )

    trainer = TrainerSupervised(
        output_dir=output_dir / "trainer",
        target_variable="time",
        task="regression",
        k_folds=2,
    )
    trainer.run(data)

    # Assertions
    assert trainer.X_train.shape[0] > 0
    assert trainer.X_test.shape[0] > 0
    assert trainer.X_val.shape[0] > 0
    assert set(trainer.model_names())

    # Load trainer and predict
    loaded_trainer = TrainerSupervised.load_trainer(output_dir / "trainer")
    preds = loaded_trainer.infer(loaded_trainer.X_val)
    assert len(preds) == len(loaded_trainer.X_val)


def test_survival_trainer_runs(radcure_clinical, tmp_path):
    radcure_clinical.rename(columns={"survival_time": "time", "death": "event"}, inplace=True)

    output_dir = tmp_path / "survival"
    data = get_analyzed_data(
        radcure_clinical,
        target="event",
        task="survival",
        output_dir=output_dir / "analyzer",
        encode=True,
    )

    trainer = TrainerSupervised(
        output_dir=output_dir / "trainer",
        target_variable=["event", "time"],
        task="survival",
    )
    trainer.run(data)

    # Assertions
    assert trainer.X_train.shape[0] > 0
    assert trainer.X_test.shape[0] > 0
    assert trainer.X_val.shape[0] > 0
    assert set(trainer.model_names())

    # Load trainer and predict
    loaded_trainer = TrainerSupervised.load_trainer(output_dir / "trainer")
    preds = loaded_trainer.infer(loaded_trainer.X_val)
    assert len(preds) == len(loaded_trainer.X_val)

