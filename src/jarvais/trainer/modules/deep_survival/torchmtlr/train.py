import logging
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import optuna
import pandas as pd
import torch
from optuna.integration import PyTorchLightningPruningCallback
from torch.utils.data import DataLoader, TensorDataset

from jarvais.loggers import logger

from .mtlr import LitMTLR
from .utils import encode_survival, make_time_bins, normalize


# Suppress PyTorch Lightning logging
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.FATAL)


def create_dataloader(data: pd.DataFrame, time_bins: torch.Tensor):
    X = torch.tensor(data.drop(["time", "event"], axis=1).values, dtype=torch.float)
    y = encode_survival(data["time"].values, data["event"].values, time_bins)

    return DataLoader(TensorDataset(X, y), batch_size=len(data), shuffle=True)

def train_mtlr(data_train: pd.DataFrame, data_val:pd.DataFrame, output_dir: Path, random_seed=42):

    in_channel = len(data_train.columns) - 2  # -2 to exclude "time" and "event"
    time_bins = make_time_bins(data_train["time"].values, event=data_train["event"].values)

    skip_cols = [
        col for col in data_train.columns if (set(data_train[col].unique()).issubset({0, 1}) or (col in ['time', 'event']))
    ]
    data_train, mean, std = normalize(data_train, skip_cols=skip_cols)
    data_val, _, _ = normalize(data_val, mean=mean, std=std, skip_cols=skip_cols)

    train_loader = create_dataloader(data_train, time_bins)
    val_loader = create_dataloader(data_val, time_bins)

    def objective(trial: optuna.trial.Trial) -> float:
        C1 = trial.suggest_categorical("C1", [c1 for c1 in np.logspace(-2, 3, 6)])
        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        dims = trial.suggest_categorical("dims", [[2**n, 2**n] for n in range(4, 10)])

        model = LitMTLR(in_channel=in_channel, num_time_bins=len(time_bins), dims=dims, dropout=dropout, C1=C1)

        trainer = pl.Trainer(
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            max_epochs=10,
            accelerator="auto",
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="valid_loss")],
        )
        trainer.fit(model, train_loader, val_loader)

        return trainer.callback_metrics["valid_loss"].item()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pruner = optuna.pruners.MedianPruner()
    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction="minimize", pruner=pruner, sampler=sampler)

    logger.info('Training MTLR...')
    study.optimize(objective, n_trials=300, timeout=600)

    trial = study.best_trial
    logger.info("Best trial: " + ", ".join(f"{key}: {value}" for key, value in trial.params.items()))

    # Train the final model with the best parameters
    model = LitMTLR(in_channel=in_channel, num_time_bins=len(time_bins), mean=mean, std=std, **trial.params)

    trainer = pl.Trainer(
        default_root_dir=output_dir,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        max_epochs=500,
        accelerator="auto",
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="valid_loss", mode="min")
        ]
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint(output_dir / 'MTLR.ckpt')

    return model
