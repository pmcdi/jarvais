import logging
from pathlib import Path

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

from .mtlr import mtlr_risk, LitMTLR
from .utils import encode_survival, make_time_bins, normalize
from lifelines.utils import concordance_index

import lightning.pytorch as pl
import optuna
from optuna.integration import PyTorchLightningPruningCallback

# Suppress PyTorch Lightning logging
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.FATAL)

def calculate_c_index(risk_pred, y, e):
    ''' Performs calculating c-index

    :param risk_pred: (np.ndarray or torch.Tensor) model prediction
    :param y: (np.ndarray or torch.Tensor) the times of event e
    :param e: (np.ndarray or torch.Tensor) flag that records whether the event occurs
    :return c_index: the c_index is calculated by (risk_pred, y, e)
    '''
    if not isinstance(y, np.ndarray):
        y = y.detach().cpu().numpy()
    if not isinstance(risk_pred, np.ndarray):
        risk_pred = risk_pred.detach().cpu().numpy()
    if not isinstance(e, np.ndarray):
        e = e.detach().cpu().numpy()
    return concordance_index(y, risk_pred, e)

def create_dataloader(data: pd.DataFrame, time_bins: torch.Tensor):
    X = torch.tensor(data.drop(["time", "event"], axis=1).values, dtype=torch.float)
    y = encode_survival(data["time"].values, data["event"].values, time_bins)

    return DataLoader(TensorDataset(X, y), batch_size=len(data), shuffle=True)

def train_mtlr(data_train: pd.DataFrame, data_val:pd.DataFrame, data_test: pd.DataFrame, output_dir: Path):

    in_channel = len(data_train.columns) - 2  # -2 to exclude "time" and "event"
    time_bins = make_time_bins(data_train["time"].values, event=data_train["event"].values)

    skip_cols = [
        col for col in data_train.columns if (set(data_train[col].unique()).issubset({0, 1}) or (col in ['time', 'event']))
    ]
    data_train, mean, std = normalize(data_train, skip_cols=skip_cols)
    data_val, _, _ = normalize(data_val, mean=mean, std=std, skip_cols=skip_cols)
    data_test, _, _ = normalize(data_test, mean=mean, std=std, skip_cols=skip_cols)

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

    pruner = optuna.pruners.MedianPruner()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", pruner=pruner)

    print('Training MTLR...')
    study.optimize(objective, n_trials=300, timeout=600)

    print("  Best trial:")
    trial = study.best_trial
    print("    Params: ")
    for key, value in trial.params.items():
        print("      {}: {}".format(key, value))

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

    # Evaluate on the test set
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(data_test.drop(["time", "event"], axis=1).values, dtype=torch.float))
        risk_pred = mtlr_risk(y_pred)
    c_index = calculate_c_index(-risk_pred, data_test['time'].values, data_test["event"].values)

    return model, c_index
