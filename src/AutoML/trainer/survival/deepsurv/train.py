import logging
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

import lightning.pytorch as pl
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from .deepsurv import LitDeepSurv
from .utils import SurvivalDataset
from .utils import calculate_c_index

# Suppress PyTorch Lightning logging
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.FATAL)

def train_deepsurv(data_train: pd.DataFrame, data_val:pd.DataFrame, data_test: pd.DataFrame, output_dir: Path):

    in_channel = len(data_train.columns) - 2 # -2 to remove time and event

    # Initialize datasets and dataloaders
    train_dataset = SurvivalDataset(data_train)
    val_dataset = SurvivalDataset(data_val)
    test_dataset = SurvivalDataset(data_test)

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    def objective(trial: optuna.trial.Trial) -> float:
        l2_reg = trial.suggest_float("l2_reg", 1e-6, 1e-2)
        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        dims = trial.suggest_categorical("dims", [[2**n, 2**n, 2**n] for n in range(4, 9)])

        model = LitDeepSurv(in_channel=in_channel, dims=dims, dropout=dropout, l2_reg=l2_reg)

        trainer = pl.Trainer(
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            max_epochs=10,
            accelerator="auto",
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="valid_c_index")],
        )
        trainer.fit(model, train_loader, val_loader)

        return trainer.callback_metrics["valid_c_index"].item()
    
    pruner = optuna.pruners.MedianPruner() 

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    print('Training DeepSurv...')
    study.optimize(objective, n_trials=100, timeout=600)

    print("  Best trial:")
    trial = study.best_trial
    print("    Params: ")
    for key, value in trial.params.items():
        print("      {}: {}".format(key, value))

    model = LitDeepSurv(in_channel=in_channel, **trial.params)

    trainer = pl.Trainer(
        default_root_dir=output_dir,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        max_epochs=500,
        accelerator="auto",
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="valid_c_index", mode="max")
        ]
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint(output_dir / 'DeepSurv.ckpt')

    # Prediction on test data
    model.eval()
    c_index = 0
    with torch.no_grad():
        for X, y, e in test_loader:
            risk_pred = model(X)  # Get risk predictions
            c_index += calculate_c_index(-risk_pred, y, e)
        c_index /= len(test_loader)
    
    return model, c_index