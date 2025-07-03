import logging
from pathlib import Path

import lightning.pytorch as pl
import optuna
import pandas as pd
from optuna.integration import PyTorchLightningPruningCallback
from torch.utils.data import DataLoader

from jarvais.loggers import logger

from .deepsurv import LitDeepSurv
from .utils import SurvivalDataset

# Suppress PyTorch Lightning logging
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.FATAL)


def train_deepsurv(data_train: pd.DataFrame, data_val:pd.DataFrame, output_dir: Path, random_seed=42):

    in_channel = len(data_train.columns) - 2 # -2 to remove time and event

    # Initialize datasets and dataloaders
    train_dataset = SurvivalDataset(data_train)
    val_dataset = SurvivalDataset(data_val)

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))

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

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=random_seed)
    pruner = optuna.pruners.MedianPruner() 
    study = optuna.create_study(direction="maximize", pruner=pruner, sampler=sampler)
    
    logger.info('Training DeepSurv...')
    study.optimize(objective, n_trials=100, timeout=600)
    trial = study.best_trial
    logger.info("Best trial: " + ", ".join(f"{key}: {value}" for key, value in trial.params.items()))

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

    return model