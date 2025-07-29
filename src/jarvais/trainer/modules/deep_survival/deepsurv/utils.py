# ------------------------------------------------------------------------------
# Code adapted from:
# https://github.com/czifan/DeepSurv.pytorch
# ------------------------------------------------------------------------------

import numpy as np
import torch
from sksurv.metrics import concordance_index_censored
from torch.utils.data import Dataset


def calculate_c_index(risk_pred, y, e):
    '''
    Calculates the concordance index using scikit-survival.

    :param risk_pred: (np.ndarray or torch.Tensor) model prediction (higher = more risk)
    :param y: (np.ndarray or torch.Tensor) time-to-event
    :param e: (np.ndarray or torch.Tensor) event occurred (1) or censored (0)
    :return: c-index score
    '''

    # Convert to NumPy if needed
    if not isinstance(risk_pred, np.ndarray):
        risk_pred = risk_pred.detach().cpu().numpy()
    if not isinstance(y, np.ndarray):
        y = y.detach().cpu().numpy()
    if not isinstance(e, np.ndarray):
        e = e.detach().cpu().numpy()

    # Flatten and ensure correct types
    risk_pred = np.ravel(risk_pred).astype(float)
    y = np.ravel(y).astype(float)
    e = np.ravel(e).astype(bool)  # Must be boolean for sksurv

    return concordance_index_censored(e, y, risk_pred)[0]


class SurvivalDataset(Dataset):
    def __init__(self, data):

        self.X = data.drop(columns=['time', 'event']).to_numpy(dtype=np.float32)
        self.e = data['event'].to_numpy(dtype=np.float32).reshape(-1, 1)
        self.y = data['time'].to_numpy(dtype=np.float32).reshape(-1, 1)

        self._normalize()

    def _normalize(self):
        '''Performs normalizing X data.'''
        range_values = self.X.max(axis=0) - self.X.min(axis=0)
        range_values[range_values == 0] = 1  # Prevent division by zero
        self.X = (self.X - self.X.min(axis=0)) / range_values

    def __getitem__(self, item):
        ''' Performs constructing torch.Tensor object'''
        # gets data with index of item
        X_item = self.X[item] # (m)
        e_item = self.e[item] # (1)
        y_item = self.y[item] # (1)
        # constructs torch.Tensor object
        X_tensor = torch.from_numpy(X_item)
        e_tensor = torch.from_numpy(e_item)
        y_tensor = torch.from_numpy(y_item)
        return X_tensor, y_tensor, e_tensor

    def __len__(self):
        return self.X.shape[0]