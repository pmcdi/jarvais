# ------------------------------------------------------------------------------
# Code adapted from:
# https://github.com/czifan/DeepSurv.pytorch
# ------------------------------------------------------------------------------

from typing import List

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import lightning.pytorch as pl

from .utils import calculate_c_index

class Regularization(object):
    def __init__(self, order: int, weight_decay: float):
        ''' The initialization of Regularization class

        :param order: (int) norm order number
        :param weight_decay: (float) weight decay rate
        '''
        super(Regularization, self).__init__()
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model: torch.nn.Module):
        ''' Performs calculates regularization(self.order) loss for model.

        :param model: (torch.nn.Module object)
        :return reg_loss: (torch.Tensor) the regularization(self.order) loss
        '''
        reg_loss = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        reg_loss = self.weight_decay * reg_loss
        return reg_loss

class DeepSurv(nn.Module):
    ''' The module class performs building network according to config'''
    def __init__(self, in_channel: int, dims: List[int], dropout: float, norm: bool = True):
        
        super(DeepSurv, self).__init__()
        self.dropout = dropout
        self.norm = norm
        self.dims = [in_channel] + dims + [1]
        self.model = self._build_network()

    def _build_network(self):
        ''' Performs building networks according to parameters'''
        layers = []
        for i in range(len(self.dims)-1):
            if i and self.dropout is not None: # adds dropout layer
                layers.append(nn.Dropout(self.dropout))
            # adds linear layer
            layers.append(nn.Linear(self.dims[i], self.dims[i+1]))
            if self.norm: # adds batchnormalize layer
                layers.append(nn.BatchNorm1d(self.dims[i+1]))
            # adds activation layer
            layers.append(nn.ReLU())
        # builds sequential network
        return nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)

class NegativeLogLikelihood(nn.Module):
    def __init__(self, l2_reg):
        super(NegativeLogLikelihood, self).__init__()
        self.L2_reg = l2_reg
        self.reg = Regularization(order=2, weight_decay=self.L2_reg)

    def forward(self, risk_pred, y, e, model):
        mask = torch.ones(y.shape[0], y.shape[0])
        mask[(y.T - y) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred-log_loss) * e) / torch.sum(e)
        l2_loss = self.reg(model)
        return neg_log_loss + l2_loss
    
class LitDeepSurv(pl.LightningModule):
    def __init__(self, in_channel: int, dims: List[int], dropout: float, l2_reg: float):
        super(LitDeepSurv, self).__init__()
        self.save_hyperparameters() 
        
        self.model = DeepSurv(in_channel, dims=dims, dropout=dropout)
        self.criterion = NegativeLogLikelihood(l2_reg)
        self.best_c_index = 0

    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        self.model.eval()
        
        with torch.no_grad():
            x = x.drop(columns=['time', 'event'], errors='ignore').to_numpy(dtype=np.float32)
            
            # Normalize input
            min_vals = x.min(axis=0)
            range_vals = x.max(axis=0) - min_vals
            range_vals[range_vals == 0] = 1  # Prevent division by zero
            
            x = (x - min_vals) / range_vals
            x = torch.from_numpy(x)
            
            y_pred = self.model(x).ravel()

        return y_pred

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.model.parameters())

    def training_step(self, batch, _):
        X, y, e = batch
        risk_pred = self(X)
        train_loss = self.criterion(risk_pred, y, e, self.model)
        train_c = calculate_c_index(-risk_pred, y, e)

        self.log('train_loss', train_loss, prog_bar=True)
        self.log('train_c_index', train_c, prog_bar=True)
        return train_loss

    def validation_step(self, batch, _):
        X, y, e = batch
        risk_pred = self(X)
        valid_loss = self.criterion(risk_pred, y, e, self.model)
        valid_c = calculate_c_index(-risk_pred, y, e)

        self.log('valid_loss', valid_loss, prog_bar=True)
        self.log('valid_c_index', valid_c, prog_bar=True)

        return valid_loss