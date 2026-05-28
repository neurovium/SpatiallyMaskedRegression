"""
Core PyTorch modules for SMR signal reconstruction.

This module exposes two model classes that are used by the training
pipeline:

    * ReconModel        -- instantaneous (lag-0) baseline.
    * LaggedReconModel  -- multi-lag variant with one linear branch per lag
                           plus an instantaneous main branch.

Both modules accept a ``mask_index`` argument at forward time to drop the
masked channels from the input before the linear projection.

The configurable lagged variant used by the training pipeline lives in
``fixed/model.py`` (`LaggedReconModelConfigurable`); it has the same
semantics as ``LaggedReconModel`` but accepts the lag set via constructor.
"""

import torch
import torch.nn as nn


class LaggedReconModel(nn.Module):
    """Multi-lag reconstruction model with one Linear per lag.

    Parameters
    ----------
    n : int
        Total number of channels.
    del_size : int
        Number of channels masked out.
    fs : float
        Sampling frequency in Hz.
    dropout_p : float
        Dropout probability (defined but not currently applied in forward,
        kept for compatibility with checkpoints from older runs).
    """

    def __init__(self, n, del_size, fs, dropout_p=0.5):
        super().__init__()
        self.fs = fs
        self.lags = [20, 30, 50, 60]   # ms

        self.fc_main = nn.Linear(n - del_size, 1)
        self.fc_lags = nn.ModuleDict({
            str(lag): nn.Linear(n - del_size, 1) for lag in self.lags
        })
        self.dropout = nn.Dropout(p=dropout_p)

    def make_lagged(self, d, lag):
        x_lagged = torch.zeros_like(d)
        sample_lag = int((lag / 1000) * self.fs)
        if sample_lag > 0:
            x_lagged[:, :, sample_lag:] = d[:, :, :-sample_lag]
        x_lagged = x_lagged.permute(0, 2, 1)
        return self.fc_lags[str(lag)](x_lagged)

    def forward(self, data, mask_index):
        keep_indices = torch.tensor(
            [i for i in range(data.size(1)) if i not in mask_index]
        )
        x = data[:, keep_indices, :]
        x = x.permute(0, 2, 1)

        y = self.fc_main(x)
        for lag in self.lags:
            y = y + self.make_lagged(data[:, keep_indices, :], lag=lag)

        return y.permute(0, 2, 1)


class ReconModel(nn.Module):
    """Instantaneous (lag-0) reconstruction baseline."""

    def __init__(self, n, del_size):
        super().__init__()
        self.fc = nn.Linear(n - del_size, 1)

    def forward(self, data, mask_index):
        keep_indices = torch.tensor(
            [i for i in range(data.size(1)) if i not in mask_index]
        )
        x = data[:, keep_indices, :]
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        x = x.permute(0, 2, 1)
        return x
