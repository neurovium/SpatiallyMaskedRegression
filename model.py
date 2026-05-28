"""
Model factory + configurable lagged variant.

The instantaneous ``ReconModel`` is reused as-is from ``new/ReconModel.py``.
``LaggedReconModelConfigurable`` is a small wrapper around the lagged model
that exposes the lag set as a constructor argument, so the manuscript's
``T = {20, 30, 50, 60} ms`` choice is just one option rather than being
hard-coded.

The ``build_model`` factory dispatches on ``settings.model_type``:

    'instantaneous' (default)  ->  ReconModel
    'lagged'                   ->  LaggedReconModelConfigurable

It also returns the key under which the *instantaneous-equivalent* weight
matrix lives in the state dict, so the downstream code in
``fixed/train.py`` can record a 2-D adjacency for both variants without
branching.

Both keys point to the same tensor shape ``(1, num_node - del_size)``:
    * 'fc.weight'      for ReconModel
    * 'fc_main.weight' for LaggedReconModelConfigurable (the lag-0 path)
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from ReconModel import ReconModel, LaggedReconModel


# ---------------------------------------------------------------------------
# Configurable lagged model
# ---------------------------------------------------------------------------
class LaggedReconModelConfigurable(nn.Module):
    """Lagged reconstruction model with a user-supplied lag set.

    Differs from ``new/ReconModel.LaggedReconModel`` only in that
    ``self.lags`` is set from the constructor argument instead of being
    hard-coded. Forward semantics are identical.

    Parameters
    ----------
    n : int
        Total number of channels.
    del_size : int
        Number of channels masked out.
    fs : float
        Sampling frequency in Hz.
    lags_ms : list of int
        Lag set in milliseconds. The instantaneous (lag-0) branch is
        always present via ``fc_main`` and is *not* included in this list.
    dropout_p : float
        Dropout probability (defined but not currently applied in forward,
        matching ``new/`` behaviour).
    """

    def __init__(self,
                 n: int,
                 del_size: int,
                 fs: float,
                 lags_ms: List[int] | None = None,
                 dropout_p: float = 0.5):
        super().__init__()
        self.fs = fs
        self.lags = list(lags_ms) if lags_ms is not None else [20, 30, 50, 60]

        self.fc_main = nn.Linear(n - del_size, 1)
        self.fc_lags = nn.ModuleDict({
            str(lag): nn.Linear(n - del_size, 1) for lag in self.lags
        })
        self.dropout = nn.Dropout(p=dropout_p)

    def make_lagged(self, d: torch.Tensor, lag: int) -> torch.Tensor:
        x_lagged = torch.zeros_like(d)
        sample_lag = int((lag / 1000) * self.fs)
        if sample_lag > 0:
            x_lagged[:, :, sample_lag:] = d[:, :, :-sample_lag]
        x_lagged = x_lagged.permute(0, 2, 1)             # (batch, time, channels)
        return self.fc_lags[str(lag)](x_lagged)

    def forward(self, data: torch.Tensor, mask_index) -> torch.Tensor:
        keep_indices = torch.tensor(
            [i for i in range(data.size(1)) if i not in mask_index]
        )
        x = data[:, keep_indices, :]
        x = x.permute(0, 2, 1)                            # (batch, time, channels)

        y = self.fc_main(x)                               # lag = 0 contribution
        for lag in self.lags:
            y = y + self.make_lagged(data[:, keep_indices, :], lag=lag)

        return y.permute(0, 2, 1)                         # (batch, 1, time)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_model(settings,
                num_node: int,
                del_size: int) -> Tuple[nn.Module, str]:
    """Return ``(model, fc_weight_key)`` for the model selected by settings.

    Parameters
    ----------
    settings : Settings
        Must expose ``model_type`` ('instantaneous' or 'lagged'). For the
        lagged variant it must also expose ``fs`` (sampling frequency) and
        optionally ``lags_ms`` (the lag set in milliseconds, default
        ``[20, 30, 50, 60]``).
    num_node : int
        Total number of channels.
    del_size : int
        Number of channels in the spatial mask (excluded predictors).

    Returns
    -------
    model : nn.Module
        The constructed model.
    fc_weight_key : str
        Key into ``model.state_dict()`` for the lag-0 / instantaneous
        weight matrix used to fill the adjacency matrix.
    """
    model_type = getattr(settings, 'model_type', 'instantaneous').lower()

    if model_type == 'instantaneous':
        return ReconModel(num_node, del_size), 'fc.weight'

    if model_type == 'lagged':
        fs = getattr(settings, 'fs', None)
        if fs is None:
            raise ValueError("settings.fs is required for model_type='lagged'.")
        lags_ms = list(getattr(settings, 'lags_ms', [20, 30, 50, 60]))
        dropout_p = float(getattr(settings, 'dropout_p', 0.5))
        model = LaggedReconModelConfigurable(
            n=num_node, del_size=del_size, fs=fs,
            lags_ms=lags_ms, dropout_p=dropout_p,
        )
        return model, 'fc_main.weight'

    raise ValueError(
        f"Unknown settings.model_type={model_type!r}; "
        "expected 'instantaneous' or 'lagged'."
    )


def collect_lagged_weights(model: LaggedReconModelConfigurable) -> torch.Tensor:
    """Return per-lag weights stacked into a tensor of shape
    ``(len(lags)+1, 1, num_predictors)``, with lag-0 in slot 0.

    Useful for saving the full lag tensor alongside the 2-D adjacency
    matrix; callers can pickle it next to ``adj_mat_*.npy``.
    """
    if not isinstance(model, LaggedReconModelConfigurable):
        raise TypeError("collect_lagged_weights expects LaggedReconModelConfigurable.")

    parts = [model.fc_main.weight.detach().cpu()]
    parts.extend(model.fc_lags[str(lag)].weight.detach().cpu()
                 for lag in model.lags)
    return torch.stack(parts, dim=0)
