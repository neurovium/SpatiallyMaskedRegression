"""
Driver for the *Local / Non-Local / All* coverage experiment (Figure 6
in the paper).

This script trains the SMR model three times per subject under three
different masking schemes:

  * ``local``     -- only the local neighborhood ``N(i)`` is available
                     as predictors of electrode ``i``. Every other
                     channel is masked.
  * ``non_local`` -- ``N(i)`` is masked. Only non-local channels are
                     available. (Equivalent to mask_intensity = 1.0 in
                     the standard sweep.)
  * ``all``       -- only the target electrode is masked. Every other
                     channel is available. (Equivalent to
                     mask_intensity = 0.0 in the standard sweep.)

Outputs are written to::

    <results_dir>/coverage/<dataset_type>/sub_<subject>/
        nbhd-<method>/model-<type>/cond-<local|non_local|all>/

with the same files the standard pipeline writes (``DCORR_<subject>.npy``,
``adj_mat_<subject>.npy``, ``best_model/``, etc.). The notebook
``fixed/notebooks/visualization.ipynb`` reads these to produce Figure 6.

Usage
-----
    python run_coverage_experiment.py

The dataset, subject count, neighborhood method, and model type are
read from ``configs/settings.yaml`` exactly as in ``main.py``.
"""

from __future__ import annotations

import os
import pickle as pkl
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from Load_preprocess_data import load_per_subject
from model import build_model, collect_lagged_weights, LaggedReconModelConfigurable
from neighborhood import find_neighborhood
from notebooks._helpers import coverage_masks  # noqa: E402  (after sys.path tweak in helper)
from path import Paths
from setting import Settings
from utils import save_settings_to_json
from visualize import analyze_weights, save_mse_all_trials


CONDITIONS = ('local', 'non_local', 'all')


def _coverage_result_dir(paths: Paths, settings: Settings, condition: str) -> str:
    """Build the per-condition output directory."""
    nbhd = getattr(settings, 'neighborhood_method', 'atlas')
    mtype = getattr(settings, 'model_type', 'instantaneous')
    rel = os.path.join(
        'coverage', settings.dataset_type,
        f'sub_{settings.train_subject}',
        f'nbhd-{nbhd}', f'model-{mtype}', f'cond-{condition}',
    )
    out = os.path.join(paths.results_dir, rel) + os.sep
    Path(out).mkdir(parents=True, exist_ok=True)
    return out


def _load_channel_metadata(settings: Settings, paths: Paths):
    if settings.dataset_type == 'EEG':
        ch_name = np.load(paths.eeg_channel_name_file())
        ch_position = np.load(paths.eeg_channel_coord_file())
    elif settings.dataset_type == 'iEEG':
        with open(paths.ieeg_channel_names_file(), 'rb') as f:
            ch_name = pkl.load(f)[settings.train_subject]
        with open(paths.ieeg_electrode_coord_file(), 'rb') as f:
            ch_position = pkl.load(f)[settings.train_subject]
    else:
        ch_name, ch_position = None, None
    return ch_name, ch_position


def _train_one_subject_condition(settings: Settings, paths: Paths,
                                 condition: str) -> None:
    """Run a single subject × condition training pass."""
    out_dir = _coverage_result_dir(paths, settings, condition)
    print(f"\n[coverage] sub={settings.train_subject} cond={condition} -> {out_dir}")

    data = load_per_subject(settings, paths, settings.train_subject)
    ch_name, ch_position = _load_channel_metadata(settings, paths)

    num_node = data.shape[1]
    data = zscore(data)

    train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_val_data, test_size=0.2, random_state=42)

    train_tensor = torch.tensor(train_data, dtype=torch.float32).to('cuda')
    val_tensor = torch.tensor(val_data, dtype=torch.float32).to('cuda')
    test_tensor = torch.tensor(test_data, dtype=torch.float32).to('cuda')

    train_loader = DataLoader(TensorDataset(train_tensor),
                              batch_size=settings.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor),
                            batch_size=settings.batch_size, shuffle=False)

    adj_mat = np.zeros((num_node, num_node))
    dcorr_all_channel = []

    method = getattr(settings, 'neighborhood_method', 'atlas')
    k = int(getattr(settings, 'neighborhood_k', 9))

    iter_range = ch_position.shape[0] if ch_position is not None else num_node
    for i in range(iter_range):
        if i >= data.shape[1]:
            break

        masks = coverage_masks(
            i, ch_name=ch_name, ch_position=ch_position,
            modality=settings.dataset_type, method=method, k=k,
        )
        list_mask = masks[condition]
        if len(list_mask) >= num_node:
            print(f"  channel {i}: mask covers all channels; skipping.")
            continue
        if len(list_mask) == 0:
            print(f"  channel {i}: empty mask; skipping.")
            continue

        model, fc_weight_key = build_model(settings, num_node, len(list_mask))
        model = model.to('cuda')
        optimizer = torch.optim.Adam(model.parameters(), lr=settings.lr)
        criterion = nn.L1Loss()

        best_val_loss = float('inf')
        patience_counter = 0
        l1_lambda = 1e-5
        l2_lambda = 1e-4
        model_path = os.path.join(out_dir, 'best_model')
        os.makedirs(model_path, exist_ok=True)

        for e in range(settings.max_epoch):
            model.train()
            loss_batch = 0
            for batch in train_loader:
                batch_data = batch[0]
                optimizer.zero_grad()
                output = model(batch_data, mask_index=list_mask)
                output = torch.squeeze(output, 1)
                target = batch_data[:, i, :]
                loss = criterion(output, target)
                l1_penalty = sum(torch.sum(torch.abs(p)) for p in model.parameters())
                l2_penalty = sum(torch.sum(p ** 2) for p in model.parameters())
                loss = loss + l1_lambda * l1_penalty + l2_lambda * l2_penalty
                loss.backward()
                optimizer.step()
                loss_batch += loss.item()

            model.eval()
            val_loss_batch = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch_data = batch[0]
                    output = model(batch_data, mask_index=list_mask)
                    output = torch.squeeze(output, 1)
                    target = batch_data[:, i, :]
                    val_loss_batch += criterion(output, target).item()
            avg_val_loss = val_loss_batch / len(val_loader)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(),
                           os.path.join(model_path, f'best_model_node_{i}.pt'))
            else:
                patience_counter += 1
            if patience_counter >= settings.patience:
                break

        model.load_state_dict(torch.load(
            os.path.join(model_path, f'best_model_node_{i}.pt')))

        # Use the shared visualize.save_mse_all_trials helper -- we need
        # to give it a paths-like object pointing at the per-condition
        # output dir for plot files.
        class _LocalPaths:
            pass
        _lp = _LocalPaths()
        _lp.result_path = out_dir

        dcorr = save_mse_all_trials(
            mask_index=list_mask, model=model, test_data=test_tensor,
            channel=i, paths=_lp, settings=settings,
        )
        dcorr_all_channel.append(dcorr)

        weights = model.state_dict()[fc_weight_key].cpu()
        m = 0
        for j in range(adj_mat.shape[1]):
            if j not in list_mask:
                adj_mat[i, j] = weights[0, m]
                m += 1

        if isinstance(model, LaggedReconModelConfigurable):
            lagged_dir = os.path.join(out_dir, 'lagged_weights')
            os.makedirs(lagged_dir, exist_ok=True)
            np.save(
                os.path.join(lagged_dir, f'lagged_weights_node_{i}.npy'),
                collect_lagged_weights(model).numpy(),
            )

    np.save(os.path.join(out_dir, f'adj_mat_{settings.train_subject}.npy'), adj_mat)
    np.save(os.path.join(out_dir, f'DCORR_{settings.train_subject}.npy'),
            np.asarray(dcorr_all_channel, dtype=object), allow_pickle=True)
    np.save(os.path.join(out_dir, 'condition.npy'), condition)

    # Reuse the shared weight visualiser.
    class _LocalPaths2:
        pass
    _lp2 = _LocalPaths2()
    _lp2.result_path = out_dir
    analyze_weights(adj_mat, settings, _lp2)


def main():
    settings = Settings()
    settings.load_settings()
    paths = Paths(settings)
    paths.load_device_paths()

    # Stash the settings JSON once at the top of the coverage tree.
    coverage_root = os.path.join(paths.results_dir, 'coverage',
                                 settings.dataset_type)
    Path(coverage_root).mkdir(parents=True, exist_ok=True)
    save_settings_to_json(settings, file_path=coverage_root + os.sep)

    for sub in range(settings.num_subject):
        settings.train_subject = sub
        for condition in CONDITIONS:
            _train_one_subject_condition(settings, paths, condition)


if __name__ == '__main__':
    main()
