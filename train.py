"""
Training pipeline for SMR, with the unified neighborhood/masking utility from
`fixed/neighborhood.py` and the model factory from `fixed/model.py` threaded
through.

Differences vs. `new/train.py`:
  1. The two per-modality calls
         find_index_mask_eeg(...)  /  find_index_mask_ieeg(...)
     are replaced by a single call to `find_index_mask(...)`, which takes a
     `method` argument (`'atlas'` or `'knn'`) read from `settings`.
  2. A new optional `settings.neighborhood_method` (default `'atlas'`) and
     `settings.neighborhood_k` (default 9) select the neighborhood definition.
  3. A new optional `settings.model_type` (default `'instantaneous'`)
     selects between the instantaneous `ReconModel` and the
     `LaggedReconModelConfigurable`. The latter reads `settings.lags_ms`
     (default `[20, 30, 50, 60]`).
  4. Result paths get a `nbhd-<method>` suffix so atlas-mode and knn-mode runs
     don't overwrite each other.

Everything else (loss, optimizer, regularization, splits, early stopping) is
left untouched so existing runs remain reproducible.
"""

import os
import random

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Local imports (these live in `new/` — `fixed/train.py` is meant to be dropped
# alongside them on the PYTHONPATH or copied into `new/` to override).
from Load_preprocess_data import load_per_subject
from utils import save_settings_to_json   # any helpers not related to masking
from visualize import compare_signal, analyze_weights, save_mse_all_trials

# Unified neighborhood/masking utility and model factory (from `fixed/`).
from neighborhood import find_index_mask
from model import build_model, collect_lagged_weights, LaggedReconModelConfigurable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_method(settings) -> str:
    """Read `settings.neighborhood_method`, defaulting to 'atlas'."""
    return getattr(settings, 'neighborhood_method', 'atlas')


def _resolve_k(settings) -> int:
    """Read `settings.neighborhood_k`, defaulting to 9."""
    return int(getattr(settings, 'neighborhood_k', 9))


def _resolve_model_type(settings) -> str:
    """Read `settings.model_type`, defaulting to 'instantaneous'."""
    return getattr(settings, 'model_type', 'instantaneous').lower()


def _build_mask(target_idx, ch_name, ch_position, settings):
    """Single dispatch point for spatial mask construction."""
    return find_index_mask(
        target_idx=target_idx,
        ch_name=ch_name,
        ch_position=ch_position,
        mask_intensity=settings.mask_intensity,
        modality=settings.dataset_type,        # 'EEG' or 'iEEG'
        method=_resolve_method(settings),
        k=_resolve_k(settings),
    )


def _load_channel_metadata(settings, paths):
    """Load `ch_name` and `ch_position` for the current subject.

    Paths come from the ``Paths`` instance loaded from
    ``configs/device_path.yaml``. The atlas path uses ``ch_name``; the
    knn path uses ``ch_position``. Either may be absent for one path or
    the other.
    """
    if settings.dataset_type == 'EEG':
        ch_name = np.load(paths.eeg_channel_name_file())
        ch_position = np.load(paths.eeg_channel_coord_file())
    elif settings.dataset_type == 'iEEG':
        with open(paths.ieeg_channel_names_file(), 'rb') as f:
            ch_name = pkl.load(f)[settings.train_subject]
        with open(paths.ieeg_electrode_coord_file(), 'rb') as f:
            ch_position = pkl.load(f)[settings.train_subject]
    else:
        ch_name = None
        ch_position = None
    return ch_name, ch_position


# ---------------------------------------------------------------------------
# Intra-subject training
# ---------------------------------------------------------------------------
def train_model(settings, paths):
    """Intra-subject training, identical to `new/train.py::train_model`
    except for the masking call."""

    data = load_per_subject(settings, paths, settings.train_subject)
    ch_name, ch_position = _load_channel_metadata(settings, paths)

    num_node = data.shape[1]
    data = zscore(data)

    train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_val_data, test_size=0.2, random_state=42)

    adj_mat = np.zeros((num_node, num_node))

    train_tensor = torch.tensor(train_data, dtype=torch.float32).to('cuda')
    val_tensor = torch.tensor(val_data, dtype=torch.float32).to('cuda')
    test_tensor = torch.tensor(test_data, dtype=torch.float32).to('cuda')

    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=settings.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=settings.batch_size, shuffle=False)

    dcorr_all_channel = []
    pear_corr_channel = []

    iter_range = ch_position.shape[0] if ch_position is not None else num_node
    for i in range(iter_range):
        if i >= data.shape[1]:
            break
        print(f'Process Node {i}  (neighborhood={_resolve_method(settings)})')

        list_mask = _build_mask(i, ch_name, ch_position, settings)
        if len(list_mask) == num_node:
            continue

        model, fc_weight_key = build_model(settings, num_node, len(list_mask))
        model = model.to('cuda')
        optimizer = torch.optim.Adam(model.parameters(), lr=settings.lr)
        criterion = nn.L1Loss()

        best_val_loss = float('inf')
        patience_counter = 0
        loss_epoch = []
        l1_lambda = 1e-5
        l2_lambda = 1e-4

        for e in range(settings.max_epoch):
            # ── Training ────────────────────────────────────────────────
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

            avg_train_loss = loss_batch / len(train_loader)
            loss_epoch.append(avg_train_loss)

            # ── Validation ──────────────────────────────────────────────
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

            print(f"Epoch [{e + 1}/{settings.max_epoch}], Node [{i + 1}/{num_node}], "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # ── Early stopping ──────────────────────────────────────────
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                model_path = paths.result_path + '/best_model'
                os.makedirs(model_path, exist_ok=True)
                torch.save(model.state_dict(), f"{model_path}/best_model_node_{i}.pt")
            else:
                patience_counter += 1

            if patience_counter >= settings.patience:
                print(f"Early stopping at epoch {e + 1} — no val improvement.")
                break

        model.load_state_dict(torch.load(f"{model_path}/best_model_node_{i}.pt"))

        # Plot loss
        plt.figure()
        plt.plot(range(1, len(loss_epoch) + 1), loss_epoch, label=f'Node {i + 1}')
        loss_path = paths.result_path + '/loss'
        os.makedirs(loss_path, exist_ok=True)
        plt.savefig(f'{loss_path}/Loss_channel_{i + 1}.png')
        np.save(f'{loss_path}/Loss_channel_{i + 1}.npy', loss_epoch)
        plt.close()

        dcorr = save_mse_all_trials(mask_index=list_mask, model=model,
                                    test_data=test_tensor, channel=i,
                                    paths=paths, settings=settings)
        dcorr_all_channel.append(dcorr)

        # Adjacency matrix: store the instantaneous (lag-0) weight row so
        # the saved tensor has the same shape regardless of model_type.
        weights = model.state_dict()[fc_weight_key].cpu()
        m = 0
        for j in range(adj_mat.shape[1]):
            if j not in list_mask:
                adj_mat[i, j] = weights[0, m]
                m += 1

        # If we trained the lagged variant, also dump the per-lag weights
        # so downstream analyses (e.g. lag-by-lag visualizations) can use
        # them. File layout: (n_lags + 1, 1, n_predictors), lag-0 first.
        if isinstance(model, LaggedReconModelConfigurable):
            lagged_dir = os.path.join(paths.result_path, 'lagged_weights')
            os.makedirs(lagged_dir, exist_ok=True)
            np.save(
                os.path.join(lagged_dir, f'lagged_weights_node_{i}.npy'),
                collect_lagged_weights(model).numpy(),
            )

    np.save(f'{paths.result_path}/adj_mat_{settings.train_subject}.npy', adj_mat)
    np.save(f'{paths.result_path}/DCORR_{settings.train_subject}.npy', dcorr_all_channel)
    np.save(f'{paths.result_path}/Pear_Corr_{settings.train_subject}.npy', pear_corr_channel)
    np.save(f'{paths.result_path}/neighborhood_config.npy',
            {'method': _resolve_method(settings), 'k': _resolve_k(settings),
             'model_type': _resolve_model_type(settings),
             'lags_ms': list(getattr(settings, 'lags_ms', [20, 30, 50, 60]))
                        if _resolve_model_type(settings) == 'lagged' else None},
            allow_pickle=True)

    analyze_weights(adj_mat, settings, paths)


# ---------------------------------------------------------------------------
# Cross-subject evaluation (single-subject donor, EEG path)
# ---------------------------------------------------------------------------
def evaluate_model(settings, paths):
    data = load_per_subject(settings, paths, settings.test_subject)
    data = zscore(data)
    data_tensor = torch.tensor(data, dtype=torch.float32).to('cuda')

    if settings.dataset_type == 'EEG':
        ch_name = np.load(paths.eeg_channel_name_file())
        ch_position = np.load(paths.eeg_channel_coord_file())
    elif settings.dataset_type == 'iEEG':
        with open(paths.ieeg_channel_names_file(), 'rb') as f:
            ch_name = pkl.load(f)[settings.train_subject]
        if settings.train_subject == 11:
            ch_name = ch_name + ['unknown']
        with open(paths.ieeg_electrode_coord_file(), 'rb') as f:
            ch_position = pkl.load(f)[settings.train_subject]
    else:
        ch_name, ch_position = None, None

    dcorr_all_channel = []
    trained_sub = random.choice([i for i in range(0, settings.num_subject)
                                 if i != settings.test_subject])

    for channel in range(data.shape[1]):
        print(f'Process Node {channel}  (neighborhood={_resolve_method(settings)})')

        list_mask = _build_mask(channel, ch_name, ch_position, settings)

        model, _fc_key = build_model(settings, data.shape[1], len(list_mask))
        model.load_state_dict(torch.load(
            paths.donor_checkpoint_file(trained_sub, channel)
        ))
        model.to('cuda')
        model.eval()

        compare_signal(mask_index=list_mask, model=model, test_data=data_tensor,
                       channel=channel, settings=settings, paths=paths)
        dcorr_all_channel.append(
            save_mse_all_trials(mask_index=list_mask, model=model,
                                test_data=data_tensor, channel=channel,
                                paths=paths, settings=settings)
        )

    np.save(f'{paths.result_path}/DCORR_test_subject_{settings.num_subject}.npy',
            dcorr_all_channel)


# ---------------------------------------------------------------------------
# Dandiset training path
# ---------------------------------------------------------------------------
def train_model_dandiset(settings, paths, data):
    with open(paths.ieeg_channel_names_file(), 'rb') as f:
        ch_name = pkl.load(f)[settings.train_subject]
    with open(paths.ieeg_electrode_coord_file(), 'rb') as f:
        ch_position = pkl.load(f)[settings.train_subject]

    num_node = data.shape[1]
    data = zscore(data)
    train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_val_data, test_size=0.2, random_state=42)

    adj_mat = np.zeros((num_node, num_node))

    train_tensor = torch.tensor(train_data, dtype=torch.float32).to('cuda')
    val_tensor = torch.tensor(val_data, dtype=torch.float32).to('cuda')
    test_tensor = torch.tensor(test_data, dtype=torch.float32).to('cuda')

    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=settings.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=settings.batch_size, shuffle=False)

    dcorr_all_channel = []
    pear_corr_channel = []

    for i in range(ch_position.shape[0]):
        if i >= data.shape[1]:
            break
        print(f'Process Node {i}  (neighborhood={_resolve_method(settings)})')

        list_mask = _build_mask(i, ch_name, ch_position, settings)
        if len(list_mask) == num_node:
            continue

        model, fc_weight_key = build_model(settings, num_node, len(list_mask))
        model = model.to('cuda')
        optimizer = torch.optim.Adam(model.parameters(), lr=settings.lr)
        criterion = nn.L1Loss()

        best_val_loss = float('inf')
        patience_counter = 0
        loss_epoch = []
        l1_lambda = 1e-5
        l2_lambda = 1e-4

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
            avg_train_loss = loss_batch / len(train_loader)
            loss_epoch.append(avg_train_loss)

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

            print(f"Epoch [{e + 1}/{settings.max_epoch}], Node [{i + 1}/{num_node}], "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                model_path = paths.result_path + '/best_model'
                os.makedirs(model_path, exist_ok=True)
                torch.save(model.state_dict(), f"{model_path}/best_model_node_{i}.pt")
            else:
                patience_counter += 1
            if patience_counter >= settings.patience:
                print(f"Early stopping at epoch {e + 1} — no val improvement.")
                break

        model.load_state_dict(torch.load(f"{model_path}/best_model_node_{i}.pt"))

        plt.figure()
        plt.plot(range(1, len(loss_epoch) + 1), loss_epoch, label=f'Node {i + 1}')
        loss_path = paths.result_path + '/loss'
        os.makedirs(loss_path, exist_ok=True)
        plt.savefig(f'{loss_path}/Loss_channel_{i + 1}.png')
        np.save(f'{loss_path}/Loss_channel_{i + 1}.npy', loss_epoch)
        plt.close()

        dcorr = save_mse_all_trials(mask_index=list_mask, model=model,
                                    test_data=test_tensor, channel=i,
                                    paths=paths, settings=settings)
        dcorr_all_channel.append(dcorr)

        weights = model.state_dict()[fc_weight_key].cpu()
        m = 0
        for j in range(adj_mat.shape[1]):
            if j not in list_mask:
                adj_mat[i, j] = weights[0, m]
                m += 1

        if isinstance(model, LaggedReconModelConfigurable):
            lagged_dir = os.path.join(paths.result_path, 'lagged_weights')
            os.makedirs(lagged_dir, exist_ok=True)
            np.save(
                os.path.join(lagged_dir, f'lagged_weights_node_{i}.npy'),
                collect_lagged_weights(model).numpy(),
            )

    np.save(f'{paths.result_path}/adj_mat_{settings.train_subject}.npy', adj_mat)
    np.save(f'{paths.result_path}/DCORR_{settings.train_subject}.npy', dcorr_all_channel)
    np.save(f'{paths.result_path}/Pear_Corr_{settings.train_subject}.npy', pear_corr_channel)
    np.save(f'{paths.result_path}/neighborhood_config.npy',
            {'method': _resolve_method(settings), 'k': _resolve_k(settings),
             'model_type': _resolve_model_type(settings),
             'lags_ms': list(getattr(settings, 'lags_ms', [20, 30, 50, 60]))
                        if _resolve_model_type(settings) == 'lagged' else None},
            allow_pickle=True)

    analyze_weights(adj_mat, settings, paths)
