"""
Visualization and reconstruction-quality helpers.

This is a straight port of ``new/visualize.py`` with the same public API
(``compare_signal``, ``analyze_weights``, ``save_mse_all_trials``,
``distance_correlation``, etc.) so the rest of ``fixed/`` can import
from it unchanged.
"""

from __future__ import annotations

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Adjacency-matrix outputs
# ---------------------------------------------------------------------------
def save_result_weights(adj_mat, ch_name, settings, paths):
    """Save a CSV of each channel's top-5 contributing channels by weight."""
    ranked_channels = []
    for row_idx, row in enumerate(adj_mat):
        top_5_indices = np.argsort(row)[-5:][::-1]
        top_5_channel_names = [ch_name[i] for i in top_5_indices]
        ranked_channels.append((ch_name[row_idx], top_5_channel_names))

    df = pd.DataFrame(ranked_channels, columns=["Row", "Top 5 Channels"])
    csv_filename = os.path.join(paths.result_path, 'ranked_channels_with_names.csv')
    df.to_csv(csv_filename, index=False)
    print(f"Top 5 ranked channels saved to {csv_filename}")


def analyze_weights(adj_mat, settings, paths):
    """Stem-plot the row of weights for each target channel; save .png + .npy."""
    weights_path = os.path.join(paths.result_path, 'weights')
    os.makedirs(weights_path, exist_ok=True)

    for i in range(adj_mat.shape[0]):
        fig, ax = plt.subplots(dpi=300)
        ax.stem(adj_mat[i, :])
        ax.set_title(f'Channel {i + 1}')
        ax.set_xlabel('Channel')
        ax.set_ylabel('Weight')
        fig.savefig(os.path.join(weights_path, f'channel{i + 1}.png'))
        np.save(os.path.join(weights_path, f'channel{i + 1}.npy'), adj_mat[i, :])
        plt.close(fig)


# ---------------------------------------------------------------------------
# Distance correlation (CPU numpy)
# ---------------------------------------------------------------------------
def distance_correlation(x, y):
    """Sample distance correlation between two 1-D arrays."""
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1D arrays.")

    n = x.shape[0]
    if y.shape[0] != n:
        raise ValueError("Arrays must have the same length.")

    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])

    A = a - a.mean(axis=0) - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0) - b.mean(axis=1)[:, None] + b.mean()

    dcov = np.sum(A * B) / (n * n)
    dvar_x = np.sum(A * A) / (n * n)
    dvar_y = np.sum(B * B) / (n * n)

    if dvar_x == 0 or dvar_y == 0:
        return 0.0
    return np.sqrt(dcov) / np.sqrt(np.sqrt(dvar_x) * np.sqrt(dvar_y))


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Per-trial / per-channel signal comparison plots
# ---------------------------------------------------------------------------
def compare_signal(mask_index, model, test_data, channel, settings, paths):
    """Plot original vs. reconstructed traces for three random test trials
    of the target channel."""
    selected_trials = [
        random.randint(0, test_data.shape[0] - 1) for _ in range(3)
    ]
    fig, ax = plt.subplots(3, 1, figsize=(15, 11), dpi=300)

    for i, trial in enumerate(selected_trials):
        model.eval()
        with torch.no_grad():
            reconstructed_signal = model(
                test_data[trial, :, :].unsqueeze(0),
                mask_index=mask_index,
            )[0, 0, :]

        original_signal = test_data[trial, :, :].cpu().numpy()[channel, :]
        reconstructed_signal = reconstructed_signal.cpu().numpy()

        original = torch.tensor(original_signal, dtype=torch.float32)
        reconstructed = torch.tensor(reconstructed_signal, dtype=torch.float32)

        epsilon = 1e-8
        mse_error = F.mse_loss(reconstructed, original).item()
        range_squared = (torch.max(original) - torch.min(original) + epsilon) ** 2
        nmse_range = (mse_error / range_squared).item()

        ax[i].plot(original_signal, label="Original Signal",
                   alpha=0.7, color='green')
        ax[i].plot(reconstructed_signal, label="Reconstructed Signal",
                   alpha=0.7, color='red')
        ax[i].set_ylabel("Signal Amplitude", fontsize=15)
        ax[i].legend(fontsize=10)

        metrics_text = f"NMSE (range): {nmse_range:.3f}\n"
        ax[i].text(
            0.01, 0.95, metrics_text,
            transform=ax[i].transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(facecolor='white', edgecolor='gray',
                      boxstyle='round,pad=0.4', alpha=0.8),
        )

    ax[0].set_title(
        f"Original vs Reconstructed Signal - Channel {channel + 1}",
        fontsize=18,
    )
    ax[2].set_xlabel("Time", fontsize=15)

    save_path = os.path.join(paths.result_path, 'compare_channel')
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, f'Channel_{channel + 1}.png'))
    fig.savefig(os.path.join(save_path, f'Channel_{channel + 1}.svg'))
    plt.close()


def save_mse_all_trials(mask_index, model, test_data, channel, paths, settings):
    """Reconstruct every test trial and the trial-mean, returning the list
    of per-trial DistCorr values. Side effects: writes the trial-mean
    figure to ``paths.result_path/compare_channel/Mean_Channel_<i>.{png,svg,npz}``."""
    model.eval()
    dcorr_all = []
    epsilon = 1e-8

    with torch.no_grad():
        mean_original_signal = torch.mean(test_data[:30, :, :], dim=0)
        mean_reconstructed_signal = model(
            mean_original_signal.unsqueeze(0), mask_index=mask_index
        )[0, 0, :].cpu().numpy()

        for i in range(test_data.shape[0]):
            reconstructed_signal = model(
                test_data[i].unsqueeze(0), mask_index=mask_index
            )[0].cpu().numpy()
            original_signal = test_data[i, channel].cpu().numpy()

            dcorr_per_trial = distance_correlation(
                np.nan_to_num(original_signal, nan=0.0, posinf=0.0, neginf=0.0),
                np.nan_to_num(reconstructed_signal[0, :],
                              nan=0.0, posinf=0.0, neginf=0.0),
            )
            dcorr_all.append(dcorr_per_trial)

    mean_original_signal = mean_original_signal[channel, :].cpu().numpy()
    Dcorr = distance_correlation(
        np.nan_to_num(mean_original_signal, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(mean_reconstructed_signal,
                      nan=0.0, posinf=0.0, neginf=0.0),
    )

    time_axis = np.arange(len(mean_original_signal)) / settings.fs

    fig, ax = plt.subplots(figsize=(15, 5), dpi=300)
    save_path = os.path.join(paths.result_path, 'compare_channel')
    os.makedirs(save_path, exist_ok=True)
    ax.plot(time_axis, mean_original_signal,
            label="Original Signal", alpha=0.9, color='blue')
    ax.plot(time_axis, mean_reconstructed_signal,
            label=f"Reconstructed Signal (DistCorr: {Dcorr:.3f})",
            alpha=0.9, color='red')
    ax.set_ylabel("Amplitude", fontsize=15)
    ax.set_xlabel("Time", fontsize=15)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    fig.savefig(os.path.join(save_path, f'Mean_Channel_{channel + 1}.png'))
    fig.savefig(os.path.join(save_path, f'Mean_Channel_{channel + 1}.svg'))
    np.savez(
        os.path.join(save_path, f'Mean_Channel_{channel + 1}.npz'),
        time_axis=time_axis,
        mean_reconstructed_signal=mean_reconstructed_signal,
        mean_original_signal=mean_original_signal,
        Dcorr=Dcorr,
    )
    plt.close()

    return dcorr_all
