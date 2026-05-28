"""
Cross-subject iEEG evaluation: per-electrode donor selection from the
transferred adjacency matrices produced by
``fixed/generate_adj_matrix_ieeg_unseen.py``.

All filesystem locations are taken from a ``Paths`` instance loaded from
``configs/device_path.yaml``; no machine-specific path lives in this
file.

Run with::

    from path import Paths
    from setting import Settings
    s = Settings(); s.load_settings()
    p = Paths(s); p.load_device_paths()
    run_cross_subject_eval(s, p)
"""

from __future__ import annotations

import os
import pickle as pkl
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Metric (copied here so the module is standalone)
# ---------------------------------------------------------------------------
def distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
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


def normalize_adj_rowwise(adj: np.ndarray) -> np.ndarray:
    adj_norm = np.zeros_like(adj)
    for i in range(adj.shape[0]):
        row_min = np.min(adj[i, :])
        row_max = np.max(adj[i, :])
        if row_max != row_min:
            adj_norm[i, :] = (adj[i, :] - row_min) / (row_max - row_min)
        else:
            adj_norm[i, :] = adj[i, :]
    return adj_norm


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------
def run_cross_subject_eval(settings, paths, fs: Optional[int] = None) -> None:
    """Evaluate every (src, dst) cross-subject reconstruction and write
    per-electrode DistCorr scores plus comparison plots.

    Parameters
    ----------
    settings : Settings
        Provides ``num_subject``.
    paths : Paths
        Resolves directory layout. Reads from
        ``paths.preprocessed_dir/iEEG`` and
        ``paths.cross_subject_adj_dir(src)``; writes to
        ``paths.results_dir/plot/sub_<src>``.
    fs : int, optional
        Sampling rate used for plotting time axes. Defaults to
        ``settings.fs``.
    """
    fs = fs if fs is not None else int(getattr(settings, 'fs', 250))

    for src_sub in range(settings.num_subject):
        adj_dir = paths.cross_subject_adj_dir(src_sub)
        plot_dir = os.path.join(paths.results_dir, 'plot', f'sub_{src_sub}')
        os.makedirs(plot_dir, exist_ok=True)

        with open(paths.ieeg_subject_file(src_sub), 'rb') as f:
            data_src = pkl.load(f)

        # 50/50 validation/test split of the target subject's trials.
        val_data, test_data = train_test_split(
            data_src, test_size=0.5, random_state=42
        )
        val_mean = np.mean(val_data, axis=0)        # (n_elec, time)
        test_mean = np.mean(test_data, axis=0)      # (n_elec, time)

        plot_data = {}

        for dst_sub in range(settings.num_subject):
            if dst_sub == src_sub:
                continue

            plot_data[dst_sub] = {
                'orig_signal_z': [],
                'recon_signal_corr': [],
                'Dcorr': [],
            }
            print(f'Source subject: {src_sub}, donor subject: {dst_sub}')

            adj_path = os.path.join(
                adj_dir,
                f'adj_mat_corr_sub{src_sub}_base_sub{dst_sub}.npy',
            )
            adj_mat_corr = np.load(adj_path)

            # Optional matched-mask file written by the strict 1-1 Hungarian.
            mask_path = os.path.join(
                adj_dir,
                f'matched_mask_sub{src_sub}_base_sub{dst_sub}.npy',
            )
            matched_mask = (np.load(mask_path)
                            if os.path.exists(mask_path)
                            else np.ones(data_src.shape[1], dtype=bool))

            Dcorr_rows = []

            for m in tqdm(range(data_src.shape[1]),
                          desc=f'electrodes for dst {dst_sub}'):
                if not matched_mask[m]:
                    Dcorr_rows.append([m, np.nan, np.nan])
                    plot_data[dst_sub]['recon_signal_corr'].append(
                        np.zeros_like(test_mean[m, :])
                    )
                    plot_data[dst_sub]['orig_signal_z'].append(
                        zscore(test_mean[m, :])
                    )
                    plot_data[dst_sub]['Dcorr'].append(np.nan)
                    continue

                weights = np.expand_dims(adj_mat_corr[m, :], axis=1)

                orig_val = zscore(val_mean[m, :])
                recon_val = zscore(np.sum(weights * val_mean, axis=0))
                Dcorr_val = distance_correlation(
                    np.nan_to_num(orig_val, nan=0.0, posinf=0.0, neginf=0.0),
                    np.nan_to_num(recon_val, nan=0.0, posinf=0.0, neginf=0.0),
                )

                orig_test = zscore(test_mean[m, :])
                recon_test = zscore(np.sum(weights * test_mean, axis=0))
                Dcorr_test = distance_correlation(
                    np.nan_to_num(orig_test, nan=0.0, posinf=0.0, neginf=0.0),
                    np.nan_to_num(recon_test, nan=0.0, posinf=0.0, neginf=0.0),
                )

                Dcorr_rows.append([m, Dcorr_val, Dcorr_test])
                plot_data[dst_sub]['recon_signal_corr'].append(recon_test)
                plot_data[dst_sub]['orig_signal_z'].append(orig_test)
                plot_data[dst_sub]['Dcorr'].append(Dcorr_test)

            df_Dcorr = pd.DataFrame(
                Dcorr_rows,
                columns=['Electrode', 'Dcorrelation_val', 'Dcorrelation_test'],
            )
            excel_path = os.path.join(plot_dir, f'Dcorr_values_{dst_sub}.xlsx')
            df_Dcorr.to_excel(excel_path, index=False)
            print(f"Saved {excel_path}")

        _plot_cross_subject(plot_data, plot_dir, fs)


def _plot_cross_subject(plot_data: dict, plot_dir: str, fs: int) -> None:
    subjects = list(plot_data.keys())
    if not subjects:
        return

    n_electrodes = len(plot_data[subjects[0]]['orig_signal_z'])
    n_rows, n_cols = 3, 4
    best_Dcorr = []

    for m in tqdm(range(n_electrodes), desc="Generating plots"):
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 10), dpi=300,
                                sharex=True, sharey=True)
        axs = axs.flatten()

        max_Dcorr = -np.inf
        idx_best = None
        for idx, dst_sub in enumerate(subjects):
            if idx >= len(axs):
                break
            recon_signal = plot_data[dst_sub]['recon_signal_corr'][m]
            orig_signal = plot_data[dst_sub]['orig_signal_z'][m]
            Dcorr = plot_data[dst_sub]['Dcorr'][m]

            val_Dcorr = pd.read_excel(
                os.path.join(plot_dir, f'Dcorr_values_{dst_sub}.xlsx')
            ).loc[m, 'Dcorrelation_val']

            if not np.isnan(val_Dcorr) and val_Dcorr > max_Dcorr:
                max_Dcorr = val_Dcorr
                idx_best = dst_sub

            axs[idx].plot(
                recon_signal,
                label=f'Reconstructed (DistCorr: {Dcorr:.4f})'
                if not np.isnan(Dcorr) else 'Reconstructed (unmatched)',
                color='blue',
            )
            axs[idx].plot(orig_signal, label='Original', color='red')
            axs[idx].set_title(f'Subject {dst_sub}')
            axs[idx].legend(fontsize=8)

        for i in range(len(subjects), len(axs)):
            fig.delaxes(axs[i])

        plt.xlabel('Time')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'Channel_{m}.png'))
        plt.close()

        if idx_best is None:
            best_Dcorr.append(np.nan)
            continue

        recon_best = plot_data[idx_best]['recon_signal_corr'][m]
        orig_best = plot_data[idx_best]['orig_signal_z'][m]
        Dcorr_best = plot_data[idx_best]['Dcorr'][m]
        time_axis = np.arange(len(orig_best)) / fs
        best_Dcorr.append(Dcorr_best)

        fig, ax = plt.subplots(figsize=(15, 5), dpi=300)
        ax.plot(time_axis, orig_best,
                label="Original Signal", alpha=0.9, color='blue')
        ax.plot(time_axis, recon_best,
                label=f"Reconstructed Signal (DistCorr: {Dcorr_best:.3f})",
                alpha=0.9, color='red')
        ax.set_ylabel("Amplitude", fontsize=15)
        ax.set_xlabel("Time", fontsize=15)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        np.save(
            os.path.join(plot_dir, f'Best_Channel_{m}.npy'),
            np.array([orig_best, recon_best, time_axis], dtype=object),
        )
        fig.savefig(os.path.join(plot_dir, f'Best_Channel_{m}.png'))
        fig.savefig(os.path.join(plot_dir, f'Best_Channel_{m}.svg'))
        plt.close()

    np.save(os.path.join(plot_dir, 'best_Dcorr.npy'),
            np.array(best_Dcorr, dtype=float))
    print('All plots generated successfully.')


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    from setting import Settings
    from path import Paths

    settings = Settings()
    settings.load_settings()
    paths = Paths(settings)
    paths.load_device_paths()
    run_cross_subject_eval(settings, paths)
