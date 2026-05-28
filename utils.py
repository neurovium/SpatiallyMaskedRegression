"""
Miscellaneous helpers for the SMR pipeline.

Mask / neighborhood logic lives in ``fixed/neighborhood.py``. This module
contains:

    * Environment / reproducibility setup (``configure_environment``).
    * Settings serialisation (``save_settings_to_json``).
    * iEEG preprocessing (``preprocess_iEEG``) used by the Dandiset path.
    * Trial epoching utilities for raw ECoG.
    * Dandiset extraction (``extract_data_dandi``).

The legacy ``find_index_mask_eeg`` / ``find_index_mask_ieeg`` and
``nearest_neighbors`` helpers from ``new/utils.py`` were intentionally
removed: their job is now done by ``fixed/neighborhood.py``.
"""

from __future__ import annotations

import json
import os
import random
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy import signal
from scipy.ndimage import median_filter


# ---------------------------------------------------------------------------
# Environment / reproducibility
# ---------------------------------------------------------------------------
def configure_environment(gpu_id: str = "0", seed: int = 42) -> None:
    """Set CUDA device order and seed Python/NumPy/PyTorch RNGs."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Settings serialisation
# ---------------------------------------------------------------------------
def save_settings_to_json(settings, file_path: str) -> None:
    """Write the public attributes of ``settings`` to ``file_path/settings.json``.

    Properties protected by name-mangling (``_Settings__*``) are unwrapped
    so the resulting JSON has clean keys.
    """
    raw = dict(settings.__dict__)
    cleaned = {}
    for k, v in raw.items():
        if k.startswith('_Settings__'):
            cleaned[k.replace('_Settings__', '')] = v
        else:
            cleaned[k] = v
    os.makedirs(file_path, exist_ok=True)
    with open(os.path.join(file_path, 'settings.json'), 'w') as f:
        json.dump(cleaned, f, indent=4, default=str)


# ---------------------------------------------------------------------------
# iEEG preprocessing
# ---------------------------------------------------------------------------
def preprocess_iEEG(data: np.ndarray,
                    fs: float = 1000,
                    band: tuple = (0.5, 150),
                    notch_freq: float = 60,
                    notch_q: float = 30,
                    new_fs: Optional[float] = None,
                    zscore: bool = True,
                    artifact_thresh: float = 5,
                    rereference: bool = True,
                    extract_envelope: bool = True,
                    use_median_denoise: bool = True,
                    high_freq_denoise: bool = True,
                    verbose: bool = True) -> np.ndarray:
    """Preprocess iEEG data: detrending, filtering, artifact handling,
    optional CAR, optional Hilbert envelope extraction.

    Returns the preprocessed array (same shape as input, or downsampled if
    ``new_fs`` < ``fs``).
    """
    def log(msg):
        if verbose:
            print(msg)

    # 1. Remove DC offset and detrend.
    log("Step 1: Removing DC offset and detrending")
    data = data - np.mean(data, axis=0)
    data = signal.detrend(data, axis=0, type='linear')

    # 2. Optional median denoise.
    if use_median_denoise:
        log("Step 2: Applying median filter (kernel size=5)")
        data = median_filter(data, size=(5, 1))
    else:
        log("Step 2: Skipping median denoising")

    # 3. Bandpass.
    log(f"Step 3: Bandpass filtering ({band[0]}-{band[1]} Hz)")
    b, a = signal.butter(4, [band[0] / (fs / 2), band[1] / (fs / 2)],
                         btype='band')
    data = signal.filtfilt(b, a, data, axis=0)

    # 4. Notch.
    log(f"Step 4: Notch filtering at {notch_freq} Hz")
    b, a = signal.iirnotch(notch_freq / (fs / 2), notch_q)
    data = signal.filtfilt(b, a, data, axis=0)

    # 5. High-frequency denoise.
    if high_freq_denoise:
        log("Step 5: Applying lowpass smoothing (cutoff=120 Hz)")
        b, a = signal.butter(4, 120 / (fs / 2), btype='low')
        data = signal.filtfilt(b, a, data, axis=0)
    else:
        log("Step 5: Skipping high-frequency denoising")

    # 6. Artifact rejection.
    log(f"Step 6: Artifact rejection (|z| > {artifact_thresh})")
    z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
    artifact_mask = z_scores > artifact_thresh
    if np.any(artifact_mask):
        medians = np.median(data, axis=0)
        data[artifact_mask] = np.take(medians, np.where(artifact_mask)[1])
    else:
        log("No artifacts detected above threshold.")

    # 7. Common Average Reference.
    if rereference:
        log("Step 7: Applying Common Average Reference (CAR)")
        data = data - np.mean(data, axis=1, keepdims=True)
    else:
        log("Step 7: Skipping re-referencing")

    # 8. Downsampling.
    if new_fs is not None and new_fs < fs:
        log(f"Step 8: Downsampling from {fs} Hz to {new_fs} Hz")
        factor = int(fs / new_fs)
        data = signal.decimate(data, factor, axis=0, ftype='fir')
        fs = new_fs
    else:
        log("Step 8: Skipping downsampling")

    # 9. Z-score.
    if zscore:
        log("Step 9: Applying z-score normalization per channel")
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    else:
        log("Step 9: Skipping z-score normalization")

    # 10. Envelope extraction.
    if extract_envelope:
        log("Step 10: Extracting signal envelope using Hilbert transform")
        analytic_signal = signal.hilbert(data, axis=0)
        data = np.abs(analytic_signal)
    else:
        log("Step 10: Skipping envelope extraction")

    log("Preprocessing complete.")
    return data


# ---------------------------------------------------------------------------
# ECoG epoching (used by the Dandiset extraction path)
# ---------------------------------------------------------------------------
def epoch_ECoG_MNE(ecog_data, events, event_id, ch_names, elec_locs, Fs,
                   epoch_times, pad_val=0.5, metadata=None):
    """Epoch raw ECoG into trials around event times using MNE."""
    import mne   # imported lazily so the rest of the package does not need MNE
    dig_ch_pos = dict(zip(ch_names, elec_locs))
    mon = mne.channels.make_dig_montage(ch_pos=dig_ch_pos, coord_frame='head')
    info = mne.create_info(ch_names=ch_names, sfreq=Fs, ch_types='eeg')
    info.set_montage(mon)

    raw = mne.io.RawArray(ecog_data, info)
    del ecog_data

    epoched_data = mne.Epochs(
        raw, events, event_id,
        tmin=epoch_times[0] - pad_val,
        tmax=epoch_times[1] + pad_val,
        baseline=None, preload=True, metadata=metadata,
    )
    del raw
    return epoched_data


def rem_bad_trials_PSD(epoched_data):
    """Drop trials whose PSD is unusually low (constant) or
    has very high-frequency power."""
    from mne.time_frequency import psd_array_multitaper
    psds, freqs = psd_array_multitaper(
        epoched_data.get_data(),
        sfreq=epoched_data.info['sfreq'],
        fmin=6, fmax=150,
    )
    psds = 10. * np.log10(psds)
    psds[np.isinf(psds)] = 0
    hi_freq_inds = np.nonzero((freqs > 115) & (freqs < 125))[0]
    ave_high_freq = np.mean(np.mean(psds[:, :, hi_freq_inds], axis=1), axis=1)
    bad_inds = np.nonzero(
        (np.squeeze(np.min(np.mean(psds, axis=1), axis=1)) < 0)
        | (ave_high_freq > (3 * np.std(ave_high_freq) + np.median(ave_high_freq)))
    )[0]
    epoched_data.drop(bad_inds)
    if epoched_data.metadata is not None:
        epoched_data.metadata.reset_index(drop=True, inplace=True)
    return epoched_data


def preprocess_ecog(ecog_data, Fs, event_times, event_labels,
                    epoch_times=(-1, 1), elec_locs=None):
    """Raw ECoG -> (n_epochs, n_channels, n_times) array of clean trials."""
    ecog_data = np.nan_to_num(ecog_data)
    n_channels = ecog_data.shape[0]
    ch_names = [f'EEG{i}' for i in range(n_channels)]
    if elec_locs is None:
        elec_locs = np.zeros((n_channels, 3))

    event_indices = [int(t * Fs) for t in event_times]
    events = np.zeros((len(event_indices), 3), dtype=int)
    for i, ind in enumerate(event_indices):
        events[i, 0] = ind
        events[i, 2] = event_labels[i]
    event_id = dict(rest=1, move=2)
    metadata_df = pd.DataFrame({'event_label': event_labels})

    epoched_data = epoch_ECoG_MNE(
        ecog_data, events, event_id, ch_names, elec_locs, Fs,
        epoch_times, metadata=metadata_df,
    )
    epoched_data = rem_bad_trials_PSD(epoched_data)
    ecog_epochs = epoched_data.get_data()
    return ecog_epochs


# ---------------------------------------------------------------------------
# Dandiset extraction
# ---------------------------------------------------------------------------
def extract_data_dandi(dandiset_id: str = "000055",
                       subject: str = "01",
                       session: str = "3",
                       sample_time_window: int = 1000,
                       cache_dir: Optional[str] = None) -> np.ndarray:
    """Fetch and preprocess one Dandiset session into reshaped trials.

    Parameters
    ----------
    dandiset_id, subject, session :
        DANDI identifiers for the recording to fetch.
    sample_time_window :
        Trial length in samples after preprocessing.
    cache_dir :
        Local cache for remfile downloads. If ``None``, a system-default
        temp directory is used (``$TMPDIR/remfile_cache``).
    """
    # Lazy imports so users without DANDI installed can still use the rest
    # of the package.
    from dandi.dandiapi import DandiAPIClient
    import h5py
    import remfile
    from pynwb import NWBHDF5IO

    filepath = f"sub-{subject}/sub-{subject}_ses-{session}_behavior+ecephys.nwb"
    with DandiAPIClient() as client:
        asset = (
            client.get_dandiset(dandiset_id, 'draft')
                  .get_asset_by_path(filepath)
        )
        s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)

    if cache_dir is None:
        cache_dir = os.path.join(
            os.environ.get('TMPDIR', '/tmp'), 'remfile_cache'
        )
    disk_cache = remfile.DiskCache(cache_dir)

    rem_file = remfile.File(s3_url, disk_cache=disk_cache)
    h5py_file = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5py_file)
    nwbfile = io.read()

    ecog_data = nwbfile.acquisition['ElectricalSeries'].data[3000000:10000000, :]
    reaches = nwbfile.intervals['reaches']
    event_labels = (list(reaches.Bimanual_class[:])
                    + [1] * len(reaches.Bimanual_class[:]))
    event_times = (list(reaches.start_time[:])
                   + [0.5] * len(reaches.start_time[:]))
    Fs = 500
    ecog_data = np.transpose(ecog_data, axes=(1, 0))

    epochs = preprocess_ecog(ecog_data, Fs, event_times, event_labels)
    preprocessed_data = preprocess_iEEG(
        epochs, fs=500, band=(0.5, 100), notch_freq=60, new_fs=10,
    )

    n_trials = preprocessed_data.shape[0] // sample_time_window
    return preprocessed_data[:n_trials * sample_time_window].reshape(
        n_trials, preprocessed_data.shape[1], sample_time_window
    )
