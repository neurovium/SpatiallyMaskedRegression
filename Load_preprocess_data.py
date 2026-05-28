"""
Subject loaders and surrogate generators.

For real data, this module loads the pickled per-subject arrays produced
by the user's preprocessing pipeline. For surrogate experiments it loads
the original signal and then applies one of three surrogate procedures
(phase-shuffle, IAAFT, block-shuffle) channel by channel.

Paths to the preprocessed data come from ``Paths`` (see ``fixed/path.py``)
and so are configurable via ``configs/device_path.yaml``.
"""

from __future__ import annotations

import pickle as pkl

import numpy as np


# ---------------------------------------------------------------------------
# Surrogate generators
# ---------------------------------------------------------------------------
def generate_phase_shuffled_surrogate(sig: np.ndarray) -> np.ndarray:
    """Phase-randomised surrogate that preserves the amplitude spectrum
    via ``np.fft.rfft``. DC and Nyquist phases are kept at zero."""
    n = len(sig)
    fft_signal = np.fft.rfft(sig)
    magnitude = np.abs(fft_signal)
    random_phases = np.random.uniform(-np.pi, np.pi, len(fft_signal))
    random_phases[0] = 0
    if n % 2 == 0:
        random_phases[-1] = 0
    shuffled_fft = magnitude * np.exp(1j * random_phases)
    return np.fft.irfft(shuffled_fft, n=n)


def shuffle_phase(sig: np.ndarray) -> np.ndarray:
    """Phase-randomised surrogate using full ``np.fft.fft`` with explicit
    Hermitian symmetry; preserves the amplitude spectrum."""
    rng = np.random.default_rng()
    n = len(sig)
    fft_data = np.fft.fft(sig)
    amplitude = np.abs(fft_data)
    phase = np.angle(fft_data)

    half = n // 2
    pos_freq_idx = np.arange(1, half)
    random_phases = rng.permutation(phase[pos_freq_idx])

    shuffled_phase = phase.copy()
    shuffled_phase[pos_freq_idx] = random_phases
    shuffled_phase[-pos_freq_idx] = -random_phases

    new_fft = amplitude * np.exp(1j * shuffled_phase)
    return np.fft.ifft(new_fft).real


def iaaft_1d(sig: np.ndarray, n_iter: int = 5) -> np.ndarray:
    """IAAFT surrogate preserving both the amplitude distribution and
    (approximately) the power spectrum.

    Runs a fixed number of iterations with no early convergence test.
    """
    x = sig
    sorted_x = np.sort(x)
    X_mag = np.abs(np.fft.fft(x))
    z = np.random.permutation(x)
    for _ in range(n_iter):
        Z = np.fft.fft(z)
        Z = X_mag * np.exp(1j * np.angle(Z))
        z_new = np.fft.ifft(Z).real
        z = sorted_x[np.argsort(np.argsort(z_new))]
    return z


def block_shuffle_1d(sig: np.ndarray,
                     block_size_samples: int,
                     rng=None) -> np.ndarray:
    """Permute contiguous blocks of length ``block_size_samples``."""
    if rng is None:
        rng = np.random.default_rng()
    n_samples = len(sig)
    n_blocks = n_samples // block_size_samples
    remainder = n_samples % block_size_samples
    blocks = [sig[i * block_size_samples:(i + 1) * block_size_samples]
              for i in range(n_blocks)]
    perm = rng.permutation(n_blocks)
    shuffled = np.concatenate([blocks[i] for i in perm])
    if remainder > 0:
        shuffled = np.concatenate([shuffled, sig[-remainder:]])
    return shuffled


# ---------------------------------------------------------------------------
# Per-subject loader
# ---------------------------------------------------------------------------
def load_per_subject(settings, paths, sub: int,
                     block_size_sec: float = 0.5,
                     fs: int | None = None) -> np.ndarray:
    """Load pickled data for subject ``sub`` (0-based), with optional
    on-the-fly surrogate generation per ``settings.dataset_type``.

    The block-shuffle path uses ``settings.fs`` for converting
    ``block_size_sec`` to samples, so the block length matches the
    dataset's native sampling rate (not the 100 Hz default that the
    previous code path was using).
    """
    fs = fs if fs is not None else int(getattr(settings, 'fs', 500))

    dataset_type = settings.dataset_type

    # --- Real EEG / iEEG ----------------------------------------------------
    if dataset_type in ('EEG', 'iEEG'):
        print(f'\n Loaded data subject {sub + 1}')
        if dataset_type == 'EEG':
            with open(paths.eeg_subject_file(sub), 'rb') as f:
                dataset = pkl.load(f)
            return np.array(dataset['data'])
        with open(paths.ieeg_subject_file(sub), 'rb') as f:
            return pkl.load(f)

    # --- Surrogate-from-real -----------------------------------------------
    if dataset_type in ('Surrogate_EEG', 'Surrogate_iEEG',
                        'IAAFT_EEG', 'IAAFT_iEEG',
                        'Block_EEG', 'Block_iEEG'):
        if 'EEG' in dataset_type and 'iEEG' not in dataset_type:
            with open(paths.eeg_subject_file(sub), 'rb') as f:
                dataset = pkl.load(f)['data']
        else:
            with open(paths.ieeg_subject_file(sub), 'rb') as f:
                dataset = pkl.load(f)

        data_per_sub = np.zeros_like(dataset)
        rng = np.random.default_rng()

        for trial in range(dataset.shape[0]):
            for ch in range(dataset.shape[1]):
                sig = dataset[trial, ch]
                if 'Surrogate' in dataset_type:
                    data_per_sub[trial, ch] = shuffle_phase(sig)
                elif 'IAAFT' in dataset_type:
                    data_per_sub[trial, ch] = iaaft_1d(sig)
                elif 'Block' in dataset_type:
                    block_size_samples = int(block_size_sec * fs)
                    data_per_sub[trial, ch] = block_shuffle_1d(
                        sig, block_size_samples, rng=rng,
                    )
        return data_per_sub

    raise ValueError(f"Unknown dataset_type: {dataset_type!r}")
