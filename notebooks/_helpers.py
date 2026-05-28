"""
Helpers for the replication notebooks.

The training driver in ``fixed/main.py`` writes outputs under

    <results_dir>/<dataset_type>_<mode>/sub_<subject>/<timestamp>/
        nbhd-<atlas|knn>/model-<instantaneous|lagged>/intesity_<percent>/

with ``percent = mask_intensity * 100`` (so ``0.25`` shows up on disk as
``intesity_25.0``). Each cell stores at least ``DCORR_<subject>.npy``
and, for the standard pipeline, ``adj_mat_<subject>.npy``.

These helpers turn "give me the DCORR for subject 0 at intensity 0.5
under the atlas neighborhood and the instantaneous model" into a path
lookup that doesn't depend on knowing the timestamped subfolder names.
If multiple runs match the (subject, intensity, neighborhood, model)
combination -- e.g. you re-trained -- the *most recent* timestamp is
returned.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

# Make the ``fixed/`` package importable when this file is used from a
# notebook in ``fixed/notebooks/``.
_PKG_ROOT = str(Path(__file__).resolve().parents[1])
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from path import Paths        # noqa: E402
from setting import Settings  # noqa: E402


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def load_paths(results_dir: Optional[str] = None) -> Paths:
    """Build a `Paths` object from the package's `configs/` files.

    Pass ``results_dir`` to override the directory configured in
    ``configs/device_path.yaml`` -- handy when the figures live in a
    different folder from training outputs.
    """
    settings = Settings()
    settings.load_settings()
    paths = Paths(settings)
    paths.load_device_paths()
    if results_dir is not None:
        paths.results_dir = str(results_dir)
    return paths


def _intensity_token(mask_intensity: float) -> str:
    """The on-disk encoding for ``intesity_<percent>`` is the float
    ``mask_intensity * 100``. Mirror that exactly."""
    return f'intesity_{mask_intensity * 100}'


def find_dcorr_file(results_dir: str,
                    dataset_type: str,
                    mode: str,
                    subject: int,
                    mask_intensity: float,
                    neighborhood_method: str = 'atlas',
                    model_type: str = 'instantaneous') -> Path:
    """Resolve the ``DCORR_<subject>.npy`` for the given run.

    Raises ``FileNotFoundError`` if no matching file exists.
    """
    pattern = (
        f'{dataset_type}_{mode}/sub_{subject}/*/'
        f'nbhd-{neighborhood_method}/model-{model_type}/'
        f'{_intensity_token(mask_intensity)}/DCORR_{subject}.npy'
    )
    matches = sorted(Path(results_dir).glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No DCORR file under {results_dir!r} matching:\n  {pattern}\n"
            "Have you run the corresponding training cell yet?"
        )
    return matches[-1]


def load_dcorr(results_dir: str,
               dataset_type: str,
               mode: str,
               subject: int,
               mask_intensity: float,
               neighborhood_method: str = 'atlas',
               model_type: str = 'instantaneous') -> np.ndarray:
    """``DCORR_<subject>.npy`` is a list-of-lists (per-electrode, per-trial).
    This helper flattens it to a 1-D array of all DistCorr values."""
    fname = find_dcorr_file(
        results_dir, dataset_type, mode, subject, mask_intensity,
        neighborhood_method, model_type,
    )
    raw = np.load(fname, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        return np.concatenate([np.asarray(r).ravel() for r in raw]).astype(float)
    return np.asarray(raw, dtype=float).ravel()


def mean_var_dcorr(results_dir: str,
                   dataset_type: str,
                   mode: str,
                   subjects: Iterable[int],
                   mask_intensities: Iterable[float],
                   neighborhood_method: str = 'atlas',
                   model_type: str = 'instantaneous') -> dict:
    """Aggregate ``DCORR`` summaries across subjects and intensities.

    Returns a dict::

        {subject: {intensity: {'mean': ..., 'var': ..., 'n': ...}}}

    Missing combinations are stored as ``None`` so the caller can decide
    how to handle them.
    """
    out = {}
    for s in subjects:
        out[s] = {}
        for m in mask_intensities:
            try:
                vals = load_dcorr(
                    results_dir, dataset_type, mode, s, m,
                    neighborhood_method, model_type,
                )
            except FileNotFoundError:
                out[s][m] = None
                continue
            out[s][m] = {
                'mean': float(np.mean(vals)),
                'var': float(np.var(vals)),
                'n': int(vals.size),
            }
    return out


def find_mean_channel_file(results_dir: str,
                           dataset_type: str,
                           mode: str,
                           subject: int,
                           mask_intensity: float,
                           channel: int,
                           neighborhood_method: str = 'atlas',
                           model_type: str = 'instantaneous') -> Path:
    """Resolve the per-channel ``Mean_Channel_<channel+1>.npz`` written
    by ``visualize.save_mse_all_trials``."""
    pattern = (
        f'{dataset_type}_{mode}/sub_{subject}/*/'
        f'nbhd-{neighborhood_method}/model-{model_type}/'
        f'{_intensity_token(mask_intensity)}/compare_channel/'
        f'Mean_Channel_{channel + 1}.npz'
    )
    matches = sorted(Path(results_dir).glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No Mean_Channel file under {results_dir!r} matching:\n  {pattern}"
        )
    return matches[-1]


def load_mean_channel(results_dir: str,
                      dataset_type: str,
                      mode: str,
                      subject: int,
                      mask_intensity: float,
                      channel: int,
                      neighborhood_method: str = 'atlas',
                      model_type: str = 'instantaneous') -> dict:
    """Load the trial-averaged original + reconstructed signals for a
    single (subject, channel, intensity) cell.

    Returns ``{'time_axis': ..., 'orig': ..., 'recon': ..., 'Dcorr': ...}``.
    """
    fname = find_mean_channel_file(
        results_dir, dataset_type, mode, subject, mask_intensity, channel,
        neighborhood_method, model_type,
    )
    z = np.load(fname)
    return {
        'time_axis': z['time_axis'],
        'orig': z['mean_original_signal'],
        'recon': z['mean_reconstructed_signal'],
        'Dcorr': float(z['Dcorr']),
        'source_file': str(fname),
    }


# ---------------------------------------------------------------------------
# Coverage-condition mask constructors (Figure 6 in the paper)
# ---------------------------------------------------------------------------
def coverage_masks(target_idx: int,
                   ch_name,
                   ch_position,
                   modality: str,
                   method: str = 'atlas',
                   k: int = 9) -> dict:
    """Return the three masks for the Figure 6 coverage experiment.

    * ``local``     -- predictors = ``N(i)`` only.
                       mask = (all channels except ``N(i)``) plus target.
    * ``non_local`` -- predictors = everything except ``N(i)``.
                       mask = ``N(i)`` plus target.  (Same as
                       ``find_index_mask`` with mask_intensity=1.0.)
    * ``all``       -- predictors = everything except target.
                       mask = {target}.  (Same as
                       ``find_index_mask`` with mask_intensity=0.0.)

    Used by ``fixed/run_coverage_experiment.py`` to drive three training
    passes on the same data.
    """
    from neighborhood import find_neighborhood

    n_total = (len(ch_name) if ch_name is not None
               else int(np.asarray(ch_position).shape[0]))

    nbhd = set(find_neighborhood(
        target_idx, ch_name, ch_position,
        modality=modality, method=method, k=k,
    ))

    all_except_target = set(range(n_total)) - {target_idx}
    non_local_channels = all_except_target - nbhd

    return {
        'local':     sorted(non_local_channels | {target_idx}),  # mask the non-local set
        'non_local': sorted(nbhd | {target_idx}),                # mask the local neighborhood
        'all':       [target_idx],                               # mask only the target
    }
