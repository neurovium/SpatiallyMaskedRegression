"""
Unified neighborhood / spatial-mask utilities for the SMR pipeline.

This module exposes a single entry point, ``find_index_mask``, that supports
both neighborhood-definition strategies used in the project so far:

    method='atlas'   -- the strategy described in the manuscript:
                          * EEG : adjacency in the standardized 10-10 / 10-5
                                  montage, encoded as fixed regional groups
                                  (matches `gihub_repo/utils.py`).
                          * iEEG: anatomical proximity from the AAL atlas,
                                  encoded as channels sharing the same
                                  anatomical label as the target.
    method='knn'     -- the strategy implemented in `new/utils.py`:
                          * The k channels closest to the target in projected
                            Euclidean coordinate space (x,y for EEG, x,z for
                            iEEG).

Either strategy returns a *neighborhood set* N(i) which is then passed
through a uniform `mask_intensity` knob, so the masking semantics described
in the manuscript

    intensity * |N(i)| local channels are randomly drawn from N(i) and
    masked together with the target electrode i,

are honoured for both methods.

Drop-in replacements `find_index_mask_eeg` and `find_index_mask_ieeg` are
provided so existing call sites in `train.py` / `evaluate_model` do not need
to be rewritten — only the new keyword arguments need to be threaded
through.

Usage
-----
    from neighborhood import find_index_mask

    list_mask = find_index_mask(
        target_idx=i,
        ch_name=ch_name,
        ch_position=ch_position,
        mask_intensity=settings.mask_intensity,
        modality=settings.dataset_type,        # 'EEG' or 'iEEG'
        method=settings.neighborhood_method,   # 'atlas' or 'knn'
        k=settings.neighborhood_k,             # used only when method='knn'
    )
"""

from __future__ import annotations

import random
from typing import Iterable, List, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# 10-10 / 10-5 regional groups for the EEG montage.
# Lifted verbatim from `gihub_repo/utils.py` so atlas-mode EEG masking is
# bit-identical to what the original (manuscript-matching) pipeline did.
# ---------------------------------------------------------------------------
_EEG_MONTAGE_GROUPS: List[List[str]] = [
    ['F3', 'F1', 'Fz', 'F2', 'F4'],
    ['FFC5h', 'FFC3h', 'FFC1h', 'FFC2h', 'FFC4h', 'FFC6h'],
    ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6'],
    ['FTT7h', 'FCC5h', 'FCC3h', 'FCC1h', 'FCC2h', 'FCC4h', 'FCC6h', 'FTT8h'],
    ['C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6'],
    ['TTP7h', 'CCP5h', 'CCP3h', 'CCP1h', 'CCP2h', 'CCP4h', 'CCP6h', 'TTP8h'],
    ['CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'],
    ['CPP5h', 'CPP3h', 'CPP1h', 'CPP2h', 'CPP4h', 'CPP6h'],
    ['P3', 'P1', 'Pz', 'P2', 'P4'],
    ['PPO1h', 'PPO2h'],
]


# ---------------------------------------------------------------------------
# Neighborhood definitions
# ---------------------------------------------------------------------------
def _neighborhood_eeg_montage(target_idx: int, ch_name: Sequence[str]) -> List[int]:
    """EEG neighborhood = the 10-10/10-5 regional group containing the target.

    Returns indices into ``ch_name`` (excluding the target itself).
    """
    target = ch_name[target_idx]
    for group in _EEG_MONTAGE_GROUPS:
        if target in group:
            return [list(ch_name).index(ch) for ch in group
                    if ch in ch_name and list(ch_name).index(ch) != target_idx]
    # Fall back to an empty neighborhood if the channel is unknown.
    return []


def _neighborhood_ieeg_aal(target_idx: int, ch_name: Sequence[str]) -> List[int]:
    """iEEG neighborhood = channels sharing the target's anatomical label.

    Assumes ``ch_name[i]`` is the AAL region label of electrode i (this is
    how the data shipped with the project encodes anatomy).
    """
    label = ch_name[target_idx]
    return [idx for idx, lab in enumerate(ch_name)
            if lab == label and idx != target_idx]


def _neighborhood_knn(target_idx: int,
                      ch_position: np.ndarray,
                      modality: str,
                      k: int) -> List[int]:
    """k-nearest electrodes by 2-D projected Euclidean distance.

    Mirrors `new/utils.py::nearest_neighbors`:
      * EEG  uses the (x, y) columns of ``ch_position`` (axial projection).
      * iEEG uses the (x, z) columns (coronal projection).

    The target electrode itself (distance zero) is excluded.
    """
    modality = modality.lower()
    pos = np.asarray(ch_position)

    if modality == 'eeg':
        dxy = pos[:, [0, 1]] - pos[target_idx, [0, 1]]
    elif modality == 'ieeg':
        dxy = pos[:, [0, 2]] - pos[target_idx, [0, 2]]
    else:
        raise ValueError(f"Unsupported modality for 'knn': {modality!r}")

    dist = np.sqrt(np.sum(dxy ** 2, axis=1))
    order = np.argsort(dist)

    neighbors: List[int] = []
    for idx in order:
        idx = int(idx)
        if dist[idx] > 0 and idx != target_idx:
            neighbors.append(idx)
        if len(neighbors) == k:
            break
    return neighbors


def find_neighborhood(target_idx: int,
                      ch_name: Sequence[str] | None,
                      ch_position: np.ndarray | None,
                      modality: str,
                      method: str = 'atlas',
                      k: int = 9) -> List[int]:
    """Return the local-neighborhood index set N(target_idx).

    Parameters
    ----------
    target_idx : int
        Index of the target electrode.
    ch_name : sequence of str or None
        Channel names. Required for ``method='atlas'``; ignored otherwise.
        For EEG, names should follow the 10-10/10-5 convention. For iEEG,
        names should encode AAL anatomical labels.
    ch_position : ndarray of shape (N, 3) or None
        Electrode coordinates. Required for ``method='knn'``; ignored
        otherwise.
    modality : {'EEG', 'iEEG'}
        Recording modality. Case-insensitive.
    method : {'atlas', 'knn'}
        Neighborhood-definition strategy.
    k : int
        Number of nearest neighbors when ``method='knn'``. Ignored for
        ``method='atlas'``.

    Returns
    -------
    list of int
        Indices of the channels considered local to ``target_idx``,
        excluding the target itself.
    """
    method = method.lower()
    mod = modality.lower()

    if method == 'atlas':
        if ch_name is None:
            raise ValueError("method='atlas' requires ch_name.")
        if mod == 'eeg':
            return _neighborhood_eeg_montage(target_idx, ch_name)
        elif mod == 'ieeg':
            return _neighborhood_ieeg_aal(target_idx, ch_name)
        else:
            raise ValueError(f"Unsupported modality for 'atlas': {modality!r}")

    if method == 'knn':
        if ch_position is None:
            raise ValueError("method='knn' requires ch_position.")
        return _neighborhood_knn(target_idx, ch_position, mod, k)

    raise ValueError(f"Unknown neighborhood method: {method!r} "
                     "(expected 'atlas' or 'knn').")


# ---------------------------------------------------------------------------
# Mask construction (uniform across methods)
# ---------------------------------------------------------------------------
def find_index_mask(target_idx: int,
                    ch_name: Sequence[str] | None,
                    ch_position: np.ndarray | None,
                    mask_intensity: float,
                    modality: str,
                    method: str = 'atlas',
                    k: int = 9,
                    rng: random.Random | None = None) -> List[int]:
    """Build the per-electrode mask used to exclude predictors during SMR.

    The neighborhood is first computed via :func:`find_neighborhood`, then
    a fraction ``mask_intensity`` of its members is sampled uniformly at
    random; the target electrode itself is always included in the returned
    mask so it never appears as its own predictor.

    With ``mask_intensity=0.0`` only the target is masked (the full
    neighborhood remains available as predictors). With
    ``mask_intensity=1.0`` the entire neighborhood plus the target are
    masked, matching the 100% condition described in the manuscript.

    Parameters
    ----------
    target_idx : int
        Index of the target electrode.
    ch_name, ch_position, modality, method, k : see :func:`find_neighborhood`.
    mask_intensity : float in [0, 1]
        Fraction of the neighborhood to mask.
    rng : random.Random, optional
        Random number generator to use for the sampling. Defaults to the
        module-level random state, matching the existing call sites.

    Returns
    -------
    list of int
        Sorted-unique indices of channels to mask, always containing
        ``target_idx``.
    """
    if not 0.0 <= mask_intensity <= 1.0:
        raise ValueError(f"mask_intensity must be in [0, 1]; got {mask_intensity!r}")

    neighborhood = find_neighborhood(
        target_idx=target_idx,
        ch_name=ch_name,
        ch_position=ch_position,
        modality=modality,
        method=method,
        k=k,
    )

    sampler = rng if rng is not None else random
    n_to_mask = int(round(len(neighborhood) * mask_intensity))
    if n_to_mask > 0 and len(neighborhood) > 0:
        sampled: Iterable[int] = sampler.sample(list(neighborhood), n_to_mask)
    else:
        sampled = []

    mask = set(int(j) for j in sampled)
    mask.add(int(target_idx))
    return sorted(mask)


# ---------------------------------------------------------------------------
# Backward-compatible wrappers matching the old call signatures
# ---------------------------------------------------------------------------
def find_index_mask_eeg(target_idx: int,
                        ch_name: Sequence[str],
                        mask_intensity: float,
                        ch_position: np.ndarray | None = None,
                        method: str = 'atlas',
                        k: int = 9) -> List[int]:
    """Backward-compatible EEG wrapper around :func:`find_index_mask`.

    Defaults to ``method='atlas'`` (the manuscript-matching behaviour).
    Pass ``method='knn'`` to reproduce `new/utils.py`'s behaviour.
    """
    return find_index_mask(
        target_idx=target_idx,
        ch_name=ch_name,
        ch_position=ch_position,
        mask_intensity=mask_intensity,
        modality='EEG',
        method=method,
        k=k,
    )


def find_index_mask_ieeg(target_idx: int,
                         ch_name: Sequence[str],
                         mask_intensity: float,
                         ch_position: np.ndarray | None = None,
                         method: str = 'atlas',
                         k: int = 9) -> List[int]:
    """Backward-compatible iEEG wrapper around :func:`find_index_mask`.

    Defaults to ``method='atlas'`` (the manuscript-matching behaviour).
    Pass ``method='knn'`` to reproduce `new/utils.py`'s behaviour.
    """
    return find_index_mask(
        target_idx=target_idx,
        ch_name=ch_name,
        ch_position=ch_position,
        mask_intensity=mask_intensity,
        modality='iEEG',
        method=method,
        k=k,
    )


__all__ = [
    'find_neighborhood',
    'find_index_mask',
    'find_index_mask_eeg',
    'find_index_mask_ieeg',
]
