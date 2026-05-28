"""
Cross-subject similarity matrices for CBEM.

This module provides Pearson-similarity utilities that match the equation
in Section *Cross-subject evaluation* of the manuscript:

    C_{ss'}(i, j) = cov(x_i^{(s)}, x_j^{(s')}) / (sigma_i^{(s)} sigma_j^{(s')}),

where x_i^{(s)} is the **full time series** of electrode i in subject s
(concatenated across trials, not the per-trial signal).

The previous implementation in `new/generate_adj_matrix_ieeg_unseen.py`
(and in `gihub_repo/Generate_Similarity_Matrix_ieeg_unseen.py`) instead
computed Pearson on each pair of *aligned* trials and then averaged
across trials. That has two problems:

    * It assumes trial i of subject s and trial i of subject s' represent
      the same event, which is not generally true in the AJILE12 cohort
      where subjects perform different numbers of movements at different
      times.

    * The loop range is taken from one of the two subjects
      (`data_dst.shape[0]` in the previous code), so when trial counts
      differ the longer subject is silently truncated and the shorter
      one's surplus trials are silently dropped.

Concatenating across trials and then computing a single Pearson per
(electrode_src, electrode_dst) pair removes both issues. When the two
subjects' total recording lengths differ, we truncate both to the common
length so the Pearson formula is well-defined; this is the only place a
length mismatch enters, and it is explicit rather than hidden.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _concat_trials(data: np.ndarray) -> np.ndarray:
    """Concatenate trials within a subject.

    Parameters
    ----------
    data : ndarray of shape (n_trials, n_channels, n_time)

    Returns
    -------
    ndarray of shape (n_channels, n_trials * n_time)
        A single 1-D time series per channel.
    """
    if data.ndim != 3:
        raise ValueError(
            f"Expected (n_trials, n_channels, n_time); got shape {data.shape}."
        )
    n_trials, n_channels, n_time = data.shape
    # (n_trials, n_channels, n_time) -> (n_channels, n_trials, n_time)
    transposed = np.transpose(data, (1, 0, 2))
    return transposed.reshape(n_channels, n_trials * n_time)


def _pearson_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation between every row of A and every row of B.

    Parameters
    ----------
    A : ndarray of shape (n_a, L)
    B : ndarray of shape (n_b, L)

    Returns
    -------
    ndarray of shape (n_a, n_b)
        Entry [i, j] is Pearson(A[i], B[j]).
    """
    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"A and B must share the time dimension; got {A.shape} vs {B.shape}."
        )
    A_c = A - A.mean(axis=1, keepdims=True)
    B_c = B - B.mean(axis=1, keepdims=True)
    A_n = np.linalg.norm(A_c, axis=1, keepdims=True)
    B_n = np.linalg.norm(B_c, axis=1, keepdims=True)

    # Avoid divide-by-zero for constant rows.
    A_n_safe = np.where(A_n == 0, 1.0, A_n)
    B_n_safe = np.where(B_n == 0, 1.0, B_n)
    A_u = A_c / A_n_safe
    B_u = B_c / B_n_safe

    C = A_u @ B_u.T
    # Rows / columns whose underlying signal was constant get NaN, matching
    # what `np.corrcoef` would have produced.
    bad_a = (A_n.squeeze(-1) == 0)
    bad_b = (B_n.squeeze(-1) == 0)
    if bad_a.any():
        C[bad_a, :] = np.nan
    if bad_b.any():
        C[:, bad_b] = np.nan
    return C


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def cross_subject_pearson(
    data_src: np.ndarray,
    data_dst: np.ndarray,
    *,
    length_policy: str = 'truncate',
) -> Tuple[np.ndarray, dict]:
    """Compute the cross-subject Pearson similarity matrix from full time series.

    The two subjects' recordings are first flattened across trials to give
    a single time series per electrode. If the resulting series differ in
    total length (different trial counts and/or different epoch lengths),
    they are reconciled per ``length_policy`` before computing Pearson.

    Parameters
    ----------
    data_src : ndarray of shape (n_trials_src, n_chan_src, n_time_src)
    data_dst : ndarray of shape (n_trials_dst, n_chan_dst, n_time_dst)
    length_policy : {'truncate', 'error'}
        How to handle a length mismatch after trial concatenation:
        - ``'truncate'`` (default): truncate both series to
          ``min(L_src, L_dst)`` samples and report the truncation in the
          returned diagnostics dict.
        - ``'error'``: raise ``ValueError`` if the two concatenated
          lengths differ.

    Returns
    -------
    C : ndarray of shape (n_chan_src, n_chan_dst)
        Pearson similarity matrix. Entries involving a constant signal
        on either side are NaN.
    info : dict
        Diagnostics: ``L_src``, ``L_dst``, ``L_used``, and ``n_nan`` (the
        number of NaN entries in ``C``).

    Notes
    -----
    This implements the manuscript equation in vectorised form:
        C[i, j] = corr(x_i^{(s)}, x_j^{(s')})
    where x is the within-subject concatenation of trial-level signals.
    No trial pairing is assumed between subjects.
    """
    src_flat = _concat_trials(np.asarray(data_src))
    dst_flat = _concat_trials(np.asarray(data_dst))

    L_src = src_flat.shape[1]
    L_dst = dst_flat.shape[1]

    if L_src != L_dst:
        if length_policy == 'error':
            raise ValueError(
                f"Concatenated lengths differ ({L_src} vs {L_dst}); "
                "pass length_policy='truncate' to allow truncation."
            )
        if length_policy != 'truncate':
            raise ValueError(
                f"Unknown length_policy={length_policy!r}; "
                "expected 'truncate' or 'error'."
            )
        L = min(L_src, L_dst)
        src_flat = src_flat[:, :L]
        dst_flat = dst_flat[:, :L]
    else:
        L = L_src

    C = _pearson_rows(src_flat, dst_flat)
    info = {
        'L_src': L_src,
        'L_dst': L_dst,
        'L_used': L,
        'n_nan': int(np.isnan(C).sum()),
    }
    return C, info


def cross_subject_pearson_legacy_per_trial_mean(
    data_src: np.ndarray,
    data_dst: np.ndarray,
) -> np.ndarray:
    """Reproduce the previous (per-trial-then-mean) similarity for comparison.

    Provided only so users can confirm the new and old similarity matrices
    differ as expected. Should not be used to generate new results.

    Trial alignment uses ``min(n_trials_src, n_trials_dst)`` rather than
    silently truncating one side — that's the only change relative to the
    original buggy loop.
    """
    src = np.asarray(data_src)
    dst = np.asarray(data_dst)
    n_trials = min(src.shape[0], dst.shape[0])
    if n_trials == 0:
        raise ValueError("At least one of the subjects has zero trials.")
    n_src, n_dst = src.shape[1], dst.shape[1]

    sim = np.zeros((n_src, n_dst))
    for m in range(n_src):
        for n in range(n_dst):
            vals = []
            for i in range(n_trials):
                a = src[i, m, :]
                b = dst[i, n, :]
                if a.std() == 0 or b.std() == 0:
                    continue
                v = np.corrcoef(a, b)[0, 1]
                if not np.isnan(v):
                    vals.append(v)
            sim[m, n] = np.mean(vals) if vals else np.nan
    return sim


__all__ = [
    'cross_subject_pearson',
    'cross_subject_pearson_legacy_per_trial_mean',
]
