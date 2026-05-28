"""
Cross-subject CBEM script with the Pearson similarity bug fixed and a
strict one-to-one Hungarian assignment.

Differences vs. `new/generate_adj_matrix_ieeg_unseen.py`:

1. Pearson similarity is computed once per electrode pair on the
   within-subject concatenation of trial signals
   (see `fixed/similarity.cross_subject_pearson`), matching the
   manuscript equation
       C_{ss'}(i, j) = cov(x_i^{(s)}, x_j^{(s')}) / (sigma_i^{(s)} sigma_j^{(s')}).
   The previous per-trial-then-mean computation is gone.

2. When the two subjects have different total recording lengths after
   trial concatenation, the longer side is truncated explicitly and the
   truncation is logged into the saved diagnostics dict.

3. The Hungarian step is now strictly one-to-one. The previous code had
   a greedy fallback when ``n_src > n_dst`` that could re-use a single
   destination electrode for multiple source electrodes, violating the
   formal constraint
       P 1 <= 1  and  P^T 1 <= 1
   in the manuscript. The fallback is removed: when one subject has
   fewer electrodes than the other, only ``min(n_src, n_dst)`` source
   electrodes receive a donor, and the unmatched rows of the transferred
   adjacency matrix are left at zero (with a boolean ``matched_mask``
   saved alongside so downstream code can skip them).
"""

from __future__ import annotations

import os
import pickle as pkl
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from similarity import cross_subject_pearson


# ---------------------------------------------------------------------------
# Strict 1-to-1 Hungarian assignment
# ---------------------------------------------------------------------------
def hungarian_strict(sim_matrix: np.ndarray) -> Tuple[Dict[int, int], np.ndarray]:
    """Solve the rectangular assignment problem on ``sim_matrix``.

    ``linear_sum_assignment`` on a non-square cost matrix already returns
    a strict one-to-one assignment of size ``min(n_src, n_dst)``; this
    function wraps it and returns:

    * ``mapping``: a dict ``{src_row -> dst_col}`` for the matched rows.
      Source rows absent from the mapping have no donor (they exceed the
      number of destination electrodes available, or vice-versa for
      destination columns absent from the dict's values).
    * ``matched_mask``: a boolean array of length ``n_src``; ``True`` at
      position ``m`` iff source row ``m`` received a donor electrode.

    Together these enforce the formal constraints
    ``P 1 <= 1`` and ``P^T 1 <= 1`` without resorting to greedy reuse.
    """
    n_src, n_dst = sim_matrix.shape

    # `linear_sum_assignment` minimises cost; we want to maximise similarity.
    # Replace NaNs with the minimum finite value so they never become the
    # optimal assignment but the routine itself doesn't fail.
    finite = sim_matrix[np.isfinite(sim_matrix)]
    fill_value = float(finite.min()) if finite.size else 0.0
    safe = np.where(np.isnan(sim_matrix), fill_value, sim_matrix)

    row_ind, col_ind = linear_sum_assignment(-safe)
    mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind)}

    matched_mask = np.zeros(n_src, dtype=bool)
    matched_mask[list(mapping.keys())] = True
    return mapping, matched_mask


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def build_cross_subject_adjacency(settings, paths) -> None:
    """Build CBEM-transferred adjacency matrices for every (src, dst) pair.

    Inputs come from ``paths.ieeg_subject_file`` and
    ``paths.donor_adj_matrix_file``; outputs go to
    ``paths.cross_subject_adj_dir(src)``.
    """
    num_subjects = settings.num_subject

    for src_sub in range(num_subjects):
        save_path = paths.cross_subject_adj_dir(src_sub)
        os.makedirs(save_path, exist_ok=True)

        with open(paths.ieeg_subject_file(src_sub), 'rb') as f:
            data_src = pkl.load(f)

        for dst_sub in range(num_subjects):
            if dst_sub == src_sub:
                continue

            print(f'Processing src={src_sub} -> dst={dst_sub}')

            with open(paths.ieeg_subject_file(dst_sub), 'rb') as f:
                data_dst = pkl.load(f)
            adj_mat_dst = np.load(paths.donor_adj_matrix_file(dst_sub))

            # --- Cross-subject similarity (concatenated time series) ---
            sim_matrix, sim_info = cross_subject_pearson(
                data_src=data_src,
                data_dst=data_dst,
                length_policy='truncate',
            )
            if sim_info['L_src'] != sim_info['L_dst']:
                print(
                    f"  length mismatch after trial concatenation: "
                    f"L_src={sim_info['L_src']}, L_dst={sim_info['L_dst']}; "
                    f"truncated to {sim_info['L_used']} samples."
                )
            if sim_info['n_nan']:
                print(f"  warning: {sim_info['n_nan']} NaN entries in C_ss'")
                # NaNs would break linear_sum_assignment; replace with the
                # minimum finite similarity so they are last in the ranking.
                finite = sim_matrix[np.isfinite(sim_matrix)]
                fill = float(finite.min()) if finite.size else 0.0
                sim_matrix = np.where(np.isnan(sim_matrix), fill, sim_matrix)

            # --- Hungarian assignment (strict 1-to-1) ---
            mapping, matched_mask = hungarian_strict(sim_matrix)
            n_matched = int(matched_mask.sum())
            n_src = data_src.shape[1]
            if n_matched < n_src:
                print(
                    f"  {n_src - n_matched} of {n_src} source electrodes have no "
                    f"donor counterpart (n_dst={data_dst.shape[1]}); their "
                    f"adjacency rows are left at zero."
                )

            # --- Transfer donor's adjacency matrix into target electrode space ---
            adj_matrix = np.zeros((n_src, n_src))
            for m in tqdm(range(n_src), desc='building adjacency matrix'):
                if m not in mapping:
                    continue   # source row has no donor → row stays zero
                index_row = mapping[m]
                for n in range(n_src):
                    if n not in mapping:
                        continue   # source column has no donor → entry stays zero
                    index_col = mapping[n]
                    adj_matrix[m, n] = adj_mat_dst[index_row, index_col]

            np.save(
                os.path.join(save_path, f'adj_mat_corr_sub{src_sub}_base_sub{dst_sub}.npy'),
                adj_matrix,
            )
            # Save which source rows were actually matched (boolean mask) so
            # downstream evaluation can skip unmatched electrodes cleanly.
            np.save(
                os.path.join(save_path, f'matched_mask_sub{src_sub}_base_sub{dst_sub}.npy'),
                matched_mask,
            )
            # Save the diagnostics next to the adjacency matrix so the
            # length-truncation policy and the mapping are recoverable
            # after the fact.
            full_info = dict(sim_info)
            full_info['mapping'] = mapping
            full_info['n_src'] = n_src
            full_info['n_dst'] = int(data_dst.shape[1])
            full_info['n_matched'] = n_matched
            np.save(
                os.path.join(save_path, f'sim_info_sub{src_sub}_base_sub{dst_sub}.npy'),
                full_info, allow_pickle=True,
            )

        print(f"Done src={src_sub}.")


if __name__ == '__main__':
    from setting import Settings
    from path import Paths

    settings = Settings()
    settings.load_settings()
    paths = Paths(settings)
    paths.load_device_paths()
    build_cross_subject_adjacency(settings, paths)
