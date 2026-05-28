"""
Direct verification that `find_index_mask` honours `mask_intensity` for
BOTH neighborhood methods, and that the curve is monotone in the
expected way.

Expected behaviour (per the manuscript and the github_repo baseline):

    mask_intensity = 0.00  -->  only the target is masked
    mask_intensity = 0.25  -->  ~25% of the neighborhood + target
    mask_intensity = 0.50  -->  ~50% of the neighborhood + target
    mask_intensity = 0.75  -->  ~75% of the neighborhood + target
    mask_intensity = 1.00  -->  full neighborhood + target

The verification builds toy `ch_name` / `ch_position` arrays so the test
does not depend on the project's pickled data.
"""

import random
import sys

# Re-use the same numpy mock used in fixed/test_neighborhood smoke tests
# when numpy isn't installed, so the atlas path can still be verified.
try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

    class _Mock:
        def __getattr__(self, name):
            raise RuntimeError("numpy not available; atlas-only smoke test")

    sys.modules.setdefault('numpy', _Mock())

from neighborhood import find_index_mask, find_neighborhood


def _expected_mask_size(neighborhood_size: int, intensity: float) -> int:
    """The github_repo / fixed formula: round(|N| * m) + the target."""
    return int(round(neighborhood_size * intensity)) + 1


def verify_atlas_eeg():
    print('\n[atlas / EEG]  10-10 montage frontal group, target=Fz')
    ch_name = ['F3', 'F1', 'Fz', 'F2', 'F4']
    target = ch_name.index('Fz')
    rng = random.Random(0)
    n_nbhd = len(find_neighborhood(target, ch_name, None,
                                   modality='EEG', method='atlas'))
    for m in [0.0, 0.25, 0.5, 0.75, 1.0]:
        mask = find_index_mask(target, ch_name, None,
                               mask_intensity=m,
                               modality='EEG', method='atlas',
                               rng=rng)
        expected = _expected_mask_size(n_nbhd, m)
        status = 'OK ' if len(mask) == expected else 'FAIL'
        print(f'  m={m:>4}  |mask|={len(mask)}  expected={expected}  {status}')


def verify_atlas_ieeg():
    print('\n[atlas / iEEG]  AAL labels, target electrode 0 (Frontal_Sup_L)')
    ch_name = (['Frontal_Sup_L'] * 4 +
               ['Parietal_Inf_R'] * 3 +
               ['Temporal_Mid_L'] * 2)
    target = 0
    rng = random.Random(0)
    n_nbhd = len(find_neighborhood(target, ch_name, None,
                                   modality='iEEG', method='atlas'))
    for m in [0.0, 0.25, 0.5, 0.75, 1.0]:
        mask = find_index_mask(target, ch_name, None,
                               mask_intensity=m,
                               modality='iEEG', method='atlas',
                               rng=rng)
        expected = _expected_mask_size(n_nbhd, m)
        status = 'OK ' if len(mask) == expected else 'FAIL'
        print(f'  m={m:>4}  |mask|={len(mask)}  expected={expected}  {status}')


def verify_knn():
    if not HAVE_NUMPY:
        print('\n[knn]  skipped (numpy not available on this interpreter)')
        return

    print('\n[knn / iEEG]  k=9 random electrode positions, target=0')
    rng_np = np.random.default_rng(0)
    ch_position = rng_np.normal(size=(20, 3))
    target = 0
    rng = random.Random(0)
    n_nbhd = len(find_neighborhood(target, None, ch_position,
                                   modality='iEEG', method='knn', k=9))
    for m in [0.0, 0.25, 0.5, 0.75, 1.0]:
        mask = find_index_mask(target, None, ch_position,
                               mask_intensity=m,
                               modality='iEEG', method='knn', k=9,
                               rng=rng)
        expected = _expected_mask_size(n_nbhd, m)
        status = 'OK ' if len(mask) == expected else 'FAIL'
        print(f'  m={m:>4}  |mask|={len(mask)}  expected={expected}  {status}')


def verify_endpoints():
    """Spot-check the two endpoints that pin the sweep:
       m=0  -> only the target
       m=1  -> full neighborhood + target  (matches github_repo behaviour)
    """
    print('\n[endpoints]  m=0 keeps everything, m=1 masks the whole neighborhood')
    ch_name = ['F3', 'F1', 'Fz', 'F2', 'F4']
    target = ch_name.index('Fz')

    mask_zero = find_index_mask(target, ch_name, None,
                                mask_intensity=0.0,
                                modality='EEG', method='atlas')
    print(f'  m=0.0 (atlas EEG): mask={mask_zero}  '
          f'{"OK " if mask_zero == [target] else "FAIL"}')

    mask_one = find_index_mask(target, ch_name, None,
                               mask_intensity=1.0,
                               modality='EEG', method='atlas')
    print(f'  m=1.0 (atlas EEG): mask={sorted(mask_one)}  '
          f'{"OK " if set(mask_one) == set(range(len(ch_name))) else "FAIL"}')


if __name__ == '__main__':
    verify_atlas_eeg()
    verify_atlas_ieeg()
    verify_knn()
    verify_endpoints()
