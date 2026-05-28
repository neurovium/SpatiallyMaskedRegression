"""
Smoke tests for `neighborhood.py`. Run with:

    python -m unittest fixed.test_neighborhood

or simply:

    python fixed/test_neighborhood.py
"""

import random
import unittest

import numpy as np

from neighborhood import (
    find_index_mask,
    find_index_mask_eeg,
    find_index_mask_ieeg,
    find_neighborhood,
)


class TestAtlasEEG(unittest.TestCase):
    def setUp(self):
        # A subset of the standard 10-10/10-5 channels used in the project.
        self.ch_name = [
            'F3', 'F1', 'Fz', 'F2', 'F4',                  # frontal group
            'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',  # FC group
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',      # central group
        ]

    def test_neighborhood_contains_only_same_group(self):
        # 'Fz' lives in the frontal group with F3, F1, F2, F4.
        target = self.ch_name.index('Fz')
        nbrs = find_neighborhood(target, self.ch_name, None,
                                 modality='EEG', method='atlas')
        nbr_names = {self.ch_name[i] for i in nbrs}
        self.assertEqual(nbr_names, {'F3', 'F1', 'F2', 'F4'})
        self.assertNotIn(target, nbrs)

    def test_intensity_zero_masks_only_target(self):
        target = self.ch_name.index('Cz')
        mask = find_index_mask(target, self.ch_name, None,
                               mask_intensity=0.0,
                               modality='EEG', method='atlas')
        self.assertEqual(mask, [target])

    def test_intensity_one_masks_full_neighborhood_plus_target(self):
        target = self.ch_name.index('Cz')
        nbrs = find_neighborhood(target, self.ch_name, None,
                                 modality='EEG', method='atlas')
        mask = find_index_mask(target, self.ch_name, None,
                               mask_intensity=1.0,
                               modality='EEG', method='atlas')
        self.assertEqual(set(mask), set(nbrs) | {target})

    def test_intermediate_intensity_sizes(self):
        target = self.ch_name.index('Cz')
        nbrs = find_neighborhood(target, self.ch_name, None,
                                 modality='EEG', method='atlas')
        # 0.5 of 6 neighbors -> 3 of them sampled, plus the target.
        rng = random.Random(0)
        mask = find_index_mask(target, self.ch_name, None,
                               mask_intensity=0.5,
                               modality='EEG', method='atlas',
                               rng=rng)
        self.assertEqual(len(mask), 4)
        self.assertIn(target, mask)
        for idx in mask:
            if idx != target:
                self.assertIn(idx, nbrs)


class TestAtlasIEEG(unittest.TestCase):
    def test_same_label_neighborhood(self):
        ch_name = ['Frontal_Sup_L'] * 3 + ['Parietal_Inf_R'] * 2 + ['Temporal_Mid_L']
        nbrs = find_neighborhood(0, ch_name, None,
                                 modality='iEEG', method='atlas')
        self.assertEqual(set(nbrs), {1, 2})

    def test_mask_intensity_one(self):
        ch_name = ['Frontal_Sup_L'] * 3 + ['Parietal_Inf_R'] * 2
        mask = find_index_mask(0, ch_name, None,
                               mask_intensity=1.0,
                               modality='iEEG', method='atlas')
        self.assertEqual(set(mask), {0, 1, 2})


class TestKNN(unittest.TestCase):
    def test_knn_picks_closest_k(self):
        # Place electrode 0 at the origin and others on a line.
        ch_position = np.array([[0, 0, 0],
                                [1, 0, 0],
                                [2, 0, 0],
                                [3, 0, 0],
                                [10, 0, 0]], dtype=float)
        nbrs = find_neighborhood(0, ch_name=None, ch_position=ch_position,
                                 modality='iEEG', method='knn', k=3)
        self.assertEqual(nbrs, [1, 2, 3])

    def test_knn_excludes_target(self):
        ch_position = np.random.RandomState(1).randn(15, 3)
        for target in range(15):
            nbrs = find_neighborhood(target, ch_name=None,
                                     ch_position=ch_position,
                                     modality='EEG', method='knn', k=5)
            self.assertNotIn(target, nbrs)
            self.assertEqual(len(nbrs), 5)

    def test_eeg_uses_xy_ieeg_uses_xz(self):
        # Place electrodes such that the y-axis matters for EEG (xy) but
        # the z-axis matters for iEEG (xz). For EEG only neighbour 1 is
        # close; for iEEG only neighbour 2 is close.
        ch_position = np.array([
            [0, 0, 0],   # target
            [0, 1, 100],  # close in (x,y), far in (x,z)
            [0, 100, 1],  # far in (x,y), close in (x,z)
        ], dtype=float)
        eeg_nbr = find_neighborhood(0, None, ch_position, 'EEG', 'knn', k=1)
        ieeg_nbr = find_neighborhood(0, None, ch_position, 'iEEG', 'knn', k=1)
        self.assertEqual(eeg_nbr, [1])
        self.assertEqual(ieeg_nbr, [2])


class TestBackwardCompatWrappers(unittest.TestCase):
    def test_eeg_wrapper_default_is_atlas(self):
        ch_name = ['F3', 'F1', 'Fz', 'F2', 'F4']
        mask = find_index_mask_eeg(2, ch_name, mask_intensity=1.0)
        self.assertEqual(set(mask), {0, 1, 2, 3, 4})

    def test_ieeg_wrapper_default_is_atlas(self):
        ch_name = ['A', 'A', 'B', 'A', 'C']
        mask = find_index_mask_ieeg(0, ch_name, mask_intensity=1.0)
        self.assertEqual(set(mask), {0, 1, 3})

    def test_eeg_wrapper_knn_path(self):
        ch_name = [f'ch{i}' for i in range(5)]
        ch_position = np.array([[0, 0, 0],
                                [1, 0, 0],
                                [2, 0, 0],
                                [3, 0, 0],
                                [10, 0, 0]], dtype=float)
        mask = find_index_mask_eeg(0, ch_name, mask_intensity=1.0,
                                   ch_position=ch_position, method='knn', k=3)
        self.assertEqual(set(mask), {0, 1, 2, 3})


class TestInputValidation(unittest.TestCase):
    def test_atlas_requires_ch_name(self):
        with self.assertRaises(ValueError):
            find_neighborhood(0, ch_name=None, ch_position=np.zeros((2, 3)),
                              modality='EEG', method='atlas')

    def test_knn_requires_ch_position(self):
        with self.assertRaises(ValueError):
            find_neighborhood(0, ch_name=['a', 'b'], ch_position=None,
                              modality='EEG', method='knn')

    def test_invalid_method(self):
        with self.assertRaises(ValueError):
            find_neighborhood(0, ch_name=['a', 'b'],
                              ch_position=np.zeros((2, 3)),
                              modality='EEG', method='nope')

    def test_invalid_intensity(self):
        with self.assertRaises(ValueError):
            find_index_mask(0, ch_name=['a', 'b'],
                            ch_position=np.zeros((2, 3)),
                            mask_intensity=1.5,
                            modality='EEG', method='atlas')


if __name__ == '__main__':
    unittest.main()
