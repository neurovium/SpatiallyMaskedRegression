"""
Smoke tests for `fixed/similarity.py`.

Run with:
    python -m unittest fixed.test_similarity
or:
    python fixed/test_similarity.py
"""

import unittest

import numpy as np

from similarity import (
    cross_subject_pearson,
    cross_subject_pearson_legacy_per_trial_mean,
)


class TestCrossSubjectPearson(unittest.TestCase):
    def test_identical_data_gives_identity_like_diagonal(self):
        rng = np.random.default_rng(0)
        data = rng.normal(size=(5, 4, 30))   # 5 trials, 4 channels, 30 samples

        C, info = cross_subject_pearson(data, data)
        self.assertEqual(C.shape, (4, 4))
        # Diagonal should be 1 within fp tolerance.
        np.testing.assert_allclose(np.diag(C), 1.0, atol=1e-6)
        self.assertEqual(info['L_src'], info['L_dst'])
        self.assertEqual(info['n_nan'], 0)

    def test_handles_different_trial_counts(self):
        rng = np.random.default_rng(1)
        src = rng.normal(size=(7, 3, 20))
        dst = rng.normal(size=(4, 5, 20))    # different trial count and channel count

        C, info = cross_subject_pearson(src, dst, length_policy='truncate')
        self.assertEqual(C.shape, (3, 5))
        self.assertEqual(info['L_src'], 7 * 20)
        self.assertEqual(info['L_dst'], 4 * 20)
        self.assertEqual(info['L_used'], 4 * 20)
        # Pearson values stay in [-1, 1]
        self.assertTrue(np.all(np.abs(C) <= 1.0 + 1e-9))

    def test_handles_different_epoch_lengths(self):
        rng = np.random.default_rng(2)
        src = rng.normal(size=(5, 3, 30))
        dst = rng.normal(size=(5, 3, 25))    # different epoch length

        C, info = cross_subject_pearson(src, dst, length_policy='truncate')
        self.assertEqual(info['L_used'], 5 * 25)
        self.assertEqual(C.shape, (3, 3))

    def test_error_policy(self):
        rng = np.random.default_rng(3)
        src = rng.normal(size=(5, 3, 30))
        dst = rng.normal(size=(4, 3, 30))    # different total length
        with self.assertRaises(ValueError):
            cross_subject_pearson(src, dst, length_policy='error')

    def test_constant_channel_gives_nan_row(self):
        rng = np.random.default_rng(4)
        src = rng.normal(size=(5, 3, 30))
        # Make channel 1 of src constant.
        src[:, 1, :] = 7.0
        dst = rng.normal(size=(5, 2, 30))

        C, info = cross_subject_pearson(src, dst)
        self.assertTrue(np.all(np.isnan(C[1, :])))
        self.assertEqual(info['n_nan'], 2)  # 1 row x 2 cols all NaN

    def test_matches_naive_concat_pearson(self):
        """Vectorised result should agree with a naive
        np.corrcoef-on-concatenated-trials baseline."""
        rng = np.random.default_rng(5)
        src = rng.normal(size=(6, 3, 25))
        dst = rng.normal(size=(6, 4, 25))

        C, _ = cross_subject_pearson(src, dst)

        # Naive reference.
        ref = np.zeros((3, 4))
        for m in range(3):
            a = src[:, m, :].reshape(-1)
            for n in range(4):
                b = dst[:, n, :].reshape(-1)
                ref[m, n] = np.corrcoef(a, b)[0, 1]

        np.testing.assert_allclose(C, ref, atol=1e-8)


class TestLegacyComparison(unittest.TestCase):
    def test_legacy_matches_old_buggy_behaviour_on_aligned_inputs(self):
        """When trial counts and epoch lengths match, the legacy function
        should produce per-trial-then-mean correlations (different from the
        concatenation-based version)."""
        rng = np.random.default_rng(6)
        src = rng.normal(size=(5, 3, 25))
        dst = rng.normal(size=(5, 4, 25))

        legacy = cross_subject_pearson_legacy_per_trial_mean(src, dst)
        new, _ = cross_subject_pearson(src, dst)

        self.assertEqual(legacy.shape, new.shape)
        # The two methods should *not* agree element-wise; the difference is
        # the whole point of the fix.
        diff = np.abs(legacy - new)
        self.assertGreater(diff.max(), 1e-3)


if __name__ == '__main__':
    unittest.main()
