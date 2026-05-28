"""
Smoke tests for `hungarian_strict` in `fixed/generate_adj_matrix_ieeg_unseen.py`.

Run with:
    python -m unittest fixed.test_hungarian
or:
    python fixed/test_hungarian.py
"""

import unittest

import numpy as np

from generate_adj_matrix_ieeg_unseen import hungarian_strict


class TestHungarianStrict(unittest.TestCase):
    def test_square_matrix_full_bijection(self):
        # Identity-like cost: source i should match dst i.
        sim = np.eye(5)
        mapping, mask = hungarian_strict(sim)
        self.assertEqual(mapping, {i: i for i in range(5)})
        self.assertTrue(mask.all())

    def test_rectangular_more_sources_than_destinations(self):
        # n_src=6, n_dst=4 -> exactly 4 assignments, 2 unmatched sources.
        rng = np.random.default_rng(0)
        sim = rng.normal(size=(6, 4))
        mapping, mask = hungarian_strict(sim)
        self.assertEqual(len(mapping), 4)
        # Each destination column is used at most once.
        cols = list(mapping.values())
        self.assertEqual(len(cols), len(set(cols)))
        # Exactly 4 source rows marked matched.
        self.assertEqual(int(mask.sum()), 4)

    def test_rectangular_more_destinations_than_sources(self):
        # n_src=3, n_dst=7 -> all sources matched, 4 destinations unused.
        rng = np.random.default_rng(1)
        sim = rng.normal(size=(3, 7))
        mapping, mask = hungarian_strict(sim)
        self.assertEqual(len(mapping), 3)
        self.assertTrue(mask.all())
        self.assertEqual(len(set(mapping.values())), 3)

    def test_no_destination_is_reused(self):
        """Direct check of the property the fallback used to violate."""
        rng = np.random.default_rng(2)
        for shape in [(10, 5), (5, 10), (8, 8), (12, 7)]:
            sim = rng.normal(size=shape)
            mapping, _ = hungarian_strict(sim)
            cols = list(mapping.values())
            self.assertEqual(
                len(cols), len(set(cols)),
                f"destination reused at shape={shape}: {cols}",
            )

    def test_handles_nans_in_similarity(self):
        sim = np.array([
            [0.9, 0.1, np.nan],
            [0.2, 0.8, 0.3],
            [0.1, 0.4, 0.7],
        ])
        mapping, mask = hungarian_strict(sim)
        # All three rows still get a destination.
        self.assertEqual(set(mapping.keys()), {0, 1, 2})
        # No destination reused.
        self.assertEqual(len(set(mapping.values())), 3)
        self.assertTrue(mask.all())

    def test_maximises_total_similarity(self):
        # A trivial case where the optimal is obvious.
        sim = np.array([
            [10.0, 1.0, 1.0],
            [1.0, 10.0, 1.0],
            [1.0, 1.0, 10.0],
        ])
        mapping, _ = hungarian_strict(sim)
        total = sum(sim[r, c] for r, c in mapping.items())
        self.assertAlmostEqual(total, 30.0)


if __name__ == '__main__':
    unittest.main()
