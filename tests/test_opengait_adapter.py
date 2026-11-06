
"""Tests for the OpenGait frame-selection adapter."""

import unittest

import numpy as np

from src.sampling.selector import METHODS
from src.sampling.opengait_adapter import select_opengait_indices


class TestOpenGaitAdapter(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(42)
        self.frames = rng.integers(
            0, 256, size=(20, 64, 64), dtype=np.uint8
        )
        self.seqs = [self.frames]

    def test_all_methods(self):
        for method in METHODS:
            with self.subTest(method=method):
                indices = select_opengait_indices(
                    self.seqs, budget=5, method=method
                )
                self.assertEqual(len(indices), 5)

    def test_indices_are_sorted_and_unique(self):
        for method in METHODS:
            with self.subTest(method=method):
                indices = select_opengait_indices(
                    self.seqs, 5, method
                )
                self.assertEqual(indices, sorted(set(indices)))

    def test_first_n_matches_project2(self):
        indices = select_opengait_indices(
            self.seqs, 5, "first_n"
        )
        self.assertEqual(indices, [0, 1, 2, 3, 4])

    def test_uniform_uses_full_sequence(self):
        indices = select_opengait_indices(
            self.seqs, 5, "uniform"
        )
        self.assertEqual(indices, [0, 4, 9, 14, 19])

    def test_short_sequence(self):
        indices = select_opengait_indices(
            [self.frames[:3]], 10, "uniform"
        )
        self.assertEqual(indices, [0, 1, 2])

    def test_original_data_unchanged(self):
        original = self.frames.copy()

        select_opengait_indices(
            self.seqs, 5, "combined"
        )

        np.testing.assert_array_equal(self.frames, original)

    def test_empty_sequence_rejected(self):
        with self.assertRaises(ValueError):
            select_opengait_indices(
                [np.empty((0, 64, 64), dtype=np.uint8)],
                5,
            )

    def test_invalid_structure(self):
        with self.assertRaises(ValueError):
            select_opengait_indices([], 5)

        with self.assertRaises(ValueError):
            select_opengait_indices(
                [self.frames, self.frames], 5
            )

        with self.assertRaises(ValueError):
            select_opengait_indices(
                [np.zeros((20, 64))], 5
            )


if __name__ == "__main__":
    unittest.main()
