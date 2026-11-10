
"""Tests for adaptive quality-diversity frame selection."""

import unittest

import numpy as np

from src.sampling.adaptive import (
    adaptive_indices,
    adaptive_quality_weight,
)
from src.sampling.combined import combined_indices


class TestAdaptiveSelection(unittest.TestCase):

    def test_clean_sequence_uses_low_quality_weight(self):
        frames = np.ones((10, 8, 8), dtype=np.uint8) * 255

        self.assertAlmostEqual(
            adaptive_quality_weight(frames), 0.25
        )

    def test_half_blank_sequence_increases_weight(self):
        frames = np.ones((10, 8, 8), dtype=np.uint8) * 255
        frames[:5] = 0

        self.assertAlmostEqual(
            adaptive_quality_weight(frames), 0.575
        )

    def test_all_blank_sequence_uses_maximum_weight(self):
        frames = np.zeros((10, 8, 8), dtype=np.uint8)

        self.assertAlmostEqual(
            adaptive_quality_weight(frames), 0.90
        )

    def test_matches_combined_at_computed_weight(self):
        rng = np.random.default_rng(42)
        frames = rng.integers(
            0, 2, size=(20, 8, 8), dtype=np.uint8
        )

        weight = adaptive_quality_weight(frames)

        self.assertEqual(
            adaptive_indices(frames, 5),
            combined_indices(
                frames, 5, quality_weight=weight
            ),
        )

    def test_sorted_unique_and_deterministic(self):
        rng = np.random.default_rng(42)
        frames = rng.integers(
            0, 256, size=(20, 8, 8), dtype=np.uint8
        )

        first = adaptive_indices(frames, 5)
        second = adaptive_indices(frames, 5)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(first, sorted(set(first)))

    def test_short_and_empty_sequences(self):
        frames = np.ones((3, 8, 8), dtype=np.uint8)

        self.assertEqual(
            adaptive_indices(frames, 5), [0, 1, 2]
        )

        empty = np.empty((0, 8, 8), dtype=np.uint8)

        self.assertEqual(adaptive_indices(empty, 5), [])

    def test_input_is_unchanged(self):
        frames = np.ones((10, 8, 8), dtype=np.uint8)
        original = frames.copy()

        adaptive_indices(frames, 5)

        np.testing.assert_array_equal(frames, original)

    def test_invalid_inputs(self):
        frames = np.zeros((5, 8, 8), dtype=np.uint8)

        for budget in [0, -1, 2.5, True]:
            with self.subTest(budget=budget):
                with self.assertRaises(ValueError):
                    adaptive_indices(frames, budget)

        with self.assertRaises(ValueError):
            adaptive_quality_weight(np.zeros((5, 8)))

        with self.assertRaises(ValueError):
            adaptive_quality_weight(
                np.full((5, 8, 8), np.nan)
            )


if __name__ == "__main__":
    unittest.main()
