
"""Unit tests for combined quality-diversity sampling."""

import unittest
import numpy as np

from src.sampling.combined import combined_indices
from src.sampling.quality import quality_indices


class TestCombinedSelection(unittest.TestCase):

    def test_empty_sequence(self):
        frames = np.empty((0, 8, 8), dtype=np.uint8)
        self.assertEqual(combined_indices(frames, 5), [])

    def test_short_sequence(self):
        frames = np.zeros((3, 8, 8), dtype=np.uint8)
        self.assertEqual(combined_indices(frames, 5), [0, 1, 2])

    def test_exact_budget(self):
        frames = np.zeros((5, 8, 8), dtype=np.uint8)
        self.assertEqual(
            combined_indices(frames, 5), [0, 1, 2, 3, 4]
        )

    def test_single_frame_prefers_quality(self):
        frames = np.zeros((3, 8, 8), dtype=np.uint8)
        frames[1, 2:6, 2:6] = 255

        self.assertEqual(combined_indices(frames, 1), [1])

    def test_quality_only_matches_quality_baseline(self):
        frames = np.zeros((5, 8, 8), dtype=np.uint8)
        frames[0, 2:6, 2:6] = 255
        frames[1, 2:6, 2:6] = 255
        frames[2, 2:6, 2:6] = 255
        frames[3] = 255

        self.assertEqual(
            combined_indices(frames, 2, quality_weight=1.0),
            quality_indices(frames, 2),
        )

    def test_diversity_changes_selection(self):
        frames = np.zeros((4, 8, 8), dtype=np.uint8)
        frames[0, 2:6, 2:6] = 255
        frames[1, 2:6, 2:6] = 255
        frames[2, 1:5, 1:5] = 255
        frames[3, 4:8, 4:8] = 255

        quality_only = combined_indices(
            frames, 2, quality_weight=1.0
        )
        diversity_only = combined_indices(
            frames, 2, quality_weight=0.0
        )

        self.assertNotEqual(quality_only, diversity_only)

    def test_sorted_unique_and_deterministic(self):
        rng = np.random.default_rng(42)
        frames = rng.integers(
            0, 2, size=(20, 8, 8), dtype=np.uint8
        )

        first = combined_indices(frames, 5)
        second = combined_indices(frames, 5)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(first, sorted(set(first)))

    def test_invalid_inputs(self):
        frames = np.zeros((5, 8, 8), dtype=np.uint8)

        for budget in [0, -1, 2.5, True]:
            with self.subTest(budget=budget):
                with self.assertRaises(ValueError):
                    combined_indices(frames, budget)

        for weight in [-0.1, 1.1, float("nan"), True]:
            with self.subTest(weight=weight):
                with self.assertRaises(ValueError):
                    combined_indices(
                        frames, 2, quality_weight=weight
                    )

        with self.assertRaises(ValueError):
            combined_indices(np.zeros((5, 8)), 2)


if __name__ == "__main__":
    unittest.main()
