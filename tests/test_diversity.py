
"""Tests for motion-diversity frame selection."""

import unittest
import numpy as np

from src.sampling.diversity import diversity_indices


class TestDiversitySelection(unittest.TestCase):

    def test_empty_sequence(self):
        frames = np.empty((0, 4, 4), dtype=np.uint8)
        self.assertEqual(diversity_indices(frames, 5), [])

    def test_sequence_shorter_than_budget(self):
        frames = np.zeros((3, 4, 4), dtype=np.uint8)
        self.assertEqual(diversity_indices(frames, 5), [0, 1, 2])

    def test_exact_budget(self):
        frames = np.zeros((5, 4, 4), dtype=np.uint8)
        self.assertEqual(diversity_indices(frames, 5), list(range(5)))

    def test_single_frame_budget(self):
        frames = np.zeros((10, 4, 4), dtype=np.uint8)
        self.assertEqual(diversity_indices(frames, 1), [5])

    def test_prefers_diverse_frames(self):
        frames = np.zeros((5, 4, 4), dtype=np.uint8)
        frames[0] = 255

        indices = diversity_indices(frames, 2)

        self.assertEqual(indices, [0, 2])

    def test_unique_and_sorted(self):
        rng = np.random.default_rng(42)
        frames = rng.integers(
            0, 256, size=(20, 8, 8), dtype=np.uint8
        )

        indices = diversity_indices(frames, 5)

        self.assertEqual(len(indices), 5)
        self.assertEqual(indices, sorted(set(indices)))

    def test_deterministic(self):
        frames = np.arange(
            10 * 4 * 4, dtype=np.float32
        ).reshape(10, 4, 4)

        first = diversity_indices(frames, 4)
        second = diversity_indices(frames, 4)

        self.assertEqual(first, second)

    def test_invalid_inputs(self):
        frames = np.zeros((5, 4, 4), dtype=np.uint8)

        for budget in [0, -1, 2.5, True]:
            with self.subTest(budget=budget):
                with self.assertRaises(ValueError):
                    diversity_indices(frames, budget)

        with self.assertRaises(ValueError):
            diversity_indices(np.zeros((5, 4)), 2)

        with self.assertRaises(ValueError):
            diversity_indices(np.full((5, 4, 4), np.nan), 2)


if __name__ == "__main__":
    unittest.main()
