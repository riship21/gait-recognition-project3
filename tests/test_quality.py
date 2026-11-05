
"""Unit tests for quality-aware silhouette selection."""

import unittest
import numpy as np

from src.sampling.quality import (
    silhouette_quality_scores,
    quality_indices,
)


class TestQualitySelection(unittest.TestCase):

    def test_empty_sequence(self):
        frames = np.empty((0, 8, 8), dtype=np.uint8)
        self.assertEqual(quality_indices(frames, 3), [])

    def test_short_sequence(self):
        frames = np.zeros((3, 8, 8), dtype=np.uint8)
        self.assertEqual(quality_indices(frames, 5), [0, 1, 2])

    def test_blank_frame_scores_zero(self):
        frames = np.zeros((3, 8, 8), dtype=np.uint8)
        frames[1, 2:6, 2:6] = 255

        scores = silhouette_quality_scores(frames)

        self.assertEqual(scores[0], 0)
        self.assertGreater(scores[1], 0)
        self.assertEqual(scores[2], 0)

    def test_prefers_nonempty_frames(self):
        frames = np.zeros((4, 8, 8), dtype=np.uint8)
        frames[1, 2:6, 2:6] = 255
        frames[3, 2:6, 2:6] = 255

        self.assertEqual(quality_indices(frames, 2), [1, 3])

    def test_penalizes_area_outlier(self):
        frames = np.zeros((4, 10, 10), dtype=np.uint8)

        frames[0, 3:7, 3:7] = 255
        frames[1, 3:7, 3:7] = 255
        frames[2, 3:7, 3:7] = 255
        frames[3] = 255

        scores = silhouette_quality_scores(frames)

        self.assertLess(scores[3], scores[0])

    def test_penalizes_position_outlier(self):
        frames = np.zeros((4, 10, 10), dtype=np.uint8)

        frames[0, 4:6, 4:6] = 255
        frames[1, 4:6, 4:6] = 255
        frames[2, 4:6, 4:6] = 255
        frames[3, 0:2, 0:2] = 255

        scores = silhouette_quality_scores(frames)

        self.assertLess(scores[3], scores[0])

    def test_sorted_unique_indices(self):
        rng = np.random.default_rng(42)
        frames = rng.integers(
            0, 2, size=(20, 8, 8), dtype=np.uint8
        )

        indices = quality_indices(frames, 5)

        self.assertEqual(len(indices), 5)
        self.assertEqual(indices, sorted(set(indices)))

    def test_invalid_inputs(self):
        frames = np.zeros((5, 8, 8), dtype=np.uint8)

        for budget in [0, -1, 2.5, True]:
            with self.subTest(budget=budget):
                with self.assertRaises(ValueError):
                    quality_indices(frames, budget)

        with self.assertRaises(ValueError):
            silhouette_quality_scores(np.zeros((5, 8)))

        with self.assertRaises(ValueError):
            silhouette_quality_scores(
                np.full((5, 8, 8), np.nan)
            )


if __name__ == "__main__":
    unittest.main()
