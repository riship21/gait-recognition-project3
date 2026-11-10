
"""Tests for the unified gait frame-selection interface."""

import unittest
import numpy as np

from src.sampling.selector import METHODS, select_frames


class TestFrameSelector(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(42)
        self.frames = rng.integers(
            0, 256, size=(20, 8, 8), dtype=np.uint8
        )

    def test_all_methods(self):
        for method in METHODS:
            with self.subTest(method=method):
                selected, indices = select_frames(
                    self.frames, 5, method
                )

                self.assertEqual(selected.shape, (5, 8, 8))
                self.assertEqual(len(indices), 5)

    def test_preserves_original_frames(self):
        for method in METHODS:
            with self.subTest(method=method):
                selected, indices = select_frames(
                    self.frames, 5, method
                )

                np.testing.assert_array_equal(
                    selected, self.frames[indices]
                )

    def test_chronological_order(self):
        for method in METHODS:
            with self.subTest(method=method):
                _, indices = select_frames(
                    self.frames, 5, method
                )

                self.assertEqual(indices, sorted(indices))

    def test_unique_indices(self):
        for method in METHODS:
            with self.subTest(method=method):
                _, indices = select_frames(
                    self.frames, 5, method
                )

                self.assertEqual(len(indices), len(set(indices)))

    def test_short_sequence(self):
        frames = self.frames[:3]

        for method in METHODS:
            with self.subTest(method=method):
                selected, indices = select_frames(
                    frames, 10, method
                )

                self.assertEqual(selected.shape[0], 3)
                self.assertEqual(indices, [0, 1, 2])

    def test_empty_sequence(self):
        frames = np.empty((0, 8, 8), dtype=np.uint8)

        for method in METHODS:
            with self.subTest(method=method):
                selected, indices = select_frames(
                    frames, 5, method
                )

                self.assertEqual(selected.shape, (0, 8, 8))
                self.assertEqual(indices, [])

    def test_does_not_modify_input(self):
        original = self.frames.copy()

        for method in METHODS:
            select_frames(self.frames, 5, method)

        np.testing.assert_array_equal(self.frames, original)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            select_frames(self.frames, 5, "unknown")

        with self.assertRaises(ValueError):
            select_frames(self.frames, 0, "uniform")

        with self.assertRaises(ValueError):
            select_frames(self.frames[:, :, 0], 5)

        with self.assertRaises(ValueError):
            select_frames(
                np.full((3, 8, 8), np.nan), 2
            )

    def test_adaptive_matches_standalone(self):
        from src.sampling.adaptive import adaptive_indices

        selected, indices = select_frames(
            self.frames, 5, method="adaptive"
        )

        expected = adaptive_indices(self.frames, 5)

        self.assertEqual(indices, expected)
        np.testing.assert_array_equal(
            selected, self.frames[expected]
        )


if __name__ == "__main__":
    unittest.main()
