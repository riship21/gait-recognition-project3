import unittest

import numpy as np

from src.corruption.occlusion import lower_body_occlusion


class TestLowerBodyOcclusion(unittest.TestCase):

    def make_frames(self, total=10):
        frames = np.zeros(
            (total, 8, 8),
            dtype=np.uint8,
        )

        frames[:, 1:7, 2:6] = 255

        return frames

    def test_input_is_not_modified(self):
        frames = self.make_frames()
        original = frames.copy()

        lower_body_occlusion(
            frames,
            frame_fraction=0.3,
            removal_fraction=0.5,
            seed=1,
        )

        np.testing.assert_array_equal(
            frames,
            original,
        )

    def test_deterministic_for_same_seed(self):
        frames = self.make_frames(20)

        first, first_indices = lower_body_occlusion(
            frames,
            seed=123,
        )

        second, second_indices = lower_body_occlusion(
            frames,
            seed=123,
        )

        self.assertEqual(
            first_indices,
            second_indices,
        )

        np.testing.assert_array_equal(
            first,
            second,
        )

    def test_expected_number_of_frames(self):
        frames = self.make_frames(10)

        _, indices = lower_body_occlusion(
            frames,
            frame_fraction=0.3,
            seed=4,
        )

        self.assertEqual(
            len(indices),
            3,
        )

    def test_only_selected_frames_change(self):
        frames = self.make_frames(10)

        corrupted, selected = lower_body_occlusion(
            frames,
            frame_fraction=0.3,
            removal_fraction=0.5,
            seed=5,
        )

        for index in range(len(frames)):
            if index in selected:
                self.assertFalse(
                    np.array_equal(
                        corrupted[index],
                        frames[index],
                    )
                )
            else:
                np.testing.assert_array_equal(
                    corrupted[index],
                    frames[index],
                )

    def test_lower_half_of_foreground_is_removed(self):
        frames = self.make_frames(1)

        corrupted, selected = lower_body_occlusion(
            frames,
            frame_fraction=1.0,
            removal_fraction=0.5,
            seed=0,
        )

        self.assertEqual(
            selected,
            [0],
        )

        # Original foreground spans rows 1 through 6.
        # Half should remain: rows 1, 2, and 3.
        self.assertTrue(
            np.all(
                corrupted[0, 1:4, 2:6] == 255
            )
        )

        self.assertTrue(
            np.all(
                corrupted[0, 4:, :] == 0
            )
        )

    def test_zero_frame_fraction_is_identity(self):
        frames = self.make_frames()

        corrupted, selected = lower_body_occlusion(
            frames,
            frame_fraction=0.0,
        )

        self.assertEqual(
            selected,
            [],
        )

        np.testing.assert_array_equal(
            corrupted,
            frames,
        )

    def test_invalid_frame_fraction(self):
        frames = self.make_frames()

        with self.assertRaises(ValueError):
            lower_body_occlusion(
                frames,
                frame_fraction=1.1,
            )

    def test_invalid_shape(self):
        frames = np.zeros(
            (8, 8),
            dtype=np.uint8,
        )

        with self.assertRaises(ValueError):
            lower_body_occlusion(frames)


if __name__ == "__main__":
    unittest.main()
