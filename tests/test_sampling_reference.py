"""Hand-calculated numerical checks for Project 3 sampling.

Expected values are literal indices or scalar mathematical expressions.
No production selector is used to generate another selector's expected
answer. These checks validate the specified algorithms, not recognition
accuracy or whether the quality heuristic measures biometric usefulness.

"""

from math import exp
import unittest

import numpy as np

from src.sampling.baselines import first_n_indices, uniform_indices
from src.sampling.diversity import diversity_indices
from src.sampling.quality import silhouette_quality_scores, quality_indices
from src.sampling.combined import combined_indices
from src.sampling.adaptive import adaptive_quality_weight, adaptive_indices


def area_reference_frames():
    """Create four 4x4 masks with foreground areas 2, 4, 8, and 0.

    Every nonempty mask has center (1.5, 1.5).
    The median nonempty area is 4, so the quality scores must be:
        [exp(-0.5), 1, exp(-1), 0].
    """
    frames = np.zeros((4, 4, 4), dtype=np.uint8)
    frames[0, 1, 1] = 255
    frames[0, 2, 2] = 255
    frames[1, 1:3, 1:3] = 255
    frames[2, 1:3, :] = 255
    return frames


class TestSamplingReference(unittest.TestCase):

    def test_first_n_literal_indices(self):
        self.assertEqual(first_n_indices(9, 4), [0, 1, 2, 3])
        self.assertEqual(first_n_indices(3, 8), [0, 1, 2])

    def test_uniform_literal_indices(self):
        # floor(i * 8 / 3), for i = 0, 1, 2, 3.
        self.assertEqual(uniform_indices(9, 4), [0, 2, 5, 8])
        self.assertEqual(uniform_indices(9, 1), [4])

    def test_quality_area_and_blank_reference(self):
        frames = area_reference_frames()
        expected = [exp(-0.5), 1.0, exp(-1.0), 0.0]

        np.testing.assert_allclose(
            silhouette_quality_scores(frames),
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_quality_position_reference(self):
        frames = np.zeros((3, 8, 8), dtype=np.uint8)
        frames[0, 2:4, 2:4] = 255
        frames[1, 2:4, 2:4] = 255
        frames[2, 4:6, 2:4] = 255

        # All areas are 4. The last center is two rows away
        # from the median center. Displacement = 2 / 8.
        # Position score = exp(-4 * 2 / 8) = exp(-1).
        expected = [1.0, 1.0, exp(-1.0)]

        np.testing.assert_allclose(
            silhouette_quality_scores(frames),
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_quality_ranking_reference(self):
        frames = area_reference_frames()

        # Quality ranking: frame 1 > frame 0 > frame 2 > frame 3.
        # Returned indices must be chronological, not quality-ranked.
        self.assertEqual(quality_indices(frames, 1), [1])
        self.assertEqual(quality_indices(frames, 2), [0, 1])
        self.assertEqual(quality_indices(frames, 3), [0, 1, 2])

    def test_diversity_maximin_reference(self):
        frames = np.zeros((5, 1, 16), dtype=np.uint8)

        for index, area in enumerate((0, 1, 5, 8, 10)):
            frames[index, 0, :area] = 255

        # Nested masks:
        # distance = abs(area_a - area_b) * 255 / 16.
        #
        # Start at frame 2 (area 5).
        # Select frame 0 (area 0), winning the tie with frame 4.
        # Select frame 4 (area 10).
        #
        # Remaining nearest-selected distances, before scaling:
        # frame 1 -> 1; frame 3 -> 2. Therefore choose frame 3.
        self.assertEqual(diversity_indices(frames, 2), [0, 2])
        self.assertEqual(diversity_indices(frames, 3), [0, 2, 4])
        self.assertEqual(diversity_indices(frames, 4), [0, 2, 3, 4])

    def test_diversity_tie_reference(self):
        frames = np.zeros((6, 4, 4), dtype=np.uint8)
        frames[:, 1:3, 1:3] = 255

        # Identical frames: all distances are zero.
        # Seed = index 3; earliest remaining indices = 0, then 1.
        self.assertEqual(diversity_indices(frames, 3), [0, 1, 3])

    def test_combined_fixed_weight_reference(self):
        frames = area_reference_frames()

        # Frame 1 is the highest-quality seed.
        # For candidates 0, 2, 3:
        # Q = [exp(-0.5), exp(-1), 0]
        # normalized D = [0.5, 1, 1]
        # S = alpha * Q + (1 - alpha) * D.
        #
        # The second choice is frame 2 at alpha 0.25 or 0.5,
        # and frame 0 at alpha 0.9 or 1.0.
        cases = (
            (0.25, [1, 2]),
            (0.5, [1, 2]),
            (0.9, [0, 1]),
            (1.0, [0, 1]),
        )

        for weight, expected in cases:
            with self.subTest(weight=weight):
                self.assertEqual(
                    combined_indices(
                        frames,
                        2,
                        quality_weight=weight,
                    ),
                    expected,
                )

    def test_zero_weight_still_has_quality_initialization(self):
        frames = area_reference_frames()

        # Combined starts at frame 1 even when alpha = 0.
        # Standalone diversity starts at middle frame 2.
        # These are intentionally different algorithms.
        self.assertEqual(
            combined_indices(frames, 2, quality_weight=0.0),
            [1, 2],
        )

        self.assertEqual(
            diversity_indices(frames, 2),
            [2, 3],
        )

    def test_adaptive_weight_and_selection_reference(self):
        frames = area_reference_frames()

        # Q = [exp(-0.5), 1, exp(-1), 0].
        # Two of four scores are below 0.5:
        # alpha = 0.25 + 0.65 * (2/4) = 0.575.
        # At this weight, candidate 2 beats candidates 0 and 3.
        self.assertAlmostEqual(
            adaptive_quality_weight(frames),
            0.575,
            places=12,
        )

        self.assertEqual(
            adaptive_indices(frames, 2),
            [1, 2],
        )

    def test_adaptive_reference_with_more_blank_frames(self):
        frames = np.concatenate(
            [
                area_reference_frames(),
                np.zeros((4, 4, 4), dtype=np.uint8),
            ],
            axis=0,
        )

        # Nonempty quality scores are unchanged.
        # Six of eight scores are below 0.5:
        # alpha = 0.25 + 0.65 * (6/8) = 0.7375.
        #
        # Candidate 0 score is approximately 0.57857.
        # Candidate 2 score is approximately 0.53381.
        # Thus adaptive selects frame 0 after seed 1.
        self.assertAlmostEqual(
            adaptive_quality_weight(frames),
            0.7375,
            places=12,
        )

        self.assertEqual(
            adaptive_indices(frames, 2),
            [0, 1],
        )

        self.assertEqual(
            combined_indices(frames, 2, quality_weight=0.5),
            [1, 2],
        )

    def test_consistent_damage_can_receive_high_quality(self):
        original = np.zeros((5, 4, 4), dtype=np.uint8)
        original[:, 1:3, 1:3] = 255

        damaged = original.copy()
        damaged[:, 2, :] = 0

        # Half the foreground is removed identically from every frame.
        # Each sequence is still internally consistent, so all scores
        # equal 1. This documents a limitation, not genuine high quality.
        for frames in (original, damaged):
            np.testing.assert_allclose(
                silhouette_quality_scores(frames),
                [1.0, 1.0, 1.0, 1.0, 1.0],
                rtol=1e-12,
                atol=1e-12,
            )

            self.assertAlmostEqual(
                adaptive_quality_weight(frames),
                0.25,
                places=12,
            )


if __name__ == "__main__":
    unittest.main()