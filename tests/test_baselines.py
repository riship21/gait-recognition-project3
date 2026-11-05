
"""Unit tests for Project 3 baseline frame-selection methods."""

import unittest

from src.sampling.baselines import (
    first_n_indices,
    uniform_indices,
)


class TestFrameSelection(unittest.TestCase):

    def test_first_n(self):
        self.assertEqual(first_n_indices(100, 5), [0, 1, 2, 3, 4])

    def test_uniform(self):
        self.assertEqual(
            uniform_indices(100, 5),
            [0, 24, 49, 74, 99],
        )

    def test_sequence_shorter_than_budget(self):
        expected = [0, 1, 2]
        self.assertEqual(first_n_indices(3, 10), expected)
        self.assertEqual(uniform_indices(3, 10), expected)

    def test_exact_budget(self):
        self.assertEqual(uniform_indices(5, 5), [0, 1, 2, 3, 4])

    def test_single_frame_budget(self):
        self.assertEqual(first_n_indices(10, 1), [0])
        self.assertEqual(uniform_indices(10, 1), [5])

    def test_empty_sequence(self):
        self.assertEqual(first_n_indices(0, 5), [])
        self.assertEqual(uniform_indices(0, 5), [])

    def test_invalid_inputs(self):
        invalid = [
            (-1, 5),
            (10, 0),
            (10, -1),
            (10.5, 5),
            (10, True),
        ]

        for total, budget in invalid:
            with self.subTest(total=total, budget=budget):
                with self.assertRaises(ValueError):
                    first_n_indices(total, budget)

                with self.assertRaises(ValueError):
                    uniform_indices(total, budget)

    def test_no_duplicate_indices(self):
        for total in [1, 5, 15, 50, 100]:
            for budget in [1, 3, 10, 30]:
                indices = uniform_indices(total, budget)

                self.assertEqual(len(indices), len(set(indices)))
                self.assertEqual(indices, sorted(indices))
                self.assertEqual(len(indices), min(total, budget))


if __name__ == "__main__":
    unittest.main()
