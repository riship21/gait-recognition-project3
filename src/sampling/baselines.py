
"""Baseline frame-selection methods for Project 3."""

from typing import List


def _validate_inputs(total_frames: int, budget: int) -> None:
    """Validate sequence length and requested frame budget."""
    if type(total_frames) is not int or total_frames < 0:
        raise ValueError("total_frames must be a non-negative integer.")

    if type(budget) is not int or budget <= 0:
        raise ValueError("budget must be a positive integer.")


def first_n_indices(total_frames: int, budget: int) -> List[int]:
    """
    Select the first N available frames.

    If the sequence contains fewer frames than the budget,
    all available frames are selected.
    """
    _validate_inputs(total_frames, budget)

    return list(range(min(total_frames, budget)))


def uniform_indices(total_frames: int, budget: int) -> List[int]:
    """
    Select frames uniformly across the complete sequence.

    The selected indices are ordered and contain no duplicates.
    If fewer frames are available than requested, use all of them.
    """
    _validate_inputs(total_frames, budget)

    if total_frames <= budget:
        return list(range(total_frames))

    if budget == 1:
        return [total_frames // 2]

    return [
        (i * (total_frames - 1)) // (budget - 1)
        for i in range(budget)
    ]
