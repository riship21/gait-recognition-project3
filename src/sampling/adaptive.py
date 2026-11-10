
"""Adaptive quality-diversity frame selection for Project 3."""

import numpy as np

from src.sampling.quality import silhouette_quality_scores
from src.sampling.combined import combined_indices


def adaptive_quality_weight(frames: np.ndarray) -> float:
    """
    Estimate the quality emphasis using sequence reliability.

    Frames with silhouette-quality scores below 0.5 are
    considered unreliable according to this heuristic.

    The returned weight is between 0.25 and 0.90.
    """
    scores = silhouette_quality_scores(frames)

    if len(scores) == 0:
        return 0.25

    unreliable_fraction = float(np.mean(scores < 0.5))

    return float(0.25 + 0.65 * unreliable_fraction)


def adaptive_indices(
    frames: np.ndarray,
    budget: int,
) -> list[int]:
    """
    Select frames with sequence-adaptive quality weighting.

    Uses the existing combined selector with a weight
    determined from the input sequence.

    Returns chronologically sorted frame indices.
    """
    if type(budget) is not int or budget <= 0:
        raise ValueError("budget must be a positive integer.")

    weight = adaptive_quality_weight(frames)

    return combined_indices(
        frames,
        budget=budget,
        quality_weight=weight,
    )
