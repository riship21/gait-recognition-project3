
"""Combined quality- and diversity-aware frame selection."""

import numpy as np

from src.sampling.quality import silhouette_quality_scores


def combined_indices(
    frames: np.ndarray,
    budget: int,
    quality_weight: float = 0.5,
) -> list[int]:
    """
    Select frames using silhouette quality and visual diversity.

    quality_weight:
        1.0 = quality only
        0.0 = diversity only after quality-based initialization

    Returns sorted, unique frame indices.
    """
    if type(budget) is not int or budget <= 0:
        raise ValueError("budget must be a positive integer.")

    if (
        isinstance(quality_weight, (bool, np.bool_))
        or not isinstance(
            quality_weight, (int, float, np.integer, np.floating)
        )
        or not np.isfinite(quality_weight)
        or not 0 <= quality_weight <= 1
    ):
        raise ValueError("quality_weight must be between 0 and 1.")

    # Also validates array shape, numeric values, and dimensions.
    quality = silhouette_quality_scores(frames)
    total = len(quality)

    if total <= budget:
        return list(range(total))

    vectors = frames.reshape(total, -1).astype(np.float32)

    # Begin with the highest-quality silhouette.
    # NumPy resolves ties by choosing the earliest index.
    first = int(np.argmax(quality))
    selected = [first]

    available = np.ones(total, dtype=bool)
    available[first] = False

    min_distances = np.mean(
        np.abs(vectors - vectors[first]), axis=1
    )

    while len(selected) < budget:
        candidates = np.flatnonzero(available)

        diversity = min_distances[candidates]
        maximum = float(np.max(diversity))

        # Normalize diversity scores into [0, 1].
        if maximum > 0:
            normalized_diversity = diversity / maximum
        else:
            normalized_diversity = np.zeros_like(diversity)

        combined_scores = (
            quality_weight * quality[candidates]
            + (1.0 - quality_weight) * normalized_diversity
        )

        # Earliest frame wins when scores tie.
        next_index = int(
            candidates[np.argmax(combined_scores)]
        )

        selected.append(next_index)
        available[next_index] = False

        distances = np.mean(
            np.abs(vectors - vectors[next_index]), axis=1
        )

        min_distances = np.minimum(
            min_distances, distances
        )

    return sorted(selected)
