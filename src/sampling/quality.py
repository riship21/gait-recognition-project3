
"""Quality-aware silhouette frame selection"""

import numpy as np


def silhouette_quality_scores(frames: np.ndarray) -> np.ndarray:
    """
    Assign a quality score to each silhouette.

    Input:
        frames: NumPy array with shape (T, H, W).

    Returns:
        Quality scores with shape (T,), between 0 and 1.

    The score considers foreground presence, area consistency,
    and foreground-center consistency.
    """
    if not isinstance(frames, np.ndarray):
        raise ValueError("frames must be a NumPy array.")

    if frames.ndim != 3:
        raise ValueError("frames must have shape (T, H, W).")

    if not np.issubdtype(frames.dtype, np.number):
        raise ValueError("frames must be numeric.")

    if not np.all(np.isfinite(frames)):
        raise ValueError("frames must contain finite values.")

    total, height, width = frames.shape

    if height == 0 or width == 0:
        raise ValueError("frame dimensions must be nonempty.")

    if total == 0:
        return np.empty(0, dtype=np.float64)

    masks = frames > 0
    areas = masks.sum(axis=(1, 2)).astype(np.float64)
    valid = areas > 0

    scores = np.zeros(total, dtype=np.float64)

    if not np.any(valid):
        return scores

    # Use the sequence's typical foreground area.
    median_area = np.median(areas[valid])

    area_scores = np.exp(
        -np.abs(areas - median_area) / median_area
    )

    # Compute the foreground center for each frame.
    y_coords = np.arange(height, dtype=np.float64)
    x_coords = np.arange(width, dtype=np.float64)

    center_y = (
        masks.sum(axis=2) @ y_coords
    ) / np.maximum(areas, 1)

    center_x = (
        masks.sum(axis=1) @ x_coords
    ) / np.maximum(areas, 1)

    # Compare positions with the sequence's typical center.
    typical_y = np.median(center_y[valid])
    typical_x = np.median(center_x[valid])

    displacement = np.sqrt(
        ((center_y - typical_y) / max(height, 1)) ** 2
        + ((center_x - typical_x) / max(width, 1)) ** 2
    )

    position_scores = np.exp(-4.0 * displacement)

    scores[valid] = (
        area_scores[valid] * position_scores[valid]
    )

    return scores


def quality_indices(frames: np.ndarray, budget: int) -> list[int]:
    """
    Select the highest-quality frames.

    Ties are resolved by the earliest frame index.
    Returned indices are sorted chronologically.
    """
    if type(budget) is not int or budget <= 0:
        raise ValueError("budget must be a positive integer.")

    scores = silhouette_quality_scores(frames)
    total = len(scores)

    if total <= budget:
        return list(range(total))

    ranked = np.lexsort((np.arange(total), -scores))
    selected = ranked[:budget]

    return sorted(int(index) for index in selected)
