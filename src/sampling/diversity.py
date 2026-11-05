
"""Motion-diversity frame selection for Project 3."""

import numpy as np


def diversity_indices(frames: np.ndarray, budget: int) -> list[int]:
    """
    Select visually diverse silhouette frames using greedy
    farthest-first selection.

    Args:
        frames: Array shaped (T, H, W).
        budget: Maximum number of frames to select.

    Returns:
        Sorted, unique frame indices.
    """
    if not isinstance(frames, np.ndarray):
        raise ValueError("frames must be a NumPy array.")

    if frames.ndim != 3:
        raise ValueError("frames must have shape (T, H, W).")

    if not np.issubdtype(frames.dtype, np.number):
        raise ValueError("frames must contain numeric values.")

    if not np.all(np.isfinite(frames)):
        raise ValueError("frames must contain only finite values.")

    if any(size == 0 for size in frames.shape[1:]):
        raise ValueError("frame dimensions must be nonempty.")

    if type(budget) is not int or budget <= 0:
        raise ValueError("budget must be a positive integer.")

    total = len(frames)

    if total <= budget:
        return list(range(total))

    # Flatten silhouettes into feature vectors.
    vectors = frames.reshape(total, -1).astype(np.float32)

    # Begin with the middle frame.
    selected = [total // 2]
    available = np.ones(total, dtype=bool)
    available[selected[0]] = False

    # Distance from each frame to its nearest selected frame.
    min_distances = np.mean(
        np.abs(vectors - vectors[selected[0]]),
        axis=1,
    )

    while len(selected) < budget:
        candidates = np.flatnonzero(available)

        # argmax selects the earliest index in a tie.
        next_index = int(
            candidates[np.argmax(min_distances[candidates])]
        )

        selected.append(next_index)
        available[next_index] = False

        distances = np.mean(
            np.abs(vectors - vectors[next_index]),
            axis=1,
        )

        min_distances = np.minimum(
            min_distances, distances
        )

    # Preserve chronological order for GaitPart inference.
    return sorted(selected)
