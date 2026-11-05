
"""Unified frame-selection interface for gait recognition."""

import numpy as np

from src.sampling.baselines import first_n_indices, uniform_indices
from src.sampling.diversity import diversity_indices
from src.sampling.quality import quality_indices
from src.sampling.combined import combined_indices


METHODS = ("first_n", "uniform", "diversity", "quality", "combined")


def select_frames(
    frames: np.ndarray,
    budget: int,
    method: str = "first_n",
    quality_weight: float = 0.5,
) -> tuple[np.ndarray, list[int]]:
    """
    Select silhouette frames using one of five methods.

    Args:
        frames: Silhouettes shaped (T, H, W).
        budget: Maximum number of selected frames.
        method: Frame-selection method.
        quality_weight: Combined-method quality weight.

    Returns:
        (selected_frames, selected_indices)

    Selected indices are always chronologically ordered.
    """

    if not isinstance(frames, np.ndarray):
        raise ValueError("frames must be a NumPy array.")

    if frames.ndim != 3:
        raise ValueError("frames must have shape (T, H, W).")

    if frames.shape[1] == 0 or frames.shape[2] == 0:
        raise ValueError("frame dimensions must be nonempty.")

    if not np.issubdtype(frames.dtype, np.number):
        raise ValueError("frames must contain numeric values.")

    if not np.all(np.isfinite(frames)):
        raise ValueError("frames must contain finite values.")

    if type(budget) is not int or budget <= 0:
        raise ValueError("budget must be a positive integer.")

    if method not in METHODS:
        raise ValueError(
            f"Unknown method: {method}. Available: {METHODS}"
        )

    total = frames.shape[0]

    if method == "first_n":
        indices = first_n_indices(total, budget)

    elif method == "uniform":
        indices = uniform_indices(total, budget)

    elif method == "diversity":
        indices = diversity_indices(frames, budget)

    elif method == "quality":
        indices = quality_indices(frames, budget)

    else:
        indices = combined_indices(
            frames, budget, quality_weight=quality_weight
        )

    # Fail if any strategy violates our shared contract.
    if (
        len(indices) != min(total, budget)
        or indices != sorted(set(indices))
        or any(i < 0 or i >= total for i in indices)
    ):
        raise RuntimeError("Frame-selection contract violated.")

    selected_frames = frames[indices]

    return selected_frames, indices
