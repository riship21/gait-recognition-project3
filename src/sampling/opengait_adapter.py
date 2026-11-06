
"""Adapter between OpenGait sequences and Project 3 frame selection.

This module does not import or modify OpenGait.
"""

import numpy as np

from src.sampling.selector import select_frames


def select_opengait_indices(
    seqs,
    budget: int,
    method: str = "first_n",
    quality_weight: float = 0.5,
) -> list[int]:
    """
    Select frame indices for OpenGait's CollateFn.

    Args:
        seqs: OpenGait sequence container. For CASIA-B,
              seqs[0] contains the silhouette frames.
        budget: Maximum number of frames to select.
        method: Project 3 sampling method.
        quality_weight: Weight for combined selection.

    Returns:
        Chronologically sorted frame indices.

    The original OpenGait sequence is never modified.
    """

    if not isinstance(seqs, (list, tuple)) or len(seqs) != 1:
        raise ValueError(
            "Expected one OpenGait silhouette feature stream."
        )

    frames = np.asarray(seqs[0])

    if frames.ndim != 3:
        raise ValueError(
            "Silhouette sequence must have shape (T, H, W)."
        )

    if frames.shape[0] == 0:
        raise ValueError(
            "Cannot run GaitPart inference on an empty sequence."
        )

    _, indices = select_frames(
        frames,
        budget=budget,
        method=method,
        quality_weight=quality_weight,
    )

    return indices
