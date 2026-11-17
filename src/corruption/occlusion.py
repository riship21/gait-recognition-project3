"""Controlled silhouette occlusion for Project 3 experiments."""

from __future__ import annotations

import numpy as np


def lower_body_occlusion(
    frames: np.ndarray,
    frame_fraction: float = 0.30,
    removal_fraction: float = 0.50,
    seed: int = 2026,
) -> tuple[np.ndarray, list[int]]:
    """Remove the lower portion of the silhouette in selected frames.

    Parameters
    ----------
    frames:
        Numeric silhouette array with shape (T, H, W).

    frame_fraction:
        Fraction of temporal frames to corrupt.

    removal_fraction:
        Fraction of each selected silhouette's vertical foreground
        extent to remove from the bottom.

    seed:
        Random seed controlling which temporal frames are corrupted.

    Returns
    -------
    corrupted:
        Copy of the input sequence containing the controlled corruption.

    indices:
        Sorted temporal indices that were selected for corruption.
    """
    frames = np.asarray(frames)

    if frames.ndim != 3:
        raise ValueError(
            f"Expected frames with shape (T, H, W), got {frames.shape}."
        )

    if not np.issubdtype(frames.dtype, np.number):
        raise TypeError("Frames must contain numeric values.")

    if not np.all(np.isfinite(frames)):
        raise ValueError("Frames must contain only finite values.")

    if not 0.0 <= frame_fraction <= 1.0:
        raise ValueError("frame_fraction must be between 0 and 1.")

    if not 0.0 <= removal_fraction <= 1.0:
        raise ValueError("removal_fraction must be between 0 and 1.")

    total_frames = len(frames)
    corrupted = frames.copy()

    if (
        total_frames == 0
        or frame_fraction == 0.0
        or removal_fraction == 0.0
    ):
        return corrupted, []

    count = int(
        np.floor(total_frames * frame_fraction + 0.5)
    )

    count = max(1, min(total_frames, count))

    rng = np.random.default_rng(seed)

    selected = np.sort(
        rng.choice(
            total_frames,
            size=count,
            replace=False,
        )
    ).tolist()

    for index in selected:
        frame = corrupted[index]

        foreground_y = np.flatnonzero(
            np.any(frame > 0, axis=1)
        )

        # Blank silhouettes remain blank.
        if len(foreground_y) == 0:
            continue

        top = int(foreground_y[0])
        bottom = int(foreground_y[-1])

        height = bottom - top + 1

        keep_fraction = 1.0 - removal_fraction

        keep_rows = int(
            np.ceil(height * keep_fraction)
        )

        cutoff = top + keep_rows

        frame[cutoff:, :] = 0

    return corrupted, selected
