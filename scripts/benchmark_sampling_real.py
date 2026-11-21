"""Benchmark Project 3 frame selection on real CASIA-B sequences.

Measures CPU frame-selection overhead only.
Dataset loading and GaitPart inference are excluded from timing.

Protocol:
- CASIA-B test subjects 075-124
- NumPy-compatible Project 2 processed silhouettes
- 200 deterministic real sequences
- 15-frame budget
- all six Project 3 selection methods
"""

from __future__ import annotations

import csv
import pickle
import random
import time
from pathlib import Path

import numpy as np

from src.sampling.selector import METHODS, select_frames


ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    ROOT.parent
    / "gait-recognition-project2"
    / "data"
    / "processed"
    / "CASIA-B-pkl-compatible"
)

OUTPUT = ROOT / "results" / "tables" / "sampling_benchmark_real.csv"

SUBJECTS = {f"{i:03d}" for i in range(75, 125)}

BUDGET = 15
SAMPLE_COUNT = 200
SEED = 2026


def load_sequence(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        frames = pickle.load(f)

    frames = np.asarray(frames)

    if frames.ndim != 3:
        raise ValueError(
            f"Expected (T,H,W), got {frames.shape} for {path}"
        )

    return frames


def main():
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(DATA_ROOT)

    candidates = sorted(
        p
        for p in DATA_ROOT.rglob("*.pkl")
        if p.parts[-4] in SUBJECTS
    )

    eligible = []

    for path in candidates:
        frames = load_sequence(path)

        if len(frames) >= BUDGET:
            eligible.append(path)

    if len(eligible) < SAMPLE_COUNT:
        raise RuntimeError(
            f"Only {len(eligible)} eligible sequences found."
        )

    rng = random.Random(SEED)
    selected_paths = sorted(
        rng.sample(eligible, SAMPLE_COUNT)
    )

    # Load before timing so disk I/O is excluded.
    sequences = [
        (path, load_sequence(path))
        for path in selected_paths
    ]

    print("Dataset:", DATA_ROOT)
    print("Eligible sequences:", len(eligible))
    print("Benchmark sequences:", len(sequences))
    print("Budget:", BUDGET)

    frame_counts = np.asarray(
        [len(frames) for _, frames in sequences],
        dtype=float,
    )

    print(
        "Sequence length: "
        f"mean={frame_counts.mean():.2f}, "
        f"median={np.median(frame_counts):.2f}, "
        f"min={frame_counts.min():.0f}, "
        f"max={frame_counts.max():.0f}"
    )

    rows = []

    for method in METHODS:
        # Warm-up outside measurements.
        select_frames(
            sequences[0][1],
            BUDGET,
            method=method,
        )

        durations = []

        for _, frames in sequences:
            start = time.perf_counter()

            selected, indices = select_frames(
                frames,
                BUDGET,
                method=method,
            )

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000.0

            if len(indices) != BUDGET:
                raise RuntimeError(
                    f"{method} returned {len(indices)} frames."
                )

            durations.append(elapsed_ms)

        values = np.asarray(
            durations,
            dtype=float,
        )

        row = {
            "method": method,
            "sequences": len(sequences),
            "budget": BUDGET,
            "mean_sequence_frames": round(
                float(frame_counts.mean()), 4
            ),
            "mean_ms": round(
                float(values.mean()), 4
            ),
            "median_ms": round(
                float(np.median(values)), 4
            ),
            "p95_ms": round(
                float(np.percentile(values, 95)), 4
            ),
            "std_ms": round(
                float(values.std()), 4
            ),
        }

        rows.append(row)

        print(
            f"{method:10s} | "
            f"mean={row['mean_ms']:8.4f} ms | "
            f"median={row['median_ms']:8.4f} ms | "
            f"p95={row['p95_ms']:8.4f} ms"
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("PASS: real-sequence benchmark completed.")
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
