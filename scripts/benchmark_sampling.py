
"""Benchmark frame-selection overhead using synthetic silhouettes.

Run from the Project 3 repository root:
    python -m scripts.benchmark_sampling

This benchmark measures CPU selection time, not GaitPart inference.
"""

import csv
import time
from pathlib import Path

import numpy as np

from src.sampling.selector import METHODS, select_frames


OUTPUT = Path("results/tables/sampling_benchmark.csv")

FRAME_COUNTS = (50, 100, 200)
BUDGETS = (15, 30, 50)
REPEATS = 10
SEED = 42


def make_synthetic_sequence(total: int) -> np.ndarray:
    """Generate a reproducible sequence of moving binary shapes."""
    frames = np.zeros((total, 64, 64), dtype=np.uint8)

    for index in range(total):
        phase = 2 * np.pi * index / 20.0
        offset = int(round(8 * np.sin(phase)))

        # Simplified moving torso and legs.
        frames[index, 10:38, 24 + offset:40 + offset] = 255
        frames[index, 38:58, 22 + offset:30 + offset] = 255
        frames[index, 38:58, 34 - offset:42 - offset] = 255

    return frames


def benchmark():
    np.random.seed(SEED)
    rows = []

    for total in FRAME_COUNTS:
        frames = make_synthetic_sequence(total)

        for budget in BUDGETS:
            for method in METHODS:

                # Warm-up outside the timed measurements.
                select_frames(frames, budget, method=method)

                durations = []

                for _ in range(REPEATS):
                    start = time.perf_counter()

                    selected, indices = select_frames(
                        frames,
                        budget,
                        method=method,
                    )

                    elapsed = (
                        time.perf_counter() - start
                    ) * 1000.0

                    durations.append(elapsed)

                rows.append({
                    "method": method,
                    "total_frames": total,
                    "budget": budget,
                    "selected_frames": len(indices),
                    "mean_ms": round(
                        float(np.mean(durations)), 4
                    ),
                    "median_ms": round(
                        float(np.median(durations)), 4
                    ),
                    "std_ms": round(
                        float(np.std(durations)), 4
                    ),
                })

                print(
                    f"{method:10s} | T={total:3d} "
                    f"| N={budget:2d} "
                    f"| median={np.median(durations):8.4f} ms"
                )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} benchmark rows to {OUTPUT}")


if __name__ == "__main__":
    benchmark()
