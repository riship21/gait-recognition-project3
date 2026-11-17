"""
Audit adaptive quality weights on the CASIA-B test partition.

"""

import csv
import pickle
from pathlib import Path

import numpy as np

from src.sampling.adaptive import (
    adaptive_indices,
    adaptive_quality_weight,
)
from src.sampling.combined import combined_indices
from src.sampling.quality import silhouette_quality_scores


PROJECT2 = (
    Path.home()
    / "COSC594"
    / "gait-recognition-project2"
)

DATASET = (
    PROJECT2
    / "data"
    / "processed"
    / "CASIA-B-pkl-compatible"
)

OUTPUT = Path("results/tables/adaptive_weight_audit.csv")

BUDGET = 15
TEST_SUBJECTS = {f"{i:03d}" for i in range(75, 125)}
FIXED_WEIGHTS = (0.25, 0.5, 0.9)


def nearest_fixed(weight: float) -> float:
    """Return the nearest comparison weight."""
    return min(
        FIXED_WEIGHTS,
        key=lambda candidate: abs(weight - candidate),
    )


def summarize(rows, label):
    """Print summary statistics for a group of rows."""
    if not rows:
        return

    weights = np.array(
        [row["adaptive_weight"] for row in rows],
        dtype=np.float64,
    )

    low_fraction = np.array(
        [row["low_quality_fraction"] for row in rows],
        dtype=np.float64,
    )

    agreement_025 = np.mean(
        [row["equals_fixed_025"] for row in rows]
    )

    agreement_050 = np.mean(
        [row["equals_fixed_050"] for row in rows]
    )

    agreement_090 = np.mean(
        [row["equals_fixed_090"] for row in rows]
    )

    nearest_counts = {
        weight: sum(
            row["nearest_fixed_weight"] == weight
            for row in rows
        )
        for weight in FIXED_WEIGHTS
    }

    print(f"\n=== {label} ===")
    print("Sequences:", len(rows))

    print(
        "Adaptive weight:"
        f" mean={weights.mean():.4f}"
        f" median={np.median(weights):.4f}"
        f" min={weights.min():.4f}"
        f" max={weights.max():.4f}"
    )

    print(
        "Low-quality frame fraction:"
        f" mean={low_fraction.mean():.4f}"
        f" median={np.median(low_fraction):.4f}"
    )

    print("Nearest fixed weight:")
    for weight in FIXED_WEIGHTS:
        count = nearest_counts[weight]
        percent = 100 * count / len(rows)

        print(
            f"  {weight:.2f}: "
            f"{count} sequences ({percent:.2f}%)"
        )

    print(
        "Exact frame-selection agreement:"
    )

    print(
        f"  adaptive == fixed 0.25: "
        f"{100 * agreement_025:.2f}%"
    )

    print(
        f"  adaptive == fixed 0.50: "
        f"{100 * agreement_050:.2f}%"
    )

    print(
        f"  adaptive == fixed 0.90: "
        f"{100 * agreement_090:.2f}%"
    )


def main():
    if not DATASET.is_dir():
        raise FileNotFoundError(DATASET)

    rows = []

    paths = sorted(DATASET.rglob("*.pkl"))

    for path in paths:
        relative = path.relative_to(DATASET)

        if len(relative.parts) < 4:
            continue

        subject = relative.parts[0]

        if subject not in TEST_SUBJECTS:
            continue

        sequence = relative.parts[1]
        view = relative.parts[2]

        condition = sequence.split("-", 1)[0].upper()

        if condition not in {"NM", "BG", "CL"}:
            continue

        with path.open("rb") as file:
            frames = pickle.load(file)

        frames = np.asarray(frames)

        if frames.ndim != 3:
            raise RuntimeError(
                f"Unexpected shape {frames.shape}: {path}"
            )

        if len(frames) == 0:
            continue

        scores = silhouette_quality_scores(frames)

        low_quality_fraction = float(
            np.mean(scores < 0.5)
        )

        weight = adaptive_quality_weight(frames)

        adaptive = adaptive_indices(
            frames,
            BUDGET,
        )

        fixed_025 = combined_indices(
            frames,
            BUDGET,
            quality_weight=0.25,
        )

        fixed_050 = combined_indices(
            frames,
            BUDGET,
            quality_weight=0.5,
        )

        fixed_090 = combined_indices(
            frames,
            BUDGET,
            quality_weight=0.9,
        )

        rows.append({
            "subject": subject,
            "sequence": sequence,
            "condition": condition,
            "view": view,
            "frames": len(frames),
            "low_quality_fraction": low_quality_fraction,
            "adaptive_weight": weight,
            "nearest_fixed_weight": nearest_fixed(weight),
            "equals_fixed_025": adaptive == fixed_025,
            "equals_fixed_050": adaptive == fixed_050,
            "equals_fixed_090": adaptive == fixed_090,
        })

    if not rows:
        raise RuntimeError(
            "No test sequences were found."
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)

    summarize(rows, "ALL TEST SEQUENCES")

    for condition in ("NM", "BG", "CL"):
        summarize(
            [
                row
                for row in rows
                if row["condition"] == condition
            ],
            condition,
        )

    print(
        f"\nSaved {len(rows)} rows to {OUTPUT}"
    )


if __name__ == "__main__":
    main()